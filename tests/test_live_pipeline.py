"""End-to-end orchestration test for the live-capture pipeline (#124, reordered
by #139 AC1).

Wires now-playing -> capture -> chromaprint -> identity waterfall -> inference
-> local store, entirely against injected fakes (this codebase's Protocol+fake
idiom). The real WASAPI/SMTC/fpcalc/onnxruntime backends are exercised in the
live smoke session with the user, not here.

AC1: capture must run *before* identity resolution (chromaprint is computed
from the PCM this process just captured) — a shared ``events`` list records
call order across the fakes to pin that down, not just the end result.
"""

from __future__ import annotations

import json

import numpy as np

from music_intel_mcp.capture import AudioFrame, FakeLoopbackCapture
from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.live_identity import (
    AcoustIdMatch,
    InMemoryAcoustIdSource,
    InMemoryMusicBrainzNameSearchSource,
    InMemoryYoutubeHistoryIndex,
    LiveIdentityResolver,
)
from music_intel_mcp.live_pipeline import (
    live_capture_journal_path,
    run_live_capture_spike,
    youtube_near_miss_journal_path,
)
from music_intel_mcp.nowplaying import InMemoryNowPlayingSource, NowPlayingInfo
from music_intel_mcp.store import UserStore, load_aliases


def _tone_frame(n: int, *, sample_rate: int = 16000, channels: int = 1, amplitude: float = 0.1):
    t = np.arange(n) / sample_rate
    tone = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    samples = np.repeat(tone.reshape(-1, 1), channels, axis=1)
    return AudioFrame(samples=samples, sample_rate=sample_rate)


class _ScriptedCapture:
    """Returns a single fixed frame from ``read()`` regardless of the
    requested duration — lets a test hand the organic path a silent or
    short buffer directly, the same fake-sink idiom ``test_replay_capture.py``
    uses for #166's gate."""

    def __init__(self, frame: AudioFrame) -> None:
        self._frame = frame

    def start(self) -> None:
        pass

    def read(self, duration_s: float) -> AudioFrame:
        return self._frame

    def stop(self) -> None:
        pass


def _fake_fingerprint_fn(events: list[str]):
    def fn(pcm, sample_rate):
        events.append("fingerprint")
        return "fp-fake", 0.25

    return fn


def test_run_live_capture_spike_captures_before_resolving_identity(tmp_path) -> None:
    """AC1: capture starts immediately and identity resolution happens after,
    using a fingerprint computed from the captured PCM."""
    events: list[str] = []

    class _TrackingCapture(FakeLoopbackCapture):
        def start(self):
            events.append("capture_start")
            super().start()

        def read(self, duration_s):
            events.append("capture_read")
            return super().read(duration_s)

    class _TrackingAcoustId(InMemoryAcoustIdSource):
        def match(self, fingerprint, duration_s):
            events.append("identity_resolve")
            return super().match(fingerprint, duration_s)

    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Around the World", artist="Daft Punk", app_id="Spotify.exe")
    )
    acoustid = _TrackingAcoustId({"fp-fake": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    live_resolver = LiveIdentityResolver(acoustid_source=acoustid)
    capture = _TrackingCapture(sample_rate=16000, channels=1)
    embedding_model = InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32))
    classifier = InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9}))
    store = UserStore(root=tmp_path)

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=capture,
        embedding_model=embedding_model,
        classifier=classifier,
        store=store,
        fingerprint_fn=_fake_fingerprint_fn(events),
    )

    assert events == ["capture_start", "capture_read", "fingerprint", "identity_resolve"]
    assert result is not None
    assert result.identity.mbid == "M-1"
    assert result.identity.level == "acoustid"
    assert result.inference.tags["genre---electronic"] == 0.9
    assert result.analysis_path.exists()
    assert result.analysis_path.is_relative_to(tmp_path)

    import json

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["raw_title"] == "Around the World"
    assert payload["provenance"]["chromaprint_fingerprint"] == "fp-fake"


def test_run_live_capture_spike_falls_through_when_fingerprinting_fails(tmp_path) -> None:
    """fpcalc missing/erroring must not kill the pipeline — the AcoustID rung
    is simply skipped and the waterfall falls through to the string chain."""

    def _failing_fingerprint_fn(pcm, sample_rate):
        raise RuntimeError("fpcalc not found")

    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Some Song", artist="Some Artist", app_id="Spotify.exe")
    )
    live_resolver = LiveIdentityResolver()  # no sources -> bottoms out at name key
    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=UserStore(root=tmp_path),
        fingerprint_fn=_failing_fingerprint_fn,
    )

    assert result is not None
    assert result.identity.level == "name"


def test_run_live_capture_spike_skips_inference_when_already_analyzed(tmp_path) -> None:
    """#126 AC1: the store is checked via the identity waterfall before
    analysis; a track that already has a stored embedding is skipped — no
    re-inference (embedding_model/classifier must not be called)."""
    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Around the World", artist="Daft Punk", app_id="Spotify.exe")
    )
    acoustid = InMemoryAcoustIdSource({"fp-fake": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    live_resolver = LiveIdentityResolver(acoustid_source=acoustid)
    embedding_model = InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32))
    classifier = InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9}))
    store = UserStore(root=tmp_path)
    store.write_audio_analysis(track_id="mbid:M-1", embedding=[0.5], tags={"genre---rock": 1.0})

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=embedding_model,
        classifier=classifier,
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
    )

    assert embedding_model.calls == 0
    assert classifier.calls == 0
    assert result is not None
    assert result.skipped is True
    assert result.identity.mbid == "M-1"
    assert result.inference is None


def test_run_live_capture_spike_key_recognized_by_backfill_selector(tmp_path) -> None:
    """#158 AC1: one function produces the key everywhere — a track captured
    live must be reported as already-analyzed by the backfill selector, i.e.
    the key ``run_live_capture_spike`` writes under and the key
    ``select_backfill_tracks`` computes from a candidate ``TrackRef`` for the
    same identity must be identical."""
    from music_intel_mcp.backfill_playlist import select_backfill_tracks
    from music_intel_mcp.models import TrackRef

    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Around the World", artist="Daft Punk", app_id="Spotify.exe")
    )
    acoustid = InMemoryAcoustIdSource({"fp-fake": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    live_resolver = LiveIdentityResolver(acoustid_source=acoustid)
    store = UserStore(root=tmp_path)

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
    )
    assert result is not None
    assert result.skipped is False

    candidate = TrackRef(mbid="M-1", name="Around the World", artist="Daft Punk")
    selected = select_backfill_tracks(
        [candidate], played_ids=set(), has_audio_analysis=store.has_audio_analysis
    )
    assert selected == []


def test_run_live_capture_spike_name_level_key_uses_normalized_name(tmp_path) -> None:
    """#158 AC1 must not regress the #139 AC4 normalization invariant
    (CONTEXT.md 'Normalization (AC4)'): when the waterfall bottoms out at the
    name rung, the stored key has to be built from the *normalized* name_key
    (feat./official-video/lyrics/remaster noise stripped), not the raw OS
    media-session title -- otherwise two plays of the same track with a
    cosmetically different title (e.g. an "(Official Video)" suffix) get
    different keys and are re-analyzed instead of deduped."""
    store = UserStore(root=tmp_path)
    live_resolver = LiveIdentityResolver()  # no sources -> always bottoms out at name key

    first = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=InMemoryNowPlayingSource(
            NowPlayingInfo(title="Strobe (Official Video)", artist="deadmau5", app_id="Spotify.exe")
        ),
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
    )
    assert first is not None
    assert first.skipped is False

    second = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=InMemoryNowPlayingSource(
            NowPlayingInfo(title="Strobe", artist="deadmau5", app_id="Spotify.exe")
        ),
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
    )
    assert second is not None
    assert second.skipped is True
    assert second.analysis_path == first.analysis_path


def test_run_live_capture_spike_writes_raw_fingerprint_sidecar(tmp_path) -> None:
    """#140 AC1: a *second*, separate fpcalc call (raw uint32 array, evidence-
    only) runs against the same captured PCM as the existing compressed-
    string fingerprint call ("one temp wav, two calls"), and is persisted to
    the ``fingerprints/<key>.json`` sidecar UserStore now exposes -- never
    into the AudioAnalysisRecord schema itself."""
    events: list[str] = []

    def _raw_fingerprint_fn(pcm, sample_rate):
        events.append("raw_fingerprint")
        return [1, 2, 3], 0.25

    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Around the World", artist="Daft Punk", app_id="Spotify.exe")
    )
    acoustid = InMemoryAcoustIdSource({"fp-fake": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    live_resolver = LiveIdentityResolver(acoustid_source=acoustid)
    store = UserStore(root=tmp_path)

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn(events),
        raw_fingerprint_fn=_raw_fingerprint_fn,
    )

    assert result is not None
    assert events == ["fingerprint", "raw_fingerprint"]
    assert store.read_fingerprint("mbid:M-1") == [1, 2, 3]


def test_run_live_capture_spike_raw_fingerprint_failure_is_non_fatal(tmp_path) -> None:
    """Mirrors the existing compressed-fingerprint fallback (fpcalc missing
    must not kill capture): a raw-fingerprint failure just skips the sidecar."""

    def _failing_raw_fingerprint_fn(pcm, sample_rate):
        raise RuntimeError("fpcalc not found")

    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Around the World", artist="Daft Punk", app_id="Spotify.exe")
    )
    acoustid = InMemoryAcoustIdSource({"fp-fake": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    live_resolver = LiveIdentityResolver(acoustid_source=acoustid)
    store = UserStore(root=tmp_path)

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
        raw_fingerprint_fn=_failing_raw_fingerprint_fn,
    )

    assert result is not None
    assert store.read_fingerprint("mbid:M-1") is None


def test_run_live_capture_spike_skips_raw_fingerprint_when_already_analyzed(tmp_path) -> None:
    """No point re-fingerprinting a track that was deduped -- no fresh PCM
    worth attaching evidence to, mirrors the inference-skip behavior."""

    def _raw_fingerprint_fn(pcm, sample_rate):
        raise AssertionError("must not be called when dedup skips the capture")

    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Around the World", artist="Daft Punk", app_id="Spotify.exe")
    )
    acoustid = InMemoryAcoustIdSource({"fp-fake": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    live_resolver = LiveIdentityResolver(acoustid_source=acoustid)
    store = UserStore(root=tmp_path)
    store.write_audio_analysis(track_id="mbid:M-1", embedding=[0.5], tags={"genre---rock": 1.0})

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
        raw_fingerprint_fn=_raw_fingerprint_fn,
    )

    assert result is not None
    assert result.skipped is True


def test_run_live_capture_spike_discards_silent_capture(tmp_path) -> None:
    """#179 AC1/AC2: a capture whose RMS is below the shared #166 threshold
    writes no audio-analysis file and is journaled with reason ``silent``."""
    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Ad Break", artist="Unknown", app_id="Spotify.exe")
    )
    store = UserStore(root=tmp_path)
    journal_path = live_capture_journal_path(store)

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=LiveIdentityResolver(),
        capture=_ScriptedCapture(_tone_frame(800, amplitude=0.0001)),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
        journal_path=journal_path,
    )

    assert result is not None
    assert result.outcome == "silent"
    assert result.analysis_path is None
    assert result.inference is None
    assert list(store.audio_analysis_dir.glob("*.json")) == []

    entries = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["outcome"] == "silent"
    assert entries[0]["reason"] is not None


def test_run_live_capture_spike_discards_short_capture(tmp_path) -> None:
    """#179 AC1/AC2: a capture shorter than the requested window writes no
    audio-analysis file and is journaled with reason ``short``, even though
    the buffer isn't silent."""
    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Cut Off", artist="Unknown", app_id="Spotify.exe")
    )
    store = UserStore(root=tmp_path)
    journal_path = live_capture_journal_path(store)

    result = run_live_capture_spike(
        duration_s=0.5,
        now_playing_source=now_playing,
        live_identity_resolver=LiveIdentityResolver(),
        capture=_ScriptedCapture(_tone_frame(800)),  # 0.05s of audio, window is 0.5s
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
        journal_path=journal_path,
    )

    assert result is not None
    assert result.outcome == "short"
    assert result.analysis_path is None
    assert result.inference is None
    assert list(store.audio_analysis_dir.glob("*.json")) == []

    entries = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["outcome"] == "short"
    assert entries[0]["reason"] is not None


def test_run_live_capture_spike_gate_uses_shared_replay_threshold(tmp_path) -> None:
    """#179 AC3: the RMS threshold is imported from #166's ``replay_capture``
    module, not redefined here — proven by monkeypatching the shared constant
    via the default argument and observing the gate's behavior change."""
    from music_intel_mcp import replay_capture

    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Quiet But Not That Quiet", artist="Unknown", app_id="Spotify.exe")
    )
    store = UserStore(root=tmp_path)

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=LiveIdentityResolver(),
        capture=_ScriptedCapture(_tone_frame(800, amplitude=0.02)),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
        rms_threshold=replay_capture.RMS_SILENCE_THRESHOLD,
    )
    assert result is not None
    assert result.outcome != "silent"

    result_stricter = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=LiveIdentityResolver(),
        capture=_ScriptedCapture(_tone_frame(800, amplitude=0.02)),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=UserStore(root=tmp_path / "second"),
        fingerprint_fn=_fake_fingerprint_fn([]),
        rms_threshold=0.03,
    )
    assert result_stricter is not None
    assert result_stricter.outcome == "silent"


def test_run_live_capture_spike_none_when_nothing_playing(tmp_path) -> None:
    result = run_live_capture_spike(
        duration_s=0.1,
        now_playing_source=InMemoryNowPlayingSource(None),
        live_identity_resolver=LiveIdentityResolver(),
        capture=FakeLoopbackCapture(),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.0])),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=UserStore(root=tmp_path),
    )
    assert result is None


# --- #170 AC4/AC5/AC6: youtube-rung track key + alias/near-miss wiring ------ #


def test_run_live_capture_spike_youtube_rung_win_uses_youtube_key(tmp_path) -> None:
    """AC4/AC5: when the history-backed youtube rung wins the waterfall (no
    higher rung resolved an mbid), the stored track key must be
    ``youtube:<id>`` -- canonical_track_id's youtube_id rung -- not the
    name-key fallback the TrackRef would otherwise bottom out at if
    ``youtube_id`` were never threaded through."""
    youtube_index = InMemoryYoutubeHistoryIndex({("Song", "Artist"): "yt-1"})
    live_resolver = LiveIdentityResolver(youtube_history_index=youtube_index)
    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Song", artist="Artist", app_id="chrome.exe")
    )
    store = UserStore(root=tmp_path)

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
    )

    assert result is not None
    assert result.identity.level == "youtube"
    assert result.identity.youtube_id == "yt-1"
    assert store.has_audio_analysis("youtube:yt-1")


def test_run_live_capture_spike_aliases_youtube_history_when_score_gated_rung_wins(
    tmp_path,
) -> None:
    """AC6: the fingerprint rung (score-gated) wins an mbid for the same
    capture the participant's own youtube history also recognizes -- the
    history match must be recorded as an alias (``youtube:<id> ->
    mbid:<id>``, tier ``history_title_match``) in the participant-root
    ``aliases.jsonl``, never the pool sidecar."""
    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Around the World", artist="Daft Punk", app_id="chrome.exe")
    )
    acoustid = InMemoryAcoustIdSource({"fp-fake": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    youtube_index = InMemoryYoutubeHistoryIndex({("Around the World", "Daft Punk"): "yt-1"})
    live_resolver = LiveIdentityResolver(
        acoustid_source=acoustid, youtube_history_index=youtube_index
    )
    store = UserStore(root=tmp_path)

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
    )

    assert result is not None
    assert result.identity.level == "acoustid"
    assert result.identity.mbid == "M-1"
    aliases = load_aliases(store.aliases_path)
    assert aliases["youtube:yt-1"] == "mbid:M-1"
    assert not youtube_near_miss_journal_path(store).exists()


def test_run_live_capture_spike_journals_near_miss_when_non_score_gated_rung_wins(
    tmp_path,
) -> None:
    """AC6: a win at ``mb_name`` (not score-gated evidence) must not alias the
    youtube history match -- it is journaled as a near-miss instead, no
    ``aliases.jsonl`` line written."""
    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Around the World", artist="Daft Punk", app_id="chrome.exe")
    )
    mb_name_search = InMemoryMusicBrainzNameSearchSource({("Around the World", "Daft Punk"): "M-2"})
    youtube_index = InMemoryYoutubeHistoryIndex({("Around the World", "Daft Punk"): "yt-1"})
    live_resolver = LiveIdentityResolver(
        mb_name_search=mb_name_search, youtube_history_index=youtube_index
    )
    store = UserStore(root=tmp_path)

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
    )

    assert result is not None
    assert result.identity.level == "mb_name"
    assert load_aliases(store.aliases_path) == {}
    journal_path = youtube_near_miss_journal_path(store)
    assert journal_path.exists()
    lines = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert lines == [
        {
            "youtube_id": "yt-1",
            "winner": "mbid:M-2",
            "winner_level": "mb_name",
            "title": "Around the World",
            "artist": "Daft Punk",
        }
    ]


def test_run_live_capture_spike_no_alias_or_near_miss_when_youtube_rung_itself_wins(
    tmp_path,
) -> None:
    """AC6 must not fire when the youtube rung is itself the waterfall's
    winner -- there is no separate winner key to alias the history match to,
    and it is not a near-miss (it is the primary result)."""
    youtube_index = InMemoryYoutubeHistoryIndex({("Song", "Artist"): "yt-1"})
    live_resolver = LiveIdentityResolver(youtube_history_index=youtube_index)
    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Song", artist="Artist", app_id="chrome.exe")
    )
    store = UserStore(root=tmp_path)

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
    )

    assert result is not None
    assert result.identity.level == "youtube"
    assert not store.aliases_path.exists()
    assert not youtube_near_miss_journal_path(store).exists()


def test_run_live_capture_spike_does_not_re_alias_already_aliased_youtube_id(tmp_path) -> None:
    """Idempotent, mirroring #178's metadata-crosswalk convention: a
    ``loser`` already present in ``aliases.jsonl`` is never re-aliased."""
    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Around the World", artist="Daft Punk", app_id="chrome.exe")
    )
    acoustid = InMemoryAcoustIdSource({"fp-fake": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    youtube_index = InMemoryYoutubeHistoryIndex({("Around the World", "Daft Punk"): "yt-1"})
    live_resolver = LiveIdentityResolver(
        acoustid_source=acoustid, youtube_history_index=youtube_index
    )
    store = UserStore(root=tmp_path)
    store.write_audio_analysis(track_id="mbid:M-1", embedding=[0.5], tags={})

    run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
    )
    run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=store,
        fingerprint_fn=_fake_fingerprint_fn([]),
    )

    lines = store.aliases_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
