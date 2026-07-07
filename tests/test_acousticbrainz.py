"""AcousticBrainz feature source (#87 AC-E, the ``MBID -> audio features`` leg).

AcousticBrainz shut down Feb 2022 and the data is frozen (CC0), but the query API
is still live. Rather than the impractical 590 GB low-level bulk dump (the only one
carrying bpm), we fetch our ~27k resolved MBIDs targeted from that live API and
distil them into the same compact ``acousticbrainz_features.jsonl`` the audio
pipeline's :class:`~music_intel_mcp.audio.AcousticBrainzDump` reads — mechanism
substitution for the same "ready-made dataset" intent (decision ``d8b488ba``).

Three layers, tested bottom-up:

1. **Pure mapping** — ``extract_ab_scalars`` pulls AB-native scalars from the
   offset-0 high/low-level docs; ``features_from_scalars`` maps them onto
   :class:`AudioFeatures` per the recorded field mapping. Fixtures mirror the real
   captured AB payload; no network.
2. **API client** — ``AcousticBrainzApiClient`` batches ``;``-joined lookups and
   backs off on 429/503. Mocked with ``respx`` (CLAUDE.md forbids live API in CI).
3. **Builder** — ``build_acousticbrainz_index`` drives the client with a resumable
   raw cache and projects the features JSONL. Driven by a fake client.
"""

from __future__ import annotations

import json

import httpx
import respx

from music_intel_mcp.acousticbrainz import (
    AB_HIGHLEVEL_URL,
    AB_LOWLEVEL_URL,
    AcousticBrainzApiClient,
    build_acousticbrainz_index,
    extract_ab_scalars,
    features_from_scalars,
)
from music_intel_mcp.audio import AcousticBrainzDump

# Real captured probe for MBID 1fc545d3-94b6-4a3d-bff3-bb8b861943d6 (the shapes
# below mirror the live AB response, trimmed to the fields the mapping reads).
_HL_DOC = {
    "metadata": {"tags": {}},
    "highlevel": {
        "danceability": {"all": {"danceable": 0.147, "not_danceable": 0.853}},
        "mood_happy": {"all": {"happy": 0.5, "not_happy": 0.5}},
        "mood_acoustic": {"all": {"acoustic": 0.960, "not_acoustic": 0.040}},
        "voice_instrumental": {"all": {"instrumental": 0.0265, "voice": 0.9735}},
        "mood_aggressive": {"all": {"aggressive": 0.0547, "not_aggressive": 0.9453}},
        "mood_relaxed": {"all": {"relaxed": 0.7, "not_relaxed": 0.3}},
    },
}
_LL_DOC = {
    "rhythm": {"bpm": 114.8},
    "lowlevel": {"average_loudness": 0.512},
}


# --------------------------------------------------------------------------- #
# Layer 1 — pure mapping
# --------------------------------------------------------------------------- #


def test_extract_ab_scalars_pulls_native_values():
    scalars = extract_ab_scalars(_HL_DOC, _LL_DOC)
    assert scalars == {
        "bpm": 114.8,
        "average_loudness": 0.512,
        "danceable": 0.147,
        "happy": 0.5,
        "acoustic": 0.960,
        "instrumental": 0.0265,
        "aggressive": 0.0547,
        "relaxed": 0.7,
    }


def test_extract_ab_scalars_none_when_both_docs_empty():
    assert extract_ab_scalars(None, None) is None
    assert extract_ab_scalars({}, {}) is None


def test_extract_ab_scalars_highlevel_only_leaves_rhythm_none():
    """A highlevel hit with no lowlevel submission still yields the mood/genre
    scalars — bpm/loudness are ``None`` (the track can't cluster, but the hit is
    honestly recorded, not dropped)."""
    scalars = extract_ab_scalars(_HL_DOC, None)
    assert scalars is not None
    assert scalars["bpm"] is None
    assert scalars["average_loudness"] is None
    assert scalars["happy"] == 0.5
    assert scalars["danceable"] == 0.147


def test_features_from_scalars_maps_to_audio_features():
    feats = features_from_scalars(extract_ab_scalars(_HL_DOC, _LL_DOC))
    assert feats is not None
    # The recorded field mapping (decision d8b488ba).
    assert feats.bpm == 114.8  # <- lowlevel rhythm.bpm
    assert feats.energy == 0.512  # <- lowlevel average_loudness
    assert feats.valence == 0.5  # <- highlevel mood_happy.all.happy
    assert feats.danceability == 0.147  # <- highlevel danceability.all.danceable
    assert feats.acousticness == 0.960  # <- highlevel mood_acoustic.all.acoustic
    assert feats.instrumentalness == 0.0265  # <- highlevel voice_instrumental.all.instrumental
    assert feats.source == "acousticbrainz_api"


def test_features_from_scalars_none_when_empty():
    assert features_from_scalars(None) is None
    assert features_from_scalars({}) is None


# --------------------------------------------------------------------------- #
# Layer 2 — API client
# --------------------------------------------------------------------------- #

_M1 = "1fc545d3-94b6-4a3d-bff3-bb8b861943d6"
_M2 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def test_client_fetch_highlevel_parses_offset_zero_and_skips_mbid_mapping():
    client = AcousticBrainzApiClient(sleep=lambda _s: None)
    body = {
        _M1: {"0": _HL_DOC},
        "mbid_mapping": {"whatever": 1},  # the API's own bookkeeping key — ignore
    }
    with respx.mock(assert_all_called=False) as router:
        router.get(AB_HIGHLEVEL_URL).mock(return_value=httpx.Response(200, json=body))
        out = client.fetch_highlevel([_M1])
    assert set(out) == {_M1}
    assert out[_M1]["highlevel"]["mood_happy"]["all"]["happy"] == 0.5


def test_client_batches_at_twenty_five_and_semicolon_joins():
    client = AcousticBrainzApiClient(sleep=lambda _s: None)
    ids = [f"mbid-{i:03d}" for i in range(30)]
    with respx.mock(assert_all_called=False) as router:
        route = router.get(AB_LOWLEVEL_URL).mock(return_value=httpx.Response(200, json={}))
        client.fetch_lowlevel(ids)
        # 30 ids -> two GET calls (25 + 5); each batch <= 25, ';'-joined.
        assert route.call_count == 2
        for call in route.calls:
            joined = call.request.url.params.get("recording_ids")
            assert len(joined.split(";")) <= 25


def test_client_backoff_retries_on_503_then_succeeds():
    slept: list[float] = []
    client = AcousticBrainzApiClient(sleep=slept.append)
    with respx.mock(assert_all_called=False) as router:
        router.get(AB_HIGHLEVEL_URL).mock(
            side_effect=[
                httpx.Response(503, headers={"Retry-After": "3"}),
                httpx.Response(200, json={_M1: {"0": _HL_DOC}}),
            ]
        )
        out = client.fetch_highlevel([_M1])
    assert set(out) == {_M1}
    assert slept == [3.0]  # honored Retry-After before the retry


# --------------------------------------------------------------------------- #
# Layer 3 — resumable builder
# --------------------------------------------------------------------------- #


class _FakeAbClient:
    """Records which mbids each leg was asked for so the highlevel-first gate is
    observable, and returns the real-shaped docs for a known set."""

    def __init__(self, highlevel: dict[str, dict], lowlevel: dict[str, dict]) -> None:
        self._hl = highlevel
        self._ll = lowlevel
        self.hl_asked: list[str] = []
        self.ll_asked: list[str] = []

    def fetch_highlevel(self, mbids):
        self.hl_asked.extend(mbids)
        return {m: self._hl[m] for m in mbids if m in self._hl}

    def fetch_lowlevel(self, mbids):
        self.ll_asked.extend(mbids)
        return {m: self._ll[m] for m in mbids if m in self._ll}


def _read_jsonl(path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["mbid"]] = row
    return out


def test_builder_writes_features_and_roundtrips_through_dump(tmp_path):
    out = tmp_path / "acousticbrainz_features.jsonl"
    raw = tmp_path / "acousticbrainz_raw.jsonl"
    client = _FakeAbClient({_M1: _HL_DOC}, {_M1: _LL_DOC})

    report = build_acousticbrainz_index([_M1], client, out_path=out, raw_cache_path=raw)

    assert report.fetched == 1
    assert report.hits == 1
    assert report.written == 1
    assert report.total_cached == 1
    assert report.cached_hits == 1
    assert report.cached_misses == 0
    # The output JSONL is exactly what AcousticBrainzDump reads.
    feats = AcousticBrainzDump(path=out).lookup(_M1)
    assert feats is not None
    assert feats.bpm == 114.8
    assert feats.valence == 0.5
    assert feats.danceability == 0.147


def test_builder_highlevel_first_gate_skips_lowlevel_for_hl_misses(tmp_path):
    """Lowlevel is fetched only for MBIDs that hit highlevel — an hl-miss can't
    cluster (no valence/danceability), so its lowlevel call is wasted."""
    out = tmp_path / "features.jsonl"
    raw = tmp_path / "raw.jsonl"
    # _M2 has neither; _M1 has both.
    client = _FakeAbClient({_M1: _HL_DOC}, {_M1: _LL_DOC})

    build_acousticbrainz_index([_M1, _M2], client, out_path=out, raw_cache_path=raw)

    assert set(client.ll_asked) == {_M1}  # _M2 (hl-miss) never reached lowlevel
    assert _M2 not in _read_jsonl(out)  # nothing to write for a total miss


def test_builder_records_misses_in_raw_cache_and_resume_skips_them(tmp_path):
    out = tmp_path / "features.jsonl"
    raw = tmp_path / "raw.jsonl"
    client = _FakeAbClient({_M1: _HL_DOC}, {_M1: _LL_DOC})

    first = build_acousticbrainz_index([_M1, _M2], client, out_path=out, raw_cache_path=raw)
    assert first.fetched == 2
    assert first.misses == 1  # _M2

    # The miss is a negative-cache entry, not re-queried on resume.
    raw_rows = _read_jsonl(raw)
    assert raw_rows[_M2]["scalars"] is None

    client2 = _FakeAbClient({_M1: _HL_DOC}, {_M1: _LL_DOC})
    second = build_acousticbrainz_index([_M1, _M2], client2, out_path=out, raw_cache_path=raw)
    assert second.fetched == 0  # everything already cached
    assert client2.hl_asked == []  # no API traffic at all on a fully-cached re-run
    # Cumulative split covers the whole cache, not just this run (fetched=0 here).
    assert second.total_cached == 2
    assert second.cached_hits == 1  # _M1
    assert second.cached_misses == 1  # _M2
    # The features JSONL is still (re)projected from the full cache.
    assert _M1 in _read_jsonl(out)


def test_builder_resume_fetches_only_new_mbids(tmp_path):
    out = tmp_path / "features.jsonl"
    raw = tmp_path / "raw.jsonl"
    # A prior (interrupted) run already cached _M1.
    raw.write_text(
        json.dumps({"mbid": _M1, "scalars": extract_ab_scalars(_HL_DOC, _LL_DOC)}) + "\n",
        encoding="utf-8",
    )
    client = _FakeAbClient({_M2: _HL_DOC}, {_M2: _LL_DOC})

    report = build_acousticbrainz_index([_M1, _M2], client, out_path=out, raw_cache_path=raw)

    assert client.hl_asked == [_M2]  # _M1 skipped (already cached)
    assert report.fetched == 1
    # Both land in the projected output (the cached one + the newly fetched one).
    assert set(_read_jsonl(out)) == {_M1, _M2}
