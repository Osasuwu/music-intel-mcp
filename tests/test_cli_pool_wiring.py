"""#263: the node-level analysis pool reaches every store the CLI and the
desktop app open.

AC1 -- every ``UserStore`` construction site resolves the pool through one
function (``--pool-dir`` > ``MUSIC_INTEL_POOL_DIR`` > no pool).
AC2 -- a root under ``participants/`` with no pool resolved is refused; any
other root works without a pool.
"""

from __future__ import annotations

import argparse
import ast
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import music_intel_mcp.cli as cli
import music_intel_mcp.desktop_app as desktop_app
from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef
from music_intel_mcp.replay_queue import MIN_VALID_PLAYS
from music_intel_mcp.shared_store import canonical_track_id
from music_intel_mcp.store import UserStore
from music_intel_mcp.stream_decode import FakeStreamDecodeSource


def _userstore_calls(module_path: Path) -> list[int]:
    """Line numbers of every direct ``UserStore(...)`` call in a module."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == "UserStore":
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("module", [cli, desktop_app], ids=["cli", "desktop_app"])
def test_entry_points_never_construct_userstore_directly(module):
    """A direct ``UserStore(...)`` call bypasses the pool resolver, which is
    exactly how #263's bug arose (no site passed ``pool_root``)."""
    assert _userstore_calls(Path(module.__file__)) == []


def _parsers_with_data_dir(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    found: dict[str, argparse.ArgumentParser] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                if any("--data-dir" in a.option_strings for a in sub._actions):
                    found[name] = sub
                found.update(_parsers_with_data_dir(sub))
    return found


def test_every_subcommand_with_data_dir_takes_pool_dir():
    parsers = _parsers_with_data_dir(cli.build_parser())
    # The walk must reach the commands the pilot runbook runs, or it proves nothing.
    assert {"analyze", "replay-capture-youtube", "purge"} <= set(parsers)
    missing = [
        name
        for name, sub in parsers.items()
        if not any("--pool-dir" in a.option_strings for a in sub._actions)
    ]
    assert missing == []


def _capture_store(monkeypatch) -> dict:
    """Stub the stream-decode leg's heavy collaborators; record the store the
    command hands to the capture loop."""
    captured: dict = {}

    def fake_process_stream_decode_queue(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "music_intel_mcp.stream_decode.process_stream_decode_queue",
        fake_process_stream_decode_queue,
    )
    monkeypatch.setattr(
        "music_intel_mcp.inference.DiscogsEffnetOnnxModel",
        lambda: InMemoryEmbeddingModel(np.array([0.0, 1.0])),
    )
    monkeypatch.setattr("music_intel_mcp.inference.MtgJamendoClassifier", lambda: object())
    return captured


def test_pool_dir_flag_reaches_the_capture_store(tmp_path, monkeypatch):
    captured = _capture_store(monkeypatch)
    root, pool = tmp_path / "root", tmp_path / "pool"

    rc = cli.main(["replay-capture-youtube", "--data-dir", str(root), "--pool-dir", str(pool)])

    assert rc == 0
    assert captured["store"].pool_root == pool


def test_pool_env_var_reaches_the_capture_store(tmp_path, monkeypatch):
    captured = _capture_store(monkeypatch)
    pool = tmp_path / "pool"
    monkeypatch.setenv("MUSIC_INTEL_POOL_DIR", str(pool))

    rc = cli.main(["replay-capture-youtube", "--data-dir", str(tmp_path / "root")])

    assert rc == 0
    assert captured["store"].pool_root == pool


def test_capture_on_a_participant_root_without_a_pool_is_refused(tmp_path, monkeypatch, capsys):
    captured = _capture_store(monkeypatch)
    root = tmp_path / "data" / "participants" / "p1"

    rc = cli.main(["replay-capture-youtube", "--data-dir", str(root)])

    assert rc == 2
    assert captured == {}  # the capture loop never started
    out = capsys.readouterr().out
    assert "--pool-dir" in out and "MUSIC_INTEL_POOL_DIR" in out


def test_analyze_on_a_participant_root_without_a_pool_is_refused(tmp_path):
    root = tmp_path / "data" / "participants" / "p1"

    rc = cli.main(["analyze", "--data-dir", str(root)])

    assert rc == 2
    assert not (root / "profiles").exists()


def test_analyze_on_a_participant_root_with_a_pool_runs(tmp_path):
    root = tmp_path / "data" / "participants" / "p1"

    rc = cli.main(["analyze", "--data-dir", str(root), "--pool-dir", str(tmp_path / "pool")])

    assert rc == 0
    assert list((root / "profiles").glob("*.json"))


def test_analyze_on_a_non_participant_root_runs_without_a_pool(tmp_path):
    root = tmp_path / "data"

    rc = cli.main(["analyze", "--data-dir", str(root)])

    assert rc == 0
    assert list((root / "profiles").glob("*.json"))


# --------------------------------------------------------------------------- #
# AC3 / AC5 -- the real capture path (fake decode + models) against a pool.
# --------------------------------------------------------------------------- #

_TRACK = TrackRef(name="Track One", artist="Artist A", youtube_id="yt1")


def _seed_history(root: Path) -> None:
    """Enough valid plays of one YouTube track for the replay queue to pick it."""
    UserStore(root=root).append_events(
        [
            ListenEvent(
                track=_TRACK,
                played_at=datetime(2026, 1, day, tzinfo=UTC),
                source="youtube_music",
                context=PlayContext(ms_played=180_000),
            )
            for day in range(1, MIN_VALID_PLAYS + 1)
        ]
    )


def _fake_capture_leg(monkeypatch) -> FakeStreamDecodeSource:
    """Keep the real queue -> decode -> analyze -> store loop; swap only the
    network decoder and the ONNX models for in-memory doubles."""
    source = FakeStreamDecodeSource(channels=2)
    monkeypatch.setattr("music_intel_mcp.stream_decode.YtDlpStreamDecodeSource", lambda: source)
    monkeypatch.setattr(
        "music_intel_mcp.inference.DiscogsEffnetOnnxModel",
        lambda: InMemoryEmbeddingModel(np.array([1.0, 2.0])),
    )
    monkeypatch.setattr(
        "music_intel_mcp.inference.MtgJamendoClassifier",
        lambda: InMemoryClassifier(ClassifierResult(tags={"genre---rock": 0.9})),
    )
    return source


def _capture(root: Path, pool: Path) -> int:
    return cli.main(["replay-capture-youtube", "--data-dir", str(root), "--pool-dir", str(pool)])


def test_capture_with_a_pool_writes_the_analysis_to_the_pool_not_the_root(tmp_path, monkeypatch):
    source = _fake_capture_leg(monkeypatch)
    root, pool = tmp_path / "data" / "participants" / "p1", tmp_path / "pool"
    _seed_history(root)

    assert _capture(root, pool) == 0

    assert source.decoded_youtube_ids == ["yt1"]  # the run really analyzed the track
    key = canonical_track_id(_TRACK)
    # A store that sees only the pool finds it; one that sees only the
    # participant root does not -- purging the root cannot lose it.
    assert UserStore(root=tmp_path / "elsewhere", pool_root=pool).has_audio_analysis(key)
    assert not UserStore(root=root).has_audio_analysis(key)


def test_a_second_participant_does_not_re_decode_a_pooled_track(tmp_path, monkeypatch):
    pool = tmp_path / "pool"
    first = tmp_path / "data" / "participants" / "p1"
    second = tmp_path / "data" / "participants" / "p2"
    _seed_history(first)
    _seed_history(second)
    first_source = _fake_capture_leg(monkeypatch)
    assert _capture(first, pool) == 0
    assert first_source.decoded_youtube_ids == ["yt1"]  # the pool now holds it

    second_source = _fake_capture_leg(monkeypatch)
    assert _capture(second, pool) == 0

    assert second_source.decoded_youtube_ids == []
