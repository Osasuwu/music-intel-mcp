import pytest

from music_intel_mcp.cli import main
from music_intel_mcp.models import GeneratedFrom, RootProfile
from music_intel_mcp.shared_store import LocalSharedStore
from music_intel_mcp.store import UserStore


def _write_profile(store: UserStore, *, timbre_coverage: float | None = 1.0) -> None:
    """A delivered profile. #263 made purge read its timbre coverage, so the
    default is a fully covered profile; ``None`` leaves the key unmeasured."""
    coverage = {} if timbre_coverage is None else {"timbre": timbre_coverage}
    store.write_profile(
        RootProfile(
            user_id=store.root.name,
            snapshot_id=f"{store.root.name}/2026-01-01T00:00:00Z",
            generated_from=GeneratedFrom(coverage_per_category=coverage),
        )
    )


# #165 AC2: purge <root> removes the participant root entirely.
def test_purge_removes_the_data_root(tmp_path):
    root = tmp_path / "participant"
    store = UserStore(root=root)
    _write_profile(store)

    rc = main(["purge", "--data-dir", str(root)])

    assert rc == 0
    assert not root.exists()


# #165 AC2: purge refuses to run while no RootProfile has been delivered for
# this root -- deleting an in-progress root before its one deliverable exists
# would silently destroy the only output the derive step produced.
def test_purge_refuses_without_delivered_profile(tmp_path, capsys):
    root = tmp_path / "participant"
    root.mkdir(parents=True)
    (root / "history.jsonl").write_text("", encoding="utf-8")  # in-progress, no profile yet

    rc = main(["purge", "--data-dir", str(root)])

    assert rc == 2
    assert root.exists()
    assert "force" in capsys.readouterr().out.lower()


# #165 AC2: --force bypasses the no-delivered-profile refusal.
def test_purge_force_bypasses_refusal_without_profile(tmp_path):
    root = tmp_path / "participant"
    root.mkdir(parents=True)
    (root / "history.jsonl").write_text("", encoding="utf-8")  # in-progress, no profile yet

    rc = main(["purge", "--data-dir", str(root), "--force"])

    assert rc == 0
    assert not root.exists()


# #165 AC2: purge never touches the pool or SharedStore -- they are sibling
# paths to the participant root, not nested under it, so a purge scoped to
# just the participant root structurally cannot reach them.
def test_purge_leaves_pool_and_shared_store_untouched(tmp_path):
    root = tmp_path / "participant"
    pool_root = tmp_path / "pool"
    store = UserStore(root=root, pool_root=pool_root)
    _write_profile(store)
    pool_root.mkdir(parents=True)
    (pool_root / "marker.txt").write_text("pool data", encoding="utf-8")

    shared_store_path = tmp_path / "shared_cache.jsonl"
    LocalSharedStore(path=shared_store_path)
    shared_store_path.write_text('{"track_id": "keep-me"}\n', encoding="utf-8")

    rc = main(["purge", "--data-dir", str(root)])

    assert rc == 0
    assert not root.exists()
    assert (pool_root / "marker.txt").read_text(encoding="utf-8") == "pool data"
    assert shared_store_path.read_text(encoding="utf-8") == '{"track_id": "keep-me"}\n'


# --------------------------------------------------------------------------- #
# #263 -- purge and the node-level analysis pool.
# --------------------------------------------------------------------------- #


def _participant(tmp_path):
    return tmp_path / "data" / "participants" / "p1"


# AC4: the analyses a participant's capture produced live in the pool, so
# deleting the participant root leaves them for every later participant.
def test_purge_of_a_participant_root_keeps_its_pooled_analyses(tmp_path):
    root, pool = _participant(tmp_path), tmp_path / "pool"
    store = UserStore(root=root, pool_root=pool)
    store.write_audio_analysis(track_id="youtube:yt1", embedding=[1.0, 2.0], tags={})
    _write_profile(store)

    rc = main(["purge", "--data-dir", str(root), "--pool-dir", str(pool)])

    assert rc == 0
    assert not root.exists()
    assert UserStore(root=tmp_path / "elsewhere", pool_root=pool).has_audio_analysis("youtube:yt1")


# AC6: a pool inside the root would be deleted with it. No flag overrides this.
@pytest.mark.parametrize("pool_name", ["", "pool"], ids=["pool-is-root", "pool-under-root"])
def test_purge_always_refuses_when_the_pool_is_inside_the_root(tmp_path, capsys, pool_name):
    root = tmp_path / "participant"
    pool = root / pool_name if pool_name else root
    _write_profile(UserStore(root=root))

    rc = main(
        [
            "purge",
            "--data-dir",
            str(root),
            "--pool-dir",
            str(pool),
            "--force",
            "--purge-capture-data",
            "--accept-low-coverage",
        ]
    )

    assert rc == 2
    assert root.exists()
    assert "pool" in capsys.readouterr().out.lower()


# AC7: the artifacts the issue names -- analyses, fingerprints, the capture
# journals and the window-probe journals. Written here as the oracle, not
# read back from the production list.
CAPTURE_ARTIFACTS = [
    "audio_analysis/youtube_yt1.json",
    "fingerprints/youtube_yt1.json",
    "live_capture_journal.jsonl",
    "replay_journal.jsonl",
    "replay_ledger.jsonl",
    "stream_decode_journal.jsonl",
    "window_probe.jsonl",
    "stream_decode_window_probe.jsonl",
]


def _root_with(tmp_path, artifact: str):
    root = tmp_path / "participant"
    _write_profile(UserStore(root=root))
    path = root / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("artifact", CAPTURE_ARTIFACTS)
def test_purge_refuses_a_root_holding_capture_data(tmp_path, capsys, artifact):
    root = _root_with(tmp_path, artifact)

    rc = main(["purge", "--data-dir", str(root), "--force"])  # --force does not override

    assert rc == 2
    assert (root / artifact).exists()
    assert "--purge-capture-data" in capsys.readouterr().out


def test_empty_analysis_directories_are_not_capture_data(tmp_path):
    root = tmp_path / "participant"
    _write_profile(UserStore(root=root))
    (root / "audio_analysis").mkdir()
    (root / "fingerprints").mkdir()

    rc = main(["purge", "--data-dir", str(root)])

    assert rc == 0
    assert not root.exists()


@pytest.mark.parametrize("artifact", CAPTURE_ARTIFACTS)
def test_purge_capture_data_flag_allows_it(tmp_path, artifact):
    root = _root_with(tmp_path, artifact)

    rc = main(["purge", "--data-dir", str(root), "--purge-capture-data"])

    assert rc == 0
    assert not root.exists()


# AC8: below 80 % timbre coverage of valid plays the pilot's criterion is
# unmet; an unmeasured coverage is not a pass.
@pytest.mark.parametrize("coverage", [0.79, 0.0, None], ids=["0.79", "0.0", "unmeasured"])
def test_purge_refuses_below_the_timbre_coverage_floor(tmp_path, capsys, coverage):
    root = tmp_path / "participant"
    _write_profile(UserStore(root=root), timbre_coverage=coverage)

    rc = main(["purge", "--data-dir", str(root)])

    assert rc == 2
    assert root.exists()
    assert "--accept-low-coverage" in capsys.readouterr().out


def test_accept_low_coverage_flag_allows_it(tmp_path):
    root = tmp_path / "participant"
    _write_profile(UserStore(root=root), timbre_coverage=0.1)

    rc = main(["purge", "--data-dir", str(root), "--accept-low-coverage"])

    assert rc == 0
    assert not root.exists()


def test_purge_at_exactly_the_coverage_floor_runs(tmp_path):
    root = tmp_path / "participant"
    _write_profile(UserStore(root=root), timbre_coverage=0.8)

    rc = main(["purge", "--data-dir", str(root)])

    assert rc == 0
