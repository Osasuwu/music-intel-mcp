from music_intel_mcp.cli import main
from music_intel_mcp.models import GeneratedFrom, RootProfile
from music_intel_mcp.shared_store import LocalSharedStore
from music_intel_mcp.store import UserStore


def _write_profile(store: UserStore) -> None:
    store.write_profile(
        RootProfile(
            user_id=store.root.name,
            snapshot_id=f"{store.root.name}/2026-01-01T00:00:00Z",
            generated_from=GeneratedFrom(),
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
