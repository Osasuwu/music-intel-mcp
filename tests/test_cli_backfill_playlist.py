"""CLI `backfill-playlist` entrypoint (#127) — ties the opt-in gate, saved-
tracks candidate pool, AC3/AC4 selection, and the Spotify playlist sync into
one daily-refresh command. Pure selection/diff logic and the HTTP-mocked
Spotify client are covered in ``test_backfill_playlist.py`` /
``test_backfill_playlist_sync.py``; this file exercises the CLI wiring only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import respx

from music_intel_mcp.cli import main
from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef

_BEARER = "opaque-bearer-fixture"


def _write_token(data_dir: Path) -> None:
    token_path = data_dir / "spotify_user_token.json"
    token_path.write_text(
        json.dumps(
            {
                "access_token": _BEARER,
                "refresh_token": "refresh-fixture",
                "expires_at": time.time() + 3600,
                "token_type": "Bearer",
            }
        ),
        encoding="utf-8",
    )


def test_backfill_playlist_disabled_by_default(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED", raising=False)
    rc = main(["backfill-playlist", "--data-dir", str(tmp_path)])
    assert rc == 1
    assert "disabled" in capsys.readouterr().out


def test_backfill_playlist_requires_authorization(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED", "true")
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    rc = main(["backfill-playlist", "--data-dir", str(tmp_path)])
    assert rc == 2
    assert "not authorized" in capsys.readouterr().out


def test_backfill_playlist_syncs_selected_tracks(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED", "true")
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    _write_token(tmp_path)

    history_path = tmp_path / "history.jsonl"
    played = TrackRef(name="Played", artist="Artist", spotify_id="played1")
    event = ListenEvent(
        track=played, played_at="2026-01-01T00:00:00Z", source="test", context=PlayContext()
    )
    history_path.write_text(event.model_dump_json() + "\n", encoding="utf-8")

    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.spotify.com/v1/me/tracks").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "track": {
                                "id": "played1",
                                "name": "Played",
                                "artists": [{"name": "Artist"}],
                                "album": {"name": "Album"},
                            }
                        },
                        {
                            "track": {
                                "id": "fresh1",
                                "name": "Fresh",
                                "artists": [{"name": "Artist"}],
                                "album": {"name": "Album"},
                            }
                        },
                    ],
                    "next": None,
                },
            )
        )
        router.get("https://api.spotify.com/v1/me").mock(
            return_value=httpx.Response(200, json={"id": "the_user"})
        )
        router.get("https://api.spotify.com/v1/me/playlists").mock(
            return_value=httpx.Response(
                200,
                json={"items": [{"id": "pl1", "name": "music-intel: to-analyze"}], "next": None},
            )
        )
        router.get("https://api.spotify.com/v1/playlists/pl1/tracks").mock(
            return_value=httpx.Response(200, json={"items": [], "next": None})
        )
        add_route = router.post("https://api.spotify.com/v1/playlists/pl1/tracks").mock(
            return_value=httpx.Response(201, json={})
        )

        rc = main(
            [
                "backfill-playlist",
                "--data-dir",
                str(tmp_path),
                "--shared-store",
                "memory",
            ]
        )

    assert rc == 0
    out = capsys.readouterr().out
    assert "2 saved tracks, 1 selected" in out
    assert add_route.called
    add_body = json.loads(add_route.calls[0].request.content)
    assert add_body["uris"] == ["spotify:track:fresh1"]


def test_backfill_playlist_loop_runs_until_stopped(tmp_path, capsys, monkeypatch):
    # AC2's daily-cadence half: --loop wires run_continuous_backfill in so a
    # single invocation keeps refreshing instead of running once and exiting.
    monkeypatch.setenv("MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED", "true")
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    _write_token(tmp_path)

    from music_intel_mcp import cli

    calls = {"n": 0}

    def fake_run_continuous_backfill(*, sync_once, interval_s, on_error=None, **_kw):
        assert interval_s == 2 * 3600.0
        calls["n"] += 1
        sync_once()  # one cycle, then stop — no real loop/sleep in a unit test

    monkeypatch.setattr(cli, "run_continuous_backfill", fake_run_continuous_backfill)

    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.spotify.com/v1/me/tracks").mock(
            return_value=httpx.Response(200, json={"items": [], "next": None})
        )
        router.get("https://api.spotify.com/v1/me").mock(
            return_value=httpx.Response(200, json={"id": "the_user"})
        )
        router.get("https://api.spotify.com/v1/me/playlists").mock(
            return_value=httpx.Response(
                200,
                json={"items": [{"id": "pl1", "name": "music-intel: to-analyze"}], "next": None},
            )
        )
        router.get("https://api.spotify.com/v1/playlists/pl1/tracks").mock(
            return_value=httpx.Response(200, json={"items": [], "next": None})
        )

        rc = main(
            [
                "backfill-playlist",
                "--data-dir",
                str(tmp_path),
                "--shared-store",
                "memory",
                "--loop",
                "--interval-hours",
                "2",
            ]
        )

    assert rc == 0
    assert calls["n"] == 1
    assert "loop mode" in capsys.readouterr().out
