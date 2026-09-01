"""CLI `automated-playback-consent` / `automated-playback` entrypoints (#128)
-- ties the consent gate, authorization check, backfill-queue selection, and
:class:`~music_intel_mcp.automated_playback.SpotifyPlaybackClient` into one
human-paced playthrough command. Pure pacing/revocation logic and the
HTTP-mocked Spotify client are covered in ``test_automated_playback.py`` /
``test_automated_playback_sync.py``; this file exercises the CLI wiring only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import respx

from music_intel_mcp.cli import main
from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef
from music_intel_mcp.store import UserStore

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


# --- AC1: consent gate -------------------------------------------------- #


def test_automated_playback_consent_grant_persists_it(tmp_path, capsys):
    rc = main(["automated-playback-consent", "--grant", "--data-dir", str(tmp_path)])
    assert rc == 0
    assert UserStore(root=tmp_path).has_automated_playback_consent() is True
    assert "granted" in capsys.readouterr().out


def test_automated_playback_consent_revoke_removes_it(tmp_path):
    UserStore(root=tmp_path).grant_automated_playback_consent(granted_at="2026-01-01T00:00:00Z")

    rc = main(["automated-playback-consent", "--revoke", "--data-dir", str(tmp_path)])

    assert rc == 0
    assert UserStore(root=tmp_path).has_automated_playback_consent() is False


def test_automated_playback_blocked_without_consent(tmp_path, capsys):
    rc = main(["automated-playback", "--data-dir", str(tmp_path)])
    assert rc == 1
    assert "consent" in capsys.readouterr().out


def test_automated_playback_requires_authorization(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    UserStore(root=tmp_path).grant_automated_playback_consent(granted_at="2026-01-01T00:00:00Z")

    rc = main(["automated-playback", "--data-dir", str(tmp_path)])

    assert rc == 2
    assert "not authorized" in capsys.readouterr().out


# --- AC2/AC4: happy path -- pacing wired to the real client, history traced #


def test_automated_playback_plays_queue_and_records_agent_originated_history(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    _write_token(tmp_path)
    UserStore(root=tmp_path).grant_automated_playback_consent(granted_at="2026-01-01T00:00:00Z")

    history_path = tmp_path / "history.jsonl"
    played = TrackRef(name="Played", artist="Artist", spotify_id="played1")
    event = ListenEvent(
        track=played, played_at="2026-01-01T00:00:00Z", source="test", context=PlayContext()
    )
    history_path.write_text(event.model_dump_json() + "\n", encoding="utf-8")

    from music_intel_mcp import cli

    monkeypatch.setattr(cli.time, "sleep", lambda s: None)

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
        router.put("https://api.spotify.com/v1/me/player/play").mock(
            return_value=httpx.Response(204)
        )
        router.get("https://api.spotify.com/v1/tracks/fresh1").mock(
            return_value=httpx.Response(200, json={"duration_ms": 1000})
        )

        rc = main(
            [
                "automated-playback",
                "--data-dir",
                str(tmp_path),
                "--shared-store",
                "memory",
            ]
        )

    assert rc == 0
    out = capsys.readouterr().out
    assert "played 1/1" in out

    events = UserStore(root=tmp_path).load_history()
    agent_events = [e for e in events if e.source == "agent_automated_playback"]
    assert len(agent_events) == 1
    assert agent_events[0].track.spotify_id == "fresh1"


def test_automated_playback_stops_early_when_nothing_to_play(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    _write_token(tmp_path)
    UserStore(root=tmp_path).grant_automated_playback_consent(granted_at="2026-01-01T00:00:00Z")

    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.spotify.com/v1/me/tracks").mock(
            return_value=httpx.Response(200, json={"items": [], "next": None})
        )

        rc = main(
            [
                "automated-playback",
                "--data-dir",
                str(tmp_path),
                "--shared-store",
                "memory",
            ]
        )

    assert rc == 0
    assert "nothing to play" in capsys.readouterr().out
