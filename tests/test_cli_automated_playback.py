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
    rc = main(
        [
            "automated-playback-consent",
            "--grant",
            "--grantor",
            "alice",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert UserStore(root=tmp_path).has_automated_playback_consent() is True
    assert "granted" in capsys.readouterr().out


# #165 AC3: --grantor is mandatory with --grant -- a consent record with no
# grantor is exactly the old, now-rejected shape.
def test_automated_playback_consent_grant_requires_grantor(tmp_path, capsys):
    rc = main(["automated-playback-consent", "--grant", "--data-dir", str(tmp_path)])
    assert rc == 2
    assert "grantor" in capsys.readouterr().out
    assert UserStore(root=tmp_path).automated_playback_consent_path.exists() is False


def test_automated_playback_consent_revoke_removes_it(tmp_path):
    UserStore(root=tmp_path).grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope="automated-playback"
    )

    rc = main(["automated-playback-consent", "--revoke", "--data-dir", str(tmp_path)])

    assert rc == 0
    assert UserStore(root=tmp_path).has_automated_playback_consent() is False


def test_automated_playback_blocked_without_consent(tmp_path, capsys):
    rc = main(
        ["automated-playback", "--data-dir", str(tmp_path), "--device-name", "replay-browser"]
    )
    assert rc == 1
    assert "consent" in capsys.readouterr().out


# #165 AC3: an old-format consent file must not crash the CLI with a raw
# traceback -- it surfaces as a clear, handled error.
def test_automated_playback_rejects_old_format_consent_file(tmp_path, capsys):
    import json

    store = UserStore(root=tmp_path)
    store.automated_playback_consent_path.parent.mkdir(parents=True, exist_ok=True)
    store.automated_playback_consent_path.write_text(
        json.dumps({"granted_at": "2026-01-01T00:00:00Z"}), encoding="utf-8"
    )

    rc = main(
        ["automated-playback", "--data-dir", str(tmp_path), "--device-name", "replay-browser"]
    )

    assert rc == 2
    assert "old" in capsys.readouterr().out.lower()


def test_automated_playback_requires_authorization(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    UserStore(root=tmp_path).grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope="automated-playback"
    )

    rc = main(
        ["automated-playback", "--data-dir", str(tmp_path), "--device-name", "replay-browser"]
    )

    assert rc == 2
    assert "not authorized" in capsys.readouterr().out


# --- AC2/AC4: happy path -- pacing wired to the real client, history traced #


def test_automated_playback_plays_queue_and_records_agent_originated_history(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    _write_token(tmp_path)
    UserStore(root=tmp_path).grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope="automated-playback"
    )

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
        router.get("https://api.spotify.com/v1/me/player/devices").mock(
            return_value=httpx.Response(
                200, json={"devices": [{"id": "dev1", "name": "replay-browser"}]}
            )
        )
        router.get("https://api.spotify.com/v1/me/player").mock(
            return_value=httpx.Response(200, json={"is_playing": False})
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
                "--device-name",
                "replay-browser",
            ]
        )

    assert rc == 0
    out = capsys.readouterr().out
    assert "played 1/1" in out

    events = UserStore(root=tmp_path).load_history()
    agent_events = [e for e in events if e.source == "agent_automated_playback"]
    assert len(agent_events) == 1
    assert agent_events[0].track.spotify_id == "fresh1"


def test_automated_playback_metadata_only_track_is_not_treated_as_analyzed(
    tmp_path, capsys, monkeypatch
):
    """#158 AC2: the "already analyzed" check must consult audio-analysis
    presence (UserStore), not SharedStore metadata presence — a track with
    only anonymous metadata cached (no audio_analysis file) must still be
    queued for automated playback."""
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    _write_token(tmp_path)
    UserStore(root=tmp_path).grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope="automated-playback"
    )

    from datetime import UTC, datetime

    from music_intel_mcp.shared_store import LocalSharedStore, TrackMetadataRecord

    shared_path = tmp_path / "shared_cache.jsonl"
    LocalSharedStore(path=shared_path).upsert_tracks(
        [
            TrackMetadataRecord(
                track_id="spotify:fresh1",
                spotify_id="fresh1",
                name="Fresh",
                artist="Artist",
                fetched_at=datetime.now(UTC),
            )
        ]
    )

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
        router.get("https://api.spotify.com/v1/me/player/devices").mock(
            return_value=httpx.Response(
                200, json={"devices": [{"id": "dev1", "name": "replay-browser"}]}
            )
        )
        router.get("https://api.spotify.com/v1/me/player").mock(
            return_value=httpx.Response(200, json={"is_playing": False})
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
                "--device-name",
                "replay-browser",
            ]
        )

    assert rc == 0
    out = capsys.readouterr().out
    assert "played 1/1" in out


# #159 AC1: device_id is mandatory -- a name that doesn't match any device on
# the account must abort the run with a clear message, not fall through to
# whatever device Spotify considers active.
def test_automated_playback_reports_error_when_device_name_not_found(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    _write_token(tmp_path)
    UserStore(root=tmp_path).grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope="automated-playback"
    )

    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.spotify.com/v1/me/tracks").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "track": {
                                "id": "fresh1",
                                "name": "Fresh",
                                "artists": [{"name": "Artist"}],
                                "album": {"name": "Album"},
                            }
                        }
                    ],
                    "next": None,
                },
            )
        )
        router.get("https://api.spotify.com/v1/me/player/devices").mock(
            return_value=httpx.Response(
                200, json={"devices": [{"id": "dev1", "name": "some-other-device"}]}
            )
        )

        rc = main(
            [
                "automated-playback",
                "--data-dir",
                str(tmp_path),
                "--device-name",
                "replay-browser",
            ]
        )

    assert rc == 3
    assert "replay-browser" in capsys.readouterr().out
    assert UserStore(root=tmp_path).load_history() == []


# #159 AC3: a 404/403 from the play endpoint must re-queue the track (bounded
# retry) and the run must continue -- proven end-to-end through the CLI by
# having the play endpoint reject once then succeed, and asserting the track
# is still recorded as played rather than lost.
def test_automated_playback_requeues_and_plays_after_transient_404(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    _write_token(tmp_path)
    UserStore(root=tmp_path).grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope="automated-playback"
    )

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
                                "id": "fresh1",
                                "name": "Fresh",
                                "artists": [{"name": "Artist"}],
                                "album": {"name": "Album"},
                            }
                        }
                    ],
                    "next": None,
                },
            )
        )
        router.get("https://api.spotify.com/v1/me/player/devices").mock(
            return_value=httpx.Response(
                200, json={"devices": [{"id": "dev1", "name": "replay-browser"}]}
            )
        )
        router.get("https://api.spotify.com/v1/me/player").mock(
            return_value=httpx.Response(200, json={"is_playing": False})
        )
        router.put("https://api.spotify.com/v1/me/player/play").mock(
            side_effect=[httpx.Response(404), httpx.Response(204)]
        )
        router.get("https://api.spotify.com/v1/tracks/fresh1").mock(
            return_value=httpx.Response(200, json={"duration_ms": 1000})
        )

        rc = main(
            [
                "automated-playback",
                "--data-dir",
                str(tmp_path),
                "--device-name",
                "replay-browser",
            ]
        )

    assert rc == 0
    out = capsys.readouterr().out
    assert "http_404" in out
    assert "played 1/1" in out

    events = UserStore(root=tmp_path).load_history()
    agent_events = [e for e in events if e.source == "agent_automated_playback"]
    assert len(agent_events) == 1
    assert agent_events[0].track.spotify_id == "fresh1"


# #128 AC3 (real-device gap caught by review, PR #151): a consent revocation
# during the CLI's session must issue a real PUT to Spotify's pause endpoint,
# not just stop the local loop.
def test_automated_playback_pauses_spotify_device_on_mid_session_revocation(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    _write_token(tmp_path)
    store = UserStore(root=tmp_path)
    store.grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope="automated-playback"
    )

    from music_intel_mcp import cli

    sleep_calls = {"n": 0}

    def fake_sleep(_seconds: float) -> None:
        sleep_calls["n"] += 1
        if sleep_calls["n"] == 1:
            store.revoke_automated_playback_consent()

    monkeypatch.setattr(cli.time, "sleep", fake_sleep)

    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.spotify.com/v1/me/tracks").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "track": {
                                "id": "fresh1",
                                "name": "Fresh",
                                "artists": [{"name": "Artist"}],
                                "album": {"name": "Album"},
                            }
                        }
                    ],
                    "next": None,
                },
            )
        )
        router.get("https://api.spotify.com/v1/me/player/devices").mock(
            return_value=httpx.Response(
                200, json={"devices": [{"id": "dev1", "name": "replay-browser"}]}
            )
        )
        router.get("https://api.spotify.com/v1/me/player").mock(
            return_value=httpx.Response(200, json={"is_playing": False})
        )
        router.put("https://api.spotify.com/v1/me/player/play").mock(
            return_value=httpx.Response(204)
        )
        router.get("https://api.spotify.com/v1/tracks/fresh1").mock(
            return_value=httpx.Response(200, json={"duration_ms": 20_000})
        )
        pause_route = router.put("https://api.spotify.com/v1/me/player/pause").mock(
            return_value=httpx.Response(204)
        )

        rc = main(
            [
                "automated-playback",
                "--data-dir",
                str(tmp_path),
                "--device-name",
                "replay-browser",
            ]
        )

    assert rc == 0
    assert pause_route.called


def test_automated_playback_stops_early_when_nothing_to_play(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    _write_token(tmp_path)
    UserStore(root=tmp_path).grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope="automated-playback"
    )

    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.spotify.com/v1/me/tracks").mock(
            return_value=httpx.Response(200, json={"items": [], "next": None})
        )

        rc = main(
            [
                "automated-playback",
                "--data-dir",
                str(tmp_path),
                "--device-name",
                "replay-browser",
            ]
        )

    assert rc == 0
    assert "nothing to play" in capsys.readouterr().out
