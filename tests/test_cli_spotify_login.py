"""CLI `spotify-login` entrypoint (#127 AC1/AC5) — completes the
authorization-code+PKCE grant so `SpotifyUserAuth`'s helpers (built for this
issue but previously wired to nothing outside tests) become reachable.
`_wait_for_oauth_callback` is the one piece of genuinely new logic (parse the
redirect querystring, enforce the PKCE `state` anti-CSRF check); it is tested
directly against a real loopback socket below, then the full command is
tested with it monkeypatched out.
"""

from __future__ import annotations

import json
import threading
import time
from urllib.request import urlopen

import httpx
import respx

from music_intel_mcp import cli
from music_intel_mcp.cli import main


def test_wait_for_oauth_callback_returns_code():
    def _fire() -> None:
        time.sleep(0.2)
        urlopen("http://127.0.0.1:8765/callback?code=abc123&state=s1", timeout=5)

    threading.Thread(target=_fire, daemon=True).start()
    code = cli._wait_for_oauth_callback(port=8765, expected_state="s1", timeout=5)
    assert code == "abc123"


def test_wait_for_oauth_callback_rejects_state_mismatch():
    def _fire() -> None:
        time.sleep(0.2)
        urlopen("http://127.0.0.1:8766/callback?code=abc123&state=wrong", timeout=5)

    threading.Thread(target=_fire, daemon=True).start()
    try:
        cli._wait_for_oauth_callback(port=8766, expected_state="s1", timeout=5)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "state mismatch" in str(exc)


def test_wait_for_oauth_callback_times_out_cleanly():
    try:
        cli._wait_for_oauth_callback(port=8767, expected_state="s1", timeout=0.3)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "timed out" in str(exc)


def test_spotify_login_requires_client_id(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    rc = main(["spotify-login", "--data-dir", str(tmp_path)])
    assert rc == 2
    assert "SPOTIFY_CLIENT_ID" in capsys.readouterr().out


def test_spotify_login_completes_exchange(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")
    monkeypatch.setattr(cli, "_wait_for_oauth_callback", lambda **_kw: "auth-code-fixture")

    with respx.mock(assert_all_called=True) as router:
        token_route = router.post("https://accounts.spotify.com/api/token").mock(
            return_value=httpx.Response(
                200,
                json={
                    "access_token": "opaque-bearer-fixture",
                    "refresh_token": "refresh-fixture",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        )
        rc = main(["spotify-login", "--data-dir", str(tmp_path)])

    assert rc == 0
    assert "authorized" in capsys.readouterr().out
    assert token_route.called
    body = token_route.calls[0].request.content.decode()
    assert "code=auth-code-fixture" in body

    token_path = tmp_path / "spotify_user_token.json"
    assert json.loads(token_path.read_text())["access_token"] == "opaque-bearer-fixture"


def test_spotify_login_surfaces_callback_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client123")

    def _raise(**_kw):
        raise RuntimeError("OAuth state mismatch — possible CSRF, aborting")

    monkeypatch.setattr(cli, "_wait_for_oauth_callback", _raise)

    rc = main(["spotify-login", "--data-dir", str(tmp_path)])
    assert rc == 2
    assert "state mismatch" in capsys.readouterr().out
