"""User-OAuth (authorization-code + PKCE) for the backfill playlist (#127
AC1/AC5) - the project's first Spotify scope beyond client-credentials.

Tokens are local-only (decision f7a9fcbd). Values below are opaque test
fixtures, not real credentials.
"""

from __future__ import annotations

import base64
import hashlib
import json

import httpx
import respx

from music_intel_mcp.spotify_user_auth import (
    PLAYBACK_SCOPES,
    PLAYLIST_SCOPES,
    SPOTIFY_AUTHORIZE_URL,
    SPOTIFY_TOKEN_URL,
    SpotifyUserAuth,
    build_authorize_url,
    generate_pkce_pair,
)

_PLACEHOLDER_ACCESS = "opaque-access-fixture"
_PLACEHOLDER_REFRESH = "opaque-refresh-fixture"
_PLACEHOLDER_STALE_ACCESS = "opaque-stale-access-fixture"
_PLACEHOLDER_FRESH_ACCESS = "opaque-fresh-access-fixture"
_PLACEHOLDER_CACHED_ACCESS = "opaque-cached-access-fixture"


def test_generate_pkce_pair_challenge_is_s256_of_verifier():
    verifier, challenge = generate_pkce_pair()
    assert 43 <= len(verifier) <= 128
    assert challenge != verifier

    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    expected = expected.rstrip(b"=").decode()
    assert challenge == expected


def test_build_authorize_url_includes_pkce_and_playlist_scopes():
    url = build_authorize_url(
        client_id="cid",
        redirect_uri="http://127.0.0.1:8765/callback",
        code_challenge="chal123",
        state="st8",
    )
    assert url.startswith(SPOTIFY_AUTHORIZE_URL)
    assert "client_id=cid" in url
    assert "code_challenge=chal123" in url
    assert "code_challenge_method=S256" in url
    assert "state=st8" in url
    for scope in PLAYLIST_SCOPES:
        assert scope in url


# #128: automated playback needs PLAYBACK_SCOPES on top of PLAYLIST_SCOPES,
# via one combined login rather than a second OAuth flow.
def test_build_authorize_url_accepts_combined_scopes():
    url = build_authorize_url(
        client_id="cid",
        redirect_uri="http://127.0.0.1:8765/callback",
        code_challenge="chal123",
        state="st8",
        scopes=PLAYLIST_SCOPES + PLAYBACK_SCOPES,
    )
    for scope in PLAYLIST_SCOPES + PLAYBACK_SCOPES:
        assert scope in url


def test_exchange_code_persists_tokens_to_local_store(tmp_path):
    token_path = tmp_path / "spotify_user_token.json"
    auth = SpotifyUserAuth(client_id="cid", token_path=token_path)

    with respx.mock(assert_all_called=True) as router:
        router.post(SPOTIFY_TOKEN_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "access_token": _PLACEHOLDER_ACCESS,
                    "refresh_token": _PLACEHOLDER_REFRESH,
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        )
        auth.exchange_code(code="authcode", redirect_uri="http://x/cb", code_verifier="v")

    assert token_path.exists()
    stored = json.loads(token_path.read_text(encoding="utf-8"))
    assert stored["access_token"] == _PLACEHOLDER_ACCESS
    assert stored["refresh_token"] == _PLACEHOLDER_REFRESH


def test_access_token_refreshes_when_expired(tmp_path):
    token_path = tmp_path / "spotify_user_token.json"
    stale = {
        "access_token": _PLACEHOLDER_STALE_ACCESS,
        "refresh_token": _PLACEHOLDER_REFRESH,
        "expires_at": 0.0,
        "token_type": "Bearer",
    }
    token_path.write_text(json.dumps(stale), encoding="utf-8")
    auth = SpotifyUserAuth(client_id="cid", token_path=token_path)

    with respx.mock(assert_all_called=True) as router:
        router.post(SPOTIFY_TOKEN_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "access_token": _PLACEHOLDER_FRESH_ACCESS,
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        )
        result = auth.access_token()

    assert result == _PLACEHOLDER_FRESH_ACCESS


def test_access_token_reuses_unexpired_token_without_network(tmp_path):
    token_path = tmp_path / "spotify_user_token.json"
    good = {
        "access_token": _PLACEHOLDER_CACHED_ACCESS,
        "refresh_token": _PLACEHOLDER_REFRESH,
        "expires_at": 9_999_999_999.0,
        "token_type": "Bearer",
    }
    token_path.write_text(json.dumps(good), encoding="utf-8")
    auth = SpotifyUserAuth(client_id="cid", token_path=token_path)

    with respx.mock(assert_all_called=False) as router:
        router.post(SPOTIFY_TOKEN_URL).mock(
            side_effect=AssertionError("must not hit the network for an unexpired token")
        )
        result = auth.access_token()

    assert result == _PLACEHOLDER_CACHED_ACCESS


def test_is_authorized_false_when_no_token_file(tmp_path):
    auth = SpotifyUserAuth(client_id="cid", token_path=tmp_path / "missing.json")
    assert auth.is_authorized() is False
