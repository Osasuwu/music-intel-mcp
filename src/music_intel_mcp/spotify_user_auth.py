"""User-OAuth (authorization-code + PKCE) for the backfill playlist (#127
AC1/AC5) — the project's first Spotify scope beyond client-credentials
(see :mod:`music_intel_mcp.spotify_api`).

Tokens are persisted to a local JSON file only (decision f7a9fcbd: personal
data never leaves the machine) — never to the shared Supabase metadata store.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

SPOTIFY_AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"

# user-library-read: candidate pool = the user's saved tracks (decision
# 8dd6ef53). playlist-modify-*: AC1's playlist create/manage.
PLAYLIST_SCOPES = (
    "user-library-read",
    "playlist-modify-public",
    "playlist-modify-private",
)

# #128: automated playback drives a real playback session (not just a
# playlist), so it needs its own scopes beyond PLAYLIST_SCOPES.
PLAYBACK_SCOPES = (
    "user-modify-playback-state",
    "user-read-playback-state",
)

_TOKEN_EXPIRY_MARGIN_S = 60


def generate_pkce_pair() -> tuple[str, str]:
    """A ``(code_verifier, code_challenge)`` pair per RFC 7636 — the
    challenge is the base64url(no-padding) SHA-256 digest of the verifier."""
    verifier = secrets.token_urlsafe(64)[:96]
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    state: str,
    scopes: tuple[str, ...] = PLAYLIST_SCOPES,
) -> str:
    """The URL to send the user to for the authorization-code+PKCE grant,
    scoped to ``scopes`` (default :data:`PLAYLIST_SCOPES`; the CLI passes
    ``PLAYLIST_SCOPES + PLAYBACK_SCOPES`` so one login covers both #127 and
    #128)."""
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
        "state": state,
        "scope": " ".join(scopes),
    }
    return f"{SPOTIFY_AUTHORIZE_URL}?{urlencode(params)}"


class SpotifyUserAuth:
    """Exchanges/refreshes a user's OAuth token and persists it to a local
    JSON file at ``token_path`` (mirrors :class:`UserStore`'s local-file-only
    convention — see :mod:`music_intel_mcp.store`)."""

    def __init__(self, *, client_id: str, token_path: Path) -> None:
        self.client_id = client_id
        self.token_path = Path(token_path)

    def is_authorized(self) -> bool:
        return self.token_path.exists()

    def exchange_code(self, *, code: str, redirect_uri: str, code_verifier: str) -> None:
        response = httpx.post(
            SPOTIFY_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": self.client_id,
                "code_verifier": code_verifier,
            },
        )
        response.raise_for_status()
        self._store(response.json())

    def access_token(self) -> str:
        record = self._load()
        if record["expires_at"] > time.time():
            return record["access_token"]
        return self._refresh(record["refresh_token"])

    def _refresh(self, refresh_token: str) -> str:
        response = httpx.post(
            SPOTIFY_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
            },
        )
        response.raise_for_status()
        payload = response.json()
        payload.setdefault("refresh_token", refresh_token)
        self._store(payload)
        return payload["access_token"]

    def _store(self, payload: dict) -> None:
        record = {
            "access_token": payload["access_token"],
            "refresh_token": payload["refresh_token"],
            "expires_at": time.time() + payload["expires_in"] - _TOKEN_EXPIRY_MARGIN_S,
            "token_type": payload.get("token_type", "Bearer"),
        }
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(record), encoding="utf-8")

    def _load(self) -> dict:
        return json.loads(self.token_path.read_text(encoding="utf-8"))
