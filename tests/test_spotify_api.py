"""Live Spotify Web API ISRC source (#87 AC-A).

Every test mocks the network with ``respx`` — CLAUDE.md forbids live API calls in
CI, and the real client-credentials flow costs a token + rate budget. The source
under test owns a persistent JSONL cache (positive *and* negative), batches the
``GET /v1/tracks`` calls at 50, and backs off on 429/503 — all verified here
against a mocked ``accounts.spotify.com`` / ``api.spotify.com``.
"""

from __future__ import annotations

import json

import httpx
import respx

from music_intel_mcp.spotify_api import (
    SPOTIFY_SEARCH_URL,
    SPOTIFY_TOKEN_URL,
    SPOTIFY_TRACKS_URL,
    SpotifyApiIsrcSource,
    SpotifySearchApiSource,
)

_CREDS = {"client_id": "cid", "client_secret": "csecret"}


def _token_route(router) -> None:
    router.post(SPOTIFY_TOKEN_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
    )


def _tracks_response(request: httpx.Request, isrcs: dict[str, str | None]) -> httpx.Response:
    """Mimic ``GET /v1/tracks?ids=`` — a ``tracks`` array positionally matching the
    requested ids, ``null`` for an unknown id, ``external_ids.isrc`` for a hit."""
    ids = request.url.params.get("ids", "").split(",")
    tracks: list[dict | None] = []
    for tid in ids:
        if tid not in isrcs:
            tracks.append(None)  # Spotify returns null for an unknown id
        else:
            isrc = isrcs[tid]
            ext = {"isrc": isrc} if isrc is not None else {}
            tracks.append({"id": tid, "external_ids": ext})
    return httpx.Response(200, json={"tracks": tracks})


def _read_cache(path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["spotify_id"]] = row["isrc"]
    return out


def test_warm_batches_fetches_and_caches_hits_and_misses(tmp_path):
    cache = tmp_path / "spotify_isrc_cache.jsonl"
    isrcs = {"A": "USABC0000001", "B": "GBXYZ0000002", "C": None}  # C is a miss
    source = SpotifyApiIsrcSource(**_CREDS, cache_path=cache, sleep=lambda _s: None)

    with respx.mock(assert_all_called=False) as router:
        _token_route(router)
        router.get(SPOTIFY_TRACKS_URL).mock(
            side_effect=lambda request: _tracks_response(request, isrcs)
        )
        fetched = source.warm(["A", "B", "C"])

    assert fetched == 3
    # Both hits and the miss land in the persistent cache (negative-cache).
    assert _read_cache(cache) == {"A": "USABC0000001", "B": "GBXYZ0000002", "C": None}
    # lookup now reads the cache, no further API traffic.
    assert source.lookup("A") == "USABC0000001"
    assert source.lookup("C") is None


def test_second_warm_makes_no_api_call_for_cached_ids(tmp_path):
    cache = tmp_path / "cache.jsonl"
    source = SpotifyApiIsrcSource(**_CREDS, cache_path=cache, sleep=lambda _s: None)
    isrcs = {"A": "USABC0000001"}

    with respx.mock(assert_all_called=False) as router:
        _token_route(router)
        route = router.get(SPOTIFY_TRACKS_URL).mock(
            side_effect=lambda request: _tracks_response(request, isrcs)
        )
        source.warm(["A"])
        first = route.call_count
        source.warm(["A"])  # already cached — must not hit the API again
        assert route.call_count == first


def test_warm_caps_batches_at_fifty(tmp_path):
    cache = tmp_path / "cache.jsonl"
    ids = [f"id{i:03d}" for i in range(60)]
    isrcs = {i: f"ISRC{i}" for i in ids}
    source = SpotifyApiIsrcSource(**_CREDS, cache_path=cache, sleep=lambda _s: None)

    with respx.mock(assert_all_called=False) as router:
        _token_route(router)
        route = router.get(SPOTIFY_TRACKS_URL).mock(
            side_effect=lambda request: _tracks_response(request, isrcs)
        )
        source.warm(ids)
        # 60 ids -> two GET calls (50 + 10); each batch <= 50 ids.
        assert route.call_count == 2
        for call in route.calls:
            assert len(call.request.url.params.get("ids").split(",")) <= 50


def test_lookup_fetches_on_demand_when_not_warmed(tmp_path):
    cache = tmp_path / "cache.jsonl"
    source = SpotifyApiIsrcSource(**_CREDS, cache_path=cache, sleep=lambda _s: None)

    with respx.mock(assert_all_called=False) as router:
        _token_route(router)
        router.get(SPOTIFY_TRACKS_URL).mock(
            side_effect=lambda request: _tracks_response(request, {"Z": "USZZZ0000009"})
        )
        assert source.lookup("Z") == "USZZZ0000009"
    # persisted for the next run
    assert _read_cache(cache) == {"Z": "USZZZ0000009"}


def test_backoff_retries_on_429_then_succeeds(tmp_path):
    cache = tmp_path / "cache.jsonl"
    slept: list[float] = []
    source = SpotifyApiIsrcSource(**_CREDS, cache_path=cache, sleep=slept.append)

    with respx.mock(assert_all_called=False) as router:
        _token_route(router)
        hit = {"tracks": [{"id": "A", "external_ids": {"isrc": "US1"}}]}
        router.get(SPOTIFY_TRACKS_URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "2"}),
                httpx.Response(200, json=hit),
            ]
        )
        assert source.lookup("A") == "US1"

    assert slept == [2.0]  # honored Retry-After before the retry


def test_resume_skips_ids_already_in_cache(tmp_path):
    cache = tmp_path / "cache.jsonl"
    # A prior (interrupted) run already resolved A and recorded B as a miss.
    cache.write_text(
        '{"spotify_id": "A", "isrc": "USOLD0000001"}\n{"spotify_id": "B", "isrc": null}\n',
        encoding="utf-8",
    )
    source = SpotifyApiIsrcSource(**_CREDS, cache_path=cache, sleep=lambda _s: None)

    with respx.mock(assert_all_called=False) as router:
        _token_route(router)
        route = router.get(SPOTIFY_TRACKS_URL).mock(
            side_effect=lambda request: _tracks_response(request, {"C": "USNEW0000003"})
        )
        fetched = source.warm(["A", "B", "C"])  # only C is unknown

    assert fetched == 1
    assert route.calls[0].request.url.params.get("ids") == "C"
    assert source.lookup("A") == "USOLD0000001"  # untouched
    assert source.lookup("C") == "USNEW0000003"


def test_lookup_strips_spotify_track_uri_prefix(tmp_path):
    cache = tmp_path / "cache.jsonl"
    source = SpotifyApiIsrcSource(**_CREDS, cache_path=cache, sleep=lambda _s: None)

    with respx.mock(assert_all_called=False) as router:
        _token_route(router)
        route = router.get(SPOTIFY_TRACKS_URL).mock(
            side_effect=lambda request: _tracks_response(request, {"A": "USABC0000001"})
        )
        assert source.lookup("spotify:track:A") == "USABC0000001"
        # the bare id, not the URI, is what hit the API and the cache
        assert route.calls[0].request.url.params.get("ids") == "A"
    assert _read_cache(cache) == {"A": "USABC0000001"}


def test_missing_credentials_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
    source = SpotifyApiIsrcSource(cache_path=tmp_path / "cache.jsonl")
    try:
        source.lookup("A")
    except RuntimeError as exc:
        assert "SPOTIFY_CLIENT_ID" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a RuntimeError for missing credentials")


def test_credentials_read_from_env_when_not_passed(tmp_path, monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "envid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "envsecret")
    cache = tmp_path / "cache.jsonl"
    source = SpotifyApiIsrcSource(cache_path=cache, sleep=lambda _s: None)

    with respx.mock(assert_all_called=False) as router:
        token = router.post(SPOTIFY_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        )
        router.get(SPOTIFY_TRACKS_URL).mock(
            side_effect=lambda request: _tracks_response(request, {"A": "USABC0000001"})
        )
        assert source.lookup("A") == "USABC0000001"
        # the token call carried a Basic auth header derived from the env creds
        assert token.called
        assert token.calls[0].request.headers["Authorization"].startswith("Basic ")


# --- SpotifySearchApiSource (#139 AC2) -------------------------------------- #


def test_search_returns_top_result_track_id():
    isrc_source = SpotifyApiIsrcSource(**_CREDS, sleep=lambda _s: None)
    source = SpotifySearchApiSource(isrc_source=isrc_source)

    with respx.mock(assert_all_called=False) as router:
        _token_route(router)
        router.get(SPOTIFY_SEARCH_URL).mock(
            return_value=httpx.Response(
                200, json={"tracks": {"items": [{"id": "SP-1"}, {"id": "SP-2"}]}}
            )
        )
        assert source.search(title="Song", artist="Artist") == "SP-1"


def test_search_returns_none_for_no_results():
    isrc_source = SpotifyApiIsrcSource(**_CREDS, sleep=lambda _s: None)
    source = SpotifySearchApiSource(isrc_source=isrc_source)

    with respx.mock(assert_all_called=False) as router:
        _token_route(router)
        router.get(SPOTIFY_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"tracks": {"items": []}})
        )
        assert source.search(title="Nonexistent", artist="Nobody") is None


def test_search_reuses_composed_isrc_source_credentials():
    """Default-constructing without ``isrc_source`` still requires credentials --
    the composed :class:`SpotifyApiIsrcSource` owns that check."""
    empty_isrc_source = SpotifyApiIsrcSource(client_id=None, client_secret=None)
    source = SpotifySearchApiSource(isrc_source=empty_isrc_source)
    try:
        source.search(title="Song", artist="Artist")
    except RuntimeError as exc:
        assert "SPOTIFY_CLIENT_ID" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a RuntimeError for missing credentials")
