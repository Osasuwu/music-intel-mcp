"""HTTP-mocked half of the backfill playlist (#127 AC1/AC2/AC3) —
:class:`SpotifyPlaylistClient` and :func:`sync_backfill_playlist` against a
mocked Spotify Web API. Pure selection/diff logic is covered separately in
``test_backfill_playlist.py``; this file exercises the network-touching
create/read/add/remove calls and the end-to-end daily-refresh orchestration.
"""

from __future__ import annotations

import httpx
import respx

from music_intel_mcp.backfill_playlist import (
    SPOTIFY_PLAYLISTS_URL,
    SPOTIFY_SAVED_TRACKS_URL,
    SpotifyPlaylistClient,
    fetch_current_user_id,
    fetch_saved_track_refs,
    playlist_tracks_url,
    sync_backfill_playlist,
)

_BEARER = "opaque-bearer-fixture"


def _client() -> SpotifyPlaylistClient:
    return SpotifyPlaylistClient(access_token=lambda: _BEARER, user_id="u1")


# --- AC1: playlist auto-created / reused ----------------------------------- #


def test_get_or_create_playlist_creates_when_absent():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYLISTS_URL).mock(
            return_value=httpx.Response(200, json={"items": [], "next": None})
        )
        router.post("https://api.spotify.com/v1/users/u1/playlists").mock(
            return_value=httpx.Response(201, json={"id": "new_playlist"})
        )
        playlist_id = client.get_or_create_playlist(name="music-intel: to-analyze")

    assert playlist_id == "new_playlist"


def test_get_or_create_playlist_reuses_existing_by_name():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYLISTS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [{"id": "existing", "name": "music-intel: to-analyze"}],
                    "next": None,
                },
            )
        )
        playlist_id = client.get_or_create_playlist(name="music-intel: to-analyze")

    assert playlist_id == "existing"


# --- candidate pool: saved/library tracks (decision 8dd6ef53) -------------- #


def test_fetch_saved_track_refs_paginates_and_maps_fields():
    page2_url = "https://api.spotify.com/v1/me/tracks?offset=50"

    def _paged_response(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("offset") == "50":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "track": {
                                "id": "b2",
                                "name": "Song B",
                                "artists": [{"name": "Artist B"}],
                                "album": {"name": "Album B"},
                            }
                        }
                    ],
                    "next": None,
                },
            )
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "track": {
                            "id": "a1",
                            "name": "Song A",
                            "artists": [{"name": "Artist A"}],
                            "album": {"name": "Album A"},
                        }
                    }
                ],
                "next": page2_url,
            },
        )

    with respx.mock(assert_all_called=True) as router:
        router.get(url__regex=r".*/me/tracks.*").mock(side_effect=_paged_response)
        refs = fetch_saved_track_refs(access_token=lambda: _BEARER)

    assert [r.spotify_id for r in refs] == ["a1", "b2"]
    assert refs[0].name == "Song A"
    assert refs[0].artist == "Artist A"
    assert refs[0].album == "Album A"


def test_fetch_saved_track_refs_uses_saved_tracks_url():
    assert SPOTIFY_SAVED_TRACKS_URL == "https://api.spotify.com/v1/me/tracks"


def test_fetch_current_user_id():
    with respx.mock(assert_all_called=True) as router:
        router.get("https://api.spotify.com/v1/me").mock(
            return_value=httpx.Response(200, json={"id": "the_user"})
        )
        user_id = fetch_current_user_id(access_token=lambda: _BEARER)

    assert user_id == "the_user"


# --- track id listing (paginated) ------------------------------------------ #


def test_get_playlist_track_ids_follows_pagination():
    client = _client()
    page2_url = "https://api.spotify.com/v1/playlists/pl1/tracks?offset=100"

    def _paged_response(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("offset") == "100":
            return httpx.Response(200, json={"items": [{"track": {"id": "c"}}], "next": None})
        return httpx.Response(
            200,
            json={
                "items": [{"track": {"id": "a"}}, {"track": {"id": "b"}}],
                "next": page2_url,
            },
        )

    with respx.mock(assert_all_called=True) as router:
        router.get(url__regex=r".*/playlists/pl1/tracks.*").mock(side_effect=_paged_response)
        track_ids = client.get_playlist_track_ids("pl1")

    assert track_ids == ["a", "b", "c"]


# --- AC3: end-to-end sync adds newly-desired, removes now-analyzed -------- #


def test_sync_backfill_playlist_adds_and_removes():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYLISTS_URL).mock(
            return_value=httpx.Response(
                200,
                json={"items": [{"id": "pl1", "name": "music-intel: to-analyze"}], "next": None},
            )
        )
        router.get(playlist_tracks_url("pl1")).mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [{"track": {"id": "spotify:stale"}}],
                    "next": None,
                },
            )
        )
        add_route = router.post(playlist_tracks_url("pl1")).mock(
            return_value=httpx.Response(201, json={})
        )
        remove_route = router.request("DELETE", playlist_tracks_url("pl1")).mock(
            return_value=httpx.Response(200, json={})
        )

        result = sync_backfill_playlist(
            client,
            desired_ids=["spotify:fresh"],
            playlist_name="music-intel: to-analyze",
        )

    assert result.to_add == ["spotify:fresh"]
    assert result.to_remove == ["spotify:stale"]
    add_body = add_route.calls[0].request.content
    assert b"spotify:track:fresh" in add_body
    remove_body = remove_route.calls[0].request.content
    assert b"spotify:track:stale" in remove_body


def test_sync_backfill_playlist_is_noop_when_membership_already_matches():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYLISTS_URL).mock(
            return_value=httpx.Response(
                200,
                json={"items": [{"id": "pl1", "name": "music-intel: to-analyze"}], "next": None},
            )
        )
        router.get(playlist_tracks_url("pl1")).mock(
            return_value=httpx.Response(
                200,
                json={"items": [{"track": {"id": "spotify:keep"}}], "next": None},
            )
        )

        result = sync_backfill_playlist(
            client,
            desired_ids=["spotify:keep"],
            playlist_name="music-intel: to-analyze",
        )

    assert result.to_add == []
    assert result.to_remove == []
