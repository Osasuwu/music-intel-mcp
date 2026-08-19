"""Tests for now-playing detection (#124 AC4): SMTC seam + identity wiring."""

from __future__ import annotations

from music_intel_mcp.identity import (
    IdentityResolver,
    InMemoryIsrcMbidIndex,
    InMemorySpotifyIsrcSource,
)
from music_intel_mcp.nowplaying import InMemoryNowPlayingSource, NowPlayingInfo, resolve_now_playing


# AC4: track correctly identified via now-playing metadata -> identity waterfall.
def test_resolve_now_playing_walks_the_identity_waterfall() -> None:
    source = InMemoryNowPlayingSource(
        NowPlayingInfo(
            title="Around the World",
            artist="Daft Punk",
            app_id="Spotify.exe",
            process_id=4242,
            spotify_id="1pKYYY0dkg23sQQXi0Q5zN",
        )
    )
    resolver = IdentityResolver(
        InMemoryIsrcMbidIndex({"FRDM19700001": "mbid-around-the-world"}),
        spotify_source=InMemorySpotifyIsrcSource({"1pKYYY0dkg23sQQXi0Q5zN": "FRDM19700001"}),
    )

    identified = resolve_now_playing(source, resolver)

    assert identified is not None
    now_playing, identity = identified
    assert now_playing.process_id == 4242
    assert identity.resolved
    assert identity.mbid == "mbid-around-the-world"
    assert identity.name == "Around the World"
    assert identity.artist == "Daft Punk"


def test_resolve_now_playing_none_when_nothing_playing() -> None:
    source = InMemoryNowPlayingSource(None)
    resolver = IdentityResolver(InMemoryIsrcMbidIndex())

    assert resolve_now_playing(source, resolver) is None
