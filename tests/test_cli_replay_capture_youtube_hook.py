"""#201 AC1 production wiring: ``replay-capture-youtube`` -- the real CLI
entry point for the stream-decode leg (there is no loopback-leg CLI command
in this repo, see ``_cmd_replay_capture_youtube``'s own docstring) -- must
pass a window-probe recorder through as ``on_capture_analyzed`` so the
pre-pilot gate rides passively on real pilot sessions instead of needing a
separate measurement run.
"""

from __future__ import annotations

import numpy as np

from music_intel_mcp.cli import main
from music_intel_mcp.inference import InMemoryEmbeddingModel
from music_intel_mcp.store import UserStore
from music_intel_mcp.window_probe import (
    WindowPair,
    load_window_pairs,
    stream_decode_window_probe_path,
)


def test_replay_capture_youtube_wires_the_window_probe_recorder(tmp_path, monkeypatch):
    captured = {}

    def fake_process_stream_decode_queue(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "music_intel_mcp.stream_decode.process_stream_decode_queue",
        fake_process_stream_decode_queue,
    )
    monkeypatch.setattr(
        "music_intel_mcp.inference.DiscogsEffnetOnnxModel",
        lambda: InMemoryEmbeddingModel(np.array([0.0, 1.0])),
    )
    monkeypatch.setattr("music_intel_mcp.inference.MtgJamendoClassifier", lambda: object())

    rc = main(["replay-capture-youtube", "--data-dir", str(tmp_path)])
    assert rc == 0

    # Fire the hook the way run_stream_decode_capture does on an accepted
    # decode: a 2 s mono buffer whose whole-track embedding is already known.
    # The recorder must journal one pair -- long leg = the passed embedding,
    # short leg = the CLI's embedding model over the (whole, < 30 s) buffer --
    # at the stream-decode leg's own journal path (#232).
    captured["on_capture_analyzed"](
        track_id="youtube:abc123",
        pcm=np.zeros(32000),
        sample_rate=16000,
        embedding=[1.0, 0.0],
        tags={},
    )

    assert load_window_pairs(stream_decode_window_probe_path(UserStore(root=tmp_path))) == [
        WindowPair(
            track_id="youtube:abc123",
            long_window_s=2.0,
            short_window_s=2.0,
            long_embedding=[1.0, 0.0],
            short_embedding=[0.0, 1.0],
        )
    ]
