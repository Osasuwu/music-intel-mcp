"""#201 AC1 production wiring: ``replay-capture-youtube`` -- the real CLI
entry point for the stream-decode leg (there is no loopback-leg CLI command
in this repo, see ``_cmd_replay_capture_youtube``'s own docstring) -- must
pass a window-probe recorder through as ``on_capture_analyzed`` so the
pre-pilot gate rides passively on real pilot sessions instead of needing a
separate measurement run.
"""

from __future__ import annotations

from music_intel_mcp.cli import main


def test_replay_capture_youtube_wires_the_window_probe_recorder(tmp_path, monkeypatch):
    captured = {}

    def fake_process_stream_decode_queue(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "music_intel_mcp.stream_decode.process_stream_decode_queue",
        fake_process_stream_decode_queue,
    )
    monkeypatch.setattr("music_intel_mcp.inference.DiscogsEffnetOnnxModel", lambda: object())
    monkeypatch.setattr("music_intel_mcp.inference.MtgJamendoClassifier", lambda: object())

    rc = main(["replay-capture-youtube", "--data-dir", str(tmp_path)])

    assert rc == 0
    assert captured["on_capture_analyzed"] is not None
    assert callable(captured["on_capture_analyzed"])
