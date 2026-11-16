"""Generate the #198 front-end fixture with Essentia's own TensorflowInputMusiCNN.

Not part of the package or the test run: Essentia has no Windows wheels and is
AGPL, so it only ever runs in a throwaway Linux container, and only its output
(data) is committed. Recipe used for the committed fixture:

    docker run --rm -v <repo>:/repo python:3.11-slim-bookworm sh -c \
      "pip install essentia 'numpy<2.5' && python -I /repo/scripts/gen_essentia_mel_fixture.py \
       /repo/tests/fixtures/essentia_tensorflow_input_musicnn.npz"

(essentia 2.1-beta6-dev). Writes an .npz with:
  signal  float32 (N,)          -- 16 kHz mono, RMS 0.1 (the front-end's normalization target)
  logmel  float32 (frames, 96)  -- es.TensorflowInputMusiCNN per frame
"""

import sys

import essentia
import essentia.standard as es
import numpy as np

SR = 16000
N = 32 * 256  # 0.512 s; a hop multiple, where Essentia and librosa center=True agree on frames

rng = np.random.default_rng(198)
t = np.arange(N) / SR
# Sparse tonal content (where power vs magnitude diverge most, per the #198
# measurement) plus a quiet broadband floor so every band has some energy.
y = (
    0.6 * np.sin(2 * np.pi * 220.0 * t)
    + 0.3 * np.sin(2 * np.pi * 1318.5 * t)
    + 0.15 * np.sin(2 * np.pi * (500.0 + 3000.0 * t) * t)
    + 0.003 * rng.standard_normal(N)
)
y = y * (0.1 / np.sqrt(np.mean(y**2)))
y = y.astype(np.float32)

tim = es.TensorflowInputMusiCNN()
frames = [tim(f) for f in es.FrameGenerator(y, frameSize=512, hopSize=256, startFromZero=False)]
logmel = np.asarray(frames, dtype=np.float32)
print("essentia", essentia.__version__, "frames", logmel.shape)
np.savez_compressed(sys.argv[1], signal=y, logmel=logmel)
