"""Offline embedding-space near-duplicate batch merge (#140).

AC3: the fingerprint-match rule (offset search ±len/2 frames, BER <= 0.25
threshold over >=240 overlapping frames) is a pure function over two raw
uint32 chromaprint arrays -- no fpcalc call, no store, no I/O (AC6).
"""

from __future__ import annotations

from music_intel_mcp.near_dup import (
    FINGERPRINT_BER_THRESHOLD,
    FINGERPRINT_MIN_OVERLAP_FRAMES,
    match_fingerprints,
)


def _fp(n: int, seed: int = 0) -> list[int]:
    """Deterministic pseudo-random uint32 "fingerprint" of length n."""
    state = seed or 1
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        out.append(state)
    return out


def test_match_fingerprints_identical_arrays_zero_offset_zero_ber():
    fp = _fp(300)

    result = match_fingerprints(fp, fp)

    assert result is not None
    assert result.offset == 0
    assert result.ber == 0.0
    assert result.overlap == 300


def test_match_fingerprints_shifted_arrays_finds_offset():
    """A recording that starts 10 frames later than the other is still
    recognized once the offset search re-aligns them (CONTEXT.md 'Offline
    fingerprint comparison' -- offset search ±len/2 frames)."""
    base = _fp(300)
    shift = 10
    # fp_b[i] == fp_a[i + shift] for i in range(len(fp_a) - shift): fp_b is
    # fp_a's tail, followed by unrelated junk so lengths still line up.
    fp_b = base[shift:] + _fp(shift, seed=999)

    result = match_fingerprints(base, fp_b)

    assert result is not None
    assert result.offset == shift
    assert result.ber == 0.0
    assert result.overlap == len(base) - shift


def test_match_fingerprints_perturbed_bits_within_threshold():
    """A handful of flipped bits (encoding noise) keeps BER well under the
    0.25 threshold at zero offset."""
    fp_a = _fp(300)
    fp_b = list(fp_a)
    # Flip one bit in 10 of the 300 frames -> 10 mismatched bits out of
    # 300*32=9600 total -> BER ~= 0.001, far under the threshold.
    for i in range(0, 300, 30):
        fp_b[i] ^= 0b1

    result = match_fingerprints(fp_a, fp_b)

    assert result is not None
    assert result.offset == 0
    # 10 mismatched bits out of 300 frames * 32 bits/frame -- pins the BER
    # denominator (bits, not frames) so a normalization bug reddens this.
    assert result.ber == 10 / (300 * 32)
    assert result.ber < FINGERPRINT_BER_THRESHOLD
    assert result.overlap == 300


def test_match_fingerprints_mismatched_pair_exceeds_threshold():
    """Two unrelated fingerprints: the best-BER offset still lands well above
    the 0.25 threshold -- the caller (scan's decision tree) is responsible for
    rejecting on the threshold, match_fingerprints just reports the measurement."""
    fp_a = _fp(300, seed=1)
    fp_b = _fp(300, seed=2)

    result = match_fingerprints(fp_a, fp_b)

    assert result is not None
    assert result.ber > FINGERPRINT_BER_THRESHOLD


def test_match_fingerprints_none_when_no_offset_reaches_minimum_overlap():
    """Arrays shorter than the minimum-overlap floor can never produce an
    eligible offset -- honest None rather than a measurement built on too
    little evidence."""
    assert FINGERPRINT_MIN_OVERLAP_FRAMES > 50
    fp_a = _fp(50)
    fp_b = _fp(50)

    result = match_fingerprints(fp_a, fp_b)

    assert result is None
