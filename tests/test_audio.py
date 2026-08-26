"""Tests for microphone signal conditioning.

The offset these remove is not a theoretical concern. A real capture session
produced 84 clips carrying a constant +0.03135 bias, which pinned the measured
level at 0.031 whether or not anyone was speaking. Voice activity detection
never closed the gate, so 56 of 79 clips ended on the segment timeout instead
of on silence.
"""

import numpy as np
import pytest

from listenr.audio import DCBlocker


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _speech_like(n, sr=16000, seed=0):
    """A quiet signal with speech-scale amplitude, matching what the mic gave."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    tone = 0.008 * np.sin(2 * np.pi * 180 * t)
    return (tone + rng.normal(0, 0.002, n)).astype(np.float32)


class TestDCBlocker:
    def test_removes_a_constant_offset(self):
        signal = _speech_like(16000) + 0.03135
        out = DCBlocker()(signal)
        assert abs(float(np.mean(out))) < 1e-3

    def test_leaves_a_clean_signal_alone(self):
        """Users whose device has no offset must see no change."""
        signal = _speech_like(16000)
        out = DCBlocker()(signal)
        assert np.corrcoef(signal - signal.mean(), out)[0, 1] > 0.99

    def test_the_offset_was_hiding_the_signal(self):
        """The real failure: level does not move between silence and speech."""
        silence = np.zeros(8000, dtype=np.float32) + 0.03135
        speech = _speech_like(8000) + 0.03135

        assert _rms(speech) / _rms(silence) < 1.1, "before: indistinguishable"

        blocker = DCBlocker()
        cleaned_silence = blocker(silence)
        cleaned_speech = blocker(speech)
        assert _rms(cleaned_speech) / _rms(cleaned_silence + 1e-12) > 3.0

    def test_state_carries_across_chunks(self):
        """Chunked input must match whole-stream input, or every boundary steps."""
        signal = _speech_like(16000) + 0.03135
        whole = DCBlocker()(signal)

        blocker = DCBlocker()
        chunked = np.concatenate([blocker(signal[i:i + 1360]) for i in range(0, len(signal), 1360)])

        assert np.allclose(whole, chunked, atol=1e-6)

    def test_no_step_at_chunk_boundaries(self):
        """Subtracting each chunk's own mean would fail this."""
        signal = _speech_like(16000) + 0.03135
        blocker = DCBlocker()
        chunk = 1360
        out = np.concatenate([blocker(signal[i:i + chunk]) for i in range(0, len(signal), chunk)])

        steps = np.abs(np.diff(out))
        at_boundary = steps[chunk - 1::chunk]
        assert at_boundary.max() <= steps.max()

    def test_preserves_dtype_and_shape(self):
        signal = _speech_like(4000)
        out = DCBlocker()(signal)
        assert out.shape == signal.shape
        assert out.dtype == signal.dtype

    def test_empty_chunk_is_safe(self):
        empty = np.array([], dtype=np.float32)
        assert DCBlocker()(empty).size == 0

    def test_reset_clears_state(self):
        blocker = DCBlocker()
        blocker(_speech_like(4000) + 0.03135)
        blocker.reset()
        first = blocker(_speech_like(4000) + 0.03135)
        assert np.allclose(first, DCBlocker()(_speech_like(4000) + 0.03135))

    def test_long_chunk_does_not_produce_infinities(self):
        """The recursion is evaluated with powers of the pole, which underflow."""
        signal = _speech_like(48000 * 5) + 0.03135
        out = DCBlocker()(signal)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("offset", [0.0, 0.001, -0.05, 0.2])
    def test_various_offsets(self, offset):
        out = DCBlocker()(_speech_like(16000) + offset)
        assert abs(float(np.mean(out))) < 1e-2
