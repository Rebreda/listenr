"""
audio.py — Signal conditioning applied between the microphone and everything else.

Public API
----------
DCBlocker  -> stateful high-pass that removes a constant offset from a stream
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, lfilter_zi

# One-pole coefficient. At 48 kHz this puts the corner near 8 Hz, far below
# speech, so it removes a constant offset and drifting bias without touching
# anything audible.
DC_BLOCKER_POLE = 0.999


class DCBlocker:
    """Remove a DC offset from a stream arriving in chunks.

    Some capture devices return audio with a constant bias. It is inaudible,
    but it dominates any energy measurement: a chunk whose real signal sits
    around 0.008 RMS reads as 0.031 when it carries a 0.031 offset, and that
    reading never changes whether or not anyone is speaking.

    That breaks voice activity detection completely. The gate sees a level
    that never drops, so it never closes, and segments only ever end on a
    timeout. It also wastes headroom before clipping.

    Subtracting each chunk's own mean would introduce a step at every chunk
    boundary. This is the standard one-pole DC blocker instead,

        y[n] = x[n] - x[n-1] + R * y[n-1]

    run through ``scipy.signal.lfilter`` with its filter state carried across
    chunks, so the output is continuous and identical to filtering the whole
    stream at once. The state is primed from the first sample so a session
    does not open with a transient.
    """

    def __init__(self, pole: float = DC_BLOCKER_POLE):
        self.pole = pole
        # y[n] = x[n] - x[n-1] + pole*y[n-1]
        self._b = np.array([1.0, -1.0])
        self._a = np.array([1.0, -pole])
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """Return *chunk* with its offset removed. Shape and dtype are preserved."""
        if chunk.size == 0:
            return chunk

        x = chunk.astype(np.float64, copy=False)

        # Start settled at the incoming level. Left at zero, the filter would
        # spend its first time constant climbing, and every session would open
        # with roughly 60 ms of un-blocked offset.
        if self._state is None:
            self._state = lfilter_zi(self._b, self._a) * float(x[0])

        y, self._state = lfilter(self._b, self._a, x, zi=self._state)
        return y.astype(chunk.dtype, copy=False)
