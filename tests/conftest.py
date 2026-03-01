"""
Shared pytest fixtures for all test files.
"""

import numpy as np
import pytest


@pytest.fixture
def silence_frames():
    """PCM-16 silence (0.5s at 16kHz) as a list of byte chunks."""
    samples = np.zeros(8000, dtype='<i2')
    return [samples.tobytes()]


@pytest.fixture
def tone_frames():
    """PCM-16 440Hz sine wave (0.5s at 16kHz) as a list of byte chunks."""
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype('<i2')
    return [samples.tobytes()]
