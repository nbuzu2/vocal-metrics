import numpy as np
from app_io.load_audio import load_audio

path_test = "data/samples/086053_damn-youwav-82631.wav"

# Contract test
def test_load_audio_returns_expected_types():
    """test that load_audio returns expected types"""
    y, sr, duration = load_audio(path_test)

    assert isinstance(y, np.ndarray)
    assert isinstance(sr, int)
    assert isinstance(duration, float)

# Sanity check
def test_load_audio_not_empty():
    """test that load_audio returns non-empty audio data"""
    y, sr, duration = load_audio(path_test)

    assert y.size > 0
    assert sr > 0
    assert duration > 0
