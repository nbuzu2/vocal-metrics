import numpy as np
from metrics.extract_pitch import  get_pitch


path_test = "data/samples/086053_damn-youwav-82631.wav"

# Contract test
def test_extract_pitch_returns_arrays():
    """test that extract_pitch returns np.ndarrays"""
    # 1 second of silence
    sr = 22050
    y = np.zeros(sr)

    f0, voiced_probs = get_pitch(y=y, sr=sr, hop_length=512)

    assert isinstance(f0, np.ndarray)
    assert isinstance(voiced_probs, np.ndarray)
