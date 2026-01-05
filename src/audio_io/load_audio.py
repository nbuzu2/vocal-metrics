import librosa
import numpy as np
from typing import Tuple

def load_audio(path: str, target_sr: int | None = None) -> Tuple[np.ndarray, int, float]:
    """
    Load an audio file from disk and return its waveform, sample rate, and duration.

    This function uses librosa to load an audio file without resampling
    (i.e., preserving the original sample rate).

    Args:
        path (str): Path to the audio file (e.g., WAV, MP3).
        target_sr (int | None): Target sampling rate. 
                                If None, the original 22050 sampling rate is preserved.

    Returns:
        Tuple[np.ndarray, int, float]:
            - y (np.ndarray): Audio time series (mono).
            - sr (int): Sampling rate of the audio.
            - duration (float): Duration of the audio in seconds.
    """
    y, sr = librosa.load(path, sr=target_sr)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, sr, duration