from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_audio_path(path: str | Path) -> Path:
    """
    Resolve an audio path in a way that works from different working directories.

    Resolution order:
    1. Absolute path as provided.
    2. Relative to the current working directory.
    3. Relative to the project root directory.
    """
    candidate = Path(path).expanduser()

    if candidate.is_absolute():
        resolved = candidate
    else:
        cwd_candidate = Path.cwd() / candidate
        resolved = cwd_candidate if cwd_candidate.exists() else PROJECT_ROOT / candidate

    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Audio file not found: '{path}'. Checked '{resolved}'."
        )

    return resolved


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
    audio_path = resolve_audio_path(path)
    y, sr = librosa.load(audio_path, sr=target_sr)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, sr, duration
