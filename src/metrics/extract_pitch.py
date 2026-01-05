"""
Frame-level audio feature extraction.

This module computes low-level vocal and spectral features
used for pitch analysis, timbre characterization, and vocal
stability metrics.
"""


import librosa
import numpy as np
import pandas as pd
from typing import Tuple


# TODO: Define voice presets for different vocal ranges
# VOICE_PRESETS = {
#     "pop_training": ("G2", "C5"),
#     "lyric_training": ("B2", "G5"),
#     "soprano_extended": ("C3", "A5"),
# }
# TODO: Future work: user-specific pitch ranges inferred from guided vocal exercises and percentile-based analysis.
# TODO: Allow user selection of voice presets in the UI.

def _extract_pitch(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate pitch (fundamental frequency) and voiced probability per frame using pYIN.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - f0 (np.ndarray): Estimated fundamental frequency per frame (Hz).
            - voiced_probs (np.ndarray): Probability that each frame contains voice.

    Note:
        These defaults are intended for general pop / speech-like singing.
        Lyric or extended vocal techniques may require wider pitch ranges.
        Voiced_flag reserved for future use
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    return f0, voiced_probs


def _extract_rms(y: np.ndarray, hop_length: int) -> np.ndarray:
    """Calculate RMS energy per frame.
    
    Returns:
        np.ndarray: RMS energy values.
    """
    return librosa.feature.rms(y=y, hop_length=hop_length)[0]


def _extract_centroid(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Calculate spectral centroid per frame.
    
    Returns:
        np.ndarray: Spectral centroid values (Hz).
    """
    return librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]


def _extract_bandwidth(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Calculate spectral bandwidth per frame.
    
    Returns:
        np.ndarray: Spectral bandwidth values (Hz).
    """
    return librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]


def _extract_flatness(y: np.ndarray, hop_length: int) -> np.ndarray:
    """Calculate spectral flatness per frame.
    
    Returns:
        np.ndarray: Spectral flatness values.
    """
    return librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]


def _extract_rolloff(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Calculate spectral rolloff per frame.
    
    Returns:
        np.ndarray: Spectral rolloff values (Hz).
    """
    return librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]


def _extract_zcr(y: np.ndarray, hop_length: int) -> np.ndarray:
    """Calculate zero-crossing rate per frame.
    
    Returns:
        np.ndarray: Zero-crossing rate values.
    """
    return librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]


def _extract_times(sr: int, hop_length: int, n_frames: int) -> np.ndarray:
    """Calculate time stamps for each frame.
    
    Returns:
        np.ndarray: Time values (seconds).
    """
    frames = range(n_frames)
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)


def extract_frame_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float = librosa.note_to_hz("G2"),
    fmax: float = librosa.note_to_hz("C5"),
    ) -> pd.DataFrame:
    
    """ Extract frame-level audio features into a DataFrame.
    Args:
        y (np.ndarray): Audio time series (mono).
        sr (int): Sampling rate of the audio.
        hop_length (int, optional): Number of samples between successive frames. Defaults to 512.
        fmin (float, optional): Minimum pitch frequency (Hz) to consider. Defaults to G2 (~98 Hz).
        fmax (float, optional): Maximum pitch frequency (Hz) to consider. Defaults to C5 (~523 Hz).
        
    Returns:
        pd.DataFrame: Frame-level representation of a single audio signal,
        where each row corresponds to a short time window (frame) and
        each column represents a low-level vocal or spectral feature
        extracted from that frame.

        The DataFrame includes the following columns:

            - time_s (float): Time stamp of each frame (seconds).
            - frequency_hz (float): Estimated fundamental frequency (Hz).
            - voiced_prob (float): Probability of voiced content (0–1).
            - rms (float): Root Mean Square energy.
            - centroid_hz (float): Spectral centroid (brightness).
            - bandwidth_hz (float): Spectral bandwidth.
            - flatness (float): Spectral flatness (noise vs tonal).
            - rolloff_hz (float): Spectral rolloff frequency.
            - zcr (float): Zero-crossing rate.
            
    Notes:
        - Frames are aligned by minimum common length across features.
        - No post-processing (voicing mask, smoothing) is applied at this stage.
    """

    f0, voiced_probs = _extract_pitch(y, sr, hop_length, fmin, fmax)
    rms = _extract_rms(y, hop_length)
    centroid = _extract_centroid(y, sr, hop_length)
    bandwidth = _extract_bandwidth(y, sr, hop_length)
    flatness = _extract_flatness(y, hop_length)
    rolloff = _extract_rolloff(y, sr, hop_length)
    zcr = _extract_zcr(y, hop_length)

    n_frames = min(len(f0), len(rms), len(centroid), len(
        bandwidth), len(flatness), len(rolloff), len(zcr))

    times = _extract_times(sr, hop_length, n_frames)
    
    df = pd.DataFrame({
        "time_s": times,
        "frequency_hz": f0[:n_frames],
        "voiced_prob": voiced_probs[:n_frames],
        "rms": rms[:n_frames],
        "centroid_hz": centroid[:n_frames],
        "bandwidth_hz": bandwidth[:n_frames],
        "flatness": flatness[:n_frames],
        "rolloff_hz": rolloff[:n_frames],
        "zcr": zcr[:n_frames]
    })
    return df
