import numpy as np
from scipy.signal import find_peaks, welch


def analyze_vibrato(df, sr, hop_length=512):
    sustained = df[df["rms"] > df["rms"].quantile(0.7)]["frequency_smoothed"].dropna()

    if len(sustained) < 50:
        return {"vibrato_rate_hz": 0, "vibrato_extent_hz": 0}

    diff = np.diff(sustained.values)
    peaks = len(find_peaks(diff)[0]) + len(find_peaks(-diff)[0])
    duration = len(sustained) * hop_length / sr

    return {
        "vibrato_rate_hz": peaks / (2 * duration),
        "vibrato_extent_hz": float(np.std(sustained)),
    }


def analyze_pitch_modulation(df, sr, hop_length=512):
    """
    Spectral decomposition of the pitch trajectory into wobble, vibrato, and flutter bands.

    Bands:
        0.5 – 3.5 Hz  → wobble  (slow, involuntary instability)
        3.5 – 8.0 Hz  → vibrato (controlled, intentional)
        8.0 – 15.0 Hz → flutter (too fast, also uncontrolled)

    Returns a wobble_score from 0 (all wobble) to 100 (no wobble).
    """
    voiced = df[df["is_voice"]]["frequency_smoothed"].dropna()
    voiced = voiced[np.isfinite(voiced) & (voiced > 0)]

    frame_rate = sr / hop_length
    if len(voiced) < int(frame_rate * 1.5):
        return {
            "wobble_score": 100.0,
            "wobble_energy_ratio": 0.0,
            "vibrato_energy_ratio": 0.0,
            "flutter_energy_ratio": 0.0,
        }

    pitch = voiced.values.astype(float)

    # Remove slow melodic trend (~0.5s moving average) to isolate modulations
    trend_window = max(3, int(frame_rate * 0.5))
    trend = np.convolve(pitch, np.ones(trend_window) / trend_window, mode="same")
    modulation = pitch - trend

    nperseg = min(len(modulation), max(int(frame_rate * 2), 64))
    freqs, psd = welch(modulation, fs=frame_rate, nperseg=nperseg)

    base_mask = freqs >= 0.5
    total = psd[base_mask].sum()
    if total == 0:
        return {
            "wobble_score": 100.0,
            "wobble_energy_ratio": 0.0,
            "vibrato_energy_ratio": 0.0,
            "flutter_energy_ratio": 0.0,
        }

    wobble_ratio = float(psd[(freqs >= 0.5) & (freqs < 3.5)].sum() / total)
    vibrato_ratio = float(psd[(freqs >= 3.5) & (freqs < 8.0)].sum() / total)
    flutter_ratio = float(psd[(freqs >= 8.0) & (freqs < 15.0)].sum() / total)

    # 0% wobble energy → 100, 100% wobble energy → 0
    wobble_score = round(max(0.0, 100.0 - wobble_ratio * 100.0), 1)

    return {
        "wobble_score": wobble_score,
        "wobble_energy_ratio": round(wobble_ratio, 3),
        "vibrato_energy_ratio": round(vibrato_ratio, 3),
        "flutter_energy_ratio": round(flutter_ratio, 3),
    }

