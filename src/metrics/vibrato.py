import numpy as np
from scipy.signal import find_peaks

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

