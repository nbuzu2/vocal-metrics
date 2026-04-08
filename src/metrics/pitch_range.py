import numpy as np
import librosa

def analyze_pitch_range(df):
    valid = df[df["is_voice"]]["frequency_smoothed"].dropna()
    valid = valid[np.isfinite(valid) & (valid > 0)]

    if valid.empty:
        return {
            "min_note": None,
            "max_note": None,
            "range_semitones": 0,
            "range_robust_semitones": 0,
        }

    min_hz, max_hz = valid.min(), valid.max()
    p5, p95 = valid.quantile([0.05, 0.95])

    range_semitones = 12 * np.log2(max_hz / min_hz) if min_hz > 0 else 0.0
    robust_range = 12 * np.log2(p95 / p5) if p5 > 0 else 0.0

    return {
        "min_note": librosa.hz_to_note(min_hz),
        "max_note": librosa.hz_to_note(max_hz),
        "range_semitones": float(range_semitones) if np.isfinite(range_semitones) else 0.0,
        "range_robust_semitones": float(robust_range) if np.isfinite(robust_range) else 0.0,
    }
