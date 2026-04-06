import numpy as np
import librosa

def analyze_pitch_range(df):
    valid = df[df["is_voice"]]["frequency_smoothed"].dropna()

    if valid.empty:
        return {
            "min_note": None,
            "max_note": None,
            "range_semitones": 0,
            "range_robust_semitones": 0,
        }

    min_hz, max_hz = valid.min(), valid.max()
    p5, p95 = valid.quantile([0.05, 0.95])

    return {
        "min_note": librosa.hz_to_note(min_hz),
        "max_note": librosa.hz_to_note(max_hz),
        "range_semitones": float(12 * np.log2(max_hz / min_hz)),
        "range_robust_semitones": float(12 * np.log2(p95 / p5)),
    }
