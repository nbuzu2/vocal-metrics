from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app_io import load_audio
from metrics import analyze_pitch_range
from metrics import analyze_vibrato
from metrics import extract_frame_features


def analyze_audio(y: np.ndarray, sr: int) -> dict[str, Any]:
    """Run the full metrics pipeline from in-memory audio."""
    duration = float(len(y) / sr) if sr else 0.0
    df = extract_frame_features(y, sr)

    return {
        "duration": duration,
        "sample_rate": int(sr),
        "frame_count": int(len(df)),
        **analyze_pitch_range(df),
        **analyze_vibrato(df, sr),
    }


def analyze_audio_file(path: str | Path) -> dict[str, Any]:
    """Load an audio file from disk and analyze it."""
    resolved_path = Path(path).expanduser()
    y, sr, duration = load_audio(str(resolved_path))
    result = analyze_audio(y, sr)
    result["duration"] = float(duration)
    result["source"] = {
        "filename": Path(path).name,
        "path": str(path),
    }
    return result
