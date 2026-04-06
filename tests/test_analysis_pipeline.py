import numpy as np

from analysis_pipeline import analyze_audio


def test_analyze_audio_returns_summary():
    sr = 22050
    y = np.zeros(sr)

    result = analyze_audio(y, sr)

    assert isinstance(result, dict)
    assert result["duration"] == 1.0
    assert result["sample_rate"] == sr
    assert "range_semitones" in result
    assert "vibrato_rate_hz" in result
