import numpy as np

from analysis_pipeline import analyze_audio


def test_analyze_audio_returns_summary():
    sr = 22050
    y = np.zeros(sr)

    result = analyze_audio(y, sr)

    assert isinstance(result, dict)
    assert result["summary"]["duration"] == 1.0
    assert result["summary"]["sample_rate"] == sr
    assert "range_semitones" in result["summary"]
    assert "vibrato_rate_hz" in result["summary"]
    assert "second_by_second" in result
    assert len(result["second_by_second"]) == 1
    assert "report_lines" in result
    assert result["analysis_mode"]["hop_length"] == 512


def test_analyze_audio_detailed_includes_frame_details():
    sr = 22050
    y = np.zeros(sr)

    result = analyze_audio(y, sr, hop_length=512, include_frame_details=True)

    assert result["analysis_mode"]["hop_length"] == 512
    assert result["analysis_mode"]["include_frame_details"] is True
    assert "second_by_second" not in result
    assert "frame_by_frame" in result
    assert len(result["frame_by_frame"]) > 0
