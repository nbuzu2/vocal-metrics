from pathlib import Path

from app_io.save_analysis import analyze_uploaded_audio


class FakeUploadedFile:
    def __init__(self, path: Path):
        self.name = path.name
        self._bytes = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._bytes


def test_analyze_uploaded_audio_returns_source_metadata():
    sample_path = Path("data/samples/086053_damn-youwav-82631.wav")
    uploaded = FakeUploadedFile(sample_path)

    result = analyze_uploaded_audio(uploaded)

    assert result["source"]["filename"] == sample_path.name
    assert result["source"]["path"] is None
    assert result["source"]["upload_bytes"] > 0
    assert "summary" in result
    assert "second_by_second" in result


def test_analyze_uploaded_audio_detailed_mode_includes_frame_details():
    sample_path = Path("data/samples/086053_damn-youwav-82631.wav")
    uploaded = FakeUploadedFile(sample_path)

    result = analyze_uploaded_audio(uploaded, hop_length=512, include_frame_details=True)

    assert result["analysis_mode"]["hop_length"] == 512
    assert "second_by_second" not in result
    assert "frame_by_frame" in result
