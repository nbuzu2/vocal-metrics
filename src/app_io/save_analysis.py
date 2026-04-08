from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from typing import Callable

from analysis_pipeline import analyze_audio_file


def _get_uploaded_name(uploaded_file: Any) -> str:
    return Path(getattr(uploaded_file, "name", "uploaded_audio.wav")).name


def _read_uploaded_bytes(uploaded_file: Any) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
        if isinstance(data, bytes):
            return data
    raise TypeError("Uploaded file must provide bytes through getvalue() or read().")


def analyze_uploaded_audio(
    uploaded_file: Any,
    hop_length: int = 512,
    include_frame_details: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Analyze an uploaded file by writing it to a temporary file first."""
    filename = _get_uploaded_name(uploaded_file)
    suffix = Path(filename).suffix or ".wav"
    file_bytes = _read_uploaded_bytes(uploaded_file)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(file_bytes)

    try:
        result = analyze_audio_file(
            temp_path,
            hop_length=hop_length,
            include_frame_details=include_frame_details,
            progress_callback=progress_callback,
        )
        result["source"] = {
            "filename": filename,
            "path": None,
            "upload_bytes": len(file_bytes),
        }
        return result
    finally:
        temp_path.unlink(missing_ok=True)
