import json
import shutil
from pathlib import Path
from uuid import uuid4

from app_io.format_to_json import save_analysis_json


def test_save_analysis_json_writes_file():
    result = {
        "duration": 1.5,
        "source": {"filename": "voice.wav", "path": None},
        "range_semitones": 3.2,
    }
    output_dir = Path("outputs/test-storage") / uuid4().hex

    try:
        saved_path = save_analysis_json(result, output_dir)

        assert saved_path.exists()
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        assert payload["duration"] == 1.5
        assert payload["source"]["filename"] == "voice.wav"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
