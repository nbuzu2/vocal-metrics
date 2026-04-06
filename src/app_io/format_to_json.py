from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


DEFAULT_OUTPUT_DIR = "outputs/analyses"


def get_output_dir() -> Path:
    output_dir = os.getenv("ANALYSIS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    path = Path(output_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_analysis_json(result: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    base_dir = Path(output_dir).expanduser() if output_dir else get_output_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = base_dir / f"analysis_{timestamp}_{uuid4().hex[:8]}.json"

    payload = _to_json_compatible(result)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)

    return file_path
