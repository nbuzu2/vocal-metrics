import argparse
import json
import os
from typing import NoReturn

from analysis_pipeline import analyze_audio_file
from app_io.format_to_json import save_analysis_json


def fail_missing_audio_path() -> NoReturn:
    raise SystemExit(
        "Missing audio path. Pass it as an argument or set the AUDIO_PATH environment variable."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a voice recording.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=os.getenv("AUDIO_PATH"),
        help="Path to the audio file. You can also provide it with AUDIO_PATH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio_path:
        fail_missing_audio_path()

    result = analyze_audio_file(args.audio_path)
    saved_path = save_analysis_json(result, os.getenv("ANALYSIS_OUTPUT_DIR"))
    print(json.dumps({"saved_json": str(saved_path), "analysis": result}, ensure_ascii=True))


if __name__ == "__main__":
    main()
