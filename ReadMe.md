# Vocal Metrics

This project analyzes vocal audio and exports the result as JSON.

Today it supports two entrypoints:

- a CLI that processes a local audio file
- a Streamlit UI where the user uploads the audio file

Both entrypoints reuse the same analysis pipeline.

## Project structure

- `src/analysis_pipeline.py`: core orchestration for audio analysis
- `src/app_io/`: app-specific input/output helpers such as loading audio, handling uploads, and saving JSON
- `src/metrics/`: pitch, range, vibrato, and frame-level feature extraction
- `src/ui/streamlit_app.py`: web UI

## Run the CLI

The CLI expects an audio path and writes the analysis JSON to disk.

```powershell
python src/main.py data/samples/086053_damn-youwav-82631.wav
```

You can also set the input and output through environment variables:

```powershell
$env:AUDIO_PATH="data/samples/086053_damn-youwav-82631.wav"
$env:ANALYSIS_OUTPUT_DIR="outputs/analyses"
python src/main.py
```

The command prints the saved JSON path and the analysis summary.

## Run the web UI

```powershell
streamlit run src/ui/streamlit_app.py
```

The UI lets the user upload an audio file, runs the same pipeline as the CLI, and saves the result automatically as JSON.

## Output JSON

By default, analysis files are written to:

```text
outputs/analyses
```

You can override that location with `ANALYSIS_OUTPUT_DIR`.

## Docker notes

For containers, keep the output directory configurable with:

```text
ANALYSIS_OUTPUT_DIR
```

For example, inside Docker you might use:

```text
/app/outputs/analyses
```
