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

## EC2 auto-start with systemd

In EC2, `restart: unless-stopped` in `docker-compose.yml` may not be enough to bring the stack back reliably after a full reboot. A simple approach is to register the Compose stack as a `systemd` service on the instance.

Create this file on the EC2 host:

```text
/etc/systemd/system/vocal-metrics.service
```

Use this content:

```ini
[Unit]
Description=Vocal Metrics Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
WorkingDirectory=/home/ubuntu/vocal-metrics
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable vocal-metrics.service
sudo systemctl start vocal-metrics.service
sudo systemctl status vocal-metrics.service
```

This file belongs on the EC2 machine, not inside the repository checkout. The repository only documents the steps.
