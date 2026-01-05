from audio_io.audio import load_audio
from metrics.extract_pitch import extract_frame_features
from metrics.pitch_range import analyze_pitch_range
from metrics.vibrato import analyze_vibrato

audio_path = "data/samples/example.wav"

y, sr, duration = load_audio(audio_path)
df = extract_frame_features(y, sr)

summary = {
    "duration": duration,
    **analyze_pitch_range(df),
    **analyze_vibrato(df, sr),
}

print(summary)