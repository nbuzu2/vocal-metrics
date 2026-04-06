from __future__ import annotations

from pathlib import Path

import streamlit as st

from app_io.format_to_json import save_analysis_json
from app_io.save_analysis import analyze_uploaded_audio


st.set_page_config(page_title="Vocal Metrics", page_icon="V", layout="centered")

st.title("Vocal Metrics")
st.write("Upload an audio file to analyze it and save the result as JSON.")

uploaded_file = st.file_uploader(
    "Audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
)

if uploaded_file is not None:
    if st.button("Analyze and save JSON", type="primary"):
        try:
            with st.spinner("Analyzing audio..."):
                result = analyze_uploaded_audio(uploaded_file)
                saved_path = save_analysis_json(result)
            st.success("Analysis completed.")
            st.write(f"Saved JSON: `{Path(saved_path)}`")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
