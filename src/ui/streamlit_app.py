from __future__ import annotations

from pathlib import Path

import streamlit as st

from app_io.format_to_json import save_analysis_json
from app_io.save_analysis import analyze_uploaded_audio


st.set_page_config(page_title="Vocal Metrics", page_icon="V", layout="centered")

st.title("Vocal Metrics")
st.write("Upload an audio file to analyze it and save the result as JSON.")
st.caption("La beta incluye dos modos: uno general segundo a segundo y otro detallado con hop mas fino.")


uploaded_file = st.file_uploader(
    "Audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
)

if uploaded_file is not None:
    general_col, detailed_col = st.columns(2)
    run_general = general_col.button("Analisis general", type="primary", use_container_width=True)
    run_detailed = detailed_col.button("Analisis detallado", use_container_width=True)

    if run_general or run_detailed:
        try:
            progress_messages: list[str] = []
            progress_container = st.empty()

            def update_progress(message: str) -> None:
                progress_messages.append(message)
                progress_container.code("\n".join(progress_messages), language="text")

            with st.spinner("Analyzing audio..."):
                if run_detailed:
                    result = analyze_uploaded_audio(
                        uploaded_file,
                        hop_length=512,
                        include_frame_details=True,
                        progress_callback=update_progress,
                    )
                    mode_label = "detallado"
                else:
                    result = analyze_uploaded_audio(
                        uploaded_file,
                        hop_length=512,
                        include_frame_details=False,
                        progress_callback=update_progress,
                    )
                    mode_label = "general"

                result["evaluator"] = {
                    "key": mode_label,
                    "label": "Analisis detallado" if run_detailed else "Analisis general",
                }
                saved_path = save_analysis_json(result)
            st.success("Analysis completed.")
            st.write(f"Saved JSON: `{Path(saved_path)}`")
            st.write(f"Mode selected: `{result['evaluator']['label']}`")
            if result.get("progress_messages"):
                st.subheader("Progreso del analisis")
                progress_container.code("\n".join(result["progress_messages"]), language="text")
            if "second_by_second" in result:
                st.subheader("Second-by-second")
                st.dataframe(result["second_by_second"], width="stretch")
            if "frame_by_frame" in result:
                st.subheader("Frame-by-frame")
                st.dataframe(result["frame_by_frame"], width="stretch")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
