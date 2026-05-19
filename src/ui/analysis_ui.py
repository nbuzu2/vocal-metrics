from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import streamlit as st

from app_io.format_to_json import save_analysis_json
from app_io.save_analysis import analyze_uploaded_audio
from report.bedrock_report import generate_ai_report


def get_audio_source() -> Any | None:
    """
    Gets the audio source from the user, either through file upload or audio recording.
    Returns:
        Any | None: The audio source, which can be a file-like object from the file uploader or audio recorder, or None if no audio source is provided.
    """
    uploaded_file = st.file_uploader(
        "Audio file",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
    )
    recorded_file = st.audio_input("O graba un audio en WAV")

    audio_source = recorded_file or uploaded_file

    if recorded_file is not None:
        st.caption("Usando audio grabado desde el navegador.")
    elif uploaded_file is not None:
        st.caption("Usando archivo subido manualmente.")

    return audio_source


def get_analysis_mode() -> str | None:
    """
    Gets the analysis mode selected by the user through two buttons: "Analisis general" and "Analisis detallado".
    Returns:
        str | None: Returns "general" if the user selects the general analysis mode, "detallado" if the user selects the detailed analysis mode, or None if no mode is selected.
    """
    general_col, detailed_col = st.columns(2)
    run_general = general_col.button("Analisis general", type="primary", use_container_width=True)
    run_detailed = detailed_col.button("Analisis detallado", use_container_width=True)

    if run_detailed:
        return "detallado"
    if run_general:
        return "general"
    return None


def run_analysis(audio_source: Any, mode: str, style: str = "libre") -> tuple[dict, str]:
    """
    Runs the audio analysis using the provided audio source and mode.
    Args:
        - audio_source (Any): The source of the audio to analyze, which can be a
            file-like object from the file uploader or audio recorder.
        - mode (str): The analysis mode, either "general" or "detallado", which
            determines the level of detail in the analysis results.

    Returns:
    - tuple[dict, str]: A tuple containing the analysis result as a dictionary
      and the file path where the analysis result JSON was saved.
    """
    progress_messages: list[str] = []
    progress_container = st.empty()

    def update_progress(message: str) -> None:
        progress_messages.append(message)
        progress_container.code("\n".join(progress_messages), language="text")

    with st.spinner("Analyzing audio..."):
        include_frame_details = mode == "detallado"
        hop_length = 128 if mode == "detallado" else 512

        result = analyze_uploaded_audio(
            audio_source,
            hop_length=hop_length,
            include_frame_details=include_frame_details,
            progress_callback=update_progress,
            style=style,
        )

        result["evaluator"] = {
            "key": mode,
            "label": "Analisis detallado" if mode == "detallado" else "Analisis general",
        }
        saved_path = save_analysis_json(result)

    progress_container.empty()
    return result, saved_path


def render_analysis_result(cached: dict) -> None:
    """
    Renders the analysis result on the Streamlit page.
    Args:
        - cached (dict): Cached analysis data from st.session_state["last_analysis"].
    """
    saved_path = cached["saved_path"]
    result_evaluator = cached.get("evaluator", {})
    progress_messages = cached.get("progress_messages", [])
    second_by_second = cached.get("second_by_second")
    frame_by_frame = cached.get("frame_by_frame")

    st.success("Analysis completed.")
    st.write(f"Saved JSON: `{Path(saved_path)}`")
    st.write(f"Mode selected: `{result_evaluator.get('label', '')}`")

    if progress_messages:
        st.subheader("Progreso del analisis")
        st.code("\n".join(progress_messages), language="text")

    if second_by_second is not None:
        st.subheader("Second-by-second")
        st.dataframe(second_by_second, width="stretch")

    if frame_by_frame is not None:
        st.subheader("Frame-by-frame")
        st.dataframe(frame_by_frame, width="stretch")


def render_ai_report_section(summary: dict) -> None:
    """Renders the AI report button and displays the generated report if available."""
    st.divider()
    st.subheader("Informe de IA")

    if st.button("Generar Informe IA", type="primary", use_container_width=True):
        with st.spinner("Generando informe con IA..."):
            try:
                report = generate_ai_report(summary)
                st.session_state["ai_report_text"] = report
            except Exception as exc:
                st.error(f"Error al generar informe: {exc}")

    if "ai_report_text" in st.session_state:
        st.markdown(st.session_state["ai_report_text"])


def get_vocal_style() -> str:
    style_labels = {
        "libre":  "Sin especificar",
        "lirico": "Lirico / Opera",
        "pop":    "Pop / Contemporaneo",
        "jazz":   "Jazz",
        "folk":   "Folk / Musica antigua",
    }
    label = st.selectbox(
        "Estilo vocal",
        options=list(style_labels.values()),
        index=0,
    )
    return next(k for k, v in style_labels.items() if v == label)


def render_analysis_page(user: dict) -> None:
    """
    Renders the analysis page where the user can upload or record an audio file, select the analysis mode, and view the results.
    Args:
        - user (dict): The authenticated user's information.
    """
    audio_source = get_audio_source()
    if audio_source is None:
        st.session_state.pop("last_analysis", None)
        st.session_state.pop("ai_report_text", None)
        return

    style = get_vocal_style()
    mode = get_analysis_mode()
    if mode is not None:
        try:
            result, saved_path = run_analysis(audio_source, mode, style)
            st.session_state["last_analysis"] = {
                "summary": result.get("summary", {}),
                "evaluator": result.get("evaluator", {}),
                "progress_messages": result.get("progress_messages", []),
                "second_by_second": result.get("second_by_second"),
                "frame_by_frame": result.get("frame_by_frame"),
                "saved_path": saved_path,
            }
            st.session_state.pop("ai_report_text", None)
            del result
            gc.collect()
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    cached = st.session_state.get("last_analysis")
    if cached:
        render_analysis_result(cached)
        render_ai_report_section(cached["summary"])
