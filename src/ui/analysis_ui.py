from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import streamlit as st

from app_io.format_to_json import save_analysis_json
from app_io.save_analysis import analyze_uploaded_audio

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


def run_analysis(audio_source: Any, mode: str) -> tuple[dict, str]:
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

        result = analyze_uploaded_audio(
            audio_source,
            hop_length=512,
            include_frame_details=include_frame_details,
            progress_callback=update_progress,
        )

        result["evaluator"] = {
            "key": mode,
            "label": "Analisis detallado" if mode == "detallado" else "Analisis general",
        }
        saved_path = save_analysis_json(result)

    return result, saved_path


def render_analysis_result(result: dict, saved_path: str) -> None:
    """
    Renders the analysis result on the Streamlit page, including success message, saved JSON path, selected mode, progress messages, and detailed dataframes if available.
    Args:
        - result (dict): The analysis result containing the evaluator information, progress messages, and optionally second-by-second and frame-by-frame data.
        - saved_path (str): The file path where the analysis result JSON was saved.
        
    UI Elements Rendered:
        - Success message indicating that the analysis is completed.
        - Display of the saved JSON file path.
    """
    st.success("Analysis completed.")
    st.write(f"Saved JSON: `{Path(saved_path)}`")
    st.write(f"Mode selected: `{result['evaluator']['label']}`")

    if result.get("progress_messages"):
        st.subheader("Progreso del analisis")
        st.code("\n".join(result["progress_messages"]), language="text")

    if "second_by_second" in result:
        st.subheader("Second-by-second")
        st.dataframe(result["second_by_second"], width="stretch")

    if "frame_by_frame" in result:
        st.subheader("Frame-by-frame")
        st.dataframe(result["frame_by_frame"], width="stretch")


def render_analysis_page(user: dict) -> None:
    """
    Renders the analysis page where the user can upload or record an audio file, select the analysis mode, and view the results.
    Args:
        - user (dict): The authenticated user's information, which can be used for personalized features 
        or access control if needed in the future.

    UI Elements Rendered:
        - File uploader for audio files (wav, mp3, ogg, flac, m4a).
        - Audio recorder input for recording audio directly from the browser.
        - Buttons to select the analysis mode (general or detailed).
        - Progress messages during the analysis process.
        - Display of the saved JSON file path and selected mode after analysis completion.
        - Dataframes showing second-by-second and frame-by-frame details if available in the result.
        
    Raises:
        - Exception: If there is an error during the analysis process, an exception will be raised
    
    """

    audio_source = get_audio_source()
    if audio_source is None:
        return

    mode = get_analysis_mode()
    if mode is None:
        return

    try:
        result, saved_path = run_analysis(audio_source, mode)
        render_analysis_result(result, saved_path)
        del result
        gc.collect()
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
