from __future__ import annotations

import streamlit as st

from ui.analysis_ui import render_analysis_page


def render_page_header(user: dict) -> None:
    """ 
    Renders the page header with the title, description, and user session information.
    Also includes a logout button that clears the authenticated user from the session state and reruns the
    app when clicked.
    """
    st.title("Vocal Metrics")
    st.write("Upload an audio file to analyze it and save the result as JSON.")
    st.caption("La beta incluye dos modos: uno general segundo a segundo y otro detallado con hop mas fino.")

    header_left, header_right = st.columns([3, 1])
    display_name = user.get("full_name") or user.get("email")
    header_left.caption(f"Sesion activa: {display_name}")

    if header_right.button("Salir", use_container_width=True):
        st.session_state.pop("authenticated_user", None)
        st.rerun()


def render_main_page(user: dict) -> None:
    render_page_header(user)
    render_analysis_page(user)
