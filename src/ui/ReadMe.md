# UI

This folder contains the Streamlit user interface split by responsibility.

## Files

- `streamlit_app.py`: Streamlit entrypoint. Sets the page config, requests the authenticated user, and renders the main page.
- `auth_ui.py`: Login flow and session guard using `st.session_state`.
- `main_ui.py`: Page-level composition such as the main header and logout button.
- `analysis_ui.py`: Audio upload/record flow, analysis mode selection, analysis execution, and result rendering.

## Design notes

- `streamlit_app.py` should stay small and orchestration-focused.
- Authentication concerns live in `auth_ui.py`.
- Page layout concerns live in `main_ui.py`.
- Analysis-specific UI and behavior live in `analysis_ui.py`.

This separation keeps the Streamlit script easier to read and change over time.
