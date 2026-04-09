from __future__ import annotations

import streamlit as st

from ui.auth_ui import require_authenticated_user
from ui.main_ui import render_main_page


st.set_page_config(page_title="Vocal Metrics", page_icon="V", layout="centered")

user = require_authenticated_user()
render_main_page(user)
