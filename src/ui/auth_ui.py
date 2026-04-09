

import streamlit as st

from auth.auth_service import authenticate_user


def render_login() -> None:
    """
    Renders the login form and handles authentication. 
    If the user is successfully authenticated, their information is stored 
    in the session state and the app is rerun.
    """
    
    st.title("Vocal Metrics")
    st.write("Inicia sesion para analizar audio")
   
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Entrar", type="primary", use_container_width=True)

    if submitted:
        try:
            user = authenticate_user(email, password)
        except Exception as exc:
            st.error(f"No se pudo validar el acceso: {exc}")
            st.stop()

        if user is None:
            st.error("Email o password incorrectos.")
            st.stop()

        st.session_state["authenticated_user"] = user.model_dump()
        st.rerun()

    st.stop()


def require_authenticated_user() -> dict:
    """ 
    Checks if there is an authenticated user in the session state. 
        - If so, returns the user information.
        - If not, renders the login form and stops execution until the user is authenticated.
    
    Returns:
        dict: The authenticated user's information.
    
    Raises:
        RuntimeError: If the function is called when the user is not authenticated and the login form is rendered.
    """
    user = st.session_state.get("authenticated_user")
    if user:
        return user

    render_login()
    raise RuntimeError("Unreachable")