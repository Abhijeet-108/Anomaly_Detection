import hashlib
import json
import re
import secrets
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
USERS_FILE = APP_DIR / "users.json"
MIN_AGE = 15
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def init_session():
    defaults = {
        "authenticated": False,
        "user_email": "",
        "user_name": "",
        "user_plan": "free",
        "subscription_expiry": None,
        "prediction_count": 0,
        "payment_link_id": None,
        "payment_status": "pending",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if not st.session_state.authenticated:
        users = _load_users()
        
        for email, user in users.items():
            if user.get("last_login", False):

                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.session_state.user_name = user["name"]

                st.session_state.user_plan = user.get("plan", "free")
                st.session_state.subscription_expiry = user.get(
                    "subscription_expiry"
                )

                break


def _load_users():
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return secrets.compare_digest(check.hex(), digest)


def signup(name: str, email: str, password: str):
    name = name.strip()
    email = email.strip().lower()

    if not name:
        return False, "Name is required."
    if not EMAIL_PATTERN.match(email):
        return False, "Enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    if email in users:
        return False, "An account with this email already exists."

    users[email] = {
        "name": name,
        "password": _hash_password(password),
        
        "plan": "free",
        "subscription_expiry": None,
        "prediction_count": 0,
        "max_free_predictions": 3,
        "payment_link_id": None,
        "payment_status": "pending",
        "last_login": False
    }
    _save_users(users)
    return True, "Account created successfully. Please log in."


def login(email: str, password: str):
    email = email.strip().lower()
    users = _load_users()
    user = users.get(email)

    if not user or not _verify_password(password, user["password"]):
        return False, "Invalid email or password."

    for user in users.values():
        user["last_login"] = False

    users[email]["last_login"] = True
    _save_users(users)

    st.session_state.authenticated = True
    st.session_state.user_email = email
    st.session_state.user_name = user["name"]

    st.session_state.user_plan = user.get("plan", "free")
    st.session_state.subscription_expiry = user.get("subscription_expiry")

    return True, "Logged in successfully."


def logout():
    users = _load_users()

    email = st.session_state.user_email

    if email in users:
        users[email]["last_login"] = False
        _save_users(users)

    st.session_state.authenticated = False
    st.session_state.user_email = ""
    st.session_state.user_name = ""
    st.session_state.user_plan = "free"
    st.session_state.subscription_expiry = None


def render_auth_page():
    st.markdown(
        """
        <div class="auth-wrap">
            <div class="auth-header">
                <div class="hero-badge"><span>●</span> Secure Access</div>
                <h1>Welcome to Kavaca Pay</h1>
                <p>Sign in or create an account to access the fraud detection dashboard.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1.2, 1.6, 1.2])
    with col_c:
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                login_email = st.text_input("Email", placeholder="you@example.com")
                login_password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
                if submitted:
                    ok, msg = login(login_email, login_password)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_signup:
            with st.form("signup_form", clear_on_submit=False):
                signup_name = st.text_input("Full Name", placeholder="Your name")
                signup_email = st.text_input("Email", placeholder="you@example.com")
                signup_password = st.text_input("Password", type="password")
                signup_confirm = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                if submitted:
                    if signup_password != signup_confirm:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = signup(signup_name, signup_email, signup_password)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)