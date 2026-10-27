"""Optional Streamlit client for the FastAPI service.

Run the API separately, then set RAG_API_URL and (for Cloud Run) API_ACCESS_TOKEN.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="Quantum Knowledge Assistant", page_icon="🧠", layout="wide")


def setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # noqa: BLE001 - Streamlit secrets are optional for local use
        return default


api_url = setting("RAG_API_URL", "http://127.0.0.1:8080").rstrip("/")
api_token = setting("API_ACCESS_TOKEN")

if "history" not in st.session_state:
    st.session_state.history = []
if "request_error" not in st.session_state:
    st.session_state.request_error = ""

with st.sidebar:
    st.subheader("Conversation")
    if st.button("New conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.request_error = ""
        st.rerun()
    st.caption(f"API: {api_url}")

st.title("🧠 Quantum Knowledge Assistant")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn.get("sources"):
            st.caption("Sources: " + ", ".join(turn["sources"]))

if st.session_state.request_error:
    st.error(st.session_state.request_error)
    st.session_state.request_error = ""

query = st.chat_input("Ask in Arabic or English")
if query:
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{api_url}/v1/chat",
                json={"query": query, "history": st.session_state.history[-5:]},
                headers=headers,
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            st.session_state.history = payload["history"]
            if st.session_state.history:
                st.session_state.history[-1]["sources"] = payload.get("sources", [])
        except requests.RequestException as error:
            detail = "The API request failed."
            if error.response is not None:
                try:
                    detail = error.response.json().get("detail", detail)
                except ValueError:
                    pass
            st.session_state.request_error = detail

    st.rerun()
