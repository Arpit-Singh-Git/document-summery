# app.py
# ------
# Streamlit UI demonstrating REST integration with NVIDIA LLM for summarization.
# - Paste or upload text
# - Construct structured prompt
# - Call NVIDIA API via requests
# - Show summary with error handling and loading indicators

from __future__ import annotations

import io
import os
from typing import Optional

import streamlit as st

from prompts import build_summarization_prompt
from nvidia_client import NvidiaLLMClient

# Optional PDF parsing using PyPDF2 (basic). Remove if not needed.
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False


# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="NVIDIA Summarizer (Demo)",
    page_icon="📝",
    layout="centered",
)

st.title("📝 NVIDIA LLM Summarizer — Demo")
st.caption("Minimal Streamlit app demonstrating REST integration with NVIDIA AI (NIM/NeMo) for document summarization.")


# -----------------------
# Helper Functions
# -----------------------
def read_text_file(uploaded) -> str:
    """Read plain text/markdown uploads as UTF-8."""
    try:
        return uploaded.read().decode("utf-8", errors="replace")
    except Exception:
        return str(uploaded.read())


def read_pdf_file(uploaded) -> str:
    """Extract simple text from a PDF using PyPDF2 (basic)."""
    if not PDF_AVAILABLE:
        return "PDF support not available. Install PyPDF2 or upload a .txt/.md file."
    text_chunks = []
    try:
        reader = PdfReader(io.BytesIO(uploaded.read()))
        for page in reader.pages:
            text_chunks.append(page.extract_text() or "")
    except Exception as e:
        return f"Failed to parse PDF: {e}"
    return "
".join(text_chunks).strip()


def get_default(key: str, fallback: Optional[str] = None) -> Optional[str]:
    """
    Retrieve defaults from Streamlit secrets or environment variables.
    Priority: st.secrets -> env -> fallback.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # st.secrets may not be available locally
        pass
    return os.getenv(key, fallback)


# -----------------------
# Sidebar (Config)
# -----------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    default_base = get_default("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
    default_model = get_default("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
    default_key = get_default("NVIDIA_API_KEY", "")

    api_base = st.text_input(
        "API Base URL",
        value=default_base or "",
        help="OpenAI-compatible base URL. Example: https://integrate.api.nvidia.com/v1",
    )
    model = st.text_input(
        "Model",
        value=default_model or "",
        help="Model name deployed on NVIDIA NIM/NeMo (e.g., meta/llama-3.1-8b-instruct)",
    )
    api_key = st.text_input(
        "API Key",
        type="password",
        value=default_key or "",
        help="Bearer token for the NVIDIA API.",
    )

    st.markdown("---")
    st.caption("Demo app for learning purposes — not production.")


# -----------------------
# Main Input Area
# -----------------------
st.subheader("1) Provide Content")
input_mode = st.radio("Choose input method:", ("Paste text", "Upload file"), horizontal=True)

user_text = ""
if input_mode == "Paste text":
    user_text = st.text_area(
        "Paste your document here",
        value="",
        height=220,
        placeholder="Paste text to summarize…",
    )
else:
    accepted_types = ["txt", "md"] + (["pdf"] if PDF_AVAILABLE else [])
    uploaded = st.file_uploader(
        "Upload a .txt, .md" + (" or .pdf" if PDF_AVAILABLE else ""),
        type=accepted_types
    )
    if uploaded:
        if uploaded.name.lower().endswith(".pdf"):
            user_text = read_pdf_file(uploaded)
        else:
            user_text = read_text_file(uploaded)

if user_text:
    st.info(f"Detected input length: **{len(user_text):,}** characters.", icon="ℹ️")


# -----------------------
# Prompt Options
# -----------------------
st.subheader("2) Summarization Options")

col1, col2, col3 = st.columns(3)
with col1:
    length = st.selectbox("Target length", ["short", "medium", "detailed"], index=0)
with col2:
    tone = st.selectbox("Tone", ["neutral", "professional", "casual"], index=0)
with col3:
    bullet_points = st.checkbox("Use bullet points", value=True)

include_title = st.checkbox("Include a title", value=True)

# Optional: preview constructed prompt
show_prompt = st.checkbox("Preview constructed prompt", value=False)


# -----------------------
# Action Button
# -----------------------
st.subheader("3) Run Summarization")
summarize_clicked = st.button("Summarize with NVIDIA", type="primary", use_container_width=True)

# Rough token targets (for demo)
length_to_tokens = {
    "short": 256,
    "medium": 512,
    "detailed": 896,
}


# -----------------------
# Trigger Summarization
# -----------------------
if summarize_clicked:
    if not api_key:
        st.error("Please provide an API key in the sidebar.", icon="🚫")
    elif not api_base:
        st.error("Please provide the API Base URL in the sidebar.", icon="🚫")
    elif not model:
        st.error("Please provide the Model name in the sidebar.", icon="🚫")
    elif not user_text.strip():
        st.error("Please paste or upload some text to summarize.", icon="📝")
    else:
        # Build prompt
        prompt = build_summarization_prompt(
            doc_text=user_text,
            length=length, tone=tone,
            bullet_points=bullet_points,
            include_title=include_title,
        )

        if show_prompt:
            with st.expander("Constructed Prompt"):
                st.code(prompt)

        # Call NVIDIA API
        with st.spinner("Contacting NVIDIA LLM and generating summary…"):
            try:
                client = NvidiaLLMClient(
                    api_key=api_key,
                    api_base=api_base,
                    model=model,
                    timeout_sec=60,
                )
                summary = client.summarize_text(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=length_to_tokens.get(length, 512),
                )

                st.success("Summary generated successfully.", icon="✅")
                st.subheader("Summary")
                st.write(summary)

            except Exception as e:
                st.error(f"Failed to generate summary: {e}", icon="⚠️")


# -----------------------
# Demo Helper (Optional)
# -----------------------
with st.expander("Try sample text (optional)"):
    demo_text = """Streamlit is an open-source Python library that makes it easy to build custom web apps for machine learning and data science.
In just a few minutes you can build and deploy powerful data apps.
It allows data scientists to turn Python scripts into interactive apps without needing any frontend experience.
Streamlit apps are written entirely in Python, using a simple API that focuses on rapid prototyping, visualization, and interactivity.
"""
    if st.button("Use sample text"):
        st.session_state["sample_text"] = demo_text
        if input_mode == "Paste text":
            st.experimental_rerun()

if "sample_text" in st.session_state and input_mode == "Paste text":
    st.text_area("Paste your document here", value=st.session_state["sample_text"], height=220, key="sample_text_area")
