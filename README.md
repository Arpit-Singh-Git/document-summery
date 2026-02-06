# 📝 NVIDIA LLM Summarizer — Streamlit Demo

Minimal Streamlit app that integrates with NVIDIA AI (NIM/NeMo) via REST to summarize documents.

## Features
- Paste or upload text/PDF
- Structured summarization prompt
- REST API call via `requests`
- Clean separation (UI vs API client)
- Basic error handling & loading state

## Project Structure
```
streamlit-nvidia-summarizer/
├─ app.py
├─ nvidia_client.py
├─ prompts.py
├─ requirements.txt
└─ README.md
```

## Local Run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scriptsctivate
pip install -r requirements.txt
streamlit run app.py
```

Configure in the sidebar or via env vars:
```
NVIDIA_API_KEY="your_key"
NVIDIA_API_BASE="https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL="meta/llama-3.1-8b-instruct"
```

## Streamlit Cloud Deployment
1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud → **New app**.
3. Select your repo, branch (`main`), and file (`app.py`).
4. In **App settings → Secrets**, add:
```
NVIDIA_API_KEY="your_key"
NVIDIA_API_BASE="https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL="meta/llama-3.1-8b-instruct"
```
5. Deploy and open your public app URL.

> Educational demo. Adjust endpoint/payload if your NVIDIA deployment differs from OpenAI-compatible `/chat/completions`.
