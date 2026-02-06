# nvidia_client.py
# ----------------
# Minimal client to call NVIDIA LLM (NIM/NeMo) using REST with requests.
# Defaults to OpenAI-compatible /chat/completions style.

from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any

import requests


class NvidiaLLMClient:
    """
    Thin wrapper over an NVIDIA LLM REST API (OpenAI-compatible).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 60,
    ) -> None:
        """
        Initialize client from args or env vars.

        Environment variables:
        - NVIDIA_API_KEY
        - NVIDIA_API_BASE (e.g., https://integrate.api.nvidia.com/v1)
        - NVIDIA_MODEL (e.g., meta/llama-3.1-8b-instruct)
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.api_base = api_base or os.getenv("NVIDIA_API_BASE", "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")
        self.model = model or os.getenv("NVIDIA_MODEL", "llama-3_2-nemoretriever-300m-embed-v1")
        self.timeout_sec = timeout_sec

        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is missing. Provide it via env var, Streamlit secrets, or constructor.")
        if not self.api_base.startswith("http"):
            raise ValueError("NVIDIA_API_BASE looks invalid. Expected a URL (e.g., https://.../v1).")

    def _headers(self) -> Dict[str, str]:
        """HTTP headers for NVIDIA API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def summarize_text(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """
        Call NVIDIA LLM via /chat/completions to get a summary.
        Adjust endpoint/payload if your deployment differs.
        """
        url = self._chat_completions_url()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a world-class summarization assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_sec)
        except requests.RequestException as net_err:
            raise RuntimeError(f"Network error talking to NVIDIA API: {net_err}") from net_err

        if resp.status_code != 200:
            # Try to surface server-provided details
            try:
                err_json = resp.json()
            except Exception:
                err_json = resp.text
            raise RuntimeError(f"NVIDIA API returned {resp.status_code}. Details: {err_json}")

        try:
            data = resp.json()
        except json.JSONDecodeError as je:
            raise RuntimeError(f"Could not parse JSON from NVIDIA API: {je}") from je

        # Common response shapes:
        content = (
            self._get_nested(data, ["choices", 0, "message", "content"]) or
            self._get_nested(data, ["choices", 0, "text"]) or
            data.get("output_text") or
            data.get("text")
        )

        if not content:
            raise RuntimeError(f"Unexpected response shape from NVIDIA API: {data}")

        return content.strip()

    def _chat_completions_url(self) -> str:
        """Build the /chat/completions URL."""
        base = self.api_base.rstrip("/")
        return f"{base}/chat/completions"

    @staticmethod
    def _get_nested(obj: Any, path: list[Any]) -> Optional[Any]:
        """
        Safely get nested values from dict/list structures.
        Example: ["choices", 0, "message", "content"]
        """
        cur = obj
        for key in path:
            try:
                if isinstance(key, int) and isinstance(cur, list):
                    cur = cur[key]
                elif isinstance(key, str) and isinstance(cur, dict):
                    cur = cur.get(key)
                else:
                    return None
            except (KeyError, IndexError, TypeError):
                return None
        return cur
