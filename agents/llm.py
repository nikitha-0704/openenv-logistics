"""Shared LLM + env-server configuration for single- and multi-agent harnesses.

This mirrors `inference.py` so submission validators see identical Phase-2
behavior whether the user runs the single-agent baseline or the multi-agent
harness:

  - When `API_BASE_URL` and `API_KEY` are both set (graders / LiteLLM proxy),
    the OpenAI client uses them with no fallback.
  - Otherwise local development falls back to HF router or OPENAI_API_KEY.

`MODEL_NAME` and `OPENENV_BASE_URL` are also exposed for convenience.
"""

from __future__ import annotations

import os

from openai import OpenAI

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

OPENENV_BASE_URL = (
    os.environ.get("OPENENV_BASE_URL")
    or os.environ.get("ENV_BASE_URL")
    or "http://localhost:7860"
)

if "API_BASE_URL" in os.environ and "API_KEY" in os.environ:
    API_BASE_URL = os.environ["API_BASE_URL"].rstrip("/")
    MODEL_NAME = os.environ.get("MODEL_NAME", MODEL_NAME)
    client = OpenAI(base_url=API_BASE_URL, api_key=os.environ["API_KEY"])
else:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
    _local_key = HF_TOKEN or os.getenv("OPENAI_API_KEY")
    if not _local_key:
        raise ValueError(
            "Set API_BASE_URL and API_KEY for submission (LiteLLM proxy), or for local "
            "runs set HF_TOKEN or OPENAI_API_KEY."
        )
    client = OpenAI(base_url=API_BASE_URL, api_key=_local_key)
