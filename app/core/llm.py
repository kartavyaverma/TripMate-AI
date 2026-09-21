"""
app/core/llm.py

Single Responsibility: construct the one shared LLM client.

Every agent and helper imports `get_llm()` instead of building its own
`ChatGoogleGenerativeAI(...)` instance. This keeps model choice, credentials,
and any future client-level settings (temperature, retries, etc.) in one place.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Return the shared ChatGoogleGenerativeAI client (constructed once, memoized)."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
    )
