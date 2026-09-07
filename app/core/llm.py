"""
app/core/llm.py

Single Responsibility: construct the one shared LLM client.

Every agent and helper imports `get_llm()` instead of building its own
`ChatGroq(...)` instance. This keeps model choice, credentials, and any
future client-level settings (temperature, retries, etc.) in one place.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from app.core.config import settings

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Return the shared ChatGroq client (constructed once, memoized)."""
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
    )
