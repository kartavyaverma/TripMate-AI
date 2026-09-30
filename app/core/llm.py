"""
app/core/llm.py

Single Responsibility: construct the shared Groq LLM clients.

Every agent and helper imports `get_llm(role)` instead of building its own
`ChatGroq(...)` instance. This keeps model choice, credentials, and
client-level settings (temperature, retries, output caps) in one place.

Roles map to separate Groq models so each draws from its own rate-limit
pool (free tier: 8K tokens per minute per model; qwen/qwen3.8-27b is also
capped at 1K *output* tokens per minute, so it only gets the short JSON calls):
  - "fast":       guardrail, supervisor routing, destination extraction
  - "specialist": flight and budget analysis
  - "writer":     draft itinerary and final polished response
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_groq import ChatGroq

from app.core.config import settings

LLMRole = Literal["fast", "specialist", "writer"]

# Output caps per role. Groq counts prompt + requested output against the
# per-minute token limit; long calls use fit_max_tokens() to trim the cap to
# whatever room the prompt leaves under LLM_REQUEST_TOKEN_LIMIT.
MAX_OUTPUT_TOKENS: dict[str, int] = {
    "fast": 450,
    "specialist": 1600,
    "writer": 4500,
}

MIN_OUTPUT_TOKENS = 1024

def fit_max_tokens(role: LLMRole, *prompt_parts: str) -> int:
    """Output cap for one call so prompt + output stays under the request limit.

    Token count is estimated conservatively at ~3 characters per token
    (tables, JSON, and symbols like the rupee sign tokenize densely).
    """
    estimated_prompt_tokens = sum(len(part) for part in prompt_parts) // 3 + 100
    room = settings.llm_request_token_limit - estimated_prompt_tokens
    return max(MIN_OUTPUT_TOKENS, min(MAX_OUTPUT_TOKENS[role], room))

def _model_for(role: LLMRole) -> str:
    if role == "fast":
        return settings.groq_fast_model
    if role == "specialist":
        return settings.groq_specialist_model
    return settings.groq_model

def _reasoning_kwargs(model: str) -> dict[str, str]:
    """Keep hidden reasoning cheap so it doesn't eat the output budget."""
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none"}
    return {}

@lru_cache(maxsize=None)
def get_llm(role: LLMRole = "writer") -> ChatGroq:
    """Return the shared ChatGroq client for a role (constructed once, memoized)."""
    model = _model_for(role)
    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature,
        max_tokens=MAX_OUTPUT_TOKENS[role],
        # Retries honour Groq's retry-after header on 429 rate limits.
        max_retries=3,
        **_reasoning_kwargs(model),
    )
