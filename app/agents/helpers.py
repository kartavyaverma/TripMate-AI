"""
app/agents/helpers.py

Single Responsibility: small utilities and constants shared across
agent modules (LLM call wrapper, JSON extraction, routing constants).
No LangGraph node functions live here — only their shared building blocks.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import LLMRole, get_llm

KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

PARALLEL_SPECIALISTS = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
]

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]

# Character budgets for tool output and earlier agent output when it is
# re-fed into later prompts (~4 chars per token for prose, ~3 for JSON).
# Raw Tavily and forecast payloads can be tens of thousands of characters;
# without these caps the budget, itinerary, and final prompts exceed Groq's
# 8K tokens-per-minute limit in a single request.
PROMPT_LIMITS = {
    "airport_data": 2000,
    "airline_data": 2000,
    "flight_results": 2400,
    "hotel_results": 3000,
    "weather_results": 2000,
    "budget_results": 2400,
    # The final polish re-reads the draft, which already folds in the tool
    # data, so the raw results get a smaller share of that prompt.
    "final_itinerary": 6000,
    "final_flight_results": 1200,
    "final_hotel_results": 1200,
    "final_weather_results": 1000,
    "final_budget_results": 1200,
}

def clip(value: Any, limit_key: str) -> str:
    """Convert tool/agent output to text and trim it to its prompt budget."""
    text = content_to_str(value).strip()
    max_chars = PROMPT_LIMITS[limit_key]
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated to fit the model's token budget]"

def warn_if_truncated(response: Any, label: str) -> None:
    """Log when a reply stopped at its output-token cap instead of finishing."""
    finish_reason = (getattr(response, "response_metadata", None) or {}).get("finish_reason")
    if finish_reason == "length":
        print(
            f"{label} WARNING: reply hit the max output token cap and may be cut off.",
            flush=True,
        )

def content_to_str(content: Any) -> str:
    """Safely convert an LLM response or tool output to a plain string.

    Models and MCP tools sometimes return a list of content parts
    like [{"type": "text", "text": "..."}] instead of a plain string.
    This normalises both cases to a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", str(part)))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)

def llm_text(
    system_prompt: str,
    user_prompt: str,
    role: LLMRole = "fast",
    max_tokens: int | None = None,
) -> str:
    """Invoke a shared LLM with a system + human message and return text.

    `max_tokens` tightens the role's output cap for short replies, which
    matters on models with a per-minute output-token limit.
    """
    kwargs = {"max_tokens": max_tokens} if max_tokens else {}
    response = get_llm(role).invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        **kwargs,
    )
    return content_to_str(response.content)

def json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model.

    Handles common model quirks:
    - JSON wrapped in markdown code fences (```json ... ``` or ``` ... ```)
    - Single-quoted Python-style dicts (ast.literal_eval fallback)
    """
    # Strip markdown code fences so raw JSON is exposed
    fenced = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Try the de-fenced text first, then the original as a final fallback
    for candidate in (fenced, text):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end < start:
            continue
        blob = candidate[start : end + 1]

        # Standard JSON parse
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass

        # Python-literal fallback (handles single quotes, trailing commas, etc.)
        try:
            result = ast.literal_eval(blob)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass

    raise ValueError(
        f"The model did not return a parseable JSON object. "
        f"Raw response (first 500 chars):\n{text[:500]}"
    )
