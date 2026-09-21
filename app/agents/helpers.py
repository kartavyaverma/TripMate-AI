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

from app.core.llm import get_llm

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

def content_to_str(content: Any) -> str:
    """Safely convert an LLM response content to a plain string.

    Gemini sometimes returns a list of content parts (multimodal format)
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

def llm_text(system_prompt: str, user_prompt: str) -> str:
    """Invoke the shared LLM with a system + human message and return text."""
    response = get_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return content_to_str(response.content)

def json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model.

    Handles common Gemini quirks:
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
