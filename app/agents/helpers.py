"""
app/agents/helpers.py

Single Responsibility: small utilities and constants shared across
agent modules (LLM call wrapper, JSON extraction, routing constants).
No LangGraph node functions live here — only their shared building blocks.
"""

from __future__ import annotations

import json
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

def llm_text(system_prompt: str, user_prompt: str) -> str:
    """Invoke the shared LLM with a system + human message and return text."""
    response = get_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)

def json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])
