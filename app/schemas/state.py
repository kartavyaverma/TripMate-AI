"""
app/schemas/state.py

Single Responsibility: define the LangGraph state shape.

This module intentionally contains NO logic — only the TypedDict that
flows through every node in the graph. Keeping it separate means agents,
the graph builder, and the API layer can all import the same contract
without pulling in unrelated dependencies (LLM clients, MCP clients, etc).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage

class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    guardrail_allowed: bool
    guardrail_reason: str
    guardrail_status: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: Annotated[int, operator.add]

def empty_constraints() -> dict[str, Any]:
    """Default/blank trip_constraints payload."""
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }
