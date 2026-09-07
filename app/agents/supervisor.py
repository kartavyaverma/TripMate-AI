"""
app/agents/supervisor.py

Single Responsibility: the input guardrail + agent routing decision.
This module decides WHICH specialists run and WHAT constraints they get;
it never calls a specialist or an MCP tool itself (see specialists.py).
"""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage

from app.agents.helpers import AGENT_ORDER, KNOWN_AGENTS, json_from_llm, llm_text
from app.monitoring.guardrail_monitor import guardrail_monitor
from app.schemas.state import TravelState, empty_constraints

def supervisor_agent(state: TravelState):
    query = state["user_query"]
    guardrail_start = time.perf_counter()

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    llm_calls = 0
    guardrail_status = "passed"

    try:
        guardrail_raw = llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
        latency_ms = (time.perf_counter() - guardrail_start) * 1000

        if allowed:
            guardrail_status = "passed"
            guardrail_monitor.record_event(
                status="allowed",
                latency_ms=latency_ms,
                query=query,
                reason=guardrail_reason,
            )
        else:
            guardrail_status = "blocked"
            guardrail_monitor.record_event(
                status="blocked",
                latency_ms=latency_ms,
                query=query,
                reason=guardrail_reason,
            )
    except Exception as exc:
        latency_ms = (time.perf_counter() - guardrail_start) * 1000
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_status = "fallback"
        guardrail_reason = "Guardrail validation fallback allowed the request."
        guardrail_monitor.record_event(
            status="fallback",
            latency_ms=latency_ms,
            query=query,
            reason=guardrail_reason,
            error=str(exc),
        )

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "guardrail_status": "blocked",
            "selected_agents": [],
            "trip_constraints": empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

If the origin is not specified in the User request, default the
"origin" in trip_constraints to "India".

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        selected_agents = AGENT_ORDER.copy()
        constraints = empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "guardrail_status": guardrail_status,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }

def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }

ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
}

def route_from_supervisor(state: TravelState) -> list[str]:
    """
    Parallel fan-out router:
    - If guardrail blocked: routes to guardrail_blocked node.
    - If guardrail allowed: launches selected independent specialists
      (flight, hotel, weather) simultaneously in parallel.
    - If no parallel specialists were selected, routes directly to budget_agent.
    """
    from app.agents.helpers import PARALLEL_SPECIALISTS

    if not state.get("guardrail_allowed", True):
        return ["guardrail_blocked"]

    selected = state.get("selected_agents", [])
    parallel_to_run = [agent for agent in PARALLEL_SPECIALISTS if agent in selected]

    if parallel_to_run:
        return parallel_to_run

    return ["budget_agent"]
