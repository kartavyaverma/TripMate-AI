"""
app/graph/builder.py

Single Responsibility: wire the LangGraph StateGraph together and attach
the PostgreSQL checkpointer. This module owns graph topology only —
node implementations live in app/agents/*, and running/serializing
results lives in app/graph/runner.py.
"""

from __future__ import annotations

import logging
import psycopg
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

from app.agents.hitl import final_agent, human_approval_agent
from app.agents.specialists import (
    budget_agent,
    flight_agent,
    hotel_agent,
    itinerary_agent,
    weather_agent,
)
from app.agents.supervisor import (
    ROUTE_MAP,
    guardrail_blocked_agent,
    route_from_supervisor,
    supervisor_agent,
)
from app.core.config import settings
from app.schemas.state import TravelState

def _build_graph() -> StateGraph:
    graph = StateGraph(TravelState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("guardrail_blocked", guardrail_blocked_agent)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("weather_agent", weather_agent)
    graph.add_node("budget_agent", budget_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("human_approval", human_approval_agent)
    graph.add_node("final_agent", final_agent)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

    graph.add_edge("flight_agent", "budget_agent")
    graph.add_edge("hotel_agent", "budget_agent")
    graph.add_edge("weather_agent", "budget_agent")

    graph.add_edge("budget_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "human_approval")
    graph.add_edge("human_approval", "final_agent")
    graph.add_edge("final_agent", END)
    graph.add_edge("guardrail_blocked", END)

    return graph

def _build_checkpointer() -> PostgresSaver | MemorySaver:
    if getattr(settings, "database_url_raw", None):
        try:
            conn = psycopg.connect(
                settings.database_url,
                autocommit=True,
                row_factory=dict_row,
                connect_timeout=5,
            )
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            logger.info("Successfully connected to PostgreSQL checkpointer.")
            return checkpointer
        except Exception as exc:
            logger.warning(
                "Could not connect to PostgreSQL (%s). "
                "Falling back to in-memory checkpointer (MemorySaver).",
                exc,
            )
    return MemorySaver()

_graph = _build_graph()
_checkpointer = _build_checkpointer()
travel_graph = _graph.compile(checkpointer=_checkpointer)
