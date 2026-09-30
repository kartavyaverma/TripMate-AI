"""
app/agents/hitl.py

Single Responsibility: the human-in-the-loop approval step and the final
response generator that runs after it. Kept separate from specialists.py
because this is the "review & finalize" stage of the pipeline, not a
data-gathering specialist.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.core.llm import fit_max_tokens, get_llm
from app.agents.helpers import clip, content_to_str, warn_if_truncated
from app.schemas.state import TravelState

def human_approval_agent(state: TravelState):
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }

def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{clip(state.get('flight_results', ''), 'final_flight_results')}

Hotels:
{clip(state.get('hotel_results', ''), 'final_hotel_results')}

Weather:
{clip(state.get('weather_results', ''), 'final_weather_results')}

Budget Analysis:
{clip(state.get('budget_results', ''), 'final_budget_results')}

Draft Itinerary:
{clip(state.get('itinerary', ''), 'final_itinerary')}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
- Keep the whole answer under about 1,300 words, using compact tables, so it is not cut off.
"""

    response = get_llm("writer").invoke(
        [
            SystemMessage(content="You are a professional AI travel booking assistant."),
            HumanMessage(content=final_prompt),
        ],
        max_tokens=fit_max_tokens("writer", final_prompt),
    )

    warn_if_truncated(response, "FINAL AGENT")

    return {
        "final_response": content_to_str(response.content),
        "messages": [response],
        "llm_calls": 1,
    }
