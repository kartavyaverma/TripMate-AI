"""
app/api/routes/travel.py

Single Responsibility: HTTP request/response handling for travel
planning and approval. All business logic lives in app/graph/runner.py —
this module only validates input, calls the runner, and shapes the
HTTP response.
"""

from __future__ import annotations

import traceback

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.graph.runner import resume_travel_agent, run_travel_agent

router = APIRouter(prefix="/api/travel", tags=["travel"])

class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None

class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""

@router.post("/plan")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Message cannot be empty."},
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        return JSONResponse(content={"success": True, **result})

    except Exception as exc:
        print("ERROR:", exc)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)},
        )

@router.post("/approve")
async def approve_travel_plan(request_data: ApprovalRequest):
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                },
            )

        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )

        return JSONResponse(content={"success": True, **result})

    except Exception as exc:
        print("APPROVAL ERROR:", exc)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)},
        )
