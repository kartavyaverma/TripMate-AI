"""
app/api/routes/monitoring.py

Single Responsibility: expose guardrail metrics over HTTP. Delegates
entirely to app/monitoring/guardrail_monitor.py for state.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.monitoring.guardrail_monitor import get_guardrail_metrics, guardrail_monitor

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

@router.get("/guardrail-metrics")
async def guardrail_metrics():
    """Real-time guardrail fallback tracking metrics, event history, and active alerts."""
    return JSONResponse(content={"success": True, "metrics": get_guardrail_metrics()})

@router.post("/guardrail-metrics/reset")
async def reset_guardrail_metrics():
    """Reset guardrail counters and alert history."""
    guardrail_monitor.reset()
    return JSONResponse(
        content={
            "success": True,
            "message": "Guardrail metrics successfully reset.",
            "metrics": get_guardrail_metrics(),
        }
    )
