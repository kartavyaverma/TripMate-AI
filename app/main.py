"""
app/main.py

FastAPI application entrypoint. Single Responsibility: application wiring —
mount routers, static files, and templates, and expose the top-level
routes (home page, health check, favicon) that don't belong in any
specific router.
"""

from pathlib import Path

import nest_asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import monitoring, travel
from app.monitoring.guardrail_monitor import get_guardrail_metrics

nest_asyncio.apply()

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="TripMate AI",
    description=(
        "LangGraph Multi-Agent Travel Planner with Parallel Execution, "
        "Supervisor, Guardrails with Alert Monitoring, "
        "Human-in-the-Loop, and FastAPI Frontend"
    ),
    version="2.2.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.include_router(travel.router)
app.include_router(monitoring.router)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.get("/health")
async def health_check():
    metrics = get_guardrail_metrics()
    return {
        "status": "ok",
        "message": "TripMate AI API is running",
        "features": [
            "supervisor_agent",
            "parallel_specialists",
            "input_guardrail_alerting",
            "human_in_the_loop",
        ],
        "guardrail": {
            "is_alerting": metrics.get("is_alerting", False),
            "fallback_rate_percent": metrics.get("window_fallback_rate", 0.0),
            "total_requests": metrics.get("total_requests", 0),
        },
    }

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})
