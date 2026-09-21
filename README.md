# TripMate AI — Multi-Agent Travel Planner

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.136+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://github.com/langchain-ai/langgraph"><img src="https://img.shields.io/badge/LangGraph-1.2+-FF4B4B.svg?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"></a>
  <a href="https://aistudio.google.com/"><img src="https://img.shields.io/badge/Gemini-Google%20AI-4285F4.svg?style=for-the-badge&logo=google&logoColor=white" alt="Google Gemini"></a>
  <a href="https://modelcontextprotocol.io/"><img src="https://img.shields.io/badge/MCP-Protocol%20Enabled-4B0082.svg?style=for-the-badge" alt="Model Context Protocol"></a>
  <a href="https://www.postgresql.org/"><img src="https://img.shields.io/badge/PostgreSQL-Checkpointer-4169E1.svg?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge" alt="License: Apache 2.0"></a>
</p>

---

**TripMate AI** is a modular multi-agent travel orchestration system powered by **LangGraph**, **Google Gemini LLM**, and the **Model Context Protocol (MCP)**. It includes an intelligent supervisor agent, automated input guardrails with real-time anomaly alerting, parallel specialist agents for flights, hotels, and weather, human-in-the-loop (HITL) approval, and a modern FastAPI web interface.

---

## Table of Contents

- [Key Features](#key-features)
- [Multi-Agent Architecture](#multi-agent-architecture)
- [Project Structure & Separation of Concerns](#project-structure--separation-of-concerns)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Virtual Environment](#2-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Configure Environment Variables](#4-configure-environment-variables)
  - [5. Run the Application](#5-run-the-application)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Model Context Protocol (MCP) Setup](#model-context-protocol-mcp-setup)
- [Environment Configuration](#environment-configuration)
- [Monitoring & Guardrails](#monitoring--guardrails)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Key Features

- **Supervisor Orchestration**: Analyzes user intent, extracts structured travel constraints (origin, destination, budget, duration), and coordinates task delegation across specialists.
- **Input Guardrail & Anomaly Alerting**: Rejects off-topic, toxic, or adversarial queries before specialist execution, tracking sliding-window alert metrics (`/api/monitoring/guardrail-metrics`).
- **Parallel Specialist Execution**:
  - **Flight Specialist**: Queries live flight schedules, status, and routes via AviationStack MCP tool / heuristic estimation.
  - **Hotel Specialist**: Discovers accommodations, amenities, and location details using Tavily search.
  - **Weather Specialist**: Fetches real-time destination forecasts via OpenWeather (supports remote HTTP MCP and local stdio MCP).
  - **Budget Specialist**: Calculates itemized expense breakdowns (flights, lodging, food, local transit) and checks compliance against user budgets.
  - **Itinerary Specialist**: Synthesizes a coherent, day-by-day travel plan.
- **Human-in-the-Loop (HITL) Review**: Pauses execution at `human_approval`, enabling users to review interim plans, suggest changes, or confirm before final generation.
- **Resilient State Checkpointing**: Supports PostgreSQL persistence (`PostgresSaver`) for multi-turn sessions with automatic, graceful fallback to `MemorySaver` when PostgreSQL is unavailable.
- **Clean Web UI**: Responsive web frontend with live execution status, quick prompt presets, and interactive human-in-the-loop controls.

---

## Multi-Agent Architecture

```mermaid
flowchart TD
    START([User Query]) --> Supervisor[Supervisor & Input Guardrail]
    
    Supervisor -->|Invalid / Blocked| GuardrailBlocked[Guardrail Blocked Agent]
    GuardrailBlocked --> END([Return Block Reason])
    
    Supervisor -->|Passed & Constraints Parsed| ParallelSpecialists{Parallel Fan-Out}
    
    subgraph Specialists [Specialist Agents]
        ParallelSpecialists --> FlightAgent[Flight Agent\nAviationStack MCP]
        ParallelSpecialists --> HotelAgent[Hotel Agent\nTavily Search]
        ParallelSpecialists --> WeatherAgent[Weather Agent\nOpenWeather MCP]
    end
    
    FlightAgent --> BudgetAgent[Budget Agent]
    HotelAgent --> BudgetAgent
    WeatherAgent --> BudgetAgent
    
    BudgetAgent --> ItineraryAgent[Itinerary Agent]
    ItineraryAgent --> HumanApproval[Human Approval\nPause & Interrupt State]
    
    HumanApproval -->|Feedback / Revision| FinalAgent[Final Agent]
    HumanApproval -->|Approved| FinalAgent
    FinalAgent --> END_SUCCESS([Complete Itinerary])
```

---

## Project Structure & Separation of Concerns

The codebase strictly adheres to the **Single Responsibility Principle (SRP)**:

```
tripmate-refactored/
├── app.py                          ← Entrypoint launcher (reads host/port/reload from config)
├── requirements.txt                ← Pinned Python dependencies
├── Dockerfile                      ← Containerization definition
├── .env.example                    ← Sanitized environment template
├── .dockerignore                   ← Docker build ignore rules
├── .gitignore                      ← Comprehensive git ignore rules
│
├── app/
│   ├── core/
│   │   ├── config.py               ← Single source of truth for configuration & env variables
│   │   └── llm.py                  ← Centralized Gemini LLM factory
│   │
│   ├── schemas/
│   │   └── state.py                ← TravelState TypedDict & state models
│   │
│   ├── monitoring/
│   │   └── guardrail_monitor.py    ← Standalone GuardrailMonitor (isolated, zero framework coupling)
│   │
│   ├── mcp/
│   │   ├── client.py               ← MCP client wrappers (Tavily, AviationStack, Weather)
│   │   └── custom_weather_mcp_server.py ← Optional local OpenWeather stdio MCP server
│   │
│   ├── agents/
│   │   ├── helpers.py              ← Shared utilities (JSON extraction, prompt helpers)
│   │   ├── supervisor.py           ← Input guardrail verification & routing logic
│   │   ├── specialists.py          ← Flight, hotel, weather, budget, & itinerary agents
│   │   └── hitl.py                 ← Human review node & final synthesis agent
│   │
│   ├── graph/
│   │   ├── builder.py              ← StateGraph topology & PostgreSQL/MemorySaver checkpointer
│   │   └── runner.py               ← Execution orchestrator (run_travel_agent, resume_travel_agent)
│   │
│   └── api/
│       └── routes/
│           ├── travel.py           ← Endpoints: /api/travel/plan, /api/travel/approve
│           └── monitoring.py       ← Endpoints: /api/monitoring/guardrail-metrics
│
├── templates/
│   └── index.html                  ← Frontend UI template
│
└── static/
    ├── script.js                   ← Asynchronous client API requests & UI state management
    └── style.css                   ← Web UI styles & layout
```

### Architectural Layer Responsibilities

| Layer | Responsibility | Never Does |
|---|---|---|
| `app/core/` | Environment variables, config validation, shared LLM client | Business logic, agent routing, HTTP |
| `app/schemas/` | `TravelState` TypedDict and data shapes | Logic, network calls |
| `app/monitoring/` | Sliding-window metric tracking and alert states | LLM invocation, API routing |
| `app/mcp/` | MCP tool transports (Tavily, AviationStack, OpenWeather) | Agent prompts, graph transitions |
| `app/agents/` | Individual agent nodes & prompt contracts | Direct graph compilation, HTTP handlers |
| `app/graph/` | Graph topology (`builder.py`) and execution (`runner.py`) | Raw prompt text, HTTP responses |
| `app/api/routes/` | HTTP request validation, status codes, response shaping | Agent execution logic |
| `app.py` | Top-level ASGI runner initialization | Business or graph logic |

---

## Prerequisites

- **Python**: Version `3.11` or newer.
- **Gemini API Key**: Required for LLM inference ([Google AI Studio](https://aistudio.google.com/app/apikey)).
- **PostgreSQL Database** *(Optional)*: Supabase, Neon, Render, or a local Docker Postgres instance. If omitted or unreachable, TripMate AI automatically falls back to `MemorySaver`.
- **uv / uvx** *(Optional)*: Required if using the AviationStack MCP server via `uvx` ([Install uv](https://docs.astral.sh/uv/)).
- **External Tool Keys** *(Optional)*:
  - [Tavily API Key](https://tavily.com/) for web search and hotel discovery.
  - [AviationStack API Key](https://aviationstack.com/) for live flight schedules.
  - [OpenWeather API Key](https://openweathermap.org/api) for destination weather forecasts.

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-username/tripmate-ai.git
cd tripmate-ai
```

### 2. Virtual Environment

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the sanitized template to `.env`:

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
DATABASE_URL=postgresql://postgres:password@localhost:5432/tripmate_db
TAVILY_API_KEY=tvly-your_tavily_key
AVIATIONSTACK_API_KEY=your_aviationstack_key
OPENWEATHER_API_KEY=your_openweather_key
```

> **Note:** If you do not have a PostgreSQL database running, leave `DATABASE_URL` as is; the system will log a warning and run using the in-memory checkpointer.

### 5. Run the Application

**Option A — Direct Launcher:**
```bash
python app.py
```

**Option B — Uvicorn with Autoreload:**
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://127.0.0.1:8000** in your browser to start planning trips.

---

## Docker Deployment

You can containerize and run the application with Docker:

```bash
# 1. Build the Docker image
docker build -t tripmate-ai .

# 2. Run container using your local .env configuration
docker run --rm -d -p 8000:8000 --env-file .env --name tripmate tripmate-ai
```

Access the interface at **http://localhost:8000**.

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web application interface |
| `POST` | `/api/travel/plan` | Initialize a travel planning workflow |
| `POST` | `/api/travel/approve` | Submit human-in-the-loop approval or feedback |
| `GET` | `/api/monitoring/guardrail-metrics` | Retrieve guardrail metrics and active alerts |
| `POST` | `/api/monitoring/guardrail-metrics/reset` | Reset sliding window guardrail metrics |
| `GET` | `/health` | System health check and feature capabilities |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/redoc` | Interactive ReDoc API documentation |

### Request & Response Examples

#### 1. Plan Trip (`POST /api/travel/plan`)
```json
// Request
{
  "message": "Plan a 7-day trip to Tokyo, Japan from Mumbai under 2.5 lakhs with flights and hotels.",
  "thread_id": "optional-custom-session-id"
}
```

```json
// Response
{
  "success": true,
  "thread_id": "787f73db-2481-4203-aa9d-1ca4f8c679a9",
  "status": "awaiting_approval",
  "itinerary_draft": "...",
  "budget_breakdown": { ... },
  "flight_recommendations": [ ... ],
  "hotel_recommendations": [ ... ],
  "weather_forecast": { ... }
}
```

#### 2. Human Approval (`POST /api/travel/approve`)
```json
// Request
{
  "thread_id": "787f73db-2481-4203-aa9d-1ca4f8c679a9",
  "approved": true,
  "feedback": "Add a sushi workshop on day 3."
}
```

---

## Model Context Protocol (MCP) Setup

TripMate AI uses the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) to decouple external tool execution from core agent logic:

### Weather MCP Modes
- **Remote Mode (`WEATHER_MCP_MODE=remote`)**: Connects over HTTP (`streamable_http`) to a hosted MCP weather server specified in `OPENWEATHER_MCP_URL`.
- **Custom Local Mode (`WEATHER_MCP_MODE=custom`)**: TripMate spawns `app/mcp/custom_weather_mcp_server.py` locally via stdio, interfacing directly with the OpenWeather API without third-party proxy dependencies.

Switch between modes simply by toggling `WEATHER_MCP_MODE` in `.env`.

---

## Environment Configuration

| Variable | Type | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | **Required** | — | API key for Google Gemini LLM inference |
| `DATABASE_URL` | **Required** | — | PostgreSQL connection URI for LangGraph state persistence |
| `GEMINI_MODEL` | Optional | `gemini-3.8-flash` | Target Gemini model (e.g. `gemini-2.0-flash`, `gemini-2.5-flash`) |
| `TAVILY_API_KEY` | Optional | `""` | Search API key for hotel & attraction discovery |
| `AVIATIONSTACK_API_KEY` | Optional | `""` | API key for flight schedules and status |
| `OPENWEATHER_API_KEY` | Optional | `""` | OpenWeather key for local or custom weather queries |
| `WEATHER_MCP_MODE` | Optional | `remote` | Weather MCP provider: `remote` or `custom` |
| `OPENWEATHER_MCP_URL` | Optional | `""` | Endpoint URL when using remote MCP transport |
| `OPENWEATHER_MCP_TRANSPORT` | Optional | `streamable_http` | Transport protocol for remote weather MCP |
| `DEFAULT_ORIGIN_DATA` | Optional | `India` | Default country of origin if unspecified in prompt |
| `APP_HOST` | Optional | `127.0.0.1` | Host binding for development server |
| `APP_PORT` | Optional | `8000` | Port for development server |
| `APP_RELOAD` | Optional | `true` | Autoreload on file changes (`true`/`false`) |

---

## Monitoring & Guardrails

The supervisor incorporates an intelligent guardrail filter:
1. **Safety & Scope Validation**: Discards non-travel or malicious prompts before downstream agents are triggered.
2. **Real-time Sliding Window Metrics**: Tracks requests, pass rates, block rates, and fallback rates over rolling time windows.
3. **Automated Alerting**: Triggers alerts when the fallback rate or error anomalies exceed predefined thresholds. Inspect the status anytime at:
   ```bash
   curl http://localhost:8000/api/monitoring/guardrail-metrics
   ```

---

## Troubleshooting

<details>
<summary><b>1. PostgreSQL connection failed (falling back to MemorySaver)</b></summary>

TripMate AI includes automatic fallback to `MemorySaver`. If your database instance is offline, waking up from sleep, or has invalid credentials, the system logs a warning and continues running in memory. For persistent cross-process state, ensure your `DATABASE_URL` includes valid credentials and reachable host permissions.
</details>

<details>
<summary><b>2. AviationStack / UVX not found</b></summary>

If `uvx` is not installed or not available on your system `PATH`, TripMate gracefully falls back to built-in heuristic flight pricing and schedules. To enable live MCP execution, install `uv` from [astral.sh/uv](https://docs.astral.sh/uv/).
</details>

<details>
<summary><b>3. SSL Certificate verification errors on Windows / macOS</b></summary>

TripMate AI automatically injects `certifi.where()` into `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` in `app/core/config.py` to prevent certificate handshake failures across corporate networks and local environments.
</details>

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

