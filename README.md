# TripMate AI — Multi-Agent Travel Planner

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.136+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://github.com/langchain-ai/langgraph"><img src="https://img.shields.io/badge/LangGraph-1.2+-FF4B4B.svg?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"></a>
  <a href="https://console.groq.com/"><img src="https://img.shields.io/badge/Groq-LLM%20Inference-F55036.svg?style=for-the-badge" alt="Groq"></a>
  <a href="https://modelcontextprotocol.io/"><img src="https://img.shields.io/badge/MCP-Protocol%20Enabled-4B0082.svg?style=for-the-badge" alt="Model Context Protocol"></a>
  <a href="https://www.postgresql.org/"><img src="https://img.shields.io/badge/PostgreSQL-Checkpointer-4169E1.svg?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge" alt="License: Apache 2.0"></a>
</p>

---

**TripMate AI** is a modular multi-agent travel orchestration system powered by **LangGraph**, **Groq-hosted LLMs** (GPT-OSS 120B / 20B, Qwen3.8 27B), and the **Model Context Protocol (MCP)**. It includes an intelligent supervisor agent, automated input guardrails with real-time anomaly alerting, parallel specialist agents for flights, hotels, and weather, human-in-the-loop (HITL) approval, and a modern FastAPI web interface.

---

## Table of Contents

- [Key Features](#key-features)
- [Multi-Agent Architecture](#multi-agent-architecture)
- [LLM Models & Token Budget](#llm-models--token-budget)
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
  - **Weather Specialist**: Fetches current conditions and a short forecast via OpenWeather, served by a bundled local stdio MCP server (a hosted HTTP MCP endpoint is optional).
  - **Budget Specialist**: Calculates itemized expense breakdowns (flights, lodging, food, local transit) and checks compliance against user budgets.
  - **Itinerary Specialist**: Synthesizes a coherent, day-by-day travel plan.
- **Human-in-the-Loop (HITL) Review**: Pauses execution at `human_approval`, enabling users to review interim plans, suggest changes, or confirm before final generation.
- **Resilient State Checkpointing**: Supports PostgreSQL persistence (`PostgresSaver`) for multi-turn sessions with automatic, graceful fallback to `MemorySaver` when PostgreSQL is unavailable.
- **Groq Multi-Model LLM Layer**: Three Groq models split by role (fast routing, specialist analysis, long-form writing), each drawing on its own rate-limit pool, with per-request token budgeting so no call exceeds the free tier's 8K tokens-per-minute limit.
- **Editorial Web UI**: A warm, magazine-style interface (Playfair Display + Plus Jakarta Sans, terracotta & sage palette) with an itinerary-flow progress bar, quick-start prompts, a 2-minute auto-approval countdown, markdown rendering, copy, and PDF export.

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

## LLM Models & Token Budget

All inference runs on **[Groq](https://console.groq.com/)** through `langchain-groq`. `app/core/llm.py` exposes one memoized client per **role**, and every agent asks for the role it needs with `get_llm(role)`:

| Role | Default model | Used by | Why this model |
|---|---|---|---|
| `fast` | `qwen/qwen3.8-27b` | Input guardrail, supervisor routing, destination extraction | Quick, clean JSON with thinking disabled (`reasoning_effort="none"`). Its 1K output-tokens-per-minute cap is fine for replies of a few hundred tokens. |
| `specialist` | `openai/gpt-oss-20b` | Flight analysis, budget feasibility | Fast mid-sized reasoning model with no output-per-minute cap (`reasoning_effort="low"`). |
| `writer` | `openai/gpt-oss-120b` | Draft itinerary, final polished plan | Strongest model on the account, reserved for the two long-form documents (`reasoning_effort="low"`). |

Every model can be swapped via `.env` (`GROQ_FAST_MODEL`, `GROQ_SPECIALIST_MODEL`, `GROQ_MODEL`).

### Why three models instead of one?

On Groq's free tier the constraint is **not** the context window (all three models accept 131K tokens) but the **rate limits, which apply per model**:

| Model | Tokens / minute (prompt + output) | Output tokens / minute | Requests / day | Tokens / day |
|---|---|---|---|---|
| `openai/gpt-oss-120b` | 8,000 | — | 1,000 | ~200K |
| `openai/gpt-oss-20b` | 8,000 | — | 1,000 | ~200K |
| `qwen/qwen3.8-27b` | 8,000 | **1,000** | 1,000 | ~200K |

A single trip makes about seven LLM calls within roughly a minute. Putting them all on one model would exceed its 8K tokens-per-minute budget; splitting them across three independent pools keeps every model comfortably inside its limit.

### How prompts are kept inside the limit

1. **Prompt budgets**: raw tool output (Tavily hotel results, forecasts, airport/airline lists) can be tens of thousands of characters. `clip()` in `app/agents/helpers.py` trims each input to a fixed character budget (`PROMPT_LIMITS`) before it is placed in a prompt. The final agent gets smaller shares because the draft it polishes already folds in the tool data.
2. **Per-call output sizing**: `fit_max_tokens()` estimates the prompt size and gives the reply whatever room is left under `LLM_REQUEST_TOKEN_LIMIT` (default `7600`), capped per role (`fast` 450, `specialist` 1,600, `writer` 4,500). Groq rejects any single request above the model's per-minute limit, so this guarantees no call fails outright.
3. **Tight caps on short calls**: the guardrail (200), supervisor (450) and destination lookup (60) together stay under Qwen's 1K output-tokens-per-minute limit.
4. **Retries & truncation warnings**: clients retry up to 3 times and honour Groq's `retry-after` header on `429`, and a warning is logged if any reply stops at its output cap.

### Measured usage (7-day Japan trip, free tier)

| Model | Calls | Tokens per trip | Largest single request |
|---|---|---|---|
| `qwen/qwen3.8-27b` | 3 | ~0.8K | ~0.5K |
| `openai/gpt-oss-20b` | 2 | ~4.7K | ~3.4K |
| `openai/gpt-oss-120b` | 2 | ~11.3K | ~5.8K |

With ~200K tokens/day per model, `gpt-oss-120b` sets the daily ceiling at roughly **15–17 complete trips per day** on the free tier (each "revise" adds one more writer call). If the user approves within a few seconds of the draft, the final call may wait ~30s for the per-minute window to reset; the retry logic handles this automatically. On a paid Groq tier, raise `LLM_REQUEST_TOKEN_LIMIT` to allow longer outputs.

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
│   │   └── llm.py                  ← Centralized Groq LLM factory (fast / specialist / writer roles)
│   │
│   ├── schemas/
│   │   └── state.py                ← TravelState TypedDict & state models
│   │
│   ├── monitoring/
│   │   └── guardrail_monitor.py    ← Standalone GuardrailMonitor (isolated, zero framework coupling)
│   │
│   ├── mcp/
│   │   ├── client.py               ← MCP client wrappers (Tavily, AviationStack, Weather)
│   │   └── custom_weather_mcp_server.py ← Default local OpenWeather stdio MCP server
│   │
│   ├── agents/
│   │   ├── helpers.py              ← Shared utilities (JSON extraction, prompt budgets via clip())
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
    ├── ui.js                       ← Presentation-only: syncs the itinerary-flow bar with app state
    └── style.css                   ← Editorial design system (tokens, layout, responsive rules)
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
- **Groq API Key**: Required for LLM inference ([Groq Console](https://console.groq.com/keys)).
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
GROQ_API_KEY=gsk_your_groq_key
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
// Response (trimmed)
{
  "success": true,
  "thread_id": "user_6fc2877ecaa4401bb936fc03e6fc8337",
  "requires_approval": true,
  "answer": "## 7-Day Tokyo Itinerary ... (draft markdown)",
  "itinerary": "... (same draft markdown)",
  "approval_request": "Please review the generated draft itinerary...",
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": { "destination": "Tokyo", "origin": "Mumbai", "duration": "7 days", "budget": "2.5 lakhs", "travel_style": "", "special_preferences": [] },
  "supervisor_reasoning": "...",
  "flight_results": "...",
  "hotel_results": "...",
  "weather_results": "...",
  "budget_results": "...",
  "guardrail_status": "passed",
  "guardrail_metrics": { "total_requests": 1, "is_alerting": false },
  "llm_calls": 6
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

Set `"approved": false` with non-empty `feedback` to request a revision (an empty rejection returns `400`). The response has the same shape as above, with `requires_approval: false` and the final plan in `answer`.

---

## Model Context Protocol (MCP) Setup

TripMate AI uses the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) to decouple external tool execution from core agent logic:

| Server | Transport | Tools used |
|---|---|---|
| **Tavily** | `streamable_http` (`mcp.tavily.com`) | `tavily_search` |
| **AviationStack** | `stdio` via `uvx aviationstack-mcp` | `list_airports`, `list_airlines` |
| **Weather** | `stdio` (local FastMCP) or `streamable_http` | `get_current_weather`, `get_weather_forecast` |

### Weather MCP Modes
- **Custom Local Mode (`WEATHER_MCP_MODE=custom`, default)**: TripMate spawns `app/mcp/custom_weather_mcp_server.py` locally over stdio. It calls the OpenWeather REST API directly, so only `OPENWEATHER_API_KEY` is needed.
- **Remote Mode (`WEATHER_MCP_MODE=remote`)**: Connects over `streamable_http` to a hosted weather MCP server at `OPENWEATHER_MCP_URL`.

If `remote` is selected but `OPENWEATHER_MCP_URL` is empty, TripMate automatically uses the local server instead (`Settings.effective_weather_mcp_mode`), so weather keeps working with just an API key.

---

## Environment Configuration

| Variable | Type | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **Required** | — | API key for Groq LLM inference |
| `DATABASE_URL` | **Required** | — | PostgreSQL connection URI for LangGraph state persistence |
| `GROQ_FAST_MODEL` | Optional | `qwen/qwen3.8-27b` | Guardrail, supervisor routing, destination extraction |
| `GROQ_SPECIALIST_MODEL` | Optional | `openai/gpt-oss-20b` | Flight and budget analysis |
| `GROQ_MODEL` | Optional | `openai/gpt-oss-120b` | Draft itinerary and final polished plan |
| `LLM_TEMPERATURE` | Optional | `0.4` | Sampling temperature for all roles |
| `LLM_REQUEST_TOKEN_LIMIT` | Optional | `7600` | Max prompt + output tokens per request; output caps shrink to fit (free tier TPM is 8000) |
| `TAVILY_API_KEY` | Optional | `""` | Search API key for hotel & attraction discovery |
| `AVIATIONSTACK_API_KEY` | Optional | `""` | API key for flight schedules and status |
| `OPENWEATHER_API_KEY` | Optional | `""` | OpenWeather key for local or custom weather queries |
| `WEATHER_MCP_MODE` | Optional | `custom` | Weather MCP provider: `custom` (bundled local server) or `remote` |
| `OPENWEATHER_MCP_URL` | Optional | `""` | Hosted weather MCP endpoint; required for `remote`, otherwise the local server is used |
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

If `uvx` is not installed or not on your `PATH`, the flight agent records "Flight information unavailable" and the rest of the pipeline continues; the budget and itinerary agents fall back to general route guidance. To enable live airport and airline data, install `uv` from [astral.sh/uv](https://docs.astral.sh/uv/) and set `AVIATIONSTACK_API_KEY`.
</details>

<details>
<summary><b>3. SSL Certificate verification errors on Windows / macOS</b></summary>

TripMate AI automatically injects `certifi.where()` into `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` in `app/core/config.py` to prevent certificate handshake failures across corporate networks and local environments.
</details>

<details>
<summary><b>4. Groq <code>429</code> rate limit / "Request too large"</b></summary>

- **Tokens per minute (TPM)**: each model allows 8K tokens per minute on the free tier. Clients retry automatically using Groq's `retry-after`, so an occasional short wait (~30s) right after a draft is normal.
- **Tokens per day (TPD)**: each model allows ~200K tokens per day. If `gpt-oss-120b` runs out, the error says so; wait for the reset or point `GROQ_MODEL` at another model on your account.
- **Output tokens per minute (OTPM)**: `qwen/qwen3.8-27b` is capped at 1K output tokens per minute, which is why it only handles the short JSON calls.
- **"Request too large"**: lower `LLM_REQUEST_TOKEN_LIMIT` if you changed models or your account's limits differ.

Check exactly which models your key can use:
```bash
curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
```
</details>

<details>
<summary><b>5. Weather unavailable (<code>getaddrinfo failed</code> / <code>ConnectError</code>)</b></summary>

The weather MCP endpoint could not be reached. Use the default `WEATHER_MCP_MODE=custom` (or leave `OPENWEATHER_MCP_URL` empty) so the bundled local server is used, and make sure `OPENWEATHER_API_KEY` is valid. When weather fails, the agent still continues with general seasonal guidance.
</details>

<details>
<summary><b>6. "reply hit the max output token cap" warning</b></summary>

A draft or final plan reached its output cap and may end mid-sentence. The prompts already ask for compact output; if it happens often, shorten the request or raise `LLM_REQUEST_TOKEN_LIMIT` on a paid tier.
</details>

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

