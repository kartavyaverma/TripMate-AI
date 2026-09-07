"""
app/mcp/client.py

Single Responsibility: own the MultiServerMCPClient and expose small,
typed async functions (`tavily_mcp_search`, `aviation_mcp_call`,
`weather_mcp_search`, `forecast_mcp_search`) that the specialist agents
call. Agents never touch MCP transport details directly.

Weather can run in two modes (see WEATHER_MCP_MODE in .env):
  - "remote" (default): talk to a third-party OpenWeather MCP endpoint
    over streamable HTTP, exactly like the original project.
  - "custom": spawn app/mcp/custom_weather_mcp_server.py locally over
    stdio, so the app has no dependency on an external MCP provider.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import settings

_MODULE_DIR = Path(__file__).resolve().parent

def _require(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(
            f"{name} is missing. Add {name}=your_key to the project .env file."
        )
    return value

def _subprocess_env(**updates: str | None) -> dict[str, str]:
    """Preserve the current environment and add MCP-specific keys."""
    import os

    env = os.environ.copy()
    for key, value in updates.items():
        if value:
            env[key] = value
    return env

def _build_weather_server_config() -> dict[str, Any]:
    if settings.weather_mcp_mode == "custom":
        return {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "app.mcp.custom_weather_mcp_server"],
            "env": _subprocess_env(
                OPENWEATHER_API_KEY=settings.openweather_api_key,
            ),
        }

    weather_url = settings.openweather_mcp_url
    if weather_url:
        if settings.openweather_api_key and "?" not in weather_url:
            weather_url = f"{weather_url.rstrip('/')}/?apiKey={settings.openweather_api_key}"
    else:
        weather_url = (
            f"https://mcp.openweather.org/mcp/?apiKey={settings.openweather_api_key or ''}"
        )

    return {
        "transport": settings.openweather_mcp_transport,
        "url": weather_url,
    }

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.tavily_api_key or ''}"
            ),
        },
        "aviationstack": {
            "transport": "stdio",
            "command": settings.uvx_path,
            "args": ["--with", "mcp<2.0.0", "aviationstack-mcp"],
            "env": _subprocess_env(
                AVIATION_STACK_API_KEY=settings.aviationstack_api_key,
            ),
        },
        "weather": _build_weather_server_config(),
    }
)

async def _get_server_tool(server_name: str, tool_name: str):
    """
    Load one tool from one MCP server.

    This prevents a broken weather or AviationStack server from crashing
    an unrelated Tavily request.
    """
    if server_name == "tavily":
        _require("TAVILY_API_KEY", settings.tavily_api_key)

    elif server_name == "aviationstack":
        _require("AVIATION_STACK_API_KEY", settings.aviationstack_api_key)
        if shutil.which("uvx") is None:
            raise RuntimeError(
                "uvx was not found. Install uv, reopen the terminal, "
                "activate the project environment, and run `uvx --version`."
            )

    elif server_name == "weather" and settings.weather_mcp_mode != "custom":
        _require("OPENWEATHER_API_KEY", settings.openweather_api_key)

    tools = await client.get_tools(server_name=server_name)
    tool = next((item for item in tools if item.name == tool_name), None)

    if tool is None:
        available_tools = ", ".join(sorted(item.name for item in tools)) or "none"
        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )

    return tool

async def get_all_tools() -> None:
    """Diagnostic helper: test every MCP server independently."""
    for server_name in ("tavily", "aviationstack", "weather"):
        try:
            tools = await client.get_tools(server_name=server_name)
            tool_names = ", ".join(tool.name for tool in tools) or "no tools"
            print(f"{server_name}: OK -> {tool_names}")
        except Exception as exc:
            print(f"{server_name}: FAILED -> {type(exc).__name__}: {exc}")

async def tavily_mcp_search(query: str):
    search_tool = await _get_server_tool("tavily", "tavily_search")
    return await search_tool.ainvoke({"query": query})

async def aviation_mcp_call(tool_name: str, tool_args: dict[str, Any] | None = None):
    aviation_tool = await _get_server_tool("aviationstack", tool_name)
    return await aviation_tool.ainvoke(tool_args or {})

async def _find_weather_tool(candidates: list[str]):
    tools = await client.get_tools(server_name="weather")

    for name in candidates:
        tool = next((t for t in tools if t.name.lower() == name.lower()), None)
        if tool:
            return tool

    if tools:
        return tools[0]

    raise RuntimeError("No tools available on the weather MCP server.")

def _prepare_weather_args(tool: Any, city: str) -> dict[str, Any]:
    schema_args = getattr(tool, "args", {}) or {}
    if "city" in schema_args:
        return {"city": city}
    if "location" in schema_args:
        return {"location": city}
    if "q" in schema_args:
        return {"q": city}
    if "query" in schema_args:
        return {"query": city}
    return {"city": city}

async def weather_mcp_search(city: str):
    if settings.weather_mcp_mode != "custom":
        _require("OPENWEATHER_API_KEY", settings.openweather_api_key)

    tool = await _find_weather_tool(
        [
            "get_current_weather",
            "get_weather",
            "current_weather",
            "weather",
            "get_weather_data",
        ]
    )
    args = _prepare_weather_args(tool, city)
    return await tool.ainvoke(args)

async def forecast_mcp_search(city: str):
    if settings.weather_mcp_mode != "custom":
        _require("OPENWEATHER_API_KEY", settings.openweather_api_key)

    tool = await _find_weather_tool(
        [
            "get_weather_forecast",
            "get_forecast",
            "forecast",
            "get_five_day_forecast",
            "five_day_forecast",
        ]
    )
    args = _prepare_weather_args(tool, city)
    return await tool.ainvoke(args)

def extract_destination(query: str) -> str:
    from app.core.llm import get_llm

    prompt = f"""
Extract only the destination city or country from the travel request.

Travel request:
{query}

Return only the destination name.
Do not add any explanation.
"""
    response = get_llm().invoke(prompt)
    destination = str(response.content).strip()

    if not destination:
        raise ValueError("The destination could not be extracted.")

    return destination
