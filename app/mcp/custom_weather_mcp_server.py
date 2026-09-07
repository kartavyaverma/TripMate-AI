"""
app/mcp/custom_weather_mcp_server.py

Single Responsibility: expose current-weather and 5-day-forecast tools
over MCP (stdio transport), backed directly by the OpenWeather REST API.

This is OPTIONAL. By default the app talks to a remote OpenWeather MCP
endpoint (WEATHER_MCP_MODE=remote in .env, the same behavior as the
original project). Set WEATHER_MCP_MODE=custom to have app/mcp/client.py
spawn this file as a local stdio MCP server instead — useful if you don't
have access to a third-party MCP endpoint, or want a fully self-hosted
weather integration.

Run it standalone for a quick manual smoke test:
    python -m app.mcp.custom_weather_mcp_server
(it will idle waiting for MCP stdio messages; Ctrl+C to exit)
"""

from __future__ import annotations

import requests
from mcp.server.fastmcp import FastMCP

from app.core.config import settings

mcp = FastMCP("custom-weather-mcp")

_BASE_URL = "https://api.openweathermap.org/data/2.5"

def _api_key() -> str:
    if not settings.openweather_api_key:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is missing. Add it to your .env file to use "
            "the custom weather MCP server."
        )
    return settings.openweather_api_key

@mcp.tool()
def get_current_weather(city: str) -> dict:
    """Return current weather conditions for the given city."""
    response = requests.get(
        f"{_BASE_URL}/weather",
        params={"q": city, "appid": _api_key(), "units": "metric"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    return {
        "city": data.get("name", city),
        "description": (data.get("weather") or [{}])[0].get("description", ""),
        "temperature_c": data.get("main", {}).get("temp"),
        "feels_like_c": data.get("main", {}).get("feels_like"),
        "humidity_percent": data.get("main", {}).get("humidity"),
        "wind_speed_mps": data.get("wind", {}).get("speed"),
    }

@mcp.tool()
def get_weather_forecast(city: str) -> dict:
    """Return a 5-day / 3-hour weather forecast summary for the given city."""
    response = requests.get(
        f"{_BASE_URL}/forecast",
        params={"q": city, "appid": _api_key(), "units": "metric"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    entries = data.get("list", [])[:8]
    forecast = [
        {
            "time": entry.get("dt_txt"),
            "description": (entry.get("weather") or [{}])[0].get("description", ""),
            "temperature_c": entry.get("main", {}).get("temp"),
        }
        for entry in entries
    ]

    return {
        "city": data.get("city", {}).get("name", city),
        "forecast": forecast,
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
