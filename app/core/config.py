"""
app/core/config.py

Single Responsibility: load & validate environment variables.

Every other module in the codebase MUST import values from `settings`
(the module-level singleton below) instead of calling `os.getenv(...)`
directly. This keeps configuration centralized, makes required-vs-optional
variables explicit, and means there is exactly one place to look when a
value is wrong.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import certifi
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"{name} is missing. Please add {name}=... to your .env file "
            f"(see .env.example)."
        )
    return value

def _optional(name: str, default: str = "") -> str:
    return os.getenv(name, default) or default

@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default_factory=lambda: _require("GEMINI_API_KEY"))
    gemini_model: str = field(
        default_factory=lambda: _optional("GEMINI_MODEL", "gemini-3.8-flash")
    )

    database_url_raw: str = field(default_factory=lambda: _require("DATABASE_URL"))

    tavily_api_key: str = field(default_factory=lambda: _optional("TAVILY_API_KEY"))

    aviationstack_api_key: str = field(
        default_factory=lambda: _optional("AVIATION_STACK_API_KEY")
        or _optional("AVIATIONSTACK_API_KEY")
    )
    uvx_path: str = field(default_factory=lambda: shutil.which("uvx") or "uvx")

    openweather_api_key: str = field(
        default_factory=lambda: _optional("OPENWEATHER_API_KEY")
    )
    openweather_mcp_url: str = field(
        default_factory=lambda: _optional("OPENWEATHER_MCP_URL")
    )
    openweather_mcp_transport: str = field(
        default_factory=lambda: _optional("OPENWEATHER_MCP_TRANSPORT", "streamable_http")
    )
    weather_mcp_mode: str = field(
        default_factory=lambda: _optional("WEATHER_MCP_MODE", "remote").lower()
    )

    default_origin: str = field(
        default_factory=lambda: _optional("DEFAULT_ORIGIN_DATA", "India")
    )
    app_host: str = field(default_factory=lambda: _optional("APP_HOST", "127.0.0.1"))
    app_port: int = field(default_factory=lambda: int(_optional("APP_PORT", "8000")))
    app_reload: bool = field(
        default_factory=lambda: _optional("APP_RELOAD", "true").lower() == "true"
    )

    @property
    def database_url(self) -> str:
        """DATABASE_URL with sslmode=require appended if not already present."""
        url = self.database_url_raw
        if "sslmode=" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}sslmode=require"
        return url

settings = Settings()
