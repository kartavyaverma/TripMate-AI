"""
app.py

Development launcher for TripMate AI.
Run with: python app.py
Or: uvicorn app:app --reload
"""

import uvicorn

from app.core.config import settings
from app.main import app

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )
