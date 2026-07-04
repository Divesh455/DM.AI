"""
Deployment launcher for DM.AI.
Starts Uvicorn using Python so hosting platforms do not depend on shell
expansion for the PORT environment variable.
"""
from __future__ import annotations

import os

import uvicorn

from app.core.config import settings


def get_runtime_port() -> int:
    """
    Returns a valid runtime port from the environment or falls back to settings.
    """
    raw_port = os.getenv("PORT", str(settings.PORT)).strip()
    try:
        return int(raw_port)
    except ValueError:
        return settings.PORT


def main() -> None:
    """
    Boots the FastAPI app with deployment-safe host and port resolution.
    """
    host = os.getenv("HOST", settings.HOST).strip() or settings.HOST
    port = get_runtime_port()
    uvicorn.run("app.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
