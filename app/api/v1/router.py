"""
Main API Router for Version 1.
Groups and registers all v1 endpoints under a unified path structure.
"""
from fastapi import APIRouter

from app.core.config import settings

api_router = APIRouter()

# Register lightweight routes first so startup stays cheap.
from app.api.v1.health import router as health_router
from app.api.v1.resume import router as resume_router

api_router.include_router(health_router, tags=["Health Operations"])
api_router.include_router(resume_router, tags=["Portfolio Assets"])

# Import the chat route after the lightweight routes to keep route wiring explicit.
from app.api.v1.chat import router as chat_router

api_router.include_router(chat_router, tags=["Chat Operations"])

if settings.INGEST_API_ENABLED:
    from app.api.v1.ingest import router as ingest_router

    api_router.include_router(ingest_router, tags=["Knowledge Base Operations"])
