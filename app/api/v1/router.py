"""
Main API Router for Version 1.
Groups and registers all v1 endpoints under a unified path structure.
"""
from fastapi import APIRouter
from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.api.v1.ingest import router as ingest_router
from app.api.v1.resume import router as resume_router

api_router = APIRouter()

# Register routes
api_router.include_router(chat_router, tags=["Chat Operations"])
api_router.include_router(health_router, tags=["Health Operations"])
api_router.include_router(ingest_router, tags=["Knowledge Base Operations"])
api_router.include_router(resume_router, tags=["Portfolio Assets"])
