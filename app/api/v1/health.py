"""
Health Check API Route.
Provides an endpoint for load balancers, Render, and monitoring tools to
verify that the API container is running and healthy.
"""
from fastapi import APIRouter
from app.core.config import settings
from app.core.constants import APP_VERSION
from app.models.schemas.health import HealthSchema
from app.models.schemas.response import BaseAPIResponse

router = APIRouter()


@router.get("/health", response_model=BaseAPIResponse[HealthSchema])
def check_health() -> BaseAPIResponse[HealthSchema]:
    """
    Checks the status of the server and returns environment status and app version.
    """
    health_data = HealthSchema(
        status="healthy",
        environment=settings.APP_ENV,
        version=APP_VERSION,
        llm_provider=settings.LLM_PROVIDER,
        vector_store_ready=settings.vector_store_index_ready,
        ingest_api_enabled=settings.INGEST_API_ENABLED,
    )
    return BaseAPIResponse(
        success=True,
        message="System is operational.",
        data=health_data
    )
