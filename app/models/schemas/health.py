"""
Health Check Schemas.
Defines structure of system health check responses, including application
status, current environment, and service versioning information.
"""
from pydantic import BaseModel


class HealthSchema(BaseModel):
    """
    Detailed system status model.
    """
    status: str
    environment: str
    version: str
    llm_provider: str
    vector_store_ready: bool
    ingest_api_enabled: bool
