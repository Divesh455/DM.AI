"""
FastAPI Application Entry Point.
Initializes the application, configures logger, registers middlewares (CORS & Request logging),
binds custom exception handlers, and mounts API routers.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.core.constants import API_V1_STR, APP_VERSION, PROJECT_NAME
from app.core.exceptions import register_exception_handlers
from app.core.logger import setup_logging
from app.api.v1.router import api_router
from app.models.schemas.response import BaseAPIResponse
from app.utils.middlewares.logging import LoggingMiddleware


# Initialize structured logging first
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown lifecycle hooks.
    """
    logger.info(f"Starting {PROJECT_NAME} on {settings.HOST}:{settings.PORT} in [{settings.APP_ENV}] mode.")
    logger.info("Knowledge directory resolved to {}", settings.knowledge_dir_path)
    logger.info("Vector store directory resolved to {}", settings.vector_store_dir_path)

    if settings.vector_store_index_ready:
        logger.info("Persisted FAISS index detected. Chat endpoint is ready for retrieval.")
    else:
        logger.warning(
            "Persisted FAISS index not found. Run local ingestion or `python scripts/build_vector_store.py` "
            "before deployment if you expect the chat endpoint to answer questions."
        )

    if settings.is_production and settings.INGEST_API_ENABLED:
        logger.warning(
            "INGEST_API_ENABLED is true in production. For free deployments, disable runtime ingestion and "
            "ship the generated FAISS files with the repository."
        )
    yield
    logger.info(f"Shutting down {PROJECT_NAME}...")


app = FastAPI(
    title=PROJECT_NAME,
    description="DM.AI - A clean, production-ready, RAG-powered personal AI assistant backend.",
    version=APP_VERSION,
    lifespan=lifespan,
    # Hide Swagger documentation in production for security
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# Configure exception handling formatting
register_exception_handlers(app)

# Register custom middlewares (order of execution is bottom-to-top)
# LoggingMiddleware intercepts requests, computes latency, and logs client IPs.
app.add_middleware(LoggingMiddleware)

# Enable Cross-Origin Resource Sharing (CORS) for Vercel/Next.js frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main version 1 router
app.include_router(api_router, prefix=API_V1_STR)


@app.get("/", tags=["Root"], response_model=BaseAPIResponse[None])
async def root() -> BaseAPIResponse[None]:
    """
    Root endpoint returning basic server greeting and confirmation.
    """
    message = (
        f"Welcome to the {PROJECT_NAME} API. Access documentation at /docs."
        if not settings.is_production
        else f"{PROJECT_NAME} is running."
    )
    return BaseAPIResponse(success=True, message=message, data=None)
