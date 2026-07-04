"""
Custom application exceptions and global handlers.
Defines clean exception schemas and mounts global exception interceptors so every
failure returns a consistent API response.
"""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger


class AppException(Exception):
    """Base application exception for all managed business logic failures."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ConfigurationException(AppException):
    """Raised when the backend is missing required configuration."""

    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class LLMException(AppException):
    """Raised when interaction with external LLM APIs fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=502)


class RAGException(AppException):
    """Raised for general vector store, ingestion, or retriever failures."""

    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class DocumentParsingException(RAGException):
    """Raised when file parsing fails during ingestion."""

    def __init__(self, message: str):
        super().__init__(message)


class IngestionException(RAGException):
    """Raised when the knowledge base cannot be converted into a usable vector index."""

    def __init__(self, message: str):
        super().__init__(message)


class VectorStoreNotReadyException(AppException):
    """Raised when retrieval is requested before a FAISS index has been created."""

    def __init__(self, message: str = "The knowledge base has not been ingested yet."):
        super().__init__(message, status_code=503)


class NotFoundException(AppException):
    """Raised when requested files, context, or endpoints do not exist."""

    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class ValidationException(AppException):
    """Raised when request bodies or parameters violate validation constraints."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class FeatureDisabledException(AppException):
    """Raised when an endpoint is intentionally disabled by configuration."""

    def __init__(self, message: str):
        super().__init__(message, status_code=403)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers handlers to intercept exceptions and transform them into standard API responses.
    """

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        logger.warning(
            "Handled app exception on {} {} - Status {}: {}",
            request.method,
            request.url.path,
            exc.status_code,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "message": exc.message,
                "data": None,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(
            "Request validation failed on {} {}: {}",
            request.method,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": "Request validation failed.",
                "data": None,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled exception on {} {}: {}",
            request.method,
            request.url.path,
            str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "An unexpected internal server error occurred.",
                "data": None,
            },
        )
