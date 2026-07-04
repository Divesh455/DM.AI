"""
Configuration management system.
Uses Pydantic Settings v2 to load environment variables from a .env file
with validation, type checking, and clean defaults.
"""
from pathlib import Path
from typing import List, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import (
    BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    RESUME_FILE_NAME,
    VECTOR_STORE_INDEX_FILES,
)


class Settings(BaseSettings):
    """
    Application settings container.
    Loads variables from the environment or a local .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # General App Config
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # RAG Config
    VECTOR_STORE_DIR: str = "data/vector_store"
    KNOWLEDGE_DIR: str = "data/knowledge"
    UPLOADS_DIR: str = "data/uploads"
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 200
    RAG_TOP_K: int = 5
    EMBEDDING_MODEL: str = DEFAULT_EMBEDDING_MODEL

    # LLM Settings
    LLM_PROVIDER: Literal["gemini", "groq"] = "gemini"
    LLM_TEMPERATURE: float = DEFAULT_LLM_TEMPERATURE
    LLM_MAX_OUTPUT_TOKENS: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS
    LLM_TIMEOUT_SECONDS: int = DEFAULT_LLM_TIMEOUT_SECONDS
    LLM_MAX_RETRIES: int = DEFAULT_LLM_MAX_RETRIES
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = DEFAULT_GEMINI_MODEL
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = ""

    # CORS Configuration
    CORS_ORIGINS: str = ""
    INGEST_API_ENABLED: bool = True

    @field_validator("APP_ENV", mode="before")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        """
        Standardizes environment strings so helper properties remain predictable.
        """
        return str(value).strip().lower()

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """
        Uppercases the log level for Loguru compatibility.
        """
        return str(value).strip().upper()

    @field_validator("VECTOR_STORE_DIR", "KNOWLEDGE_DIR", "UPLOADS_DIR", mode="before")
    @classmethod
    def validate_directory_fields(cls, value: str) -> str:
        """
        Guards against blank path configuration values.
        """
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Directory path configuration cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        """
        Ensures the configured RAG and LLM settings are valid.
        """
        if self.RAG_CHUNK_SIZE <= 0:
            raise ValueError("RAG_CHUNK_SIZE must be greater than 0.")
        if self.RAG_CHUNK_OVERLAP < 0:
            raise ValueError("RAG_CHUNK_OVERLAP cannot be negative.")
        if self.RAG_CHUNK_OVERLAP >= self.RAG_CHUNK_SIZE:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE.")
        if self.RAG_TOP_K <= 0:
            raise ValueError("RAG_TOP_K must be greater than 0.")
        if not 0 <= self.LLM_TEMPERATURE <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0.0 and 2.0.")
        if self.LLM_MAX_OUTPUT_TOKENS <= 0:
            raise ValueError("LLM_MAX_OUTPUT_TOKENS must be greater than 0.")
        if self.LLM_TIMEOUT_SECONDS <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be greater than 0.")
        if self.LLM_MAX_RETRIES < 0:
            raise ValueError("LLM_MAX_RETRIES cannot be negative.")
        return self

    @property
    def cors_origins_list(self) -> List[str]:
        """
        Parses the comma-separated CORS_ORIGINS string into a list of origins.
        """
        if not self.CORS_ORIGINS.strip():
            return []
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        """
        Returns True when the backend is running in production mode.
        """
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        """
        Returns True when the backend is running in development mode.
        """
        return self.APP_ENV == "development"

    @property
    def base_dir(self) -> Path:
        """
        Returns the project root for resolving relative filesystem paths.
        """
        return BASE_DIR

    def resolve_path(self, value: str) -> Path:
        """
        Resolves relative paths against the project root and preserves absolute paths.
        """
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (self.base_dir / candidate).resolve()

    @property
    def knowledge_dir_path(self) -> Path:
        """
        Absolute path to the knowledge base directory.
        """
        return self.resolve_path(self.KNOWLEDGE_DIR)

    @property
    def vector_store_dir_path(self) -> Path:
        """
        Absolute path to the FAISS persistence directory.
        """
        return self.resolve_path(self.VECTOR_STORE_DIR)

    @property
    def uploads_dir_path(self) -> Path:
        """
        Absolute path to the uploads directory reserved for future use.
        """
        return self.resolve_path(self.UPLOADS_DIR)

    @property
    def resume_file_path(self) -> Path:
        """
        Absolute path to the portfolio resume file served by the API.
        """
        return self.knowledge_dir_path / RESUME_FILE_NAME

    @property
    def vector_store_index_ready(self) -> bool:
        """
        Returns True when the persisted FAISS index files are available on disk.
        """
        return all((self.vector_store_dir_path / name).exists() for name in VECTOR_STORE_INDEX_FILES)


settings = Settings()
