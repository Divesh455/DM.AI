"""
Embedding factory for the RAG layer.
Creates the sentence-transformer embedding model used by FAISS and retrieval.
"""
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

from app.core.config import settings
from app.core.exceptions import IngestionException


class EmbeddingFactory:
    """
    Creates embedding clients for the configured sentence-transformer model.
    """

    def __init__(self, model_name: str | None = None):
        """
        Stores the model name so service-level code does not know vendor details.
        """
        self._model_name = model_name or settings.EMBEDDING_MODEL

    def create(self) -> Embeddings:
        """
        Returns a LangChain-compatible embeddings object.
        """
        logger.info("Initializing embedding model: {}", self._model_name)
        return self._create_cached_embeddings(self._model_name)

    @staticmethod
    @lru_cache(maxsize=4)
    def _create_cached_embeddings(model_name: str) -> Embeddings:
        """
        Caches the embedding model in memory so it is not rebuilt on every request.
        """
        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            logger.exception("Failed to initialize embedding model {}: {}", model_name, exc)
            raise IngestionException(
                f"Failed to initialize embedding model '{model_name}'. "
                "Ensure the model is cached locally or that outbound access to Hugging Face is available."
            ) from exc
