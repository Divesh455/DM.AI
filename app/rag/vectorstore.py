"""
FAISS persistence layer.
Owns index creation, loading, and saving so higher layers stay storage-agnostic.
"""
from pathlib import Path
from typing import ClassVar

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from loguru import logger

from app.core.config import settings
from app.core.constants import VECTOR_STORE_INDEX_FILES
from app.core.exceptions import IngestionException, VectorStoreNotReadyException


class VectorStoreManager:
    """
    Manages the lifecycle of the local FAISS vector store.
    """

    _vector_store_cache: ClassVar[dict[str, FAISS]] = {}

    def __init__(self, store_dir: str | Path | None = None):
        """
        Configures the location where the FAISS index is persisted.
        """
        self._store_dir = Path(store_dir or settings.vector_store_dir_path).resolve()

    @property
    def store_dir(self) -> Path:
        """
        Returns the absolute path used for FAISS persistence.
        """
        return self._store_dir

    def ensure_store_dir(self) -> None:
        """
        Creates the vector-store directory if it does not already exist.
        """
        self._store_dir.mkdir(parents=True, exist_ok=True)

    def index_exists(self) -> bool:
        """
        Checks whether the persisted FAISS files are available on disk.
        """
        return all((self._store_dir / file_name).exists() for file_name in VECTOR_STORE_INDEX_FILES)

    def build(self, documents: list[Document], embeddings: Embeddings) -> FAISS:
        """
        Creates a new FAISS index from retrieval chunks.
        """
        if not documents:
            raise IngestionException("Cannot create a vector store without document chunks.")

        try:
            logger.info("Creating FAISS index from {} chunks.", len(documents))
            return FAISS.from_documents(documents, embeddings)
        except Exception as exc:
            logger.exception("Failed to build FAISS vector store: {}", exc)
            raise IngestionException("Failed to build the FAISS vector store.") from exc

    def save(self, vector_store: FAISS) -> None:
        """
        Persists the FAISS index locally for future retrieval requests.
        """
        self.ensure_store_dir()

        try:
            vector_store.save_local(str(self._store_dir))
            self._vector_store_cache[str(self._store_dir)] = vector_store
            logger.info("Saved FAISS vector store to {}.", self._store_dir)
        except Exception as exc:
            logger.exception("Failed to save FAISS vector store: {}", exc)
            raise IngestionException("Failed to persist the FAISS vector store.") from exc

    def load(self, embeddings: Embeddings) -> FAISS:
        """
        Loads an existing FAISS index from disk.
        """
        if not self.index_exists():
            raise VectorStoreNotReadyException(
                "The knowledge base has not been ingested yet. Run POST /api/v1/ingest first."
            )

        cache_key = str(self._store_dir)
        cached_vector_store = self._vector_store_cache.get(cache_key)
        if cached_vector_store is not None:
            logger.debug("Using cached FAISS vector store from {}.", self._store_dir)
            return cached_vector_store

        try:
            # LangChain stores FAISS metadata in a local pickle alongside the index.
            vector_store = FAISS.load_local(
                folder_path=str(self._store_dir),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            self._vector_store_cache[cache_key] = vector_store
            return vector_store
        except Exception as exc:
            logger.exception("Failed to load FAISS vector store: {}", exc)
            raise VectorStoreNotReadyException("The stored FAISS index could not be loaded.") from exc

    def document_count(self, vector_store: FAISS) -> int:
        """
        Returns the number of vectors stored inside the FAISS index.
        """
        return int(vector_store.index.ntotal)
