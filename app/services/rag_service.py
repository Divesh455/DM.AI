"""
Service layer for portfolio knowledge-base ingestion.
Coordinates the lower RAG components while keeping API routes thin.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from loguru import logger

from app.core.config import settings
from app.core.exceptions import IngestionException, NotFoundException
from app.models.schemas.ingest import IngestSummary
from app.rag.loader import DocumentLoader
from app.rag.splitter import DocumentSplitter

if TYPE_CHECKING:
    from app.rag.embeddings import EmbeddingFactory
    from app.rag.vectorstore import VectorStoreManager


class RAGService:
    """
    Orchestrates document loading, chunking, embedding, and FAISS persistence.
    """

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        splitter: DocumentSplitter | None = None,
        embedding_factory: EmbeddingFactory | None = None,
        vector_store_manager: VectorStoreManager | None = None,
        knowledge_dir: str | Path | None = None,
    ):
        """
        Injects all RAG dependencies so the service remains easy to test and extend.
        """
        if embedding_factory is None:
            from app.rag.embeddings import EmbeddingFactory

            embedding_factory = EmbeddingFactory()
        if vector_store_manager is None:
            from app.rag.vectorstore import VectorStoreManager

            vector_store_manager = VectorStoreManager()

        self._loader = loader or DocumentLoader()
        self._splitter = splitter or DocumentSplitter()
        self._embedding_factory = embedding_factory
        self._vector_store_manager = vector_store_manager
        self._knowledge_dir = Path(knowledge_dir or settings.knowledge_dir_path).resolve()

    def ingest_knowledge_base(self, rebuild_index: bool = True) -> IngestSummary:
        """
        Builds or refreshes the FAISS index from the knowledge-base files.
        """
        started_at = perf_counter()

        if not self._knowledge_dir.is_dir():
            raise NotFoundException(f"Knowledge directory not found: {self._knowledge_dir}")

        logger.info(
            "Starting knowledge-base ingestion from {} with rebuild_index={}.",
            self._knowledge_dir,
            rebuild_index,
        )

        documents = self._loader.load_directory(self._knowledge_dir)
        if not documents:
            raise IngestionException(
                "No supported portfolio documents were found in the knowledge directory."
            )

        chunks = self._splitter.split_documents(documents)
        if not chunks:
            raise IngestionException("Portfolio documents were loaded, but no chunks were generated.")

        sources = sorted({document.metadata.get("source", "unknown") for document in documents})
        status = "completed"

        if rebuild_index or not self._vector_store_manager.index_exists():
            embeddings = self._embedding_factory.create()
            vector_store = self._vector_store_manager.build(chunks, embeddings)
            self._vector_store_manager.save(vector_store)
            chunk_count = self._vector_store_manager.document_count(vector_store)
        else:
            logger.info("Skipping FAISS rebuild because an index already exists on disk.")
            status = "skipped"
            chunk_count = len(chunks)

        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "Knowledge ingestion finished with status={} in {}ms.",
            status,
            duration_ms,
        )

        return IngestSummary(
            status=status,
            knowledge_sources=sources,
            document_count=len(documents),
            chunk_count=chunk_count,
            vector_store_path=str(self._vector_store_manager.store_dir),
            chunk_size=self._splitter.chunk_size,
            chunk_overlap=self._splitter.chunk_overlap,
            top_k=settings.RAG_TOP_K,
            duration_ms=duration_ms,
        )
