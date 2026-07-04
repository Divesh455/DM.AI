"""
Document splitting layer.
Uses RecursiveCharacterTextSplitter to turn long documents into retrieval-friendly chunks.
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.core.config import settings
from app.core.exceptions import IngestionException


class DocumentSplitter:
    """
    Splits parent documents into overlapping chunks for semantic search.
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        """
        Creates the underlying recursive splitter with project defaults.
        """
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        logger.info(
            "Initialized DocumentSplitter with chunk_size={} and chunk_overlap={}.",
            self.chunk_size,
            self.chunk_overlap,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Splits documents and enriches each chunk with a chunk index.
        """
        if not documents:
            logger.warning("DocumentSplitter received an empty document list.")
            return []

        try:
            raw_chunks = self._splitter.split_documents(documents)
        except Exception as exc:
            logger.exception("Document splitting failed: {}", exc)
            raise IngestionException("Failed to split knowledge documents into chunks.") from exc

        chunks: list[Document] = []
        for index, chunk in enumerate(raw_chunks):
            if not chunk.page_content.strip():
                continue

            chunk.metadata = {
                **chunk.metadata,
                "chunk_index": index,
            }
            chunks.append(chunk)

        logger.info(
            "Split {} source documents into {} retrieval chunks.",
            len(documents),
            len(chunks),
        )
        return chunks

