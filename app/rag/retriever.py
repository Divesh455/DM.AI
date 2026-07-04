"""
Retriever abstraction for the portfolio knowledge base.
Wraps FAISS retrieval so the chat service can ask for relevant chunks without storage details.
"""
from dataclasses import dataclass

from langchain_core.documents import Document
from loguru import logger

from app.core.config import settings
from app.core.exceptions import ValidationException
from app.rag.embeddings import EmbeddingFactory
from app.rag.vectorstore import VectorStoreManager


@dataclass(slots=True)
class RetrievedDocument:
    """
    Represents a retrieved chunk together with its raw similarity score.
    """

    document: Document
    score: float


class PortfolioRetriever:
    """
    Retrieves the most relevant portfolio chunks for a user question.
    """

    def __init__(
        self,
        embedding_factory: EmbeddingFactory | None = None,
        vector_store_manager: VectorStoreManager | None = None,
        top_k: int | None = None,
    ):
        """
        Accepts RAG dependencies via constructor injection for easier testing.
        """
        self._embedding_factory = embedding_factory or EmbeddingFactory()
        self._vector_store_manager = vector_store_manager or VectorStoreManager()
        self._top_k = top_k or settings.RAG_TOP_K

    def get_relevant_documents(self, query: str) -> list[Document]:
        """
        Executes similarity search against the stored FAISS index.
        """
        return [result.document for result in self.retrieve(query)]

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        """
        Executes similarity search and keeps the raw similarity scores for logging and debugging.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValidationException("The search query cannot be empty.")

        embeddings = self._embedding_factory.create()
        vector_store = self._vector_store_manager.load(embeddings)
        documents_and_scores = vector_store.similarity_search_with_score(
            cleaned_query,
            k=self._top_k,
        )
        results = [
            RetrievedDocument(document=document, score=float(score))
            for document, score in documents_and_scores
        ]

        logger.info(
            "Retriever returned {} documents for query '{}'.",
            len(results),
            cleaned_query,
        )
        for index, result in enumerate(results, start=1):
            logger.debug(
                "Retriever result {} => source={} score={}",
                index,
                result.document.metadata.get("source", "unknown"),
                result.score,
            )
        return results
