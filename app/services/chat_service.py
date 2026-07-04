"""
Chat service layer for DM.AI.
Coordinates retrieval and LLM generation without placing business logic inside the API route.
"""
from collections.abc import Sequence
from time import perf_counter

from langchain_core.documents import Document
from loguru import logger

from app.core.exceptions import ValidationException
from app.core.prompts import INSUFFICIENT_CONTEXT_REPLY
from app.models.schemas.chat import ChatResponse
from app.rag.retriever import PortfolioRetriever
from app.services.llm_service import LLMService


class ChatService:
    """
    Orchestrates the end-to-end RAG chat workflow for portfolio questions.
    """

    def __init__(
        self,
        retriever: PortfolioRetriever | None = None,
        llm_service: LLMService | None = None,
    ):
        """
        Injects retrieval and generation dependencies so the service can be tested in isolation.
        """
        self._retriever = retriever or PortfolioRetriever()
        self._llm_service = llm_service or LLMService()

    def answer_question(self, message: str) -> ChatResponse:
        """
        Retrieves relevant knowledge chunks and asks the LLM for a grounded answer.
        """
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValidationException("The message field cannot be empty.")

        started_at = perf_counter()
        documents = self._retriever.get_relevant_documents(cleaned_message)

        if not documents:
            logger.info("No retrieval context was found for query '{}'.", cleaned_message)
            return ChatResponse(answer=INSUFFICIENT_CONTEXT_REPLY)

        context = self._build_context(documents)
        sources = [self._format_source_label(document) for document in documents]
        logger.info(
            "Generating chat response for query '{}' using {} context chunks: {}",
            cleaned_message,
            len(documents),
            ", ".join(sources),
        )

        answer = self._llm_service.generate_answer(question=cleaned_message, context=context)
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info("Chat response generated in {}ms.", duration_ms)
        return ChatResponse(answer=answer)

    @staticmethod
    def _build_context(documents: Sequence[Document]) -> str:
        """
        Formats retrieved chunks into a prompt-ready context string with source labels.
        """
        sections: list[str] = []
        for index, document in enumerate(documents, start=1):
            source_label = ChatService._format_source_label(document)
            sections.append(
                f"[Context {index}] {source_label}\n{document.page_content.strip()}"
            )
        return "\n\n".join(section for section in sections if section.strip())

    @staticmethod
    def _format_source_label(document: Document) -> str:
        """
        Builds a human-readable label for each retrieved chunk.
        """
        metadata = document.metadata
        source = metadata.get("source", "unknown")
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")

        parts = [str(source)]
        if page is not None:
            parts.append(f"page {page}")
        if chunk_index is not None:
            parts.append(f"chunk {chunk_index}")
        return " | ".join(parts)
