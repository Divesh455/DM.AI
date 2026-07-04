"""
Knowledge-base ingestion API route.
Exposes the RAG indexing workflow without leaking business logic into the route layer.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import APIRouter, Body, Depends

from app.core.config import settings
from app.core.exceptions import FeatureDisabledException
from app.models.schemas.ingest import IngestRequest, IngestSummary
from app.models.schemas.response import BaseAPIResponse

if TYPE_CHECKING:
    from app.services.rag_service import RAGService

router = APIRouter()


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """
    Factory dependency for the ingestion service.
    """
    from app.services.rag_service import RAGService

    return RAGService()


@router.post("/ingest", response_model=BaseAPIResponse[IngestSummary])
def ingest_knowledge_base(
    payload: IngestRequest = Body(default_factory=IngestRequest),
    rag_service: RAGService = Depends(get_rag_service),
) -> BaseAPIResponse[IngestSummary]:
    """
    Rebuilds the local FAISS index from the portfolio knowledge directory.
    """
    if not settings.INGEST_API_ENABLED:
        raise FeatureDisabledException(
            "The ingest endpoint is disabled in this environment. Build the vector store locally with "
            "`python scripts/build_vector_store.py` and deploy the generated FAISS files."
        )

    result = rag_service.ingest_knowledge_base(rebuild_index=payload.rebuild_index)
    message = (
        "Knowledge base ingested successfully."
        if result.status == "completed"
        else "Existing vector store kept; ingestion was skipped."
    )
    return BaseAPIResponse(success=True, message=message, data=result)
