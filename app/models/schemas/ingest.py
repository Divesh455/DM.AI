"""
Schemas for knowledge-base ingestion requests and responses.
These models keep the API contract stable while the internal RAG pipeline evolves.
"""
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """
    Request body for the ingest endpoint.
    """
    rebuild_index: bool = Field(
        default=True,
        description="Whether the FAISS index should be regenerated from the current knowledge files.",
    )


class IngestSummary(BaseModel):
    """
    Result metadata returned after an ingestion run.
    """
    status: str
    knowledge_sources: list[str]
    document_count: int
    chunk_count: int
    vector_store_path: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    duration_ms: float
