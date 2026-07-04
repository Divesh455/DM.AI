from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app.api.v1.ingest import get_rag_service
from app.core.config import settings
from app.main import app
from app.rag.embeddings import EmbeddingFactory
from app.rag.vectorstore import VectorStoreManager
from app.services.rag_service import RAGService


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(index + 1), 1.0] for index, text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 1.0]


class FakeEmbeddingFactory(EmbeddingFactory):
    def create(self) -> Embeddings:
        return FakeEmbeddings()


def create_test_service(tmp_path: Path) -> RAGService:
    knowledge_dir = tmp_path / "knowledge"
    vector_store_dir = tmp_path / "vector_store"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    (knowledge_dir / "about.md").write_text(
        "# About\nDivesh Matkar is an AI engineer who builds FastAPI and RAG systems.",
        encoding="utf-8",
    )
    (knowledge_dir / "projects.md").write_text(
        "# Projects\nSkillCart is a portfolio project focused on intelligent skill matching.",
        encoding="utf-8",
    )

    return RAGService(
        embedding_factory=FakeEmbeddingFactory(),
        vector_store_manager=VectorStoreManager(store_dir=vector_store_dir),
        knowledge_dir=knowledge_dir,
    )


def test_ingest_endpoint_builds_vector_store(tmp_path: Path) -> None:
    app.dependency_overrides[get_rag_service] = lambda: create_test_service(tmp_path)
    client = TestClient(app)

    response = client.post("/api/v1/ingest", json={"rebuild_index": True})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()

    assert payload["success"] is True
    assert payload["data"]["status"] == "completed"
    assert payload["data"]["document_count"] == 2
    assert payload["data"]["chunk_count"] >= 2


def test_ingest_endpoint_respects_disabled_configuration(tmp_path: Path) -> None:
    original_value = settings.INGEST_API_ENABLED
    settings.INGEST_API_ENABLED = False
    app.dependency_overrides[get_rag_service] = lambda: create_test_service(tmp_path)
    client = TestClient(app)

    response = client.post("/api/v1/ingest", json={"rebuild_index": True})

    app.dependency_overrides.clear()
    settings.INGEST_API_ENABLED = original_value

    assert response.status_code == 403
    payload = response.json()

    assert payload["success"] is False
    assert "disabled" in payload["message"].lower()
