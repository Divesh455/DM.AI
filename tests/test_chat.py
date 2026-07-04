from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api.v1.chat import get_chat_service
from app.core.prompts import INSUFFICIENT_CONTEXT_REPLY
from app.main import app
from app.models.schemas.chat import ChatResponse
from app.rag.retriever import PortfolioRetriever
from app.services.chat_service import ChatService


class FakeChatService:
    def answer_question(self, message: str) -> ChatResponse:
        return ChatResponse(answer=f"Divesh is described in the portfolio. Query: {message}")


class EmptyContextChatService:
    def answer_question(self, message: str) -> ChatResponse:
        return ChatResponse(answer=INSUFFICIENT_CONTEXT_REPLY)


class ExplodingEmbeddingFactory:
    def create(self) -> Document:
        raise AssertionError("Embeddings should not be initialized when the vector store is missing.")


class MissingVectorStoreManager:
    def index_exists(self) -> bool:
        return False


def test_chat_endpoint_returns_generated_answer() -> None:
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/api/v1/chat", json={"message": "Who is Divesh?"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()

    assert payload["success"] is True
    assert payload["message"] == "Response generated successfully."
    assert "Divesh is described in the portfolio" in payload["data"]["answer"]


def test_chat_endpoint_returns_consistent_validation_error() -> None:
    client = TestClient(app)

    response = client.post("/api/v1/chat", json={"message": "   "})

    assert response.status_code == 422
    payload = response.json()

    assert payload["success"] is False
    assert payload["message"] == "Request validation failed."


def test_chat_endpoint_returns_503_when_vector_store_is_missing() -> None:
    retriever = PortfolioRetriever(
        embedding_factory=ExplodingEmbeddingFactory(),
        vector_store_manager=MissingVectorStoreManager(),
    )
    chat_service = ChatService(retriever=retriever)
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    client = TestClient(app)

    response = client.post("/api/v1/chat", json={"message": "Who is Divesh?"})

    app.dependency_overrides.clear()

    assert response.status_code == 503
    payload = response.json()

    assert payload["success"] is False
    assert "knowledge base is not ready" in payload["message"].lower()
