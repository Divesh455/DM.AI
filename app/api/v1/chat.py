"""
Chat API route for portfolio question answering.
This route stays thin and delegates RAG orchestration to the chat service layer.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from app.models.schemas.chat import ChatRequest, ChatResponse
from app.models.schemas.response import BaseAPIResponse

if TYPE_CHECKING:
    from app.services.chat_service import ChatService

router = APIRouter()


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """
    Factory dependency for the chat service.
    """
    from app.services.chat_service import ChatService

    return ChatService()


@router.post("/chat", response_model=BaseAPIResponse[ChatResponse])
def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> BaseAPIResponse[ChatResponse]:
    """
    Answers a portfolio question using retrieval-augmented generation and Gemini.
    """
    result = chat_service.answer_question(payload.message)
    return BaseAPIResponse(
        success=True,
        message="Response generated successfully.",
        data=result,
    )
