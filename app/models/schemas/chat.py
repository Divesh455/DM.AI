"""
Schemas for chat request and response payloads.
These models define the public contract for the DM.AI chat endpoint.
"""
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """
    Request body accepted by the chat endpoint.
    """

    message: str = Field(..., min_length=1, description="Visitor question for DM.AI.")

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        """
        Trims whitespace so blank messages are rejected consistently.
        """
        return str(value).strip()


class ChatResponse(BaseModel):
    """
    Response payload returned after the assistant generates an answer.
    """

    answer: str
