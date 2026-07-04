"""
Unified API Response Schema.
Enforces a consistent payload contract for all DM.AI API outputs,
improving client consumption on the Next.js frontend.
"""
from typing import Generic, Optional, TypeVar
from pydantic import BaseModel

# Generic type variable for the payload data field
T = TypeVar("T")


class BaseAPIResponse(BaseModel, Generic[T]):
    """
    Base wrapper for all API responses.
    Ensures consistent success flags, message notices, and nested response data.
    """
    success: bool
    message: str
    data: Optional[T] = None
