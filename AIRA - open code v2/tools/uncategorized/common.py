"""Common response schemas used across the API."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response wrapper."""

    status: str = "success"
    data: T


class ErrorResponse(BaseModel):
    """Standard error response wrapper."""

    status: str = "error"
    error: str
    data: dict[str, Any] | None = None


class ActionResponse(BaseModel):
    """Response for action endpoints (start, stop, etc.)."""

    status: str
    message: str
    data: dict[str, Any] | None = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    status: str = "success"
    data: list[T]
    total: int
    limit: int
    offset: int


class MessageResponse(BaseModel):
    """Simple message response."""

    status: str = "success"
    message: str
