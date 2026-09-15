"""Alert schemas."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.models.alert import AlertSeverity, AlertType


class AlertBase(BaseModel):
    """Base alert schema."""

    alert_type: AlertType
    severity: AlertSeverity = AlertSeverity.INFO
    title: str = Field(..., max_length=200)
    message: str


class AlertCreate(AlertBase):
    """Schema for creating alerts."""

    bot_id: Optional[str] = None
    bot_name: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class AlertData(AlertBase):
    """Alert response data."""

    id: str
    bot_id: Optional[str]
    bot_name: Optional[str]
    is_read: bool
    is_dismissed: bool
    created_at: datetime
    metadata: Optional[dict[str, Any]] = None

    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    """Single alert response."""

    status: str = "success"
    data: AlertData


class AlertListResponse(BaseModel):
    """List of alerts response."""

    status: str = "success"
    data: list[AlertData]
    total: int
    unread_count: int


class AlertCountResponse(BaseModel):
    """Alert count response."""

    status: str = "success"
    data: dict[str, int]


class AlertMarkReadRequest(BaseModel):
    """Request to mark alerts as read."""

    alert_ids: list[str]


class AlertDismissRequest(BaseModel):
    """Request to dismiss alerts."""

    alert_ids: list[str]
