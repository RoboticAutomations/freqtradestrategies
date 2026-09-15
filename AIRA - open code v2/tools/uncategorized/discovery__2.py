"""Discovery-related Pydantic schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from src.schemas.bot import BotResponse


class DiscoveryStatusResponse(BaseModel):
    """Response schema for discovery status endpoint."""

    status: str = "success"
    data: "DiscoveryStatusData"


class DiscoveryStatusData(BaseModel):
    """Discovery service status data."""

    docker_enabled: bool
    docker_available: bool
    filesystem_enabled: bool
    filesystem_available: bool
    last_scan: Optional[datetime] = None
    scan_interval_seconds: int
    next_scan: Optional[datetime] = None


class DiscoveryTriggerResponse(BaseModel):
    """Response schema for manual discovery trigger."""

    status: str = "success"
    data: "DiscoveryResultData"


class DiscoveryResultData(BaseModel):
    """Discovery scan result data."""

    discovered: int
    new: int
    updated: int
    removed: int
    bots: list[BotResponse] = []


class ManualBotRequest(BaseModel):
    """Request schema for manual bot registration."""

    name: str
    api_url: str
    username: Optional[str] = None
    password: Optional[str] = None


class ManualBotResponse(BaseModel):
    """Response schema for manual bot registration."""

    status: str = "success"
    data: BotResponse
    message: str = "Bot registered successfully"
