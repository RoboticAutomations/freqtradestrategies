"""Bot-related Pydantic schemas."""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, validator

from src.models.bot import BotEnvironment, HealthState, SourceMode, TradingMode


def _ensure_utc_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure datetimes are timezone-aware UTC.

    Many DB drivers / ORM configs return naive datetimes even when the value
    is meant to be UTC. Naive values cause 1-hour offsets in the frontend.
    """
    if dt is None:
        return None
    if isinstance(dt, datetime) and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class BotResponse(BaseModel):
    """Bot response schema."""

    id: str
    name: str
    environment: BotEnvironment
    host: Optional[str] = None
    api_url: Optional[str] = None
    health_state: HealthState
    source_mode: SourceMode
    exchange: Optional[str] = None
    strategy: Optional[str] = None
    trading_mode: Optional[TradingMode] = None
    is_dryrun: bool
    tags: list[str]
    last_seen: Optional[datetime] = None

    @validator("last_seen", pre=True, always=True)
    def _v_last_seen(cls, v):
        return _ensure_utc_aware(v)

    class Config:
        from_attributes = True


class BotListResponse(BaseModel):
    """Bot list response schema."""

    status: str = "success"
    data: list[BotResponse]


class BotDetailResponse(BaseModel):
    """Bot detail response with metrics."""

    status: str = "success"
    data: "BotDetailData"


class BotDetailData(BotResponse):
    """Extended bot data with additional details."""

    container_id: Optional[str] = None
    user_data_path: Optional[str] = None
    discovered_at: datetime
    created_at: datetime

    @validator("discovered_at", "created_at", pre=True, always=True)
    def _v_dt_fields(cls, v):
        return _ensure_utc_aware(v)


class BotUpdateRequest(BaseModel):
    """Request schema for updating bot settings."""

    name: Optional[str] = None
    tags: Optional[list[str]] = None
    source_mode: Optional[SourceMode] = None


class BotCredentialsRequest(BaseModel):
    """Request schema for updating bot API credentials."""

    username: str
    password: str


class BotCredentialsResponse(BaseModel):
    """Response schema for credentials update."""

    status: str = "success"
    message: str
    api_available: bool


class BotMetricsResponse(BaseModel):
    """Bot metrics response schema."""

    status: str = "success"
    data: "BotMetricsData"


class BotMetricsData(BaseModel):
    """Bot metrics data."""

    bot_id: str
    timestamp: datetime
    equity: Optional[float] = None
    profit_abs: Optional[float] = None
    profit_pct: Optional[float] = None
    profit_realized: Optional[float] = None
    profit_unrealized: Optional[float] = None
    open_positions: int = 0
    closed_trades: int = 0
    win_rate: Optional[float] = None
    balance: Optional[float] = None
    drawdown: Optional[float] = None
    data_source: SourceMode
    health_state: HealthState

    class Config:
        from_attributes = True


class BotHealthResponse(BaseModel):
    """Bot health status response."""

    status: str = "success"
    data: "BotHealthData"


class BotHealthData(BaseModel):
    """Bot health status data."""

    bot_id: str
    health_state: HealthState
    source_mode: SourceMode
    active_source: SourceMode
    api_available: bool
    sqlite_available: bool
    api_success_rate: float
    sqlite_success_rate: float
    api_avg_latency_ms: float
    sqlite_avg_latency_ms: float
    last_check: Optional[datetime] = None
    state_changed_at: Optional[datetime] = None
