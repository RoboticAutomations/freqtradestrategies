"""Pydantic schemas for trade data."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from src.models.bot import SourceMode


class TradeData(BaseModel):
    """Trade information schema."""

    id: int
    pair: str
    is_open: bool
    open_date: datetime
    close_date: Optional[datetime] = None
    open_rate: float
    close_rate: Optional[float] = None
    amount: float
    stake_amount: float
    close_profit: Optional[float] = None
    close_profit_abs: Optional[float] = None
    enter_tag: Optional[str] = None
    exit_reason: Optional[str] = None
    leverage: float = 1.0
    is_short: bool = False
    data_source: SourceMode


class TradeListResponse(BaseModel):
    """Response schema for trade list."""

    status: str = "success"
    data: list[TradeData]
