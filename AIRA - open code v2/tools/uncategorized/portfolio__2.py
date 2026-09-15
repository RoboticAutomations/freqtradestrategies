"""Portfolio-related Pydantic schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class PortfolioSummaryResponse(BaseModel):
    """Response schema for portfolio summary."""

    status: str = "success"
    data: "PortfolioSummaryData"


class PortfolioSummaryData(BaseModel):
    """Portfolio summary data."""

    timestamp: datetime
    # Bot counts (all registered bots)
    total_bots: int
    healthy_bots: int
    degraded_bots: int = 0  # Deprecated - kept for compatibility
    unreachable_bots: int
    # Excluded bots (not counted in portfolio totals)
    hyperopt_bots: int = 0
    backtest_bots: int = 0
    # Portfolio totals (only healthy trading bots contribute)
    portfolio_bots: int = 0  # Number of bots in portfolio totals
    total_profit_abs: float
    total_profit_pct: float
    total_balance: float
    total_open_positions: int
    total_closed_trades: int
    avg_win_rate: Optional[float] = None
    best_performer: Optional[str] = None
    worst_performer: Optional[str] = None


class ExchangeBreakdownResponse(BaseModel):
    """Response schema for exchange breakdown."""

    status: str = "success"
    data: list["ExchangeBreakdownData"]


class ExchangeBreakdownData(BaseModel):
    """Exchange breakdown data."""

    exchange: str
    bot_count: int
    profit_abs: float
    profit_pct: float
    balance: float
    open_positions: int


class StrategyBreakdownResponse(BaseModel):
    """Response schema for strategy breakdown."""

    status: str = "success"
    data: list["StrategyBreakdownData"]


class StrategyBreakdownData(BaseModel):
    """Strategy breakdown data."""

    strategy: str
    bot_count: int
    profit_abs: float
    profit_pct: float
    open_positions: int
    closed_trades: int
    win_rate: Optional[float] = None
