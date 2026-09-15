"""Abstract base connector for bot data access."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ConnectorResult:
    """Result from a connector operation."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BotStatus:
    """Current bot status from connector."""

    state: str  # running, stopped, etc.
    strategy: Optional[str] = None
    exchange: Optional[str] = None
    trading_mode: Optional[str] = None
    is_dryrun: bool = True
    version: Optional[str] = None


@dataclass
class BotProfit:
    """Profit metrics from connector."""

    profit_closed_coin: float = 0.0
    profit_closed_percent: float = 0.0
    profit_closed_fiat: float = 0.0
    profit_all_coin: float = 0.0
    profit_all_percent: float = 0.0
    profit_all_fiat: float = 0.0
    trade_count: int = 0
    closed_trade_count: int = 0
    first_trade_date: Optional[datetime] = None
    latest_trade_date: Optional[datetime] = None
    winning_trades: int = 0
    losing_trades: int = 0


@dataclass
class BotBalance:
    """Balance information from connector."""

    currency: str
    total: float = 0.0
    free: float = 0.0
    used: float = 0.0
    stake_currency: str = "USDT"
    stake_currency_balance: float = 0.0


@dataclass
class Trade:
    """Trade information from connector."""

    trade_id: int
    pair: str
    is_open: bool
    open_rate: float
    open_date: datetime
    close_rate: Optional[float] = None
    close_date: Optional[datetime] = None
    stake_amount: float = 0.0
    amount: float = 0.0
    profit_abs: Optional[float] = None
    profit_ratio: Optional[float] = None
    stop_loss: Optional[float] = None
    stop_loss_abs: Optional[float] = None
    take_profit: Optional[float] = None
    sell_reason: Optional[str] = None
    min_rate: Optional[float] = None
    max_rate: Optional[float] = None
    enter_tag: Optional[str] = None
    exit_reason: Optional[str] = None
    leverage: float = 1.0
    is_short: bool = False


class BaseConnector(ABC):
    """Abstract base class for bot data connectors.

    Implementations provide data access to Freqtrade bots via different
    mechanisms (REST API, SQLite database).
    """

    def __init__(self, bot_id: str):
        """Initialize connector.

        Args:
            bot_id: UUID of the bot this connector serves.
        """
        self.bot_id = bot_id
        self._available = False
        self._last_check: Optional[datetime] = None
        self._last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        """Check if connector is currently available."""
        return self._available

    @property
    def last_error(self) -> Optional[str]:
        """Get last error message if any."""
        return self._last_error

    @abstractmethod
    async def check_health(self) -> ConnectorResult:
        """Check connector health/availability.

        Returns:
            ConnectorResult with success=True if healthy.
        """
        pass

    @abstractmethod
    async def get_status(self) -> ConnectorResult:
        """Get bot status.

        Returns:
            ConnectorResult with BotStatus data if successful.
        """
        pass

    @abstractmethod
    async def get_profit(self) -> ConnectorResult:
        """Get profit metrics.

        Returns:
            ConnectorResult with BotProfit data if successful.
        """
        pass

    @abstractmethod
    async def get_balance(self) -> ConnectorResult:
        """Get balance information.

        Returns:
            ConnectorResult with BotBalance data if successful.
        """
        pass

    @abstractmethod
    async def get_trades(
        self,
        limit: int = 50,
        offset: int = 0,
        is_open: Optional[bool] = None,
    ) -> ConnectorResult:
        """Get trades list.

        Args:
            limit: Maximum trades to return.
            offset: Pagination offset.
            is_open: Filter by open/closed status.

        Returns:
            ConnectorResult with list[Trade] data if successful.
        """
        pass

    @abstractmethod
    async def get_open_trades(self) -> ConnectorResult:
        """Get currently open trades.

        Returns:
            ConnectorResult with list[Trade] data if successful.
        """
        pass

    async def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass
