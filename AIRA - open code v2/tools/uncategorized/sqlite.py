"""SQLite connector for read-only access to Freqtrade database."""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite
import structlog

from src.services.connectors.base import (
    BaseConnector,
    BotBalance,
    BotProfit,
    BotStatus,
    ConnectorResult,
    Trade,
)

logger = structlog.get_logger()


class SQLiteConnector(BaseConnector):
    """Connector for read-only SQLite database access.

    Provides data access when API is unavailable by reading
    the bot's tradesv3.sqlite database directly.
    """

    def __init__(self, bot_id: str, db_path: str):
        """Initialize SQLite connector.

        Args:
            bot_id: UUID of the bot.
            db_path: Path to the tradesv3.sqlite database file.
        """
        super().__init__(bot_id)
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None

    async def _get_connection(self) -> Optional[aiosqlite.Connection]:
        """Get or create database connection.

        Opens in read-only mode.

        NOTE: Do NOT use immutable=1 because Freqtrade keeps writing to this DB.
        Using immutable would cause stale reads after the file updates.
        """
        if self._connection is not None:
            return self._connection

        if not os.path.exists(self.db_path):
            self._available = False
            self._last_error = "Database file not found"
            return None

        try:
            # Open in read-only mode (allow seeing updates written by Freqtrade)
            uri = f"file:{self.db_path}?mode=ro"
            self._connection = await aiosqlite.connect(uri, uri=True)
            self._connection.row_factory = aiosqlite.Row
            self._available = True
            return self._connection

        except Exception as e:
            self._available = False
            self._last_error = str(e)
            logger.error(
                "SQLite connection error",
                bot_id=self.bot_id,
                db_path=self.db_path,
                error=str(e),
            )
            return None

    async def _execute(
        self,
        query: str,
        params: tuple = (),
    ) -> ConnectorResult:
        """Execute query with timing and error handling.

        Args:
            query: SQL query to execute.
            params: Query parameters.

        Returns:
            ConnectorResult with rows or error.
        """
        start = time.perf_counter()

        try:
            conn = await self._get_connection()
            if conn is None:
                return ConnectorResult(
                    success=False,
                    error=self._last_error or "No database connection",
                )

            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            latency = (time.perf_counter() - start) * 1000

            self._available = True
            self._last_error = None

            return ConnectorResult(
                success=True,
                data=[dict(row) for row in rows],
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self._available = False
            self._last_error = str(e)
            logger.error(
                "SQLite query error",
                bot_id=self.bot_id,
                query=query[:100],
                error=str(e),
            )
            return ConnectorResult(
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    async def check_health(self) -> ConnectorResult:
        """Check database health by verifying file exists and is readable."""
        start = time.perf_counter()

        if not os.path.exists(self.db_path):
            self._available = False
            self._last_error = "Database file not found"
            return ConnectorResult(
                success=False,
                error="Database file not found",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Try to query version table
        result = await self._execute(
            "SELECT * FROM sqlite_master WHERE type='table' LIMIT 1"
        )

        if result.success:
            self._last_check = datetime.utcnow()

        return result

    async def get_status(self) -> ConnectorResult:
        """Get bot status from pairlocks/trades table.

        SQLite doesn't have full config, so we infer what we can.
        """
        # Try to get strategy from most recent trade
        result = await self._execute(
            """
            SELECT strategy, stake_currency, is_short
            FROM trades
            ORDER BY open_date DESC
            LIMIT 1
            """
        )

        if not result.success:
            return result

        strategy = None
        trading_mode = "spot"
        stake_currency = "USDT"

        if result.data:
            row = result.data[0]
            strategy = row.get("strategy")
            stake_currency = row.get("stake_currency", "USDT")
            if row.get("is_short"):
                trading_mode = "futures"

        return ConnectorResult(
            success=True,
            data=BotStatus(
                state="unknown",  # Can't determine from SQLite
                strategy=strategy,
                exchange=None,  # Not stored in trades
                trading_mode=trading_mode,
                is_dryrun=True,  # Assume dryrun for safety
                version=None,
            ),
            latency_ms=result.latency_ms,
        )

    async def get_profit(self) -> ConnectorResult:
        """Calculate profit metrics from trades table."""
        # Get profit data and starting capital estimate
        result = await self._execute(
            """
            SELECT
                SUM(CASE WHEN is_open = 0 THEN close_profit_abs ELSE 0 END) as profit_closed,
                SUM(close_profit_abs) as profit_all,
                COUNT(*) as trade_count,
                SUM(CASE WHEN is_open = 0 THEN 1 ELSE 0 END) as closed_count,
                SUM(CASE WHEN is_open = 0 AND close_profit_abs > 0 THEN 1 ELSE 0 END) as winning,
                SUM(CASE WHEN is_open = 0 AND close_profit_abs <= 0 THEN 1 ELSE 0 END) as losing,
                MIN(open_date) as first_trade,
                MAX(COALESCE(close_date, open_date)) as latest_trade
            FROM trades
            """
        )

        if not result.success:
            return result

        data = result.data[0] if result.data else {}

        first_trade = None
        latest_trade = None

        if data.get("first_trade"):
            try:
                first_trade = datetime.fromisoformat(data["first_trade"])
            except (ValueError, TypeError):
                pass

        if data.get("latest_trade"):
            try:
                latest_trade = datetime.fromisoformat(data["latest_trade"])
            except (ValueError, TypeError):
                pass

        # Get starting capital estimate from first 5 trades (to average out any variations)
        starting_capital = 0.0
        if data.get("trade_count", 0) > 0:
            capital_result = await self._execute(
                """
                SELECT AVG(stake_amount) as avg_stake
                FROM (
                    SELECT stake_amount
                    FROM trades
                    ORDER BY open_date ASC
                    LIMIT 5
                )
                """
            )
            if capital_result.success and capital_result.data:
                starting_capital = capital_result.data[0].get("avg_stake") or 0.0

        # Calculate profit percentages based on starting capital
        profit_closed = data.get("profit_closed") or 0.0
        profit_all = data.get("profit_all") or 0.0

        # Calculate percentages (avoid division by zero)
        profit_closed_percent = 0.0
        profit_all_percent = 0.0
        if starting_capital > 0:
            profit_closed_percent = (profit_closed / starting_capital) * 100
            profit_all_percent = (profit_all / starting_capital) * 100

        return ConnectorResult(
            success=True,
            data=BotProfit(
                profit_closed_coin=profit_closed,
                profit_closed_percent=profit_closed_percent,
                profit_closed_fiat=0.0,  # No fiat conversion in SQLite
                profit_all_coin=profit_all,
                profit_all_percent=profit_all_percent,
                profit_all_fiat=0.0,
                trade_count=data.get("trade_count") or 0,
                closed_trade_count=data.get("closed_count") or 0,
                first_trade_date=first_trade,
                latest_trade_date=latest_trade,
                winning_trades=data.get("winning") or 0,
                losing_trades=data.get("losing") or 0,
            ),
            latency_ms=result.latency_ms,
        )

    async def get_balance(self) -> ConnectorResult:
        """Get balance estimate from trades.

        SQLite doesn't track actual balance, so we estimate
        from stake amounts in open trades.
        """
        result = await self._execute(
            """
            SELECT
                stake_currency,
                SUM(CASE WHEN is_open = 1 THEN stake_amount ELSE 0 END) as used,
                SUM(close_profit_abs) as total_profit
            FROM trades
            GROUP BY stake_currency
            LIMIT 1
            """
        )

        if not result.success:
            return result

        data = result.data[0] if result.data else {}
        stake_currency = data.get("stake_currency", "USDT")

        return ConnectorResult(
            success=True,
            data=BotBalance(
                currency=stake_currency,
                total=0.0,  # Unknown from SQLite
                free=0.0,  # Unknown
                used=data.get("used") or 0.0,
                stake_currency=stake_currency,
                stake_currency_balance=0.0,
            ),
            latency_ms=result.latency_ms,
        )

    async def get_trades(
        self,
        limit: int = 50,
        offset: int = 0,
        is_open: Optional[bool] = None,
    ) -> ConnectorResult:
        """Get trades from database."""
        where_clause = ""
        params: list[Any] = []

        if is_open is not None:
            where_clause = "WHERE is_open = ?"
            params.append(1 if is_open else 0)

        result = await self._execute(
            f"""
            SELECT *
            FROM trades
            {where_clause}
            ORDER BY open_date DESC
            LIMIT ? OFFSET ?
            """,
            tuple(params + [limit, offset]),
        )

        if not result.success:
            return result

        trades = [self._parse_trade(row) for row in result.data]

        return ConnectorResult(
            success=True,
            data=trades,
            latency_ms=result.latency_ms,
        )

    async def get_open_trades(self) -> ConnectorResult:
        """Get currently open trades."""
        return await self.get_trades(limit=100, is_open=True)

    async def get_closed_trades(self, limit: int = 50) -> ConnectorResult:
        """Get closed/historical trades."""
        return await self.get_trades(limit=limit, is_open=False)

    def _parse_trade(self, data: dict) -> Trade:
        """Parse trade data from database row."""
        open_date = datetime.utcnow()
        if data.get("open_date"):
            try:
                open_date = datetime.fromisoformat(str(data["open_date"]))
            except (ValueError, TypeError):
                pass

        close_date = None
        if data.get("close_date"):
            try:
                close_date = datetime.fromisoformat(str(data["close_date"]))
            except (ValueError, TypeError):
                pass

        return Trade(
            trade_id=data.get("id", 0),
            pair=data.get("pair", ""),
            is_open=bool(data.get("is_open", True)),
            open_rate=float(data.get("open_rate") or 0.0),
            open_date=open_date,
            close_rate=float(data["close_rate"]) if data.get("close_rate") else None,
            close_date=close_date,
            stake_amount=float(data.get("stake_amount") or 0.0),
            amount=float(data.get("amount") or 0.0),
            profit_abs=float(data["close_profit_abs"]) if data.get("close_profit_abs") else None,
            profit_ratio=float(data["close_profit"]) if data.get("close_profit") else None,
            stop_loss=float(data["stop_loss"]) if data.get("stop_loss") else None,
            stop_loss_abs=float(data["stop_loss_abs"]) if data.get("stop_loss_abs") else None,
            take_profit=float(data["take_profit"]) if data.get("take_profit") else None,
            sell_reason=data.get("exit_reason") or data.get("sell_reason"),
            min_rate=float(data["min_rate"]) if data.get("min_rate") else None,
            max_rate=float(data["max_rate"]) if data.get("max_rate") else None,
        )

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
