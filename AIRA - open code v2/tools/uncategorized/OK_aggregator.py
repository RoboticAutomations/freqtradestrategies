"""Aggregator service for portfolio-wide calculations."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.bot import Bot, HealthState
from src.models.metrics import BotMetrics
from src.services.connectors.manager import connector_manager
from src.db.analytics import analytics_session_maker
from sqlalchemy import select, desc, text

logger = structlog.get_logger()


class BotType(str, Enum):
    """Bot operational type."""

    TRADING = "trading"  # Active trading bot (dryrun or live)
    HYPEROPT = "hyperopt"  # Hyperopt optimization process
    BACKTEST = "backtest"  # Backtest process
    DOWNLOAD = "download"  # Data download process
    UNKNOWN = "unknown"


def detect_bot_type(bot: Bot) -> BotType:
    """Detect bot type based on name/container patterns.

    Args:
        bot: Bot instance to check.

    Returns:
        Detected BotType.
    """
    # Check bot name and container_id for type indicators
    name_lower = (bot.name or "").lower()
    container_lower = (bot.container_id or "").lower()

    # Check for hyperopt
    if "hyperopt" in name_lower or "hyperopt" in container_lower:
        return BotType.HYPEROPT

    # Check for backtest
    if "backtest" in name_lower or "backtest" in container_lower:
        return BotType.BACKTEST

    # Check for data download
    if "download" in name_lower or "download" in container_lower:
        return BotType.DOWNLOAD

    # Default to trading bot
    return BotType.TRADING


def is_portfolio_bot(bot: Bot) -> bool:
    """Check if bot should be included in portfolio calculations.

    Only healthy trading bots are included in portfolio totals.

    Args:
        bot: Bot instance to check.

    Returns:
        True if bot should be included in portfolio calculations.
    """
    # Must be a trading bot (not hyperopt/backtest/download)
    bot_type = detect_bot_type(bot)
    if bot_type != BotType.TRADING:
        return False

    # Must be healthy to be included in portfolio totals
    if bot.health_state != HealthState.HEALTHY:
        return False

    return True


@dataclass
class BotMetricsSummary:
    """Metrics summary for a single bot."""

    bot_id: str
    bot_name: str
    exchange: Optional[str]
    strategy: Optional[str]
    health_state: HealthState
    profit_abs: float = 0.0
    profit_pct: float = 0.0
    open_positions: int = 0
    closed_trades: int = 0
    win_rate: Optional[float] = None
    balance: float = 0.0
    is_available: bool = True


@dataclass
class PortfolioSummary:
    """Aggregated portfolio summary."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    # Bot counts (all registered bots)
    total_bots: int = 0
    healthy_bots: int = 0
    degraded_bots: int = 0
    unreachable_bots: int = 0
    # Excluded bot counts (not included in portfolio totals)
    hyperopt_bots: int = 0
    backtest_bots: int = 0
    # Portfolio totals (only healthy trading bots)
    portfolio_bots: int = 0  # Number of bots contributing to portfolio totals
    total_profit_abs: float = 0.0
    total_profit_pct: float = 0.0
    total_balance: float = 0.0
    total_open_positions: int = 0
    total_closed_trades: int = 0
    avg_win_rate: Optional[float] = None
    best_performer: Optional[str] = None
    worst_performer: Optional[str] = None


@dataclass
class ExchangeBreakdown:
    """Breakdown by exchange."""

    exchange: str
    bot_count: int = 0
    profit_abs: float = 0.0
    profit_pct: float = 0.0
    balance: float = 0.0
    open_positions: int = 0


@dataclass
class StrategyBreakdown:
    """Breakdown by strategy."""

    strategy: str
    bot_count: int = 0
    profit_abs: float = 0.0
    profit_pct: float = 0.0
    open_positions: int = 0
    closed_trades: int = 0
    win_rate: Optional[float] = None


class AggregatorService:
    """Service for aggregating metrics across multiple bots."""

    async def get_latest_cached_metrics(
        self,
        db: AsyncSession,
        bot_id: str,
    ) -> BotMetrics | None:
        """Get latest cached metrics for a bot from database.

        First tries local bot_metrics table, then falls back to
        analytics database bot_snapshots table.

        Args:
            db: Database session.
            bot_id: Bot ID to get metrics for.

        Returns:
            Latest BotMetrics or None if no cached data.
        """
        # First try local bot_metrics table
        result = await db.execute(
            select(BotMetrics)
            .where(BotMetrics.bot_id == bot_id)
            .order_by(desc(BotMetrics.timestamp))
            .limit(1)
        )
        local_metrics = result.scalar_one_or_none()
        if local_metrics:
            return local_metrics

        # Fallback: try analytics database bot_snapshots
        try:
            async with analytics_session_maker() as analytics_db:
                # Get bot name to match with snapshots
                bot_result = await db.execute(select(Bot).where(Bot.id == bot_id))
                bot = bot_result.scalar_one_or_none()
                if not bot or not bot.name:
                    logger.debug("No bot name found for metrics lookup", bot_id=bot_id)
                    return None

                # Try exact match first
                row = await analytics_db.execute(
                    text("""
                        SELECT profit_all, balance, open_trades, trade_count, winrate, timestamp
                        FROM bot_snapshots
                        WHERE bot_name = :bot_name
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """),
                    {"bot_name": bot.name}
                )
                snapshot = row.fetchone()
                
                # If no exact match, try partial match (bot name contains the snapshot name or vice versa)
                if not snapshot:
                    row = await analytics_db.execute(
                        text("""
                            SELECT profit_all, balance, open_trades, trade_count, winrate, timestamp, bot_name
                            FROM bot_snapshots
                            WHERE bot_name ILIKE :pattern
                            ORDER BY timestamp DESC
                            LIMIT 1
                        """),
                        {"pattern": f"%{bot.name}%"}
                    )
                    snapshot = row.fetchone()
                
                # Try reverse - snapshot name contains bot name
                if not snapshot:
                    row = await analytics_db.execute(
                        text("""
                            SELECT profit_all, balance, open_trades, trade_count, winrate, timestamp, bot_name
                            FROM bot_snapshots
                            WHERE :bot_name ILIKE '%' || bot_name || '%'
                            ORDER BY timestamp DESC
                            LIMIT 1
                        """),
                        {"bot_name": bot.name}
                    )
                    snapshot = row.fetchone()
                
                if snapshot:
                    logger.info("Found analytics snapshot", bot_id=bot_id, bot_name=bot.name)
                    # Convert snapshot to BotMetrics format
                    return BotMetrics(
                        bot_id=bot_id,
                        timestamp=snapshot.timestamp,
                        profit_abs=float(snapshot.profit_all) if snapshot.profit_all else 0.0,
                        profit_pct=0.0,  # Not available in snapshots
                        balance=float(snapshot.balance) if snapshot.balance else 0.0,
                        open_positions=int(snapshot.open_trades) if snapshot.open_trades else 0,
                        closed_trades=int(snapshot.trade_count) if snapshot.trade_count else 0,
                        win_rate=float(snapshot.winrate) / 100 if snapshot.winrate else None,
                        data_source="api",  # Source doesn't matter for display
                    )
                else:
                    logger.debug("No analytics snapshot found", bot_id=bot_id, bot_name=bot.name)
        except Exception as e:
            logger.warning("Failed to fetch from analytics DB", bot_id=bot_id, error=str(e))

        return None

    async def get_portfolio_summary_from_cache(
        self,
        db: AsyncSession,
    ) -> PortfolioSummary:
        """Get portfolio summary from cached database metrics.

        Uses bot_metrics table instead of live bot connections.
        This provides persistent summary even when bots are offline.

        Args:
            db: Database session.

        Returns:
            Aggregated portfolio metrics from cached data.
        """
        result = await db.execute(select(Bot))
        bots = result.scalars().all()

        summary = PortfolioSummary(
            total_bots=len(bots),
        )

        if not bots:
            return summary

        bot_metrics: list[BotMetricsSummary] = []
        win_rates: list[float] = []

        for bot in bots:
            bot_type = detect_bot_type(bot)

            # Count by bot type
            if bot_type == BotType.HYPEROPT:
                summary.hyperopt_bots += 1
                continue
            elif bot_type == BotType.BACKTEST:
                summary.backtest_bots += 1
                continue
            elif bot_type == BotType.DOWNLOAD:
                continue

            # Count by health state - simplified: only healthy or unreachable
            if bot.health_state == HealthState.HEALTHY:
                summary.healthy_bots += 1
            else:
                summary.unreachable_bots += 1

            # Try to get cached metrics from database
            logger.debug("Looking for cached metrics", bot_id=bot.id, bot_name=bot.name)
            cached = await self.get_latest_cached_metrics(db, bot.id)
            
            if cached:
                logger.info("Found cached metrics for bot", bot_id=bot.id, bot_name=bot.name, profit=cached.profit_abs)
                summary.portfolio_bots += 1
            else:
                logger.debug("No cached metrics found for bot", bot_id=bot.id, bot_name=bot.name)

            metrics = BotMetricsSummary(
                bot_id=bot.id,
                bot_name=bot.name,
                exchange=bot.exchange,
                strategy=bot.strategy,
                health_state=bot.health_state,
                is_available=cached is not None,
            )

            if cached:
                metrics.profit_abs = cached.profit_abs or 0.0
                metrics.profit_pct = cached.profit_pct or 0.0
                metrics.balance = cached.balance or 0.0
                metrics.open_positions = cached.open_positions or 0
                metrics.closed_trades = cached.closed_trades or 0
                metrics.win_rate = cached.win_rate

                if metrics.win_rate is not None:
                    win_rates.append(metrics.win_rate)

                bot_metrics.append(metrics)

                # Aggregate totals
                summary.total_profit_abs += metrics.profit_abs
                summary.total_balance += metrics.balance
                summary.total_open_positions += metrics.open_positions
                summary.total_closed_trades += metrics.closed_trades
            else:
                # No cached data - bot hasn't reported metrics yet
                bot_metrics.append(metrics)

        # Calculate averages
        if win_rates:
            summary.avg_win_rate = sum(win_rates) / len(win_rates)

        if bot_metrics:
            available_metrics = [m for m in bot_metrics if m.is_available and m.balance > 0]
            if available_metrics:
                best = max(available_metrics, key=lambda m: m.profit_abs)
                worst = min(available_metrics, key=lambda m: m.profit_abs)
                summary.best_performer = best.bot_name
                summary.worst_performer = worst.bot_name

                if summary.total_balance > 0:
                    summary.total_profit_pct = (
                        summary.total_profit_abs / summary.total_balance * 100
                    )

        logger.info(
            "Portfolio summary from cache",
            total_bots=summary.total_bots,
            portfolio_bots=summary.portfolio_bots,
            cached_count=len([m for m in bot_metrics if m.is_available]),
        )

        return summary

    async def get_portfolio_summary(
        self,
        db: AsyncSession,
        use_cache: bool = True,
    ) -> PortfolioSummary:
        """Get aggregated portfolio summary.

        Args:
            db: Database session.
            use_cache: If True, use cached metrics from database.
                      If False, try to fetch live from bots.

        Returns:
            Aggregated portfolio metrics.
        """
        # Use cached version by default for persistent summary
        if use_cache:
            return await self.get_portfolio_summary_from_cache(db)

        # Legacy live bot connection method
        return await self._get_portfolio_summary_live(db)

    async def _get_portfolio_summary_live(
        self,
        db: AsyncSession,
    ) -> PortfolioSummary:
        """Get portfolio summary from live bot connections (legacy method).

        Args:
            db: Database session.

        Returns:
            Aggregated portfolio metrics from live bots.
        """
        result = await db.execute(select(Bot))
        bots = result.scalars().all()

        summary = PortfolioSummary(
            total_bots=len(bots),
        )

        if not bots:
            return summary

        # Collect metrics from each bot
        bot_metrics: list[BotMetricsSummary] = []
        win_rates: list[float] = []

        for bot in bots:
            # Detect bot type
            bot_type = detect_bot_type(bot)

            # Count by bot type (excluded from portfolio)
            if bot_type == BotType.HYPEROPT:
                summary.hyperopt_bots += 1
                continue
            elif bot_type == BotType.BACKTEST:
                summary.backtest_bots += 1
                continue
            elif bot_type == BotType.DOWNLOAD:
                continue

            # Count by health state - simplified: only healthy or unreachable
            if bot.health_state == HealthState.HEALTHY:
                summary.healthy_bots += 1
            else:
                summary.unreachable_bots += 1
                continue  # Skip unreachable bots

            # This is a healthy trading bot - include in portfolio
            summary.portfolio_bots += 1

            # Try to get metrics
            try:
                connector, _ = await connector_manager.get_connector(bot)

                profit_result = await connector.get_profit()
                balance_result = await connector.get_balance()
                open_trades_result = await connector.get_open_trades()

                metrics = BotMetricsSummary(
                    bot_id=bot.id,
                    bot_name=bot.name,
                    exchange=bot.exchange,
                    strategy=bot.strategy,
                    health_state=bot.health_state,
                )

                if profit_result.success and profit_result.data:
                    profit = profit_result.data
                    metrics.profit_abs = profit.profit_all_coin
                    metrics.profit_pct = profit.profit_all_percent
                    metrics.closed_trades = profit.closed_trade_count

                    total_closed = profit.winning_trades + profit.losing_trades
                    if total_closed > 0:
                        metrics.win_rate = profit.winning_trades / total_closed
                        win_rates.append(metrics.win_rate)

                if balance_result.success and balance_result.data:
                    metrics.balance = balance_result.data.stake_currency_balance

                if open_trades_result.success and open_trades_result.data:
                    metrics.open_positions = len(open_trades_result.data)

                bot_metrics.append(metrics)

                # Aggregate totals (only healthy trading bots)
                summary.total_profit_abs += metrics.profit_abs
                summary.total_balance += metrics.balance
                summary.total_open_positions += metrics.open_positions
                summary.total_closed_trades += metrics.closed_trades

            except Exception as e:
                logger.warning(
                    "Failed to get metrics for bot",
                    bot_id=bot.id,
                    error=str(e),
                )
                bot_metrics.append(
                    BotMetricsSummary(
                        bot_id=bot.id,
                        bot_name=bot.name,
                        exchange=bot.exchange,
                        strategy=bot.strategy,
                        health_state=bot.health_state,
                        is_available=False,
                    )
                )

        # Calculate averages and find best/worst performers
        if win_rates:
            summary.avg_win_rate = sum(win_rates) / len(win_rates)

        if bot_metrics:
            available_metrics = [m for m in bot_metrics if m.is_available]
            if available_metrics:
                best = max(available_metrics, key=lambda m: m.profit_abs)
                worst = min(available_metrics, key=lambda m: m.profit_abs)
                summary.best_performer = best.bot_name
                summary.worst_performer = worst.bot_name

                # Calculate weighted average profit percentage
                if summary.total_balance > 0:
                    summary.total_profit_pct = (
                        summary.total_profit_abs / summary.total_balance * 100
                    )

        return summary

    async def get_exchange_breakdown(
        self,
        db: AsyncSession,
    ) -> list[ExchangeBreakdown]:
        """Get metrics breakdown by exchange using cached data.

        Uses bot_metrics table for persistence even when bots are offline.

        Args:
            db: Database session.

        Returns:
            List of exchange-wise metric breakdowns.
        """
        result = await db.execute(select(Bot))
        bots = result.scalars().all()

        exchange_data: dict[str, ExchangeBreakdown] = {}

        for bot in bots:
            # Skip non-trading bots (hyperopt, backtest, download)
            if not is_portfolio_bot(bot):
                continue

            # Skip unreachable bots
            if bot.health_state != HealthState.HEALTHY:
                continue

            exchange = bot.exchange or "Unknown"

            if exchange not in exchange_data:
                exchange_data[exchange] = ExchangeBreakdown(exchange=exchange)

            breakdown = exchange_data[exchange]
            breakdown.bot_count += 1

            # Use cached metrics instead of live connection
            cached = await self.get_latest_cached_metrics(db, bot.id)
            if cached:
                breakdown.profit_abs += cached.profit_abs or 0
                breakdown.balance += cached.balance or 0
                breakdown.open_positions += cached.open_positions or 0

        # Calculate profit percentages
        for breakdown in exchange_data.values():
            if breakdown.balance > 0:
                breakdown.profit_pct = breakdown.profit_abs / breakdown.balance * 100

        return list(exchange_data.values())

    async def get_strategy_breakdown(
        self,
        db: AsyncSession,
    ) -> list[StrategyBreakdown]:
        """Get metrics breakdown by strategy using cached data.

        Uses bot_metrics table for persistence even when bots are offline.

        Args:
            db: Database session.

        Returns:
            List of strategy-wise metric breakdowns.
        """
        result = await db.execute(select(Bot))
        bots = result.scalars().all()

        strategy_data: dict[str, StrategyBreakdown] = {}
        strategy_win_rates: dict[str, list[float]] = {}

        for bot in bots:
            # Skip non-trading bots (hyperopt, backtest, download)
            if not is_portfolio_bot(bot):
                continue

            # Skip unreachable bots
            if bot.health_state != HealthState.HEALTHY:
                continue

            strategy = bot.strategy or "Unknown"

            if strategy not in strategy_data:
                strategy_data[strategy] = StrategyBreakdown(strategy=strategy)
                strategy_win_rates[strategy] = []

            breakdown = strategy_data[strategy]
            breakdown.bot_count += 1

            # Use cached metrics instead of live connection
            cached = await self.get_latest_cached_metrics(db, bot.id)
            if cached:
                breakdown.profit_abs += cached.profit_abs or 0
                breakdown.closed_trades += cached.closed_trades or 0
                breakdown.open_positions += cached.open_positions or 0
                if cached.win_rate is not None:
                    strategy_win_rates[strategy].append(cached.win_rate)

        # Calculate average win rates per strategy
        for strategy, breakdown in strategy_data.items():
            rates = strategy_win_rates.get(strategy, [])
            if rates:
                breakdown.win_rate = sum(rates) / len(rates)

        return list(strategy_data.values())


# Singleton instance
aggregator_service = AggregatorService()
