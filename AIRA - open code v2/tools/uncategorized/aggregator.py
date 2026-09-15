"""Aggregator service for portfolio-wide calculations."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
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


def is_active_bot(bot: Bot, max_age_hours: int = 24) -> bool:
    """Check if bot has recent activity (updated within max_age_hours).
    
    Args:
        bot: Bot instance to check.
        max_age_hours: Maximum age of last update to consider active.
    
    Returns:
        True if bot has been updated recently.
    """
    if not bot.updated_at:
        return False
    
    age = datetime.utcnow() - bot.updated_at
    return age <= timedelta(hours=max_age_hours)


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
    last_updated: Optional[datetime] = None


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
    last_updated: Optional[datetime] = None  # Track when strategy last had activity


@dataclass
class DailyStrategyProfit:
    """Daily profit breakdown by strategy."""
    
    date: str  # YYYY-MM-DD
    strategy: str
    daily_profit: float
    daily_loss: float  # Negative values
    win_count: int
    loss_count: int
    win_rate: float
    total_trades: int


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
                # Get bot to extract port from API URL
                bot_result = await db.execute(select(Bot).where(Bot.id == bot_id))
                bot = bot_result.scalar_one_or_none()
                if not bot:
                    return None

                # Extract port from API URL for matching
                import re
                port = None
                if bot.api_url:
                    match = re.search(r':(\d+)', bot.api_url)
                    if match:
                        port = match.group(1)
                
                if not port:
                    logger.debug("No port found in bot API URL", bot_id=bot_id, api_url=bot.api_url)
                    return None

                # Match by port number in bot_name (e.g., "Strategy-9000" or "9000")
                # Get the most recent snapshot with actual balance data (not 0)
                row = await analytics_db.execute(
                    text("""
                        SELECT profit_all, balance, open_trades, trade_count, winrate, timestamp, bot_name
                        FROM bot_snapshots
                        WHERE bot_name LIKE :port_pattern
                          AND balance > 0
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """),
                    {"port_pattern": f"%-{port}"}
                )
                snapshot = row.fetchone()
                
                if snapshot:
                    logger.info("Found analytics snapshot by port", bot_id=bot_id, port=port, snapshot_name=snapshot.bot_name)
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
                    logger.debug("No analytics snapshot found for port", bot_id=bot_id, port=port, bot_name=bot.name)
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
            
            # Include bot in portfolio if we have cached metrics, regardless of health state
            if cached:
                logger.info("Found cached metrics for bot", bot_id=bot.id, bot_name=bot.name, profit=cached.profit_abs, health=bot.health_state)
                summary.portfolio_bots += 1
            else:
                logger.debug("No cached metrics found for bot", bot_id=bot.id, bot_name=bot.name)
                # Only skip if no cached data AND bot is unreachable
                if bot.health_state != HealthState.HEALTHY:
                    continue

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
                metrics.last_updated = cached.timestamp

                if metrics.win_rate is not None:
                    win_rates.append(float(metrics.win_rate))

                bot_metrics.append(metrics)

                # Aggregate totals - convert Decimal to float
                summary.total_profit_abs += float(metrics.profit_abs) if metrics.profit_abs else 0
                summary.total_balance += float(metrics.balance) if metrics.balance else 0
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
            total_profit=summary.total_profit_abs,
            total_balance=summary.total_balance,
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

            exchange = bot.exchange or "Unknown"

            # Use cached metrics if available, regardless of health state
            cached = await self.get_latest_cached_metrics(db, bot.id)
            
            # Only skip if no cached data AND bot is unreachable
            if not cached and bot.health_state != HealthState.HEALTHY:
                continue

            if exchange not in exchange_data:
                exchange_data[exchange] = ExchangeBreakdown(exchange=exchange)

            breakdown = exchange_data[exchange]
            breakdown.bot_count += 1

            if cached:
                breakdown.profit_abs += float(cached.profit_abs) if cached.profit_abs else 0
                breakdown.balance += float(cached.balance) if cached.balance else 0

                breakdown.open_positions += cached.open_positions or 0

        # Calculate profit percentages
        for breakdown in exchange_data.values():
            if breakdown.balance > 0:
                breakdown.profit_pct = breakdown.profit_abs / breakdown.balance * 100

        return list(exchange_data.values())

    async def get_strategy_breakdown(
        self,
        db: AsyncSession,
        only_active: bool = True,
        max_age_hours: int = 48,
    ) -> list[StrategyBreakdown]:
        """Get metrics breakdown by strategy using cached data.

        Uses bot_metrics table for persistence even when bots are offline.
        
        Args:
            db: Database session.
            only_active: If True, only include strategies with recent activity.
            max_age_hours: Maximum age of last update to consider active.

        Returns:
            List of strategy-wise metric breakdowns.
        """
        result = await db.execute(select(Bot))
        bots = result.scalars().all()

        strategy_data: dict[str, StrategyBreakdown] = {}
        strategy_win_rates: dict[str, list[float]] = {}
        strategy_last_updated: dict[str, datetime] = {}

        for bot in bots:
            # Skip non-trading bots (hyperopt, backtest, download)
            if not is_portfolio_bot(bot):
                continue

            strategy = bot.strategy or "Unknown"

            # Use cached metrics if available, regardless of health state
            cached = await self.get_latest_cached_metrics(db, bot.id)
            
            # Only skip if no cached data AND bot is unreachable
            if not cached and bot.health_state != HealthState.HEALTHY:
                continue

            if strategy not in strategy_data:
                strategy_data[strategy] = StrategyBreakdown(strategy=strategy)
                strategy_win_rates[strategy] = []
                strategy_last_updated[strategy] = None

            breakdown = strategy_data[strategy]
            breakdown.bot_count += 1

            if cached:
                breakdown.profit_abs += float(cached.profit_abs) if cached.profit_abs else 0
                breakdown.closed_trades += cached.closed_trades or 0
                breakdown.open_positions += cached.open_positions or 0
                if cached.win_rate is not None:
                    strategy_win_rates[strategy].append(float(cached.win_rate))
                
                # Track last update time
                if cached.timestamp:
                    if strategy_last_updated[strategy] is None or cached.timestamp > strategy_last_updated[strategy]:
                        strategy_last_updated[strategy] = cached.timestamp

        # Calculate average win rates per strategy
        for strategy, breakdown in strategy_data.items():
            rates = strategy_win_rates.get(strategy, [])
            if rates:
                breakdown.win_rate = sum(rates) / len(rates)
            breakdown.last_updated = strategy_last_updated.get(strategy)

        # Filter to only active strategies if requested
        if only_active:
            cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
            active_strategies = []
            for strategy, breakdown in strategy_data.items():
                if breakdown.last_updated and breakdown.last_updated >= cutoff:
                    active_strategies.append(breakdown)
                else:
                    logger.info("Filtering out inactive strategy", strategy=strategy, last_updated=breakdown.last_updated)
            return active_strategies

        return list(strategy_data.values())

    async def get_daily_strategy_profits(
        self,
        db: AsyncSession,
        days: int = 30,
    ) -> list[DailyStrategyProfit]:
        """Get daily profit breakdown by strategy from daily_performance table.
        
        Uses bot_name as the strategy identifier (since strategy column can be empty).
        
        Args:
            db: Database session.
            days: Number of days to fetch.
        
        Returns:
            List of daily strategy profit records.
        """
        results: list[DailyStrategyProfit] = []
        
        try:
            # Query daily_performance directly - use bot_name as strategy name
            # Use proper PostgreSQL interval syntax
            result = await db.execute(
                text("""
                    SELECT 
                        date,
                        bot_name as strategy,
                        COALESCE(SUM(profit), 0) as daily_profit,
                        COALESCE(SUM(win_count), 0) as win_count,
                        COALESCE(SUM(loss_count), 0) as loss_count,
                        COALESCE(SUM(total_trades), 0) as total_trades,
                        COALESCE(AVG(win_rate), 0) as win_rate
                    FROM daily_performance
                    WHERE date >= (CURRENT_DATE - make_interval(days => :days))
                      AND bot_name IS NOT NULL
                    GROUP BY date, bot_name
                    ORDER BY date DESC, SUM(profit) DESC
                """),
                {"days": days}
            )
            
            rows = result.mappings().all()
            
            for row in rows:
                daily_profit = float(row["daily_profit"] or 0)
                win_count = int(row["win_count"] or 0)
                loss_count = int(row["loss_count"] or 0)
                total_trades = int(row["total_trades"] or 0)
                win_rate = float(row["win_rate"] or 0)
                
                results.append(DailyStrategyProfit(
                    date=str(row["date"]),
                    strategy=row["strategy"] or "Unknown",
                    daily_profit=daily_profit if daily_profit > 0 else 0,
                    daily_loss=daily_profit if daily_profit < 0 else 0,
                    win_count=win_count,
                    loss_count=loss_count,
                    win_rate=win_rate,
                    total_trades=total_trades,
                ))
                    
        except Exception as e:
            logger.error("Failed to fetch daily strategy profits", error=str(e))
            # Return empty list - the frontend will show "no data"
            pass
        
        return results


# Singleton instance
aggregator_service = AggregatorService()
