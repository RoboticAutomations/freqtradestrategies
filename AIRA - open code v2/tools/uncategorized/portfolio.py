"""Portfolio API endpoints."""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import CurrentUser
from src.models import get_db
from src.models.bot import Bot
from src.schemas.portfolio import (
    ExchangeBreakdownData,
    ExchangeBreakdownResponse,
    PortfolioSummaryData,
    PortfolioSummaryResponse,
    StrategyBreakdownData,
    StrategyBreakdownResponse,
)
from src.services.aggregator import aggregator_service, is_portfolio_bot
from src.services.connectors.manager import connector_manager

router = APIRouter()


@router.get("/summary", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PortfolioSummaryResponse:
    """Get aggregated portfolio summary.

    Returns overall portfolio metrics including total profit,
    balance, trade counts, and bot health status breakdown.

    Returns:
        Aggregated portfolio metrics.
    """
    summary = await aggregator_service.get_portfolio_summary(db)

    return PortfolioSummaryResponse(
        data=PortfolioSummaryData(
            timestamp=summary.timestamp,
            total_bots=summary.total_bots,
            healthy_bots=summary.healthy_bots,
            degraded_bots=0,  # Deprecated - no more degraded state
            unreachable_bots=summary.unreachable_bots,
            hyperopt_bots=summary.hyperopt_bots,
            backtest_bots=summary.backtest_bots,
            portfolio_bots=summary.portfolio_bots,
            total_profit_abs=summary.total_profit_abs,
            total_profit_pct=summary.total_profit_pct,
            total_balance=summary.total_balance,
            total_open_positions=summary.total_open_positions,
            total_closed_trades=summary.total_closed_trades,
            avg_win_rate=summary.avg_win_rate,
            best_performer=summary.best_performer,
            worst_performer=summary.worst_performer,
        )
    )


@router.get("/by-exchange", response_model=ExchangeBreakdownResponse)
async def get_exchange_breakdown(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ExchangeBreakdownResponse:
    """Get portfolio breakdown by exchange.

    Returns metrics grouped by exchange, showing profit,
    balance, and position counts per exchange.

    Returns:
        Exchange-wise breakdown of portfolio metrics.
    """
    breakdowns = await aggregator_service.get_exchange_breakdown(db)

    return ExchangeBreakdownResponse(
        data=[
            ExchangeBreakdownData(
                exchange=b.exchange,
                bot_count=b.bot_count,
                profit_abs=b.profit_abs,
                profit_pct=b.profit_pct,
                balance=b.balance,
                open_positions=b.open_positions,
            )
            for b in breakdowns
        ]
    )


@router.get("/by-strategy", response_model=StrategyBreakdownResponse)
async def get_strategy_breakdown(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    only_active: bool = Query(True, description="Only include strategies with recent activity"),
    max_age_hours: int = Query(48, description="Maximum hours since last update to consider active"),
) -> StrategyBreakdownResponse:
    """Get portfolio breakdown by strategy.

    Returns metrics grouped by trading strategy, showing profit,
    trade counts, and win rates per strategy.

    By default only shows active strategies (updated within last 48 hours).

    Args:
        only_active: If True, filter to only strategies with recent activity.
        max_age_hours: Maximum age of last update to consider active.

    Returns:
        Strategy-wise breakdown of portfolio metrics.
    """
    breakdowns = await aggregator_service.get_strategy_breakdown(db, only_active=only_active, max_age_hours=max_age_hours)

    return StrategyBreakdownResponse(
        data=[
            StrategyBreakdownData(
                strategy=b.strategy,
                bot_count=b.bot_count,
                profit_abs=b.profit_abs,
                profit_pct=b.profit_pct,
                open_positions=b.open_positions,
                closed_trades=b.closed_trades,
                win_rate=b.win_rate,
            )
            for b in breakdowns
        ]
    )


@router.get("/top-performers")
async def get_top_performers(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(10, ge=1, le=50, description="Number of top performers to return"),
    sort_by: str = Query("profit_pct", description="Sort by: profit_pct, profit_abs, win_rate"),
) -> dict:
    """Get top performing bots using cached data.

    Uses analytics database bot_snapshots for persistent metrics
    even when bots are offline.

    Args:
        limit: Number of bots to return.
        sort_by: Metric to sort by.

    Returns:
        List of top performing bots with metrics.
    """
    from src.services.aggregator import aggregator_service

    # Get all bots
    result = await db.execute(select(Bot))
    all_bots = list(result.scalars())

    # Filter to trading bots only
    bots = [bot for bot in all_bots if is_portfolio_bot(bot)]

    # Get cached metrics for each bot
    valid_metrics = []
    for bot in bots:
        cached = await aggregator_service.get_latest_cached_metrics(db, bot.id)
        if cached and cached.balance > 0:  # Only include bots with data
            # Calculate profit_pct from profit_abs and balance (analytics doesn't store it)
            profit_pct = cached.profit_pct or 0
            if profit_pct == 0 and cached.profit_abs and cached.balance > 0:
                profit_pct = (cached.profit_abs / cached.balance) * 100

            valid_metrics.append({
                "bot_id": bot.id,
                "bot_name": bot.name,
                "exchange": bot.exchange,
                "strategy": bot.strategy,
                "profit_abs": round(cached.profit_abs or 0, 2),
                "profit_pct": round(profit_pct, 2),
                "closed_trades": cached.closed_trades or 0,
                "win_rate": round(cached.win_rate or 0, 2),
                "data_source": "cached",
            })

    # Sort by requested metric
    sort_key = sort_by if sort_by in ["profit_pct", "profit_abs", "win_rate"] else "profit_pct"
    sorted_metrics = sorted(
        valid_metrics,
        key=lambda x: x.get(sort_key) or 0,
        reverse=True
    )

    return {
        "status": "success",
        "data": {
            "top_performers": sorted_metrics[:limit],
            "worst_performers": sorted_metrics[-limit:][::-1] if len(sorted_metrics) >= limit else sorted_metrics[::-1],
            "total_bots_analyzed": len(valid_metrics),
        }
    }


@router.get("/daily-strategy-profits")
async def get_daily_strategy_profits(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    days: int = Query(30, ge=1, le=90, description="Number of days to fetch"),
) -> dict:
    """Get daily profit breakdown by strategy.

    Returns daily performance metrics for each strategy including:
    - Daily profit/loss
    - Win/loss counts
    - Win rate percentage

    Args:
        days: Number of days to fetch (1-90).

    Returns:
        Daily strategy profit breakdown sorted by date and profit.
    """
    from src.services.aggregator import aggregator_service, DailyStrategyProfit

    daily_profits = await aggregator_service.get_daily_strategy_profits(db, days=days)

    # Group by date for easier frontend consumption
    grouped_by_date: dict[str, list[dict]] = {}
    for profit in daily_profits:
        date_key = profit.date
        if date_key not in grouped_by_date:
            grouped_by_date[date_key] = []
        grouped_by_date[date_key].append({
            "strategy": profit.strategy,
            "daily_profit": round(profit.daily_profit, 2),
            "daily_loss": round(profit.daily_loss, 2),
            "win_count": profit.win_count,
            "loss_count": profit.loss_count,
            "win_rate": round(profit.win_rate, 2),
            "total_trades": profit.total_trades,
        })

    return {
        "status": "success",
        "data": {
            "by_date": grouped_by_date,
            "raw": [
                {
                    "date": p.date,
                    "strategy": p.strategy,
                    "daily_profit": round(p.daily_profit, 2),
                    "daily_loss": round(p.daily_loss, 2),
                    "win_count": p.win_count,
                    "loss_count": p.loss_count,
                    "win_rate": round(p.win_rate, 2),
                    "total_trades": p.total_trades,
                }
                for p in daily_profits
            ],
        },
    }


@router.get("/active-bots")
async def get_active_bots(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Get list of active trading bots with their strategy and performance data.

    Returns individual bot details (not aggregated by strategy) including:
    - Bot name and ID
    - Strategy name
    - Exchange
    - API URL for FreqUI access
    - Profit, trades, open positions

    Returns:
        List of active bots with performance metrics.
    """
    from src.services.aggregator import aggregator_service

    result = await db.execute(select(Bot))
    bots = result.scalars().all()

    active_bots = []

    for bot in bots:
        # Only include healthy trading bots
        if not is_portfolio_bot(bot):
            continue

        # Get cached metrics
        cached = await aggregator_service.get_latest_cached_metrics(db, bot.id)

        if cached:
            active_bots.append({
                "bot_id": bot.id,
                "bot_name": bot.name,
                "strategy": bot.strategy,
                "exchange": bot.exchange,
                "api_url": bot.api_url,
                "profit_abs": round(cached.profit_abs or 0, 2),
                "closed_trades": cached.closed_trades or 0,
                "open_positions": cached.open_positions or 0,
            })

    return {
        "status": "success",
        "data": {
            "bots": active_bots,
        },
    }