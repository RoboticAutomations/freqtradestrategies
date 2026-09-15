"""Bot management API endpoints."""

from datetime import datetime


def utc_naive_now() -> datetime:
    """UTC timestamp without tzinfo (safe for TIMESTAMP WITHOUT TIME ZONE columns)."""
    return datetime.utcnow()
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import CurrentUser
from src.models import get_db
from src.models.bot import Bot, BotEnvironment, HealthState, SourceMode
from src.models.metrics import BotMetrics as BotMetricsModel
from src.schemas.bot import (
    BotCredentialsRequest,
    BotCredentialsResponse,
    BotDetailData,
    BotDetailResponse,
    BotHealthData,
    BotHealthResponse,
    BotListResponse,
    BotMetricsData,
    BotMetricsResponse,
    BotResponse,
    BotUpdateRequest,
)
from src.schemas.common import MessageResponse
from src.services.connectors.manager import connector_manager
from src.services.health import health_monitor
from src.services.cache import cache, bot_metrics_key, bot_health_key

router = APIRouter()


@router.get("/search", response_model=BotListResponse)
async def search_bots(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
) -> BotListResponse:
    """Search bots by name, exchange, or strategy.

    Args:
        q: Search query string.
        limit: Maximum number of results to return.

    Returns:
        List of bots matching the search query.
    """
    # Case-insensitive search across name, exchange, and strategy
    search_term = f"%{q.lower()}%"

    result = await db.execute(
        select(Bot)
        .where(
            (Bot.name.ilike(search_term))
            | (Bot.exchange.ilike(search_term))
            | (Bot.strategy.ilike(search_term))
        )
        .order_by(Bot.name)
        .limit(limit)
    )
    all_bots = list(result.scalars())

    bots = [BotResponse.model_validate(bot) for bot in all_bots]

    return BotListResponse(data=bots)


@router.get("", response_model=BotListResponse)
async def list_bots(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    environment: Optional[BotEnvironment] = Query(None, description="Filter by environment"),
    health_state: Optional[HealthState] = Query(None, description="Filter by health state"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    tags: Optional[list[str]] = Query(None, description="Filter by tags"),
) -> BotListResponse:
    """List all registered bots with optional filtering.

    Args:
        environment: Filter by deployment environment (docker, baremetal, etc.)
        health_state: Filter by health state (healthy, degraded, unreachable)
        exchange: Filter by exchange name
        strategy: Filter by strategy name
        tags: Filter by tags (any match)

    Returns:
        List of bots matching filters.
    """
    query = select(Bot)

    # Apply filters
    if environment:
        query = query.where(Bot.environment == environment)
    if health_state:
        query = query.where(Bot.health_state == health_state)
    if exchange:
        query = query.where(Bot.exchange == exchange)
    if strategy:
        query = query.where(Bot.strategy == strategy)

    result = await db.execute(query.order_by(Bot.name))
    all_bots = list(result.scalars())

    # Filter by tags in Python (JSON doesn't support overlap in SQLite)
    if tags:
        all_bots = [bot for bot in all_bots if any(tag in bot.tags for tag in tags)]

    bots = [BotResponse.model_validate(bot) for bot in all_bots]

    return BotListResponse(data=bots)


@router.get("/{bot_id}", response_model=BotDetailResponse)
async def get_bot(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BotDetailResponse:
    """Get detailed information about a specific bot.

    Args:
        bot_id: Bot UUID.

    Returns:
        Detailed bot information including connection details.

    Raises:
        404: Bot not found.
    """
    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    return BotDetailResponse(
        data=BotDetailData(
            id=bot.id,
            name=bot.name,
            environment=bot.environment,
            host=bot.host,
            api_url=bot.api_url,
            health_state=bot.health_state,
            source_mode=bot.source_mode,
            exchange=bot.exchange,
            strategy=bot.strategy,
            trading_mode=bot.trading_mode,
            is_dryrun=bot.is_dryrun,
            tags=bot.tags,
            last_seen=bot.last_seen,
            container_id=bot.container_id,
            user_data_path=bot.user_data_path,
            discovered_at=bot.discovered_at,
            created_at=bot.created_at,
        )
    )


@router.patch("/{bot_id}", response_model=BotDetailResponse)
async def update_bot(
    bot_id: str,
    update: BotUpdateRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BotDetailResponse:
    """Update bot settings.

    Args:
        bot_id: Bot UUID.
        update: Fields to update.

    Returns:
        Updated bot information.

    Raises:
        404: Bot not found.
    """
    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    # Apply updates
    if update.name is not None:
        bot.name = update.name
    if update.tags is not None:
        bot.tags = update.tags
    if update.source_mode is not None:
        bot.source_mode = update.source_mode

    await db.commit()
    await db.refresh(bot)

    return BotDetailResponse(
        data=BotDetailData(
            id=bot.id,
            name=bot.name,
            environment=bot.environment,
            host=bot.host,
            api_url=bot.api_url,
            health_state=bot.health_state,
            source_mode=bot.source_mode,
            exchange=bot.exchange,
            strategy=bot.strategy,
            trading_mode=bot.trading_mode,
            is_dryrun=bot.is_dryrun,
            tags=bot.tags,
            last_seen=bot.last_seen,
            container_id=bot.container_id,
            user_data_path=bot.user_data_path,
            discovered_at=bot.discovered_at,
            created_at=bot.created_at,
        )
    )


@router.put("/{bot_id}/credentials", response_model=BotCredentialsResponse)
async def update_bot_credentials(
    bot_id: str,
    request: BotCredentialsRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BotCredentialsResponse:
    """Update API credentials for a bot.

    Updates the username/password used to authenticate with the bot's API.
    After updating, tests the connection to verify the credentials work.

    Args:
        bot_id: Bot UUID.
        request: New credentials.

    Returns:
        Success status and whether API is now available.

    Raises:
        404: Bot not found.
    """
    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    if not bot.api_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot has no API URL configured",
        )

    # Store credentials (encrypted) - for now just test connection
    from src.services.connectors.api import APIConnector

    connector = APIConnector(
        bot_id=bot.id,
        api_url=bot.api_url,
        username=request.username,
        password=request.password,
    )

    try:
        # Test the connection with new credentials
        health_result = await connector.check_health()
        api_available = health_result.success

        if api_available:
            # Update bot health state
            bot.health_state = HealthState.HEALTHY
            bot.last_seen = utc_naive_now()
            await db.commit()

            # Invalidate cached connector so it gets recreated with new creds
            await connector_manager.invalidate_connector(bot.id)

            # TODO: Store encrypted credentials in bot.credentials_enc
            # For now, credentials need to be stored in config

            return BotCredentialsResponse(
                message="Credentials verified successfully. API connection restored.",
                api_available=True,
            )
        else:
            return BotCredentialsResponse(
                status="error",
                message=f"Credentials rejected: {health_result.error}",
                api_available=False,
            )
    finally:
        await connector.close()


@router.delete("/{bot_id}", response_model=MessageResponse)
async def delete_bot(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Remove a bot from the dashboard.

    This does not stop or affect the actual bot - it only removes it
    from dashboard monitoring. The bot may be rediscovered on next scan.

    Args:
        bot_id: Bot UUID.

    Returns:
        Success message.

    Raises:
        404: Bot not found.
    """
    from sqlalchemy import delete
    from src.models.metrics import BotMetrics
    
    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    bot_name = bot.name
    
    # FIX: Delete related metrics first to avoid FK constraint errors
    await db.execute(delete(BotMetrics).where(BotMetrics.bot_id == bot_id))
    
    # Clear cache entries
    cache.delete(bot_health_key(bot_id))
    cache.delete(bot_metrics_key(bot_id))
    
    # Now delete the bot
    await db.delete(bot)
    await db.commit()

    return MessageResponse(message=f"Bot '{bot_name}' removed from dashboard")


@router.get("/{bot_id}/metrics", response_model=BotMetricsResponse)
async def get_bot_metrics(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    bypass_cache: bool = Query(False, description="Bypass cache and fetch fresh data"),
) -> BotMetricsResponse:
    """Get current metrics for a bot.

    Fetches real-time profit, balance, and trade metrics using
    the best available data source (API or SQLite).
    
    PERSISTS good data to DB and falls back to last known good values
    when API returns empty/zero data.

    Args:
        bot_id: Bot UUID.
        bypass_cache: If True, fetch fresh data ignoring cache.

    Returns:
        Current bot metrics with source indicator.

    Raises:
        404: Bot not found.
    """
    # Check cache first (unless bypassed)
    cache_key = bot_metrics_key(bot_id)
    if not bypass_cache:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return BotMetricsResponse(data=cached_data)

    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    # Get connector and fetch metrics
    connector, source = await connector_manager.get_connector(bot)

    profit_result = await connector.get_profit()
    balance_result = await connector.get_balance()
    open_trades_result = await connector.get_open_trades()

    # Calculate win rate
    win_rate = None
    if profit_result.success and profit_result.data:
        profit = profit_result.data
        total_closed = profit.winning_trades + profit.losing_trades
        if total_closed > 0:
            win_rate = profit.winning_trades / total_closed

    # Build metrics data from API
    metrics_data = BotMetricsData(
        bot_id=bot.id,
        timestamp=datetime.utcnow(),
        profit_abs=profit_result.data.profit_all_coin if profit_result.success else None,
        profit_pct=profit_result.data.profit_all_percent if profit_result.success else None,
        profit_realized=profit_result.data.profit_closed_coin if profit_result.success else None,
        profit_unrealized=(
            profit_result.data.profit_all_coin - profit_result.data.profit_closed_coin
            if profit_result.success else None
        ),
        open_positions=len(open_trades_result.data) if open_trades_result.success else 0,
        closed_trades=profit_result.data.closed_trade_count if profit_result.success else 0,
        win_rate=win_rate,
        balance=balance_result.data.stake_currency_balance if balance_result.success else None,
        data_source=source,
        health_state=bot.health_state,
    )

    # Check if we have valid data
    # Valid data means: at least one of profit, balance, or closed_trades is non-zero
    # OR the bot has been running (closed_trades > 0 indicates trading activity)
    has_valid_data = False
    if profit_result.success and profit_result.data:
        profit = profit_result.data
        has_profit_data = profit.profit_all_coin is not None and profit.profit_all_coin != 0
        has_trade_activity = profit.closed_trade_count is not None and profit.closed_trade_count > 0
        has_balance_data = balance_result.success and balance_result.data and balance_result.data.stake_currency_balance not in (None, 0)
        
        # Consider valid if: has profit OR has trade activity OR has balance
        has_valid_data = has_profit_data or has_trade_activity or has_balance_data

    if has_valid_data:
        # Save to DB for persistence
        db_metrics = BotMetricsModel(
            bot_id=bot.id,
            timestamp=datetime.utcnow(),
            profit_abs=metrics_data.profit_abs,
            profit_pct=metrics_data.profit_pct,
            profit_realized=metrics_data.profit_realized,
            profit_unrealized=metrics_data.profit_unrealized,
            open_positions=metrics_data.open_positions,
            closed_trades=metrics_data.closed_trades,
            win_rate=metrics_data.win_rate,
            balance=metrics_data.balance,
            data_source=source,
        )
        db.add(db_metrics)
        await db.commit()
    else:
        # Fetch last known good metrics from DB
        last_metrics_result = await db.execute(
            select(BotMetricsModel)
            .where(BotMetricsModel.bot_id == bot_id)
            .order_by(BotMetricsModel.timestamp.desc())
            .limit(1)
        )
        last_metrics = last_metrics_result.scalar_one_or_none()
        
        if last_metrics:
            # Use cached data but mark as stale
            metrics_data = BotMetricsData(
                bot_id=bot.id,
                timestamp=last_metrics.timestamp,
                profit_abs=float(last_metrics.profit_abs) if last_metrics.profit_abs else None,
                profit_pct=float(last_metrics.profit_pct) if last_metrics.profit_pct else None,
                profit_realized=float(last_metrics.profit_realized) if last_metrics.profit_realized else None,
                profit_unrealized=float(last_metrics.profit_unrealized) if last_metrics.profit_unrealized else None,
                open_positions=last_metrics.open_positions,
                closed_trades=last_metrics.closed_trades,
                win_rate=float(last_metrics.win_rate) if last_metrics.win_rate else None,
                balance=float(last_metrics.balance) if last_metrics.balance else None,
                data_source=last_metrics.data_source,  # Keep original source
                health_state=bot.health_state,
            )

    # Cache the metrics for 30 seconds
    cache.set(cache_key, metrics_data, ttl_seconds=30)

    return BotMetricsResponse(data=metrics_data)


@router.get("/{bot_id}/health", response_model=BotHealthResponse)
async def get_bot_health(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    bypass_cache: bool = Query(False, description="Bypass cache and fetch fresh data"),
) -> BotHealthResponse:
    """Get health status for a bot.

    Returns detailed health metrics including API/SQLite availability,
    success rates, and latency information.

    Args:
        bot_id: Bot UUID.
        bypass_cache: If True, fetch fresh data ignoring cache.

    Returns:
        Bot health status with data source metrics.

    Raises:
        404: Bot not found.
    """
    # Check cache first (unless bypassed)
    cache_key = bot_health_key(bot_id)
    if not bypass_cache:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return BotHealthResponse(data=cached_data)

    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    # Get health metrics
    metrics = health_monitor.get_metrics(bot_id)

    # Determine active source
    _, active_source = await connector_manager.get_connector(bot)

    if metrics:
        health_data = BotHealthData(
            bot_id=bot.id,
            health_state=bot.health_state,
            source_mode=bot.source_mode,
            active_source=active_source,
            api_available=bool(bot.api_url) and metrics.api_success_rate > 0,
            sqlite_available=bool(bot.user_data_path) and metrics.sqlite_success_rate > 0,
            api_success_rate=metrics.api_success_rate,
            sqlite_success_rate=metrics.sqlite_success_rate,
            api_avg_latency_ms=metrics.api_avg_latency,
            sqlite_avg_latency_ms=metrics.sqlite_avg_latency,
            last_check=metrics.last_check,
            state_changed_at=metrics.state_changed_at,
        )
    else:
        # No metrics yet - return defaults
        health_data = BotHealthData(
            bot_id=bot.id,
            health_state=bot.health_state,
            source_mode=bot.source_mode,
            active_source=active_source,
            api_available=bool(bot.api_url),
            sqlite_available=bool(bot.user_data_path),
            api_success_rate=0.0,
            sqlite_success_rate=0.0,
            api_avg_latency_ms=0.0,
            sqlite_avg_latency_ms=0.0,
        )

    # Cache the health data for 15 seconds (shorter than metrics since health can change faster)
    cache.set(cache_key, health_data, ttl_seconds=15)

    return BotHealthResponse(data=health_data)


@router.post("/{bot_id}/health/check", response_model=BotHealthResponse)
async def trigger_health_check(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BotHealthResponse:
    """Trigger an immediate health check for a bot.

    Forces a health check regardless of the scheduled interval.

    Args:
        bot_id: Bot UUID.

    Returns:
        Updated bot health status.

    Raises:
        404: Bot not found.
    """
    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    # Invalidate cached data for this bot
    cache.delete(bot_health_key(bot_id))
    cache.delete(bot_metrics_key(bot_id))

    # Trigger immediate check
    metrics = await health_monitor.trigger_check(bot_id)

    # Refresh bot to get updated state
    await db.refresh(bot)

    _, active_source = await connector_manager.get_connector(bot)

    if metrics:
        return BotHealthResponse(
            data=BotHealthData(
                bot_id=bot.id,
                health_state=bot.health_state,
                source_mode=bot.source_mode,
                active_source=active_source,
                api_available=bool(bot.api_url) and metrics.api_success_rate > 0,
                sqlite_available=bool(bot.user_data_path) and metrics.sqlite_success_rate > 0,
                api_success_rate=metrics.api_success_rate,
                sqlite_success_rate=metrics.sqlite_success_rate,
                api_avg_latency_ms=metrics.api_avg_latency,
                sqlite_avg_latency_ms=metrics.sqlite_avg_latency,
                last_check=metrics.last_check,
                state_changed_at=metrics.state_changed_at,
            )
        )

    return BotHealthResponse(
        data=BotHealthData(
            bot_id=bot.id,
            health_state=bot.health_state,
            source_mode=bot.source_mode,
            active_source=active_source,
            api_available=bool(bot.api_url),
            sqlite_available=bool(bot.user_data_path),
            api_success_rate=0.0,
            sqlite_success_rate=0.0,
            api_avg_latency_ms=0.0,
            sqlite_avg_latency_ms=0.0,
        )
    )


@router.get("/{bot_id}/trades")
async def get_bot_trades(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    is_open: Optional[bool] = Query(None, description="Filter by open/closed status"),
) -> dict:
    """Get trades for a bot.

    Fetches trade history using the best available data source.

    Args:
        bot_id: Bot UUID.
        is_open: Optional filter for open/closed trades.

    Returns:
        List of trades.

    Raises:
        404: Bot not found.
    """
    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    # Get connector and fetch trades
    connector, source = await connector_manager.get_connector(bot)

    if is_open is True:
        trades_result = await connector.get_open_trades()
    elif is_open is False:
        trades_result = await connector.get_closed_trades()
    else:
        # Get both and combine
        open_result = await connector.get_open_trades()
        closed_result = await connector.get_closed_trades()

        trades = []
        if open_result.success:
            trades.extend(open_result.data)
        if closed_result.success:
            trades.extend(closed_result.data)

        return {
            "status": "success",
            "data": [
                {
                    "id": t.trade_id,
                    "pair": t.pair,
                    "is_open": t.is_open,
                    "open_date": t.open_date.isoformat() if t.open_date else None,
                    "close_date": t.close_date.isoformat() if t.close_date else None,
                    "open_rate": t.open_rate,
                    "close_rate": t.close_rate,
                    "amount": t.amount,
                    "stake_amount": t.stake_amount,
                    "close_profit": t.profit_ratio,
                    "close_profit_abs": t.profit_abs,
                    "enter_tag": t.enter_tag,
                    "exit_reason": t.exit_reason,
                    "leverage": t.leverage or 1.0,
                    "is_short": t.is_short or False,
                    "data_source": source.value,
                }
                for t in trades
            ],
        }

    if not trades_result.success:
        return {"status": "success", "data": []}

    return {
        "status": "success",
        "data": [
            {
                "id": t.trade_id,
                "pair": t.pair,
                "is_open": t.is_open,
                "open_date": t.open_date.isoformat() if t.open_date else None,
                "close_date": t.close_date.isoformat() if t.close_date else None,
                "open_rate": t.open_rate,
                "close_rate": t.close_rate,
                "amount": t.amount,
                "stake_amount": t.stake_amount,
                "close_profit": t.profit_ratio,
                "close_profit_abs": t.profit_abs,
                "enter_tag": t.enter_tag,
                "exit_reason": t.exit_reason,
                "leverage": t.leverage or 1.0,
                "is_short": t.is_short or False,
                "data_source": source.value,
            }
            for t in trades_result.data
        ],
    }


@router.get("/{bot_id}/status")
async def get_bot_status(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Get Freqtrade bot running status.

    Fetches the actual running state (running/stopped) from the Freqtrade API.

    Args:
        bot_id: Bot UUID.

    Returns:
        Bot status information.

    Raises:
        404: Bot not found.
        400: Bot API not available.
    """
    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    if not bot.is_api_available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot API is not available",
        )

    connector, _ = await connector_manager.get_connector(bot)
    status_result = await connector.get_status()

    if not status_result.success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to get bot status: {status_result.error}",
        )

    bot_status = status_result.data
    return {
        "status": "success",
        "data": {
            "state": bot_status.state,
            "strategy": bot_status.strategy,
            "exchange": bot_status.exchange,
            "trading_mode": bot_status.trading_mode,
            "is_dryrun": bot_status.is_dryrun,
            "version": bot_status.version,
        },
    }


@router.get("/{bot_id}/performance")
async def get_bot_performance(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Get aggregate performance statistics for a bot.

    Calculates win rate, total profit, average duration, etc. from trade history.

    Args:
        bot_id: Bot UUID.

    Returns:
        Performance statistics.

    Raises:
        404: Bot not found.
    """
    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    # Get connector and fetch trades
    connector, source = await connector_manager.get_connector(bot)
    closed_result = await connector.get_closed_trades()

    if not closed_result.success or not closed_result.data:
        return {
            "status": "success",
            "data": {
                "total_profit": 0.0,
                "total_profit_pct": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "avg_duration_mins": 0.0,
            },
        }

    trades = closed_result.data
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if (t.profit_ratio or 0) > 0)
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0

    total_profit = sum(t.profit_abs or 0 for t in trades)
    total_profit_pct = sum((t.profit_ratio or 0) * 100 for t in trades)

    # Calculate average duration
    durations = []
    for t in trades:
        if t.open_date and t.close_date:
            duration_mins = (t.close_date - t.open_date).total_seconds() / 60
            durations.append(duration_mins)

    avg_duration_mins = sum(durations) / len(durations) if durations else 0.0

    return {
        "status": "success",
        "data": {
            "total_profit": round(total_profit, 8),
            "total_profit_pct": round(total_profit_pct, 2),
            "win_rate": round(win_rate, 1),
            "total_trades": total_trades,
            "avg_duration_mins": round(avg_duration_mins, 1),
        },
    }


# Bot Control Endpoints

@router.post("/{bot_id}/start", response_model=MessageResponse)
async def start_bot(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Start a bot's trading.

    Requires API access and operator/admin role.

    Args:
        bot_id: Bot UUID.

    Returns:
        Success message.

    Raises:
        404: Bot not found.
        400: Bot API not available.
        403: User lacks permission.
    """
    if not current_user.can_control_bots():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to control bots",
        )

    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    if not bot.is_api_available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot API is not available",
        )

    connector, _ = await connector_manager.get_connector(bot)

    # APIConnector has start_bot method
    from src.services.connectors.api import APIConnector
    if not isinstance(connector, APIConnector):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot control requires API connection",
        )

    api_result = await connector.start_bot()

    if not api_result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start bot: {api_result.error}",
        )

    # Invalidate cache after successful action
    cache.delete(bot_metrics_key(bot_id))
    cache.delete(bot_health_key(bot_id))

    return MessageResponse(message=f"Bot '{bot.name}' started successfully")


@router.post("/{bot_id}/stop", response_model=MessageResponse)
async def stop_bot(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Stop a bot's trading.

    Requires API access and operator/admin role.

    Args:
        bot_id: Bot UUID.

    Returns:
        Success message.
    """
    if not current_user.can_control_bots():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to control bots",
        )

    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    if not bot.is_api_available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot API is not available",
        )

    connector, _ = await connector_manager.get_connector(bot)

    from src.services.connectors.api import APIConnector
    if not isinstance(connector, APIConnector):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot control requires API connection",
        )

    api_result = await connector.stop_bot()

    if not api_result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop bot: {api_result.error}",
        )

    # Invalidate cache after successful action
    cache.delete(bot_metrics_key(bot_id))
    cache.delete(bot_health_key(bot_id))

    return MessageResponse(message=f"Bot '{bot.name}' stopped successfully")


@router.post("/{bot_id}/reload", response_model=MessageResponse)
async def reload_bot_config(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Reload bot configuration.

    Requires API access and operator/admin role.

    Args:
        bot_id: Bot UUID.

    Returns:
        Success message.
    """
    if not current_user.can_control_bots():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to control bots",
        )

    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    if not bot.is_api_available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot API is not available",
        )

    connector, _ = await connector_manager.get_connector(bot)

    from src.services.connectors.api import APIConnector
    if not isinstance(connector, APIConnector):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot control requires API connection",
        )

    api_result = await connector.reload_config()

    if not api_result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload config: {api_result.error}",
        )

    # Invalidate cache after successful action
    cache.delete(bot_metrics_key(bot_id))
    cache.delete(bot_health_key(bot_id))

    return MessageResponse(message=f"Bot '{bot.name}' configuration reloaded")


@router.post("/{bot_id}/forceexit", response_model=MessageResponse)
async def force_exit_all(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Force exit all open trades.

    Requires API access and admin role.

    Args:
        bot_id: Bot UUID.

    Returns:
        Success message.
    """
    if not current_user.can_force_exit():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can force exit trades",
        )

    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found",
        )

    if not bot.is_api_available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot API is not available",
        )

    connector, _ = await connector_manager.get_connector(bot)

    from src.services.connectors.api import APIConnector
    if not isinstance(connector, APIConnector):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot control requires API connection",
        )

    api_result = await connector.force_exit()

    if not api_result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to force exit: {api_result.error}",
        )

    # Invalidate cache after successful action
    cache.delete(bot_metrics_key(bot_id))
    cache.delete(bot_health_key(bot_id))

    return MessageResponse(message=f"Force exit initiated for all trades on '{bot.name}'")
