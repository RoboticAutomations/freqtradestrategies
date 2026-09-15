"""Trade monitoring service for real-time trade updates."""

import asyncio
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models import async_session_maker
from src.models.bot import Bot, HealthState
from src.services.connectors.api import APIConnector
from src.services.websocket import ws_manager

logger = structlog.get_logger()


class TradeMonitor:
    """Background service for monitoring trade changes across all bots."""

    def __init__(self):
        """Initialize trade monitor."""
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._connectors: dict[str, APIConnector] = {}
        self._trade_cache: dict[str, dict[int, dict]] = {}  # bot_id -> {trade_id -> trade_data}
        self._interval = 15  # Check every 15 seconds

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    async def start(self) -> None:
        """Start the trade monitoring task."""
        if self._running:
            logger.warning("Trade monitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Trade monitor started", interval=self._interval)

    async def stop(self) -> None:
        """Stop the trade monitoring task."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Clean up connectors
        for connector in self._connectors.values():
            await connector.close()
        self._connectors.clear()
        self._trade_cache.clear()

        logger.info("Trade monitor stopped")

    async def _run_loop(self) -> None:
        """Main monitoring loop."""
        # Wait for other services to initialize
        await asyncio.sleep(10)

        while self._running:
            try:
                await self._check_all_bots()
                await asyncio.sleep(self._interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Trade monitor error", error=str(e))
                await asyncio.sleep(30)

    async def _check_all_bots(self) -> None:
        """Check trades for all healthy bots."""
        try:
            async with async_session_maker() as session:
                # Get all healthy bots with API access
                result = await session.execute(
                    select(Bot).where(
                        Bot.health_state == HealthState.HEALTHY,
                        Bot.api_url.isnot(None),
                    )
                )
                bots = result.scalars().all()

                for bot in bots:
                    await self._check_bot_trades(bot)

        except Exception as e:
            logger.error("Trade check batch failed", error=str(e))

    async def _check_bot_trades(self, bot: Bot) -> None:
        """Check trades for a single bot and broadcast changes."""
        try:
            connector = await self._get_connector(bot)
            if not connector:
                return

            # Fetch current open trades
            trades_result = await connector.get_open_trades()
            if not trades_result.success:
                return

            current_trades = trades_result.data or []

            # Initialize cache for this bot if needed
            if bot.id not in self._trade_cache:
                self._trade_cache[bot.id] = {}
                # First run - just populate cache without broadcasting
                for trade in current_trades:
                    self._trade_cache[bot.id][trade.trade_id] = self._trade_to_dict(trade)
                return

            cached_trades = self._trade_cache[bot.id]
            current_trade_ids = {trade.trade_id for trade in current_trades}
            cached_trade_ids = set(cached_trades.keys())

            # Detect new trades (opened)
            new_trade_ids = current_trade_ids - cached_trade_ids
            for trade in current_trades:
                if trade.trade_id in new_trade_ids:
                    trade_data = self._trade_to_dict(trade)
                    cached_trades[trade.trade_id] = trade_data

                    await ws_manager.broadcast_bot_update(
                        bot_id=bot.id,
                        event_type="trade_update",
                        data={
                            "action": "opened",
                            "trade": trade_data,
                        },
                    )
                    logger.debug(
                        "Trade opened",
                        bot_id=bot.id,
                        trade_id=trade.trade_id,
                        pair=trade.pair,
                    )

            # Detect closed trades
            closed_trade_ids = cached_trade_ids - current_trade_ids
            for trade_id in closed_trade_ids:
                closed_trade = cached_trades.pop(trade_id)

                await ws_manager.broadcast_bot_update(
                    bot_id=bot.id,
                    event_type="trade_update",
                    data={
                        "action": "closed",
                        "trade": closed_trade,
                    },
                )
                logger.debug(
                    "Trade closed",
                    bot_id=bot.id,
                    trade_id=trade_id,
                    pair=closed_trade.get("pair"),
                )

            # Check for trade updates (profit changes on open trades)
            for trade in current_trades:
                if trade.trade_id in cached_trade_ids:
                    new_data = self._trade_to_dict(trade)
                    old_data = cached_trades[trade.trade_id]

                    # Check if profit changed significantly (>0.1%)
                    old_profit = old_data.get("current_profit_pct", 0) or 0
                    new_profit = new_data.get("current_profit_pct", 0) or 0

                    if abs(new_profit - old_profit) > 0.1:
                        cached_trades[trade.trade_id] = new_data

                        await ws_manager.broadcast_bot_update(
                            bot_id=bot.id,
                            event_type="trade_update",
                            data={
                                "action": "updated",
                                "trade": new_data,
                            },
                        )

        except Exception as e:
            logger.debug("Failed to check trades for bot", bot_id=bot.id, error=str(e))

    async def _get_connector(self, bot: Bot) -> Optional[APIConnector]:
        """Get or create API connector for bot."""
        if not bot.api_url:
            return None

        if bot.id not in self._connectors:
            self._connectors[bot.id] = APIConnector(
                bot_id=bot.id,
                api_url=bot.api_url,
                timeout=settings.health.request_timeout_seconds,
            )

        return self._connectors[bot.id]

    def _trade_to_dict(self, trade) -> dict:
        """Convert trade object to dict for broadcasting."""
        return {
            "trade_id": trade.trade_id,
            "pair": trade.pair,
            "is_short": trade.is_short,
            "open_date": trade.open_date,
            "open_rate": trade.open_rate,
            "stake_amount": trade.stake_amount,
            "amount": trade.amount,
            "current_profit_pct": getattr(trade, "profit_pct", None),
            "current_profit_abs": getattr(trade, "profit_abs", None),
            "stop_loss": getattr(trade, "stop_loss", None),
            "take_profit": getattr(trade, "take_profit", None),
            "leverage": trade.leverage,
            "enter_tag": trade.enter_tag,
        }


# Singleton instance
trade_monitor = TradeMonitor()
