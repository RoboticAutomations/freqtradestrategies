"""Health monitoring service with hysteresis-based state transitions."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta


def utc_naive_now() -> datetime:
    """UTC timestamp without tzinfo (safe for TIMESTAMP WITHOUT TIME ZONE columns)."""
    return datetime.utcnow()
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models import async_session_maker
from src.models.bot import Bot, HealthState, SourceMode
from src.models.alert import Alert, AlertType, AlertSeverity
from src.models.metrics import BotMetrics
from src.services.connectors.api import APIConnector
from src.services.connectors.base import ConnectorResult
from src.services.websocket import ws_manager

logger = structlog.get_logger()


@dataclass
class HealthMetrics:
    """Sliding window health metrics for a single bot."""

    bot_id: str
    window_size: int = 20  # Larger window = more tolerant
    success_threshold: float = 0.6  # 60% success rate for healthy (was 80%)
    degraded_threshold: float = 0.3  # 30% success rate for degraded (was 50%)

    # Sliding window of recent health checks
    api_results: deque = field(default_factory=lambda: deque(maxlen=10))
    sqlite_results: deque = field(default_factory=lambda: deque(maxlen=10))

    # Latency tracking
    api_latencies: deque = field(default_factory=lambda: deque(maxlen=10))
    sqlite_latencies: deque = field(default_factory=lambda: deque(maxlen=10))

    # State tracking
    current_state: HealthState = HealthState.HEALTHY
    current_source: SourceMode = SourceMode.API
    last_check: Optional[datetime] = None
    state_changed_at: Optional[datetime] = None

    # Hysteresis counters (require N consecutive states to transition)
    consecutive_healthy: int = 0
    consecutive_degraded: int = 0
    consecutive_unreachable: int = 0

    def record_api_check(self, result: ConnectorResult) -> None:
        """Record an API health check result."""
        self.api_results.append(result.success)
        if result.success:
            self.api_latencies.append(result.latency_ms)
        self.last_check = datetime.utcnow()

    def record_sqlite_check(self, result: ConnectorResult) -> None:
        """Record a SQLite health check result."""
        self.sqlite_results.append(result.success)
        if result.success:
            self.sqlite_latencies.append(result.latency_ms)

    @property
    def api_success_rate(self) -> float:
        """Calculate API success rate from sliding window."""
        if not self.api_results:
            return 0.0
        return sum(self.api_results) / len(self.api_results)

    @property
    def sqlite_success_rate(self) -> float:
        """Calculate SQLite success rate from sliding window."""
        if not self.sqlite_results:
            return 0.0
        return sum(self.sqlite_results) / len(self.sqlite_results)

    @property
    def api_avg_latency(self) -> float:
        """Calculate average API latency."""
        if not self.api_latencies:
            return 0.0
        return sum(self.api_latencies) / len(self.api_latencies)

    @property
    def sqlite_avg_latency(self) -> float:
        """Calculate average SQLite latency."""
        if not self.sqlite_latencies:
            return 0.0
        return sum(self.sqlite_latencies) / len(self.sqlite_latencies)

    @property
    def api_available(self) -> bool:
        """Check if API is available based on recent success rate."""
        return self.api_success_rate >= self.degraded_threshold

    @property
    def last_api_error(self) -> Optional[str]:
        """Return last API error if any."""
        if self.api_results and not self.api_results[-1]:
            return "API check failed"
        return None

    def evaluate_state(self) -> tuple[HealthState, SourceMode]:
        """Evaluate health state - simplified: only HEALTHY or UNREACHABLE.

        Returns:
            Tuple of (new_health_state, recommended_source).
        """
        api_rate = self.api_success_rate
        sqlite_rate = self.sqlite_success_rate

        # Simplified: Either healthy (can connect) or unreachable (can't connect)
        # No more "degraded" state that hides bots from summary
        if api_rate >= self.success_threshold or sqlite_rate >= self.success_threshold:
            target_state = HealthState.HEALTHY
            target_source = SourceMode.API if api_rate >= sqlite_rate else SourceMode.SQLITE
            self.consecutive_healthy += 1
            self.consecutive_unreachable = 0
        else:
            target_state = HealthState.UNREACHABLE
            target_source = SourceMode.API
            self.consecutive_unreachable += 1
            self.consecutive_healthy = 0

        # Apply hysteresis - require 3 consecutive checks to change state
        hysteresis_count = 3
        new_state = self.current_state
        new_source = self.current_source

        if target_state == HealthState.HEALTHY and self.consecutive_healthy >= hysteresis_count:
            new_state = HealthState.HEALTHY
            new_source = target_source

        elif target_state == HealthState.UNREACHABLE and self.consecutive_unreachable >= hysteresis_count:
            new_state = HealthState.UNREACHABLE
            new_source = target_source

        # Track state changes
        if new_state != self.current_state:
            self.state_changed_at = datetime.utcnow()
            logger.info(
                "Bot health state changed",
                bot_id=self.bot_id,
                old_state=self.current_state.value,
                new_state=new_state.value,
                api_rate=api_rate,
                sqlite_rate=sqlite_rate,
            )

        self.current_state = new_state
        self.current_source = new_source

        return new_state, new_source


class HealthMonitor:
    """Background health monitoring service."""

    def __init__(self):
        """Initialize health monitor."""
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._metrics: dict[str, HealthMetrics] = {}
        self._connectors: dict[str, APIConnector] = {}
        self._interval = settings.health.check_interval_seconds

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    def get_metrics(self, bot_id: str) -> Optional[HealthMetrics]:
        """Get health metrics for a bot."""
        return self._metrics.get(bot_id)

    async def start(self) -> None:
        """Start the health monitoring task."""
        if self._running:
            logger.warning("Health monitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Health monitor started", interval=self._interval)

    async def stop(self) -> None:
        """Stop the health monitoring task."""
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

        logger.info("Health monitor stopped")

    async def _run_loop(self) -> None:
        """Main monitoring loop."""
        # Initial check after short delay
        await asyncio.sleep(5)
        await self._check_all_bots()

        while self._running:
            try:
                await asyncio.sleep(self._interval)

                if not self._running:
                    break

                await self._check_all_bots()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error", error=str(e))
                await asyncio.sleep(10)

    async def _check_all_bots(self) -> None:
        """Check health of all registered bots."""
        try:
            async with async_session_maker() as session:
                # Get all bots
                result = await session.execute(select(Bot))
                bots = result.scalars().all()

                for bot in bots:
                    await self._check_bot(session, bot)

                await session.commit()

        except Exception as e:
            logger.error("Health check batch failed", error=str(e))

    async def _check_bot(self, session: AsyncSession, bot: Bot) -> None:
        """Check health of a single bot."""
        try:
            # Get or create metrics tracker
            if bot.id not in self._metrics:
                self._metrics[bot.id] = HealthMetrics(bot_id=bot.id)
            metrics = self._metrics[bot.id]

            # Check API if URL available
            bot_metrics_data = None
            if bot.api_url:
                connector = await self._get_api_connector(bot, session)
                if connector:
                    api_result = await connector.check_health()
                    metrics.record_api_check(api_result)

                    # Fetch, save, and broadcast live metrics if healthy
                    if api_result.success:
                        bot_metrics_data = await self._fetch_and_broadcast_metrics(bot, connector, session)

            # Check SQLite if path available
            if bot.user_data_path:
                sqlite_result = await self._check_sqlite(bot)
                metrics.record_sqlite_check(sqlite_result)

            # Evaluate and update state
            new_state, new_source = metrics.evaluate_state()

            # Update bot record if state changed
            old_state = bot.health_state
            if bot.health_state != new_state or bot.source_mode != new_source:
                bot.health_state = new_state
                if bot.source_mode == SourceMode.AUTO:
                    # Only update if in auto mode
                    pass  # Source selection happens at query time

                # Create alert for state change
                if old_state != new_state:
                    await self._create_health_alert(session, bot, old_state, new_state)

                # Broadcast health update via WebSocket
                await ws_manager.broadcast_bot_update(
                    bot_id=bot.id,
                    event_type="health_update",
                    data={
                        "health_state": new_state.value,
                        "previous_state": old_state.value,
                        "source_mode": new_source.value,
                        "api_success_rate": metrics.api_success_rate,
                        "sqlite_success_rate": metrics.sqlite_success_rate,
                    },
                )

            bot.last_seen = utc_naive_now()

        except Exception as e:
            logger.error(
                "Bot health check failed",
                bot_id=bot.id,
                error=str(e),
            )

    async def _fetch_and_broadcast_metrics(self, bot: Bot, connector: APIConnector, session: AsyncSession) -> Optional[dict]:
        """Fetch bot metrics, save to database, and broadcast via WebSocket.

        Args:
            bot: Bot to fetch metrics for.
            connector: API connector to use.
            session: Database session for persisting metrics.

        Returns:
            Metrics dict if successful, None otherwise.
        """
        try:
            # Fetch profit data
            profit_result = await connector.get_profit()
            if not profit_result.success:
                return None

            profit = profit_result.data

            # Fetch balance
            balance_result = await connector.get_balance()
            balance = balance_result.data.stake_currency_balance if balance_result.success and balance_result.data else 0

            # Fetch open trades
            trades_result = await connector.get_open_trades()
            open_trades = len(trades_result.data) if trades_result.success and trades_result.data else 0

            # Calculate win rate
            total_closed = profit.winning_trades + profit.losing_trades
            win_rate = profit.winning_trades / total_closed if total_closed > 0 else 0

            metrics_data = {
                "profit_abs": profit.profit_all_coin,
                "profit_pct": profit.profit_all_percent,
                "profit_today": profit.profit_closed_coin,
                "balance": balance,
                "open_trades": open_trades,
                "closed_trades": profit.closed_trade_count,
                "winning_trades": profit.winning_trades,
                "losing_trades": profit.losing_trades,
                "win_rate": win_rate,
            }

            # Save metrics to database for persistence
            try:
                bot_metrics = BotMetrics(
                    bot_id=bot.id,
                    timestamp=datetime.utcnow(),
                    profit_abs=profit.profit_all_coin,
                    profit_pct=profit.profit_all_percent,
                    balance=balance,
                    open_positions=open_trades,
                    closed_trades=profit.closed_trade_count,
                    win_rate=win_rate if total_closed > 0 else None,
                    data_source=SourceMode.API,
                )
                session.add(bot_metrics)
                await session.commit()
                logger.debug("Metrics saved to database", bot_id=bot.id)
            except Exception as db_error:
                logger.warning("Failed to save metrics to database", bot_id=bot.id, error=str(db_error))
                await session.rollback()

            # Broadcast metrics update
            await ws_manager.broadcast_bot_update(
                bot_id=bot.id,
                event_type="metrics_update",
                data=metrics_data,
            )

            return metrics_data

        except Exception as e:
            logger.debug("Failed to fetch metrics for broadcast", bot_id=bot.id, error=str(e))
            return None

    async def _get_api_connector(self, bot: Bot, session: AsyncSession) -> Optional[APIConnector]:
        """Get or create API connector for bot."""
        if not bot.api_url:
            return None

        if bot.id not in self._connectors:
            # Fetch API credentials from settings
            from src.models.settings import SystemSetting
            from sqlalchemy import select
            username_result = await session.execute(select(SystemSetting).where(SystemSetting.key == "api_username"))
            password_result = await session.execute(select(SystemSetting).where(SystemSetting.key == "api_password"))
            
            username_setting = username_result.scalar_one_or_none()
            password_setting = password_result.scalar_one_or_none()
            
            username = username_setting.value if username_setting else None
            password = password_setting.value if password_setting else None
            
            logger.info(
                "API credentials loaded",
                bot_id=bot.id,
                has_username=bool(username),
                has_password=bool(password),
            )
            
            self._connectors[bot.id] = APIConnector(
                bot_id=bot.id,
                api_url=bot.api_url,
                username=username,
                password=password,
                timeout=settings.health.request_timeout_seconds,
            )

        return self._connectors[bot.id]

    async def _check_sqlite(self, bot: Bot) -> ConnectorResult:
        """Check SQLite database availability."""
        import os

        if not bot.user_data_path:
            return ConnectorResult(success=False, error="No database path")

        db_path = os.path.join(bot.user_data_path, "tradesv3.sqlite")

        if not os.path.exists(db_path):
            return ConnectorResult(success=False, error="Database not found")

        # Just check file exists and is readable
        try:
            with open(db_path, "rb") as f:
                f.read(16)  # Read SQLite header
            return ConnectorResult(success=True)
        except Exception as e:
            return ConnectorResult(success=False, error=str(e))

    async def _create_health_alert(
        self,
        session: AsyncSession,
        bot: Bot,
        old_state: HealthState,
        new_state: HealthState,
    ) -> None:
        """Create an alert for bot health state change.

        Note: Alerts are optional. If DISABLE_ALERTS=true (default), we skip
        creating Alert rows entirely to avoid failing startup when the alerts
        feature/migration is not deployed.
        """
        import os

        if os.getenv("DISABLE_ALERTS", "true").lower() == "true":
            return

        # Determine alert type and severity
        if new_state == HealthState.UNREACHABLE:
            alert_type = AlertType.BOT_OFFLINE
            severity = AlertSeverity.CRITICAL
            title = f"Bot Offline: {bot.name}"
            message = f"Bot '{bot.name}' is unreachable. Previous state: {old_state.value}."
        elif new_state == HealthState.DEGRADED:
            alert_type = AlertType.BOT_DEGRADED
            severity = AlertSeverity.WARNING
            title = f"Bot Degraded: {bot.name}"
            message = f"Bot '{bot.name}' is experiencing issues. Using fallback data source."
        elif new_state == HealthState.HEALTHY and old_state in [HealthState.UNREACHABLE, HealthState.DEGRADED]:
            alert_type = AlertType.BOT_ONLINE
            severity = AlertSeverity.INFO
            title = f"Bot Online: {bot.name}"
            message = f"Bot '{bot.name}' has recovered and is now healthy."
        else:
            # No alert needed for other transitions
            return

        # Create the alert (best-effort; don't break monitoring if alerts table is missing)
        try:
            alert = Alert(
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                bot_id=bot.id,
                bot_name=bot.name,
            )
            session.add(alert)

            logger.info(
                "Health alert created",
                bot_id=bot.id,
                bot_name=bot.name,
                alert_type=alert_type.value,
                severity=severity.value,
            )
        except Exception as e:
            logger.warning(
                "Health alert skipped",
                bot_id=bot.id,
                bot_name=bot.name,
                error=str(e),
            )

    async def trigger_check(self, bot_id: str) -> Optional[HealthMetrics]:
        """Trigger an immediate health check for a specific bot.

        Args:
            bot_id: Bot UUID to check.

        Returns:
            Updated health metrics or None if bot not found.
        """
        try:
            async with async_session_maker() as session:
                result = await session.execute(
                    select(Bot).where(Bot.id == bot_id)
                )
                bot = result.scalar_one_or_none()

                if not bot:
                    return None

                await self._check_bot(session, bot)
                await session.commit()

                return self._metrics.get(bot_id)

        except Exception as e:
            logger.error(
                "Manual health check failed",
                bot_id=bot_id,
                error=str(e),
            )
            return None


# Singleton instance
health_monitor = HealthMonitor()
