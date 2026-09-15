"""Log monitoring service for rate limit detection.

Only monitors running trading bots (not hyperopt/backtest).
Tracks active rate limits with auto-expiration.
"""

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import structlog
from sqlalchemy import select

from src.config import settings
from src.models import async_session_maker
from src.models.bot import Bot, HealthState
from src.services.aggregator import is_portfolio_bot

logger = structlog.get_logger()

# Rate limit detection patterns (case insensitive)
# These patterns are designed to catch actual rate limit ERRORS, not config settings
RATE_LIMIT_PATTERNS = [
    r"HTTP.*429",  # HTTP 429 status code in response
    r"status.*429",  # Status 429
    r"code.*429",  # Error code 429
    r"rate.?limit.*(hit|exceed|error|reach|block)",  # Rate limit hit/exceeded/error
    r"(hit|exceed|reach).*rate.?limit",  # Hit rate limit
    r"too.?many.?requests",  # Too many requests error
    r"RateLimitExceeded",  # Exception name
    r"DDos.?Guard",  # DDoS Guard blocks
    r"RetryAfter",  # Retry-After header (rate limit response)
    r"request.?limit.*(exceed|reach|hit)",  # Request limit exceeded
    r"API.?rate.?limit",  # API rate limit (but not enableRateLimit config)
    r"throttl.*(request|api|call)",  # Throttled requests
    r"banned.*(IP|temporarily)",  # IP banned due to rate limit
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in RATE_LIMIT_PATTERNS]


class ActiveRateLimit:
    """Tracks an active rate limit for a bot."""

    def __init__(self, bot_id: str, bot_name: str, exchange: str | None, context: str):
        self.bot_id = bot_id
        self.bot_name = bot_name
        self.exchange = exchange
        self.context = context
        self.first_seen = datetime.utcnow()
        self.last_seen = datetime.utcnow()
        self.occurrence_count = 1

    def update(self, context: str):
        """Update with new occurrence."""
        self.last_seen = datetime.utcnow()
        self.context = context
        self.occurrence_count += 1

    def is_expired(self, timeout_minutes: int = 10) -> bool:
        """Check if this rate limit has expired (no new occurrences)."""
        return datetime.utcnow() - self.last_seen > timedelta(minutes=timeout_minutes)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "bot_id": self.bot_id,
            "bot_name": self.bot_name,
            "exchange": self.exchange,
            "context": self.context[:200],
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "occurrence_count": self.occurrence_count,
            "age_seconds": (datetime.utcnow() - self.first_seen).total_seconds(),
        }


class LogMonitor:
    """Monitor bot logs for rate limit events.

    Only monitors running trading bots (healthy/degraded, not hyperopt/backtest).
    Uses in-memory tracking with auto-expiration for active rate limits.
    """

    def __init__(self, check_interval: int = 30, expiration_minutes: int = 10):
        """Initialize log monitor.

        Args:
            check_interval: Seconds between log checks.
            expiration_minutes: Minutes after which inactive rate limits expire.
        """
        self._check_interval = check_interval
        self._expiration_minutes = expiration_minutes
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # In-memory tracking of active rate limits (bot_id -> ActiveRateLimit)
        self._active_rate_limits: dict[str, ActiveRateLimit] = {}

        # Track last check time per bot for incremental log reading
        self._last_check_time: dict[str, datetime] = {}
        self._last_file_position: dict[str, int] = {}

        # Docker client (lazy init)
        self._docker_client = None

    def _get_docker_client(self):
        """Lazy initialization of Docker client."""
        if self._docker_client is None:
            try:
                import docker
                socket_path = settings.discovery.docker.socket
                self._docker_client = docker.DockerClient(base_url=socket_path)
            except Exception as e:
                logger.debug("Failed to initialize Docker client for log monitoring", error=str(e))
                self._docker_client = None
        return self._docker_client

    async def start(self) -> None:
        """Start the log monitoring background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Log monitor started",
                   check_interval=self._check_interval,
                   expiration_minutes=self._expiration_minutes)

    async def stop(self) -> None:
        """Stop the log monitoring background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Log monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_bots()
                self._cleanup_expired()
            except Exception as e:
                logger.error("Error in log monitor loop", error=str(e))

            await asyncio.sleep(self._check_interval)

    def _cleanup_expired(self) -> None:
        """Remove expired rate limits."""
        expired_ids = [
            bot_id for bot_id, rl in self._active_rate_limits.items()
            if rl.is_expired(self._expiration_minutes)
        ]
        for bot_id in expired_ids:
            bot_name = self._active_rate_limits[bot_id].bot_name
            del self._active_rate_limits[bot_id]
            logger.info("Rate limit expired", bot_id=bot_id, bot_name=bot_name)

            # Broadcast expiration via WebSocket
            asyncio.create_task(self._broadcast_rate_limit_cleared(bot_id, bot_name))

    async def _check_all_bots(self) -> None:
        """Check logs for all running trading bots."""
        async with async_session_maker() as db:
            # Get all bots
            result = await db.execute(select(Bot))
            all_bots = list(result.scalars())

            # Filter to only running portfolio bots (trading, not hyperopt/backtest)
            running_bots = [
                bot for bot in all_bots
                if is_portfolio_bot(bot) and bot.health_state in [HealthState.HEALTHY, HealthState.DEGRADED]
            ]

            for bot in running_bots:
                try:
                    await self._check_bot_logs(bot)
                except Exception as e:
                    logger.debug("Error checking logs for bot", bot_id=bot.id, error=str(e))

    async def _check_bot_logs(self, bot: Bot) -> None:
        """Check logs for a specific bot (only new entries since last check).

        Args:
            bot: Bot to check logs for.
        """
        new_logs = None

        # Try Docker logs first
        if bot.container_id and bot.environment.value == "docker":
            new_logs = await self._get_docker_logs_since(bot.id, bot.container_id)

        # Try filesystem logs
        if not new_logs and bot.user_data_path:
            new_logs = await self._get_filesystem_logs_since(bot.id, bot.user_data_path)

        if not new_logs:
            return

        # Check for rate limit patterns in NEW logs only
        rate_limit_context = self._check_for_rate_limits(new_logs)

        if rate_limit_context:
            self._record_rate_limit(bot, rate_limit_context)

    async def _get_docker_logs_since(self, bot_id: str, container_id: str) -> Optional[str]:
        """Get NEW logs from a Docker container since last check.

        Args:
            bot_id: Bot ID for tracking.
            container_id: Docker container ID.

        Returns:
            New log contents or None.
        """
        client = self._get_docker_client()
        if not client:
            return None

        try:
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                None,
                lambda: client.containers.get(container_id)
            )

            # Check if container is running
            if container.status != "running":
                return None

            # Get logs since last check (or last 30 seconds for first check)
            last_check = self._last_check_time.get(bot_id)
            if last_check:
                since = last_check
            else:
                since = datetime.utcnow() - timedelta(seconds=30)

            # Update last check time
            self._last_check_time[bot_id] = datetime.utcnow()

            logs = await loop.run_in_executor(
                None,
                lambda: container.logs(since=since, tail=50).decode('utf-8', errors='ignore')
            )
            return logs if logs.strip() else None

        except Exception as e:
            logger.debug("Failed to get Docker logs", container_id=container_id[:12], error=str(e))
            return None

    async def _get_filesystem_logs_since(self, bot_id: str, user_data_path: str) -> Optional[str]:
        """Get NEW logs from filesystem since last check.

        Args:
            bot_id: Bot ID for tracking.
            user_data_path: Path to user_data directory.

        Returns:
            New log contents or None.
        """
        try:
            log_dir = Path(user_data_path) / "logs"
            if not log_dir.exists():
                return None

            # Find most recent log file
            log_files = sorted(
                log_dir.glob("*.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if not log_files:
                return None

            log_file = log_files[0]
            current_size = log_file.stat().st_size

            # Get last read position
            last_position = self._last_file_position.get(bot_id, 0)

            # If file was truncated/rotated, reset position
            if current_size < last_position:
                last_position = 0

            # No new content
            if current_size <= last_position:
                return None

            loop = asyncio.get_event_loop()

            def read_new_content():
                with open(log_file, 'r', errors='ignore') as f:
                    f.seek(last_position)
                    # Read at most 50KB of new content
                    content = f.read(50000)
                    return content, f.tell()

            content, new_position = await loop.run_in_executor(None, read_new_content)
            self._last_file_position[bot_id] = new_position

            return content if content.strip() else None

        except Exception as e:
            logger.debug("Failed to get filesystem logs", path=user_data_path, error=str(e))
            return None

    def _check_for_rate_limits(self, logs: str) -> Optional[str]:
        """Check log content for rate limit patterns.

        Args:
            logs: Log content to check.

        Returns:
            Matched context or None.
        """
        for pattern in COMPILED_PATTERNS:
            match = pattern.search(logs)
            if match:
                # Get some context around the match
                start = max(0, match.start() - 30)
                end = min(len(logs), match.end() + 70)
                context = logs[start:end].replace('\n', ' ').strip()
                return context
        return None

    def _record_rate_limit(self, bot: Bot, context: str) -> None:
        """Record a rate limit occurrence for a bot.

        Args:
            bot: Bot that hit rate limit.
            context: Log context showing the rate limit.
        """
        if bot.id in self._active_rate_limits:
            # Update existing
            self._active_rate_limits[bot.id].update(context)
            logger.debug("Rate limit updated", bot_id=bot.id, bot_name=bot.name,
                        count=self._active_rate_limits[bot.id].occurrence_count)
        else:
            # New rate limit
            self._active_rate_limits[bot.id] = ActiveRateLimit(
                bot_id=bot.id,
                bot_name=bot.name,
                exchange=bot.exchange,
                context=context,
            )
            logger.warning("New rate limit detected", bot_id=bot.id, bot_name=bot.name, exchange=bot.exchange)

            # Broadcast via WebSocket
            asyncio.create_task(self._broadcast_rate_limit_alert(bot, context))

    async def _broadcast_rate_limit_alert(self, bot: Bot, context: str) -> None:
        """Broadcast rate limit alert via WebSocket."""
        try:
            from src.services.websocket import ws_manager

            await ws_manager.broadcast({
                "type": "rate_limit_alert",
                "bot_id": bot.id,
                "bot_name": bot.name,
                "exchange": bot.exchange,
                "message": f"Rate limit detected: {bot.name}",
                "context": context[:100],
                "timestamp": datetime.utcnow().isoformat() + 'Z',
            })
        except Exception as e:
            logger.debug("Failed to broadcast rate limit alert", error=str(e))

    async def _broadcast_rate_limit_cleared(self, bot_id: str, bot_name: str) -> None:
        """Broadcast rate limit cleared via WebSocket."""
        try:
            from src.services.websocket import ws_manager

            await ws_manager.broadcast({
                "type": "rate_limit_cleared",
                "bot_id": bot_id,
                "bot_name": bot_name,
                "message": f"Rate limit cleared: {bot_name}",
                "timestamp": datetime.utcnow().isoformat() + 'Z',
            })
        except Exception as e:
            logger.debug("Failed to broadcast rate limit cleared", error=str(e))

    def get_active_rate_limits(self) -> list[dict]:
        """Get currently active (non-expired) rate limits.

        Returns:
            List of active rate limit info.
        """
        # Clean up expired first
        self._cleanup_expired()

        return [rl.to_dict() for rl in self._active_rate_limits.values()]

    def get_rate_limit_count(self) -> int:
        """Get count of active rate limits."""
        self._cleanup_expired()
        return len(self._active_rate_limits)

    def has_active_rate_limits(self) -> bool:
        """Check if there are any active rate limits."""
        self._cleanup_expired()
        return len(self._active_rate_limits) > 0

    def clear_rate_limit(self, bot_id: str) -> bool:
        """Manually clear a rate limit for a bot.

        Args:
            bot_id: Bot ID to clear.

        Returns:
            True if cleared, False if not found.
        """
        if bot_id in self._active_rate_limits:
            del self._active_rate_limits[bot_id]
            return True
        return False


# Singleton instance
log_monitor = LogMonitor(check_interval=30, expiration_minutes=10)
