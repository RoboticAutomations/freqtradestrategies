"""Simple in-memory caching service with TTL support."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """Cache entry with value and expiration."""

    value: Any
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


class CacheService:
    """In-memory cache with TTL support and automatic cleanup."""

    def __init__(self, default_ttl_seconds: int = 60):
        """Initialize cache service.

        Args:
            default_ttl_seconds: Default TTL for cache entries.
        """
        self._cache: dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl_seconds
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start background cleanup task."""
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Cache service started", default_ttl=self._default_ttl)

    async def stop(self) -> None:
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        self._cache.clear()
        logger.info("Cache service stopped")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired entries."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cache cleanup error", error=str(e))

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.debug("Cache cleanup", removed_count=len(expired_keys))

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self._cache[key]
            return None
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl_seconds: TTL in seconds (uses default if not specified).
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> bool:
        """Delete entry from cache.

        Args:
            key: Cache key.

        Returns:
            True if entry was deleted, False if not found.
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def delete_pattern(self, pattern: str) -> int:
        """Delete entries matching a pattern.

        Args:
            pattern: Pattern to match (supports * wildcard at end).

        Returns:
            Number of entries deleted.
        """
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            keys_to_delete = [
                key for key in self._cache.keys()
                if key.startswith(prefix)
            ]
        else:
            keys_to_delete = [pattern] if pattern in self._cache else []

        for key in keys_to_delete:
            del self._cache[key]

        return len(keys_to_delete)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._cache)


# Singleton instance with 30-second default TTL for metrics
cache = CacheService(default_ttl_seconds=30)


# Cache key generators
def bot_metrics_key(bot_id: str) -> str:
    """Generate cache key for bot metrics."""
    return f"bot_metrics:{bot_id}"


def bot_health_key(bot_id: str) -> str:
    """Generate cache key for bot health."""
    return f"bot_health:{bot_id}"


def bots_list_key(filters_hash: str = "") -> str:
    """Generate cache key for bots list."""
    return f"bots_list:{filters_hash}"


def portfolio_key() -> str:
    """Generate cache key for portfolio summary."""
    return "portfolio_summary"
