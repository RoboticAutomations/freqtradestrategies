"""Periodic discovery scheduler for background bot discovery."""

import asyncio
from typing import Optional

import structlog

from src.config import settings
from src.models import async_session_maker

logger = structlog.get_logger()


class DiscoveryScheduler:
    """Background scheduler for periodic bot discovery."""

    def __init__(self):
        """Initialize the discovery scheduler."""
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._interval = settings.discovery.interval_seconds

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    async def start(self) -> None:
        """Start the periodic discovery task."""
        if self._running:
            logger.warning("Discovery scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Discovery scheduler started", interval=self._interval)

    async def stop(self) -> None:
        """Stop the periodic discovery task."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Discovery scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        from src.services.discovery.orchestrator import discovery_orchestrator

        # Run initial discovery
        await self._run_discovery()

        # Then run periodically
        while self._running:
            try:
                await asyncio.sleep(self._interval)

                if not self._running:
                    break

                await self._run_discovery()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Discovery scheduler error", error=str(e))
                # Continue running even on error
                await asyncio.sleep(10)

    async def _run_discovery(self) -> None:
        """Execute a single discovery scan."""
        from src.services.discovery.orchestrator import discovery_orchestrator

        try:
            async with async_session_maker() as session:
                result = await discovery_orchestrator.discover_all(session)
                logger.info(
                    "Scheduled discovery completed",
                    discovered=result["discovered"],
                    new=result["new"],
                    updated=result["updated"],
                )
        except Exception as e:
            logger.error("Discovery scan failed", error=str(e))

    async def trigger_manual_scan(self) -> dict:
        """Trigger an immediate discovery scan.

        Returns:
            Discovery result summary.
        """
        from src.services.discovery.orchestrator import discovery_orchestrator

        logger.info("Manual discovery scan triggered")

        async with async_session_maker() as session:
            return await discovery_orchestrator.discover_all(session)


# Singleton instance
discovery_scheduler = DiscoveryScheduler()
