"""
Discovery orchestrator combining Docker and filesystem discovery.
V9: Added discovery_host_ip setting support
"""

from datetime import datetime


def utc_naive_now() -> datetime:
    """UTC timestamp without tzinfo (safe for TIMESTAMP WITHOUT TIME ZONE columns)."""
    return datetime.utcnow()
from typing import Optional
from uuid import uuid4

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models.bot import Bot, BotEnvironment, HealthState, SourceMode
from src.models.settings import SystemSetting
from src.services.discovery import DiscoveryResult
from src.services.discovery.docker import DockerDiscovery
from src.services.discovery.filesystem import FilesystemDiscovery
from src.services.connectors.api import APIConnector
from src.services.websocket import ws_manager

logger = structlog.get_logger()


class DiscoveryOrchestrator:
    """Orchestrates bot discovery from multiple sources."""

    def __init__(self):
        """Initialize discovery orchestrator with all discovery sources."""
        self.docker_discovery = DockerDiscovery()
        self.filesystem_discovery = FilesystemDiscovery()
        self._last_scan: Optional[datetime] = None

    @property
    def last_scan(self) -> Optional[datetime]:
        """Get timestamp of last discovery scan."""
        return self._last_scan

    async def _get_discovery_host(self, db: AsyncSession) -> str:
        """Get discovery host IP from settings. Defaults to localhost."""
        from sqlalchemy import select
        result = await db.execute(select(SystemSetting).where(SystemSetting.key == "discovery_host_ip"))
        setting = result.scalar_one_or_none()
        if setting and setting.value:
            return setting.value
        return "localhost"

    async def _get_api_credentials(self, db: AsyncSession) -> tuple[str | None, str | None]:
        """Get configured Freqtrade API credentials from settings."""
        result = await db.execute(
            select(SystemSetting).where(SystemSetting.key.in_(("api_username", "api_password")))
        )
        settings_map = {setting.key: setting.value for setting in result.scalars().all()}
        return settings_map.get("api_username"), settings_map.get("api_password")

    async def _enrich_result_from_api(
        self,
        result: DiscoveryResult,
        username: str | None,
        password: str | None,
    ) -> DiscoveryResult:
        """Fill missing metadata from the live bot API when possible."""
        if not result.api_url or (result.exchange and result.strategy):
            return result
        if not username or not password:
            return result

        connector = APIConnector(
            bot_id=result.container_id or result.name,
            api_url=result.api_url,
            username=username,
            password=password,
        )

        try:
            status_result = await connector.get_status()
            if not status_result.success or not status_result.data:
                return result

            bot_status = status_result.data
            if not result.exchange:
                result.exchange = bot_status.exchange
            if not result.strategy:
                result.strategy = bot_status.strategy
            result.is_dryrun = bot_status.is_dryrun
            return result
        except Exception as exc:
            logger.debug("API enrichment failed", api_url=result.api_url, error=str(exc))
            return result
        finally:
            await connector.close()

    async def discover_all(self, db: AsyncSession) -> dict:
        """Run discovery across all sources and update database.

        Args:
            db: Database session for persisting results.

        Returns:
            Summary of discovery results.
        """
        logger.info("Starting discovery scan")
        self._last_scan = datetime.utcnow()

        # Get discovery host IP from settings
        discovery_host = await self._get_discovery_host(db)
        api_username, api_password = await self._get_api_credentials(db)
        logger.info("Using discovery host", host=discovery_host)

        all_results: list[DiscoveryResult] = []

        # Run Docker discovery with host IP
        if await self.docker_discovery.is_available():
            docker_results = await self.docker_discovery.discover(host=discovery_host)
            all_results.extend(docker_results)
            logger.info("Docker discovery found bots", count=len(docker_results))

        # Run filesystem discovery
        if await self.filesystem_discovery.is_available():
            fs_results = await self.filesystem_discovery.discover()
            all_results.extend(fs_results)
            logger.info("Filesystem discovery found bots", count=len(fs_results))

        enriched_results: list[DiscoveryResult] = []
        for result in all_results:
            enriched_results.append(
                await self._enrich_result_from_api(result, api_username, api_password)
            )

        # Process results and update database
        summary = await self._process_results(db, enriched_results)

        logger.info(
            "Discovery scan completed",
            discovered=summary["discovered"],
            new=summary["new"],
            updated=summary["updated"],
            removed=summary["removed"],
        )

        return summary

    async def get_status(self) -> dict:
        """Return current discovery service status."""
        docker_available = await self.docker_discovery.is_available()
        filesystem_available = await self.filesystem_discovery.is_available()
        scan_interval_seconds = settings.discovery.interval_seconds
        next_scan = None

        if self._last_scan:
            from datetime import timedelta

            next_scan = self._last_scan + timedelta(seconds=scan_interval_seconds)

        return {
            "docker_enabled": True,
            "docker_available": docker_available,
            "filesystem_enabled": True,
            "filesystem_available": filesystem_available,
            "last_scan": self._last_scan,
            "scan_interval_seconds": scan_interval_seconds,
            "next_scan": next_scan,
        }

    async def _process_results(
        self, db: AsyncSession, results: list[DiscoveryResult]
    ) -> dict:
        """Process discovery results and sync with database.

        Args:
            db: Database session.
            results: List of discovery results.

        Returns:
            Summary dict with counts.
        """
        summary = {
            "discovered": len(results),
            "new": 0,
            "updated": 0,
            "removed": 0,
            "errors": 0,
        }

        # Get existing bots for comparison
        existing_bots = await db.execute(select(Bot))
        existing_bots = existing_bots.scalars().all()
        
        # Map by container_id and api_url for deduplication
        container_map: dict[str, Bot] = {}
        url_map: dict[str, Bot] = {}
        
        for bot in existing_bots:
            if bot.container_id:
                container_map[bot.container_id] = bot
            if bot.api_url:
                url_map[bot.api_url] = bot

        # Process each discovered result
        processed_ids = set()
        
        for result in results:
            try:
                # Skip if already processed this URL
                if result.api_url and result.api_url in processed_ids:
                    continue
                if result.api_url:
                    processed_ids.add(result.api_url)

                # Check for existing bot
                existing = None
                if result.container_id and result.container_id in container_map:
                    existing = container_map[result.container_id]
                elif result.api_url and result.api_url in url_map:
                    existing = url_map[result.api_url]

                if existing:
                    # Update existing bot
                    await self._update_bot(existing, result, db)
                    summary["updated"] += 1
                else:
                    # Create new bot
                    await self._create_bot(result, db)
                    summary["new"] += 1

            except Exception as e:
                logger.error("Failed to process discovery result", error=str(e))
                summary["errors"] += 1

        # Commit all changes
        await db.commit()

        # Notify via WebSocket
        await ws_manager.broadcast({
            "type": "discovery_complete",
            "data": summary
        })

        return summary

    async def _create_bot(self, result: DiscoveryResult, db: AsyncSession) -> Bot:
        """Create a new bot from discovery result."""
        bot = Bot(
            id=str(uuid4()),
            name=result.name,
            environment=BotEnvironment(result.environment),
            host=result.host,
            container_id=result.container_id,
            user_data_path=result.user_data_path,
            api_url=result.api_url,
            api_port=result.api_port,
            exchange=result.exchange,
            strategy=result.strategy,
            is_dryrun=result.is_dryrun,
            health_state=HealthState.UNKNOWN,
            source_mode=SourceMode.AUTO,
        )
        db.add(bot)
        return bot

    async def _update_bot(self, bot: Bot, result: DiscoveryResult, db: AsyncSession) -> None:
        """Update existing bot with discovery result."""
        # Update fields that may have changed
        bot.host = result.host
        bot.container_id = result.container_id
        bot.user_data_path = result.user_data_path
        bot.api_url = result.api_url
        bot.api_port = result.api_port
        bot.exchange = result.exchange
        bot.strategy = result.strategy
        bot.is_dryrun = result.is_dryrun
        bot.last_seen = utc_naive_now()


# Singleton instance for global access
discovery_orchestrator = DiscoveryOrchestrator()
