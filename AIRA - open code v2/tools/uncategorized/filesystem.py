"""Filesystem-based bot discovery for bare-metal installations."""

import asyncio
import json
import os
from glob import glob
from pathlib import Path
from typing import Optional

import structlog

from src.config import settings
from src.services.discovery import BaseDiscovery, DiscoveryResult

logger = structlog.get_logger()


class FilesystemDiscovery(BaseDiscovery):
    """Discover Freqtrade bots by scanning filesystem paths."""

    def __init__(self):
        """Initialize filesystem discovery."""
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if filesystem discovery is enabled and paths exist."""
        if self._available is not None:
            return self._available

        if not settings.discovery.filesystem.enabled:
            self._available = False
            return False

        # Check if at least one scan path exists
        for path_pattern in settings.discovery.filesystem.scan_paths:
            expanded = os.path.expanduser(path_pattern)
            if glob(expanded):
                self._available = True
                logger.info("Filesystem discovery available", paths=len(settings.discovery.filesystem.scan_paths))
                return True

        logger.info("No filesystem scan paths found")
        self._available = False
        return False

    async def discover(self) -> list[DiscoveryResult]:
        """Discover Freqtrade bots by scanning configured paths.

        Looks for:
        1. SQLite database files (tradesv3.sqlite, tradesv3.dryrun.sqlite)
        2. config.json files
        3. Valid user_data directory structure

        Returns:
            List of discovered Freqtrade bot installations.
        """
        if not await self.is_available():
            return []

        results: list[DiscoveryResult] = []
        seen_paths: set[str] = set()

        loop = asyncio.get_event_loop()

        for path_pattern in settings.discovery.filesystem.scan_paths:
            try:
                expanded = os.path.expanduser(path_pattern)

                # Find all matching paths
                matching_paths = await loop.run_in_executor(None, glob, expanded)

                for user_data_path in matching_paths:
                    user_data_path = os.path.abspath(user_data_path)

                    if user_data_path in seen_paths:
                        continue

                    if not os.path.isdir(user_data_path):
                        continue

                    seen_paths.add(user_data_path)

                    # Try to create a discovery result
                    result = await self._scan_user_data_dir(user_data_path)
                    if result:
                        results.append(result)

            except Exception as e:
                logger.warning("Error scanning path pattern", pattern=path_pattern, error=str(e))

        logger.info("Filesystem discovery completed", discovered_count=len(results))
        return results

    async def _scan_user_data_dir(self, user_data_path: str) -> Optional[DiscoveryResult]:
        """Scan a user_data directory for Freqtrade bot.

        Args:
            user_data_path: Path to user_data directory.

        Returns:
            DiscoveryResult or None if not a valid bot installation.
        """
        try:
            # Check for SQLite databases
            sqlite_files = []
            for pattern in settings.discovery.filesystem.patterns:
                # Support glob patterns
                if '*' in pattern or '?' in pattern or '[' in pattern:
                    # Use glob for wildcard patterns
                    matches = glob(os.path.join(user_data_path, pattern))
                    sqlite_files.extend(matches)
                else:
                    db_path = os.path.join(user_data_path, pattern)
                    if os.path.exists(db_path):
                        sqlite_files.append(db_path)

            if not sqlite_files:
                return None

            # Prefer live database over dryrun
            sqlite_path = None
            is_dryrun = True
            for db_file in sqlite_files:
                if "dryrun" not in db_file.lower():
                    sqlite_path = db_file
                    is_dryrun = False
                    break
            if sqlite_path is None:
                sqlite_path = sqlite_files[0]
                is_dryrun = True

            # Try to read config
            config_data = await self._read_config(user_data_path)

            # Generate bot name from path
            parent_dir = Path(user_data_path).parent.name
            bot_name = config_data.get("bot_name") if config_data else None
            if not bot_name:
                bot_name = f"bot_{parent_dir}"

            # Extract info from config
            exchange = None
            strategy = None
            api_url = None
            api_port = None

            if config_data:
                exchange = config_data.get("exchange", {}).get("name")
                strategy = config_data.get("strategy")

                # Check for API server config
                api_server = config_data.get("api_server", {})
                if api_server.get("enabled"):
                    listen_ip = api_server.get("listen_ip_address", "127.0.0.1")
                    api_port = api_server.get("listen_port", 8080)
                    api_url = f"http://{listen_ip}:{api_port}"

            return DiscoveryResult(
                environment="baremetal",
                name=bot_name,
                host="localhost",
                user_data_path=user_data_path,
                api_url=api_url,
                api_port=api_port,
                api_available=api_url is not None,
                sqlite_path=sqlite_path,
                sqlite_available=True,
                config_data=config_data,
                exchange=exchange,
                strategy=strategy,
                is_dryrun=is_dryrun,
            )

        except Exception as e:
            logger.warning(
                "Failed to scan user_data directory",
                path=user_data_path,
                error=str(e),
            )
            return None

    async def _read_config(self, user_data_path: str) -> Optional[dict]:
        """Read and parse config.json from user_data directory.

        Args:
            user_data_path: Path to user_data directory.

        Returns:
            Parsed config dict or None.
        """
        # Look for config in parent directory (standard Freqtrade layout)
        parent_dir = Path(user_data_path).parent
        config_paths = [
            parent_dir / "config.json",
            parent_dir / "config_private.json",
            Path(user_data_path) / "config.json",
        ]

        for config_path in config_paths:
            try:
                if config_path.exists():
                    loop = asyncio.get_event_loop()
                    content = await loop.run_in_executor(
                        None, config_path.read_text
                    )
                    return json.loads(content)
            except Exception:
                continue

        return None
