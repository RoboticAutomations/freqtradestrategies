"""Docker-based bot discovery using Docker SDK."""

import asyncio
import fnmatch
import json
import re
from typing import Optional

import structlog

from src.config import settings
from src.services.discovery import BaseDiscovery, DiscoveryResult

logger = structlog.get_logger()


class DockerDiscovery(BaseDiscovery):
    """Discover Freqtrade bots running in Docker containers."""

    def __init__(self):
        """Initialize Docker discovery with client from settings."""
        self._client = None
        self._available: Optional[bool] = None

    def _get_client(self):
        """Lazy initialization of Docker client."""
        if self._client is None:
            try:
                import docker

                socket_path = settings.discovery.docker.socket
                self._client = docker.DockerClient(base_url=socket_path)
            except Exception as e:
                logger.warning("Failed to initialize Docker client", error=str(e))
                self._client = None
        return self._client

    async def is_available(self) -> bool:
        """Check if Docker is available."""
        if self._available is not None:
            return self._available

        if not settings.discovery.docker.enabled:
            self._available = False
            return False

        try:
            client = self._get_client()
            if client is None:
                self._available = False
                return False

            # Run ping in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, client.ping)
            self._available = True
            logger.info("Docker discovery available")
        except Exception as e:
            logger.warning("Docker not available", error=str(e))
            self._available = False

        return self._available

    async def discover(self, **kwargs) -> list[DiscoveryResult]:
        """Discover Freqtrade containers.

        Finds containers by:
        1. Matching image patterns (freqtrade*, freqtradeorg/freqtrade*, etc.)
        2. Matching configured labels (com.docker.compose.service, etc.)

        Args:
            **kwargs: Additional arguments including 'host' for API URL

        Returns:
            List of discovered Freqtrade bot containers.
        """
        if not await self.is_available():
            return []

        client = self._get_client()
        if client is None:
            return []

        results: list[DiscoveryResult] = []
        seen_ids: set[str] = set()

        loop = asyncio.get_event_loop()

        try:
            # Get only running containers for pattern matching
            # (Use all=False to exclude stopped/exited containers like backtest runs)
            all_containers = await loop.run_in_executor(
                None,
                lambda: client.containers.list(all=False),
            )

            # Find by image patterns using fnmatch for wildcard support
            for container in all_containers:
                if container.id in seen_ids:
                    continue

                try:
                    # Get image name - handle both image tags and IDs
                    image_tags = container.image.tags if container.image.tags else []
                    image_name = image_tags[0] if image_tags else str(container.image.id)[:12]

                    # Also get short image name without tag
                    image_short = image_name.split(":")[0] if ":" in image_name else image_name

                    # Check against patterns
                    for pattern in settings.discovery.docker.image_patterns:
                        if (fnmatch.fnmatch(image_name, pattern) or
                            fnmatch.fnmatch(image_short, pattern) or
                            fnmatch.fnmatch(image_name.lower(), pattern.lower())):
                            seen_ids.add(container.id)
                            result = await self._extract_container_info(container, **kwargs)
                            if result:
                                results.append(result)
                                logger.debug(
                                    "Found container by image pattern",
                                    container=container.name,
                                    image=image_name,
                                    pattern=pattern,
                                )
                            break
                except Exception as e:
                    logger.debug(
                        "Error checking container image",
                        container_id=container.id[:12],
                        error=str(e),
                    )

            # Find by labels
            for label in settings.discovery.docker.labels:
                try:
                    containers = await loop.run_in_executor(
                        None,
                        lambda l=label: client.containers.list(all=True, filters={"label": l}),
                    )
                    for container in containers:
                        if container.id not in seen_ids:
                            seen_ids.add(container.id)
                            result = await self._extract_container_info(container, **kwargs)
                            if result:
                                results.append(result)
                                logger.debug(
                                    "Found container by label",
                                    container=container.name,
                                    label=label,
                                )
                except Exception as e:
                    logger.debug("Error finding containers by label", label=label, error=str(e))

            logger.info("Docker discovery completed", discovered_count=len(results))

        except Exception as e:
            logger.error("Docker discovery failed", error=str(e))

        return results

    async def _extract_container_info(self, container, **kwargs) -> Optional[DiscoveryResult]:
        """Extract Freqtrade bot info from a Docker container.

        Args:
            container: Docker container object.
            **kwargs: Additional arguments including 'host' for API URL

        Returns:
            DiscoveryResult or None if not a valid Freqtrade container.
        """
        try:
            attrs = container.attrs
            labels = attrs.get("Config", {}).get("Labels", {})

            # Get container name (strip leading /)
            container_name = attrs.get("Name", "").lstrip("/")
            if not container_name:
                container_name = container.id[:12]

            config = self._read_container_config(container)

            # Check for custom name label/config
            bot_name = (
                labels.get("com.freqtrade.bot_name")
                or config.get("bot_name")
                or container_name
            )

            # Extract port mappings. Host-networked Freqtrade bots expose no
            # Docker port bindings, so fall back to reading /freqtrade/config.json.
            api_port = self._extract_api_port(attrs)
            if api_port is None:
                api_port = self._extract_api_port_from_config(container, config)
            
            # Use provided host or default to localhost
            # Host is passed from orchestrator based on discovery_host_ip setting
            host = kwargs.get('host', 'localhost')

            # Build API URL if port found
            api_url = f"http://{host}:{api_port}" if api_port else None

            # Extract user_data path from mounts
            user_data_path = self._extract_user_data_path(attrs)

            # Extract config from labels first, then the in-container config.
            exchange = labels.get("com.freqtrade.exchange") or (config.get("exchange") or {}).get("name")
            strategy = labels.get("com.freqtrade.strategy") or config.get("strategy")
            dryrun_label = labels.get("com.freqtrade.dry_run")
            if dryrun_label is not None:
                is_dryrun = dryrun_label.lower() == "true"
            else:
                is_dryrun = bool(config.get("dry_run", True))

            return DiscoveryResult(
                environment="docker",
                name=bot_name,
                host=host,
                container_id=container.id,
                user_data_path=user_data_path,
                api_url=api_url,
                api_port=api_port,
                api_available=container.status == "running" and api_port is not None,
                sqlite_path=f"{user_data_path}/tradesv3.sqlite" if user_data_path else None,
                sqlite_available=user_data_path is not None,
                labels=labels,
                exchange=exchange,
                strategy=strategy,
                is_dryrun=is_dryrun,
            )

        except Exception as e:
            logger.warning(
                "Failed to extract container info",
                container_id=container.id[:12],
                error=str(e),
            )
            return None

    def _extract_api_port(self, attrs: dict) -> Optional[int]:
        """Extract API port from container port mappings.

        Args:
            attrs: Container attributes dict.

        Returns:
            Host port number or None.
        """
        try:
            ports = attrs.get("NetworkSettings", {}).get("Ports", {})

            # Look for common Freqtrade API ports (8080, 8081, etc.)
            for container_port, bindings in ports.items():
                if bindings and container_port.startswith("8"):
                    # Get first binding's host port
                    host_port = bindings[0].get("HostPort")
                    if host_port:
                        return int(host_port)

            # Fallback: any mapped port
            for container_port, bindings in ports.items():
                if bindings:
                    host_port = bindings[0].get("HostPort")
                    if host_port:
                        return int(host_port)

        except Exception:
            pass

        return None

    def _read_container_config(self, container) -> dict:
        """Read bot config from inside the container."""
        try:
            exit_code, output = container.exec_run("cat /freqtrade/config.json")
            if exit_code != 0 or not output:
                return {}
            return json.loads(output.decode("utf-8"))
        except Exception:
            return {}

    def _extract_api_port_from_config(self, container, config: Optional[dict] = None) -> Optional[int]:
        """Extract API port from the bot's config inside the container."""
        try:
            if config is None:
                config = self._read_container_config(container)
            if not config:
                return None
            api_server = config.get("api_server") or {}
            if not api_server.get("enabled"):
                return None

            listen_port = api_server.get("listen_port")
            return int(listen_port) if listen_port else None
        except Exception:
            return None

    def _extract_user_data_path(self, attrs: dict) -> Optional[str]:
        """Extract user_data path from container mounts.

        Args:
            attrs: Container attributes dict.

        Returns:
            Host path to user_data directory or None.
        """
        try:
            mounts = attrs.get("Mounts", [])

            for mount in mounts:
                destination = mount.get("Destination", "")
                source = mount.get("Source", "")

                # Check if this is a user_data mount
                if "user_data" in destination or "user_data" in source:
                    return source

                # Check for common Freqtrade paths
                if destination.endswith("/freqtrade/user_data"):
                    return source

        except Exception:
            pass

        return None
