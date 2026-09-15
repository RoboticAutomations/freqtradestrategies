"""Bot discovery services for Docker and filesystem scanning."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiscoveryResult:
    """Result from a discovery scan for a single bot."""

    environment: str  # docker, baremetal
    name: str
    host: Optional[str] = None
    container_id: Optional[str] = None
    user_data_path: Optional[str] = None
    api_url: Optional[str] = None
    api_port: Optional[int] = None
    api_available: bool = False
    sqlite_path: Optional[str] = None
    sqlite_available: bool = False
    config_data: Optional[dict] = None
    labels: dict = field(default_factory=dict)
    exchange: Optional[str] = None
    strategy: Optional[str] = None
    is_dryrun: bool = True


class BaseDiscovery(ABC):
    """Abstract base class for bot discovery implementations."""

    @abstractmethod
    async def discover(self) -> list[DiscoveryResult]:
        """Discover Freqtrade bots.

        Returns:
            List of discovered bots.
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this discovery method is available.

        Returns:
            True if discovery method can be used.
        """
        pass
