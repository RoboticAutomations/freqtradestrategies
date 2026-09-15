"""Bot model for registered Freqtrade instances."""

from datetime import datetime
from enum import Enum

import sqlalchemy as sa
from sqlalchemy import ARRAY, Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models import Base


def _enum_values(enum_cls):
    return [e.value for e in enum_cls]


class BotEnvironment(str, Enum):
    """Bot deployment environment type."""

    DOCKER = "docker"
    BAREMETAL = "baremetal"
    K8S = "k8s"
    MANUAL = "manual"


class HealthState(str, Enum):
    """Bot API health state."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


class SourceMode(str, Enum):
    """Data source mode for the bot."""

    API = "api"
    SQLITE = "sqlite"
    MIXED = "mixed"
    AUTO = "auto"


class TradingMode(str, Enum):
    """Trading mode."""

    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"


class Bot(Base):
    """Registered Freqtrade instance (discovered or manually added)."""

    __tablename__ = "bots"

    # Identity
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    environment: Mapped[BotEnvironment] = mapped_column(
        sa.Enum(BotEnvironment, name="botenvironment", values_callable=_enum_values),
        nullable=False,
    )

    # Connection details
    host: Mapped[str | None] = mapped_column(String(255))
    container_id: Mapped[str | None] = mapped_column(String(64))
    user_data_path: Mapped[str | None] = mapped_column(String(500))
    api_url: Mapped[str | None] = mapped_column(String(255))
    api_port: Mapped[int | None] = mapped_column(Integer)
    credentials_enc: Mapped[str | None] = mapped_column(Text)

    # Data source configuration
    source_mode: Mapped[SourceMode] = mapped_column(
        sa.Enum(SourceMode, name="sourcemode", values_callable=_enum_values),
        default=SourceMode.AUTO,
        nullable=False,
    )
    health_state: Mapped[HealthState] = mapped_column(
        sa.Enum(HealthState, name="healthstate", values_callable=_enum_values),
        default=HealthState.UNKNOWN,
        nullable=False,
    )

    # Bot metadata
    exchange: Mapped[str | None] = mapped_column(String(50), index=True)
    strategy: Mapped[str | None] = mapped_column(String(100), index=True)
    trading_mode: Mapped[TradingMode | None] = mapped_column(
        sa.Enum(TradingMode, name="tradingmode", values_callable=_enum_values),
        nullable=True,
    )
    is_dryrun: Mapped[bool] = mapped_column(Boolean, default=True)

    # Postgres schema uses varchar[] (ARRAY)
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        server_default=sa.text("'{}'::varchar[]"),
        nullable=False,
    )

    # Timestamps
    last_seen: Mapped[datetime | None] = mapped_column()
    discovered_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)

    # Relationships
    metrics = relationship("BotMetrics", back_populates="bot", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<Bot {self.name} ({self.environment.value}, {self.health_state.value})>"

    @property
    def is_api_available(self) -> bool:
        """Check if bot API is available for control actions."""
        return self.health_state in [HealthState.HEALTHY, HealthState.DEGRADED] and self.api_url

    @property
    def effective_source(self) -> SourceMode:
        """Get the effective data source based on health state and configuration."""
        if self.source_mode != SourceMode.AUTO:
            return self.source_mode

        if self.health_state == HealthState.HEALTHY:
            return SourceMode.API
        if self.health_state == HealthState.DEGRADED:
            return SourceMode.MIXED
        return SourceMode.SQLITE
