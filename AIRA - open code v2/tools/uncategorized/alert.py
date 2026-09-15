"""Alert model for notifications and alerts."""

from datetime import datetime
from enum import Enum
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models import Base


class AlertType(str, Enum):
    """Type of alert."""

    BOT_OFFLINE = "bot_offline"
    BOT_ONLINE = "bot_online"
    BOT_DEGRADED = "bot_degraded"
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    PROFIT_THRESHOLD = "profit_threshold"
    LOSS_THRESHOLD = "loss_threshold"
    RATE_LIMIT = "rate_limit"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class AlertSeverity(str, Enum):
    """Alert severity level."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Alert(Base):
    """Alert/notification entry."""

    __tablename__ = "alerts"

    # Alert metadata
    alert_type: Mapped[AlertType] = mapped_column(nullable=False)
    severity: Mapped[AlertSeverity] = mapped_column(default=AlertSeverity.INFO, nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    # Associated bot (optional)
    bot_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("bots.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    bot_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Status
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_dismissed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Target user (optional - for user-specific alerts)
    user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Additional data as JSON
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships - no backref to prevent errors when alerts table doesn't exist
    bot = relationship("Bot", lazy="select")
    user = relationship("User", lazy="select")

    def __repr__(self) -> str:
        return f"<Alert {self.alert_type.value}: {self.title[:30]}>"

    @classmethod
    def create_bot_alert(
        cls,
        alert_type: AlertType,
        bot_id: str,
        bot_name: str,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        metadata: Optional[dict] = None,
    ) -> "Alert":
        """Create a bot-related alert."""
        import json

        return cls(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            bot_id=bot_id,
            bot_name=bot_name,
            metadata_json=json.dumps(metadata) if metadata else None,
        )

    @classmethod
    def create_system_alert(
        cls,
        alert_type: AlertType,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        metadata: Optional[dict] = None,
    ) -> "Alert":
        """Create a system-wide alert."""
        import json

        return cls(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
