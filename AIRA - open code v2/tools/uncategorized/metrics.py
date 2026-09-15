"""BotMetrics model for cached performance snapshots."""

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import ForeignKey, Index, Integer, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models import Base
from src.models.bot import SourceMode, _enum_values


class BotMetrics(Base):
    """Cached performance metrics snapshot for a bot."""

    __tablename__ = "bot_metrics"

    # Foreign key
    bot_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("bots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True)

    # Financial metrics
    equity: Mapped[float | None] = mapped_column(Numeric(18, 8))
    profit_abs: Mapped[float | None] = mapped_column(Numeric(18, 8))
    profit_pct: Mapped[float | None] = mapped_column(Numeric(8, 4))
    profit_realized: Mapped[float | None] = mapped_column(Numeric(18, 8))
    profit_unrealized: Mapped[float | None] = mapped_column(Numeric(18, 8))
    balance: Mapped[float | None] = mapped_column(Numeric(18, 8))
    drawdown: Mapped[float | None] = mapped_column(Numeric(8, 4))

    # Trade statistics
    open_positions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    closed_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    win_rate: Mapped[float | None] = mapped_column(Numeric(5, 2))

    # Data source tracking
    data_source: Mapped[SourceMode] = mapped_column(
        sa.Enum(SourceMode, name="sourcemode", values_callable=_enum_values),
        nullable=False,
    )

    # Relationships
    bot = relationship("Bot", back_populates="metrics")

    # Indexes for efficient queries
    __table_args__ = (
        Index("idx_metrics_bot_timestamp", "bot_id", timestamp.desc()),
        Index("idx_metrics_timestamp", timestamp.desc()),
    )

    def __repr__(self) -> str:
        return f"<BotMetrics {self.bot_id} @ {self.timestamp}>"

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for API responses."""
        return {
            "bot_id": self.bot_id,
            "timestamp": self.timestamp.isoformat(),
            "equity": float(self.equity) if self.equity else None,
            "profit_abs": float(self.profit_abs) if self.profit_abs else None,
            "profit_pct": float(self.profit_pct) if self.profit_pct else None,
            "profit_realized": float(self.profit_realized) if self.profit_realized else None,
            "profit_unrealized": float(self.profit_unrealized) if self.profit_unrealized else None,
            "open_positions": self.open_positions,
            "closed_trades": self.closed_trades,
            "win_rate": float(self.win_rate) if self.win_rate else None,
            "balance": float(self.balance) if self.balance else None,
            "drawdown": float(self.drawdown) if self.drawdown else None,
            "data_source": self.data_source.value,
        }
