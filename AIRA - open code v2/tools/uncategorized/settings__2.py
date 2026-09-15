"""System settings model for dashboard configuration."""

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models import Base


class SystemSetting(Base):
    """Key-value store for persisted runtime settings."""

    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
    value_type: Mapped[str] = mapped_column(String(20), default="string", nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
