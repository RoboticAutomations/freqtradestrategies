"""User model for dashboard authentication."""

from enum import Enum

import sqlalchemy as sa
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import ENUM, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.models import Base


class UserRole(str, Enum):
    """User role levels."""

    ADMIN = "admin"
    OPERATOR = "operator"
    READONLY = "readonly"


class User(Base):
    """Dashboard user account with role-based permissions."""

    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    # IMPORTANT: Postgres enum values are lowercase strings ('admin','operator','readonly')
    # SQLAlchemy must persist Enum .value (not the Enum member name like 'ADMIN').
    role: Mapped[UserRole] = mapped_column(
        ENUM(
            UserRole,
            name="userrole",
            create_type=False,
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        default=UserRole.READONLY,
        nullable=False,
    )

    # Postgres schema uses jsonb.
    preferences: Mapped[dict] = mapped_column(
        JSONB,
        server_default=sa.text("'{}'::jsonb"),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role.value})>"

    def has_permission(self, required_roles: list[UserRole]) -> bool:
        """Check if user has one of the required roles."""
        return self.role in required_roles

    def can_control_bots(self) -> bool:
        """Check if user can perform bot control actions."""
        return self.role in [UserRole.ADMIN, UserRole.OPERATOR]

    def can_force_exit(self) -> bool:
        """Check if user can force exit positions."""
        return self.role == UserRole.ADMIN

    def can_manage_users(self) -> bool:
        """Check if user can manage other users."""
        return self.role == UserRole.ADMIN
