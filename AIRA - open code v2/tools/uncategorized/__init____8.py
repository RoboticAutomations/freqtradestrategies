"""SQLAlchemy models and database connection."""

from datetime import datetime
from typing import AsyncGenerator
from uuid import uuid4

from sqlalchemy import MetaData
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


from src.config import settings

# Naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    metadata = metadata

    # Use UUID in Postgres to match Alembic schema (uuid type)
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


# Determine database URL and engine options based on database type
db_url = settings.database.url
engine_options: dict = {"echo": settings.database.echo}

if db_url.startswith("postgresql"):
    # PostgreSQL configuration
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
    engine_options["pool_size"] = settings.database.pool_size
elif db_url.startswith("sqlite"):
    # SQLite doesn't support pool_size - use check_same_thread for async
    pass

# Create async engine
engine = create_async_engine(db_url, **engine_options)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database sessions."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Import models to register them with SQLAlchemy
from src.models.bot import Bot  # noqa: E402, F401
from src.models.metrics import BotMetrics  # noqa: E402, F401
from src.models.user import User  # noqa: E402, F401
from src.models.alert import Alert  # noqa: E402, F401
from src.models.pairlist import PairlistJob, PairlistResult, PairlistPairResult  # noqa: E402, F401
from src.models.settings import SystemSetting  # noqa: E402, F401

__all__ = ["Base", "engine", "async_session_maker", "get_db", "User", "Bot", "BotMetrics", "Alert", "PairlistJob", "PairlistResult", "PairlistPairResult", "SystemSetting"]
