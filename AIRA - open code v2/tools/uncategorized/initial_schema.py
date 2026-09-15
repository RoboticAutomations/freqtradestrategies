"""Initial schema with User, Bot, and BotMetrics tables.

Revision ID: 001
Revises:
Create Date: 2025-12-05

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    
    # Check if users table already exists (SQL init scripts may have created it)
    result = conn.execute(sa.text(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'users'"
    ))
    if result.scalar():
        # Schema already exists, skip creation
        return
    
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    
    # Create enum types (idempotent via DO block)
    op.execute("""DO $$ BEGIN
        CREATE TYPE userrole AS ENUM ('admin', 'operator', 'readonly');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$;""")
    op.execute("""DO $$ BEGIN
        CREATE TYPE botenvironment AS ENUM ('docker', 'baremetal', 'k8s', 'manual');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$;""")
    op.execute("""DO $$ BEGIN
        CREATE TYPE healthstate AS ENUM ('healthy', 'degraded', 'unreachable', 'unknown');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$;""")
    op.execute("""DO $$ BEGIN
        CREATE TYPE sourcemode AS ENUM ('api', 'sqlite', 'mixed', 'auto');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$;""")
    op.execute("""DO $$ BEGIN
        CREATE TYPE tradingmode AS ENUM ('spot', 'futures', 'margin');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$;""")

    # Create users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("username", sa.String(50), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column(
            "role",
            postgresql.ENUM("admin", "operator", "readonly", name="userrole", create_type=False),
            nullable=False,
            server_default="readonly",
        ),
        sa.Column("preferences", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_users_username", "users", ["username"])

    # Create bots table
    op.create_table(
        "bots",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column(
            "environment",
            postgresql.ENUM(
                "docker", "baremetal", "k8s", "manual", name="botenvironment", create_type=False
            ),
            nullable=False,
        ),
        sa.Column("host", sa.String(255)),
        sa.Column("container_id", sa.String(64)),
        sa.Column("user_data_path", sa.String(500)),
        sa.Column("api_url", sa.String(255)),
        sa.Column("api_port", sa.Integer),
        sa.Column("credentials_enc", sa.Text),
        sa.Column(
            "source_mode",
            postgresql.ENUM("api", "sqlite", "mixed", "auto", name="sourcemode", create_type=False),
            nullable=False,
            server_default="auto",
        ),
        sa.Column(
            "health_state",
            postgresql.ENUM(
                "healthy", "degraded", "unreachable", "unknown",
                name="healthstate",
                create_type=False,
            ),
            nullable=False,
            server_default="unknown",
        ),
        sa.Column("exchange", sa.String(50)),
        sa.Column("strategy", sa.String(100)),
        sa.Column(
            "trading_mode",
            postgresql.ENUM("spot", "futures", "margin", name="tradingmode", create_type=False),
        ),
        sa.Column("is_dryrun", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("tags", postgresql.ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("last_seen", sa.DateTime),
        sa.Column("discovered_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_bot_environment", "bots", ["environment"])
    op.create_index("idx_bot_health_state", "bots", ["health_state"])
    op.create_index("idx_bot_exchange", "bots", ["exchange"])
    op.create_index("idx_bot_strategy", "bots", ["strategy"])
    op.create_index("idx_bot_last_seen", "bots", ["last_seen"])
    op.create_index("idx_bot_tags", "bots", ["tags"], postgresql_using="gin")

    # Create bot_metrics table
    op.create_table(
        "bot_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "bot_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("bots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("timestamp", sa.DateTime, nullable=False),
        sa.Column("equity", sa.Numeric(18, 8)),
        sa.Column("profit_abs", sa.Numeric(18, 8)),
        sa.Column("profit_pct", sa.Numeric(8, 4)),
        sa.Column("profit_realized", sa.Numeric(18, 8)),
        sa.Column("profit_unrealized", sa.Numeric(18, 8)),
        sa.Column("balance", sa.Numeric(18, 8)),
        sa.Column("drawdown", sa.Numeric(8, 4)),
        sa.Column("open_positions", sa.Integer, nullable=False, server_default="0"),
        sa.Column("closed_trades", sa.Integer, nullable=False, server_default="0"),
        sa.Column("win_rate", sa.Numeric(5, 2)),
        sa.Column(
            "data_source",
            postgresql.ENUM("api", "sqlite", "mixed", "auto", name="sourcemode", create_type=False),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_metrics_bot_id", "bot_metrics", ["bot_id"])
    op.create_index("idx_metrics_timestamp", "bot_metrics", [sa.text("timestamp DESC")])
    op.create_index(
        "idx_metrics_bot_timestamp",
        "bot_metrics",
        ["bot_id", sa.text("timestamp DESC")],
    )


def downgrade() -> None:
    op.drop_table("bot_metrics")
    op.drop_table("bots")
    op.drop_table("users")

    op.execute("DROP TYPE IF EXISTS tradingmode")
    op.execute("DROP TYPE IF EXISTS sourcemode")
    op.execute("DROP TYPE IF EXISTS healthstate")
    op.execute("DROP TYPE IF EXISTS botenvironment")
    op.execute("DROP TYPE IF EXISTS userrole")
