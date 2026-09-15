"""Add replay_runs table for freqtrade-replay integration.

Revision ID: 003
Revises: v6_strategy_lab
Create Date: 2025-05-19

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "v6_strategy_lab"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(conn, table_name):
    """Check if a table exists using to_regclass (most reliable)."""
    result = conn.execute(sa.text(
        "SELECT to_regclass(:table_name) IS NOT NULL"
    ), {"table_name": table_name})
    return result.scalar()


def upgrade() -> None:
    conn = op.get_bind()
    
    # Skip if table already exists
    if _table_exists(conn, 'replay_runs'):
        return
    
    # Replay runs tracking — mirrors optimization_runs structure with replay-specific fields
    op.create_table(
        'replay_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('bot_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('strategy_name', sa.String(length=100), nullable=False),
        sa.Column('timerange', sa.String(length=50), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('profit_pct', sa.Float(), nullable=True),
        sa.Column('profit_abs', sa.Float(), nullable=True),
        sa.Column('win_rate', sa.Float(), nullable=True),
        sa.Column('total_trades', sa.Integer(), nullable=True),
        sa.Column('max_drawdown_pct', sa.Float(), nullable=True),
        sa.Column('sharpe', sa.Float(), nullable=True),
        sa.Column('profit_factor', sa.Float(), nullable=True),
        sa.Column('docker_container_id', sa.String(length=64), nullable=True),
        sa.Column('docker_log_path', sa.String(length=500), nullable=True),
        sa.Column('report_html_path', sa.String(length=500), nullable=True),
        sa.Column('report_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['bot_id'], ['bots.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('idx_replay_runs_bot', 'replay_runs', ['bot_id'])
    op.create_index('idx_replay_runs_strategy', 'replay_runs', ['strategy_name'])
    op.create_index('idx_replay_runs_status', 'replay_runs', ['status'])
    op.create_index('idx_replay_runs_started', 'replay_runs', ['started_at'])


def downgrade() -> None:
    op.drop_index('idx_replay_runs_started', table_name='replay_runs')
    op.drop_index('idx_replay_runs_status', table_name='replay_runs')
    op.drop_index('idx_replay_runs_strategy', table_name='replay_runs')
    op.drop_index('idx_replay_runs_bot', table_name='replay_runs')
    op.drop_table('replay_runs')
