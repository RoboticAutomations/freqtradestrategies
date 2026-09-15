"""Add backtest_results and fix optimization_runs schema.

Revision ID: 004
Revises: 003
Create Date: 2025-05-19
"""
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(conn, table_name):
    result = conn.execute(sa.text(
        "SELECT to_regclass(:table_name) IS NOT NULL"
    ), {"table_name": table_name})
    return result.scalar()


def _column_exists(conn, table_name, column_name):
    result = conn.execute(sa.text(
        """SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = :table_name AND column_name = :column_name
        )"""
    ), {"table_name": table_name, "column_name": column_name})
    return result.scalar()


def upgrade() -> None:
    conn = op.get_bind()

    # ── backtest_results ───────────────────────────────────────────
    if not _table_exists(conn, 'backtest_results'):
        op.create_table(
            'backtest_results',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('strategy_name', sa.String(length=100), nullable=False),
            sa.Column('timeframe', sa.String(length=20), nullable=True),
            sa.Column('timerange', sa.String(length=50), nullable=True),
            sa.Column('start_balance', sa.Float(), nullable=True),
            sa.Column('final_balance', sa.Float(), nullable=True),
            sa.Column('total_profit_pct', sa.Float(), nullable=True),
            sa.Column('total_profit_abs', sa.Float(), nullable=True),
            sa.Column('total_trades', sa.Integer(), nullable=True),
            sa.Column('win_rate', sa.Float(), nullable=True),
            sa.Column('avg_profit_pct', sa.Float(), nullable=True),
            sa.Column('max_drawdown_pct', sa.Float(), nullable=True),
            sa.Column('sharpe', sa.Float(), nullable=True),
            sa.Column('sortino', sa.Float(), nullable=True),
            sa.Column('calmar', sa.Float(), nullable=True),
            sa.Column('profit_factor', sa.Float(), nullable=True),
            sa.Column('best_pair', sa.String(length=50), nullable=True),
            sa.Column('worst_pair', sa.String(length=50), nullable=True),
            sa.Column('backtest_date', sa.DateTime(), nullable=True),
            sa.Column('config_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column('pairlist_used', sa.Text(), nullable=True),
            sa.Column('hyperopt_id', sa.Integer(), nullable=True),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('idx_backtest_strategy', 'backtest_results', ['strategy_name'])
        op.create_index('idx_backtest_date', 'backtest_results', ['backtest_date'])
        op.create_index('idx_backtest_profit', 'backtest_results', ['total_profit_pct'])

    # ── optimization_runs fix ─────────────────────────────────────
    if _table_exists(conn, 'optimization_runs'):
        # Columns added in 004 migration
        cols_to_add = [
            ('process_type', sa.Column('process_type', sa.String(length=20), nullable=True)),
            ('result_profit_pct', sa.Column('result_profit_pct', sa.Float(), nullable=True)),
            ('result_drawdown', sa.Column('result_drawdown', sa.Float(), nullable=True)),
            ('result_trade_count', sa.Column('result_trade_count', sa.Integer(), nullable=True)),
            ('duration_seconds', sa.Column('duration_seconds', sa.Float(), nullable=True)),
            ('config', sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True)),
            ('error_message', sa.Column('error_message', sa.Text(), nullable=True)),
        ]
        for col_name, col_def in cols_to_add:
            if not _column_exists(conn, 'optimization_runs', col_name):
                op.add_column('optimization_runs', col_def)
    else:
        # Create optimization_runs if it doesn't exist at all
        op.create_table(
            'optimization_runs',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('bot_id', postgresql.UUID(as_uuid=False), nullable=True),
            sa.Column('strategy_name', sa.String(length=100), nullable=False),
            sa.Column('process_type', sa.String(length=20), nullable=True),
            sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
            sa.Column('started_at', sa.DateTime(), nullable=True),
            sa.Column('completed_at', sa.DateTime(), nullable=True),
            sa.Column('duration_seconds', sa.Float(), nullable=True),
            sa.Column('result_profit_pct', sa.Float(), nullable=True),
            sa.Column('result_drawdown', sa.Float(), nullable=True),
            sa.Column('result_trade_count', sa.Integer(), nullable=True),
            sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('idx_optimization_bot', 'optimization_runs', ['bot_id'])
        op.create_index('idx_optimization_status', 'optimization_runs', ['status'])


def downgrade() -> None:
    # Intentionally empty — don't drop production data tables
    pass
