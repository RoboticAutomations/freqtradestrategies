"""Database migration for Strategy Lab (V6)
Add tables for optimization tracking
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'v6_strategy_lab'
down_revision = '002'  # Set to last V5 migration
branch_labels = None
depends_on = None


def _table_exists(conn, table_name):
    """Check if a table exists using to_regclass (most reliable)."""
    result = conn.execute(sa.text(
        "SELECT to_regclass(:table_name) IS NOT NULL"
    ), {"table_name": table_name})
    return result.scalar()


def upgrade():
    conn = op.get_bind()
    
    # Skip if tables already exist (SQL init scripts may have created them)
    if _table_exists(conn, 'optimization_runs'):
        return
    
    # Optimization runs tracking
    op.create_table(
        'optimization_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('bot_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('strategy_name', sa.String(length=100), nullable=False),
        sa.Column('run_type', sa.String(length=20), nullable=False),  # backtest, hyperopt, workflow
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('profit_pct', sa.Float(), nullable=True),
        sa.Column('winrate_pct', sa.Float(), nullable=True),
        sa.Column('max_drawdown_pct', sa.Float(), nullable=True),
        sa.Column('sharpe', sa.Float(), nullable=True),
        sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('fthypt_path', sa.String(length=500), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['bot_id'], ['bots.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_optimization_runs_bot', 'optimization_runs', ['bot_id'])
    op.create_index('idx_optimization_runs_strategy', 'optimization_runs', ['strategy_name'])
    op.create_index('idx_optimization_runs_status', 'optimization_runs', ['status'])
    
    # Workflow schedules per bot
    op.create_table(
        'workflow_schedules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('bot_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('strategy_name', sa.String(length=100), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('cron_expression', sa.String(length=50), nullable=True),
        sa.Column('last_run_at', sa.DateTime(), nullable=True),
        sa.Column('next_run_at', sa.DateTime(), nullable=True),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['bot_id'], ['bots.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('bot_id', 'strategy_name', name='uix_workflow_bot_strategy')
    )
    
    # Hyperopt epochs from .fthypt files
    op.create_table(
        'hyperopt_epochs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('epoch_number', sa.Integer(), nullable=False),
        sa.Column('trades', sa.Integer(), nullable=True),
        sa.Column('profit_total_pct', sa.Float(), nullable=True),
        sa.Column('profit_total_abs', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('max_drawdown_abs', sa.Float(), nullable=True),
        sa.Column('wins', sa.Integer(), nullable=True),
        sa.Column('losses', sa.Integer(), nullable=True),
        sa.Column('draws', sa.Integer(), nullable=True),
        sa.Column('avg_profit', sa.Float(), nullable=True),
        sa.Column('objective', sa.Float(), nullable=True),
        sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_best', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('extracted', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['run_id'], ['optimization_runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id', 'epoch_number', name='uix_epoch_run_number')
    )
    op.create_index('idx_hyperopt_epochs_run', 'hyperopt_epochs', ['run_id'])
    op.create_index('idx_hyperopt_epochs_best', 'hyperopt_epochs', ['is_best'])


def downgrade():
    op.drop_table('hyperopt_epochs')
    op.drop_table('workflow_schedules')
    op.drop_table('optimization_runs')
