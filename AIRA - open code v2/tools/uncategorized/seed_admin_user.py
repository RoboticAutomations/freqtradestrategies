"""Seed default admin user.

Revision ID: 002
Revises: 001
Create Date: 2026-02-03

This migration inserts an initial admin user if the users table is empty.
Credentials are taken from env vars at runtime of migration:
- DASHBOARD_ADMIN_USERNAME (default: admin)
- DASHBOARD_ADMIN_PASSWORD (default: admin)

Note: bcrypt has a 72-byte limit; we truncate to 72 bytes.
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # check if users table has any rows
    user_count = conn.execute(sa.text("SELECT COUNT(*) FROM users")).scalar() or 0
    if user_count > 0:
        return

    username = os.environ.get("DASHBOARD_ADMIN_USERNAME", "admin")
    password = os.environ.get("DASHBOARD_ADMIN_PASSWORD", "admin")

    # normalize + bcrypt safety
    password = password.strip()
    password = password.encode("utf-8")[:72].decode("utf-8", errors="ignore")

    # passlib bcrypt hashing
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    password_hash = pwd_context.hash(password)

    # Insert admin user. Preferences column is jsonb.
    conn.execute(
        sa.text(
            """
            INSERT INTO users (id, username, password_hash, role, preferences, created_at, updated_at)
            VALUES (gen_random_uuid(), :username, :password_hash, 'admin'::userrole, '{}'::jsonb, now(), now())
            """
        ),
        {
            "username": username,
            "password_hash": password_hash,
        },
    )


def downgrade() -> None:
    # remove seeded admin user if it exists (best-effort)
    conn = op.get_bind()
    conn.execute(sa.text("DELETE FROM users WHERE username = 'admin'"))
