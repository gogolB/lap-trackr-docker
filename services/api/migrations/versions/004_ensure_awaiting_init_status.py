"""Ensure awaiting_init exists in the sessionstatus enum.

Revision ID: 004
Revises: 003
Create Date: 2026-03-08
"""

from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE sessionstatus ADD VALUE IF NOT EXISTS 'awaiting_init' AFTER 'export_failed'")


def downgrade() -> None:
    # PostgreSQL does not support removing values from enums.
    pass
