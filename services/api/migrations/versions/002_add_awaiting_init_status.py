"""Add awaiting_init session status.

Revision ID: 002
Revises: 001
Create Date: 2026-03-07
"""

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE sessionstatus ADD VALUE IF NOT EXISTS 'awaiting_init' AFTER 'export_failed'")


def downgrade() -> None:
    # PostgreSQL does not support removing values from enums.
    pass
