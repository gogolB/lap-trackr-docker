"""Add session name and grading warnings columns.

Revision ID: 005
Revises: 004
Create Date: 2026-03-08
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    session_columns = {column["name"] for column in inspector.get_columns("sessions")}
    if "name" not in session_columns:
        op.add_column(
            "sessions",
            sa.Column("name", sa.String(length=255), nullable=False, server_default="Untitled Session"),
        )
        op.alter_column("sessions", "name", server_default=None)

    grading_columns = {column["name"] for column in inspector.get_columns("grading_results")}
    if "warnings" not in grading_columns:
        op.add_column(
            "grading_results",
            sa.Column("warnings", JSONB(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    grading_columns = {column["name"] for column in inspector.get_columns("grading_results")}
    if "warnings" in grading_columns:
        op.drop_column("grading_results", "warnings")

    session_columns = {column["name"] for column in inspector.get_columns("sessions")}
    if "name" in session_columns:
        op.drop_column("sessions", "name")
