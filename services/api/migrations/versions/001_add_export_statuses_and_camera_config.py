"""Add exporting/export_failed session statuses and camera_config table.

Revision ID: 001
Revises:
Create Date: 2026-03-07
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new values to the sessionstatus enum
    op.execute("ALTER TYPE sessionstatus ADD VALUE IF NOT EXISTS 'exporting' AFTER 'completed'")
    op.execute("ALTER TYPE sessionstatus ADD VALUE IF NOT EXISTS 'export_failed' AFTER 'exporting'")

    # Create camera_config table
    op.create_table(
        "camera_config",
        sa.Column("id", sa.Integer(), primary_key=True, default=1),
        sa.Column("on_axis_serial", sa.String(32), nullable=False),
        sa.Column("off_axis_serial", sa.String(32), nullable=False),
        sa.Column("on_axis_swap_eyes", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("off_axis_swap_eyes", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("on_axis_flip", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("off_axis_flip", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("id = 1", name="single_row_camera_config"),
    )


def downgrade() -> None:
    op.drop_table("camera_config")
    # Note: PostgreSQL does not support removing values from enums.
    # To fully downgrade, you'd need to recreate the enum type.
