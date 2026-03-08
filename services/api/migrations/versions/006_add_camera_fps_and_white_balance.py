"""Add camera fps and white balance controls.

Revision ID: 006
Revises: 005
Create Date: 2026-03-08
"""

import sqlalchemy as sa
from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("camera_config")}

    additions = [
        ("camera_fps", sa.Column("camera_fps", sa.Integer(), nullable=False, server_default="60")),
        (
            "on_axis_whitebalance_auto",
            sa.Column("on_axis_whitebalance_auto", sa.Boolean(), nullable=False, server_default=sa.true()),
        ),
        (
            "off_axis_whitebalance_auto",
            sa.Column("off_axis_whitebalance_auto", sa.Boolean(), nullable=False, server_default=sa.true()),
        ),
        (
            "on_axis_whitebalance_temperature",
            sa.Column("on_axis_whitebalance_temperature", sa.Integer(), nullable=False, server_default="4600"),
        ),
        (
            "off_axis_whitebalance_temperature",
            sa.Column("off_axis_whitebalance_temperature", sa.Integer(), nullable=False, server_default="4600"),
        ),
    ]

    for name, column in additions:
        if name not in columns:
            op.add_column("camera_config", column)

    for name in (
        "camera_fps",
        "on_axis_whitebalance_auto",
        "off_axis_whitebalance_auto",
        "on_axis_whitebalance_temperature",
        "off_axis_whitebalance_temperature",
    ):
        op.alter_column("camera_config", name, server_default=None)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("camera_config")}

    for name in (
        "off_axis_whitebalance_temperature",
        "on_axis_whitebalance_temperature",
        "off_axis_whitebalance_auto",
        "on_axis_whitebalance_auto",
        "camera_fps",
    ):
        if name in columns:
            op.drop_column("camera_config", name)
