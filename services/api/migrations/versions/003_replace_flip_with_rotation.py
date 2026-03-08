"""Replace flip boolean with rotation integer (0, 90, 180, 270).

Revision ID: 003
Revises: 002
Create Date: 2026-03-07
"""

import sqlalchemy as sa
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new rotation columns
    op.add_column("camera_config", sa.Column("on_axis_rotation", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("camera_config", sa.Column("off_axis_rotation", sa.Integer(), nullable=False, server_default="0"))

    # Migrate flip=True to rotation=180
    op.execute("UPDATE camera_config SET on_axis_rotation = 180 WHERE on_axis_flip = true")
    op.execute("UPDATE camera_config SET off_axis_rotation = 180 WHERE off_axis_flip = true")

    # Add flip columns (horizontal / vertical mirror)
    op.add_column("camera_config", sa.Column("on_axis_flip_h", sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("camera_config", sa.Column("on_axis_flip_v", sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("camera_config", sa.Column("off_axis_flip_h", sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("camera_config", sa.Column("off_axis_flip_v", sa.Boolean(), nullable=False, server_default="false"))

    # Drop old flip columns (replaced by rotation + flip_h/flip_v)
    op.drop_column("camera_config", "on_axis_flip")
    op.drop_column("camera_config", "off_axis_flip")


def downgrade() -> None:
    op.add_column("camera_config", sa.Column("on_axis_flip", sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("camera_config", sa.Column("off_axis_flip", sa.Boolean(), nullable=False, server_default="false"))

    op.execute("UPDATE camera_config SET on_axis_flip = true WHERE on_axis_rotation = 180")
    op.execute("UPDATE camera_config SET off_axis_flip = true WHERE off_axis_rotation = 180")

    op.drop_column("camera_config", "on_axis_rotation")
    op.drop_column("camera_config", "off_axis_rotation")
    op.drop_column("camera_config", "on_axis_flip_h")
    op.drop_column("camera_config", "on_axis_flip_v")
    op.drop_column("camera_config", "off_axis_flip_h")
    op.drop_column("camera_config", "off_axis_flip_v")
