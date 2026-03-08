"""Shared data container threaded through all pipeline passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PassData:
    """Mutable container carrying intermediate results across passes."""

    session_dir: Path
    on_frames: list[np.ndarray]       # BGR frames from on-axis camera
    off_frames: list[np.ndarray]      # BGR frames from off-axis camera
    fps: float

    # Pass 1 outputs — per-instrument RLE-encoded masks (or None per frame)
    on_masks: dict[str, list[np.ndarray | None]] = field(default_factory=dict)
    off_masks: dict[str, list[np.ndarray | None]] = field(default_factory=dict)

    # Pass 2 outputs — per-instrument 2D tracks and visibility
    on_tracks: dict[str, np.ndarray] = field(default_factory=dict)    # {label: (T, 2) xy}
    off_tracks: dict[str, np.ndarray] = field(default_factory=dict)
    on_visibility: dict[str, np.ndarray] = field(default_factory=dict)  # {label: (T,) float}
    off_visibility: dict[str, np.ndarray] = field(default_factory=dict)

    # Pass 3 updates on_tracks/off_tracks/visibility in-place

    # Pass 4 outputs
    trajectories_3d: dict[str, np.ndarray] = field(default_factory=dict)  # {label: (T, 3)}
    reprojection_errors: dict[str, np.ndarray] = field(default_factory=dict)  # {label: (T,)}

    # Pass 5 outputs
    smoothed_3d: dict[str, np.ndarray] = field(default_factory=dict)  # {label: (T, 3)}

    # Pass 6 outputs
    identity_verified: bool = False
    swap_map: dict[str, str] | None = None

    # Calibration data (loaded once, used by passes 4+)
    stereo_calib: dict | None = None
    on_calib: dict | None = None
    off_calib: dict | None = None

    # Depth maps for single-camera fallback in pass 4
    on_depth: list[np.ndarray] = field(default_factory=list)
    off_depth: list[np.ndarray] = field(default_factory=list)
