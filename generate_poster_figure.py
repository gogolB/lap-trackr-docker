#!/usr/bin/env python3
"""Generate a publication-quality 4-panel composite figure for SAGES poster.

Panels:
  A) Video frame with 2D tip-track overlay
  B) 3D trajectory with convex-hull wireframe
  C) Velocity profile over time
  D) Radar chart of skill-assessment dimensions

Usage:
    python generate_poster_figure.py <session_dir> [options]
    python generate_poster_figure.py data/sessions/2026-03-08_19-32-00 -o poster.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# ── Colour palette ──────────────────────────────────────────────────────────
GREEN = "#2ecc71"
PINK = "#e74c8c"
BG_DARK = "#1a1a2e"
BG_PANEL = "#16213e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"
ACCENT = "#0f3460"


# ── Data loaders ────────────────────────────────────────────────────────────

def load_poses(results_dir: Path) -> list[dict]:
    with open(results_dir / "poses.json") as f:
        return json.load(f)


def load_metrics(results_dir: Path) -> dict:
    with open(results_dir / "metrics.json") as f:
        return json.load(f)


def load_detections(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "detections_on_axis.csv"
    if not csv_path.exists():
        sys.exit(f"Error: {csv_path} not found")
    return pd.read_csv(csv_path)


def grab_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Error: cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = min(frame_idx, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"Error: cannot read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


# ── Panel A: Video frame with 2D tip overlay ────────────────────────────────

def panel_video_overlay(ax: plt.Axes, session_dir: Path, results_dir: Path,
                        frame_idx: int | None) -> None:
    video_path = session_dir / "on_axis_left.mp4"
    if not video_path.exists():
        ax.text(0.5, 0.5, "on_axis_left.mp4\nnot found",
                transform=ax.transAxes, ha="center", va="center",
                color=TEXT_COLOR, fontsize=14)
        ax.set_facecolor(BG_PANEL)
        ax.set_title("A) Video Frame with Tip Overlay", color=TEXT_COLOR,
                      fontsize=13, fontweight="bold", pad=10)
        return

    total_frames = get_video_frame_count(video_path)
    if frame_idx is None:
        frame_idx = total_frames // 2

    frame = grab_frame(video_path, frame_idx)
    det = load_detections(results_dir)

    # Draw trails up to the chosen frame
    trail_window = min(frame_idx, 60)  # last N frames of trail
    trail_start = max(0, frame_idx - trail_window)

    ax.imshow(frame, aspect="auto")

    for label, color in [("green_tip", GREEN), ("pink_tip", PINK)]:
        subset = det[(det["label"] == label) &
                     (det["frame_idx"] >= trail_start) &
                     (det["frame_idx"] <= frame_idx)].sort_values("frame_idx")
        if subset.empty:
            continue

        xs = subset["x"].values
        ys = subset["y"].values

        # Gradient trail: increasing alpha toward current frame
        n = len(xs)
        for i in range(1, n):
            alpha = 0.15 + 0.85 * (i / n)
            lw = 1.0 + 2.5 * (i / n)
            ax.plot(xs[i-1:i+1], ys[i-1:i+1], color=color,
                    alpha=alpha, linewidth=lw, solid_capstyle="round")

        # Current position marker
        ax.plot(xs[-1], ys[-1], "o", color=color, markersize=10,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        nice_label = label.replace("_", " ").title()
        ax.annotate(nice_label, (xs[-1], ys[-1]),
                    xytext=(12, -12), textcoords="offset points",
                    color="white", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.8))

    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A) Video Frame with Tip Overlay", color=TEXT_COLOR,
                  fontsize=13, fontweight="bold", pad=10)


# ── Panel B: 3D trajectory with convex hull ─────────────────────────────────

def panel_3d_trajectory(ax: plt.Axes, poses: list[dict]) -> None:
    tips = {}
    for label, color in [("green_tip", GREEN), ("pink_tip", PINK)]:
        pts = np.array([p[label] for p in poses if p.get(label) is not None],
                       dtype=np.float64)
        if len(pts) == 0:
            continue
        pts_mm = pts * 1000.0  # meters → mm
        tips[label] = (pts_mm, color)

    if not tips:
        ax.text2D(0.5, 0.5, "No 3D data", transform=ax.transAxes,
                  ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        ax.set_title("B) 3D Trajectory", color=TEXT_COLOR,
                      fontsize=13, fontweight="bold", pad=10)
        return

    all_pts = np.concatenate([v[0] for v in tips.values()], axis=0)

    # Draw convex hull wireframe
    if len(all_pts) >= 4:
        try:
            hull = ConvexHull(all_pts)
            for simplex in hull.simplices:
                tri = all_pts[simplex]
                verts = [list(zip(tri[:, 0], tri[:, 1], tri[:, 2]))]
                poly = art3d.Poly3DCollection(verts, alpha=0.04,
                                              facecolor="#4ecdc4",
                                              edgecolor="#4ecdc4",
                                              linewidth=0.3)
                ax.add_collection3d(poly)
        except Exception:
            pass

    # Draw tip paths
    for label, (pts_mm, color) in tips.items():
        ax.plot(pts_mm[:, 0], pts_mm[:, 1], pts_mm[:, 2],
                color=color, linewidth=1.5, alpha=0.85,
                label=label.replace("_", " ").title())
        # Start / end markers (no extra legend entries)
        ax.scatter(*pts_mm[0], color=color, s=60, marker="^",
                   edgecolors="white", linewidths=0.8, zorder=5)
        ax.scatter(*pts_mm[-1], color=color, s=60, marker="s",
                   edgecolors="white", linewidths=0.8, zorder=5)

    ax.set_xlabel("X (mm)", color=TEXT_COLOR, fontsize=9, labelpad=5)
    ax.set_ylabel("Y (mm)", color=TEXT_COLOR, fontsize=9, labelpad=5)
    ax.set_zlabel("Z (mm)", color=TEXT_COLOR, fontsize=9, labelpad=5)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_COLOR)
    ax.yaxis.pane.set_edgecolor(GRID_COLOR)
    ax.zaxis.pane.set_edgecolor(GRID_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    ax.view_init(elev=25, azim=-45)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.6,
              labelcolor=TEXT_COLOR, facecolor=BG_PANEL, edgecolor=GRID_COLOR)
    ax.set_title("B) 3D Instrument Trajectory", color=TEXT_COLOR,
                  fontsize=13, fontweight="bold", pad=10)


# ── Panel C: Velocity profile ───────────────────────────────────────────────

def panel_velocity(ax: plt.Axes, poses: list[dict]) -> None:
    ax.set_facecolor(BG_PANEL)

    for label, color in [("green_tip", GREEN), ("pink_tip", PINK)]:
        pts = []
        timestamps = []
        for p in poses:
            if p.get(label) is not None:
                pts.append(p[label])
                timestamps.append(p["timestamp"])

        if len(pts) < 2:
            continue

        pts = np.array(pts, dtype=np.float64) * 1000.0  # m → mm
        ts = np.array(timestamps)

        # Finite differences for speed (mm/s)
        diffs = np.diff(pts, axis=0)
        dt = np.diff(ts)
        dt[dt == 0] = 1e-6  # avoid division by zero
        speeds = np.linalg.norm(diffs, axis=1) / dt

        t_mid = (ts[:-1] + ts[1:]) / 2.0

        # Light smoothing for display
        if len(speeds) > 5:
            kernel = np.ones(5) / 5
            speeds_smooth = np.convolve(speeds, kernel, mode="same")
        else:
            speeds_smooth = speeds

        nice_label = label.replace("_", " ").title()
        ax.plot(t_mid, speeds_smooth, color=color, linewidth=1.5,
                alpha=0.9, label=nice_label)
        ax.fill_between(t_mid, 0, speeds_smooth, color=color, alpha=0.1)

        # Annotate peak
        peak_idx = np.argmax(speeds_smooth)
        ax.annotate(f"Peak: {speeds_smooth[peak_idx]:.0f} mm/s",
                    xy=(t_mid[peak_idx], speeds_smooth[peak_idx]),
                    xytext=(0, 12), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1))

    ax.set_xlabel("Time (s)", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Speed (mm/s)", color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    ax.legend(fontsize=9, framealpha=0.6, labelcolor=TEXT_COLOR,
              facecolor=BG_PANEL, edgecolor=GRID_COLOR)
    ax.set_title("C) Tip Velocity Profile", color=TEXT_COLOR,
                  fontsize=13, fontweight="bold", pad=10)


# ── Panel D: Radar chart ────────────────────────────────────────────────────

_RADAR_DIMENSIONS = [
    "Workspace\nVolume",
    "Economy of\nMotion",
    "Smoothness",
    "Speed\nControl",
    "Path\nEfficiency",
]


def _normalize_metrics(metrics: dict) -> tuple[list[float], list[str]]:
    """Normalize raw metrics to 0-1 for each radar dimension.

    Returns (normalized_values, annotation_strings).
    """
    ws_vol = metrics.get("workspace_volume", 0)       # cm³
    economy = metrics.get("economy_of_motion", 0)      # 0-1
    max_jerk = metrics.get("max_jerk", 1)               # mm/s³
    avg_speed = metrics.get("avg_speed", 0)             # mm/s
    path_length = metrics.get("path_length", 0)         # mm
    total_time = metrics.get("total_time", 1)           # s

    # 1) Workspace volume: 0 → 0, 5000 cm³ → 1
    ws_norm = min(ws_vol / 5000.0, 1.0)

    # 2) Economy of motion: direct 0-1
    eco_norm = economy

    # 3) Smoothness: inverse of jerk (log scale)
    #    Lower jerk = smoother. Use log10 mapping.
    if max_jerk > 0:
        # log10(1e8) = 8 → score 0, log10(1) = 0 → score 1
        smooth_norm = max(1.0 - np.log10(max(max_jerk, 1)) / 8.0, 0.0)
    else:
        smooth_norm = 1.0

    # 4) Speed control: penalize very high avg speed (>200 mm/s is fast)
    #    Ideal range is 20-100 mm/s for deliberate movement
    speed_norm = max(1.0 - abs(avg_speed - 60) / 200.0, 0.0)

    # 5) Path efficiency: economy * time factor
    #    Shorter total path relative to workspace extent = more efficient
    if ws_vol > 0:
        # Approximate workspace extent as cube root of volume (cm → mm)
        extent_mm = (ws_vol ** (1/3)) * 10.0
        path_ratio = path_length / max(extent_mm, 1.0)
        path_eff = max(1.0 - path_ratio / 50.0, 0.0)
    else:
        path_eff = 0.0

    values = [ws_norm, eco_norm, smooth_norm, speed_norm, path_eff]
    def _compact(val: float, unit: str) -> str:
        if abs(val) >= 1e6:
            return f"{val:.1e} {unit}"
        if abs(val) >= 1000:
            return f"{val:.0f} {unit}"
        return f"{val:.1f} {unit}"

    annotations = [
        _compact(ws_vol, "cm³"),
        f"{economy:.3f}",
        _compact(max_jerk, "mm/s³"),
        _compact(avg_speed, "mm/s"),
        _compact(path_length, "mm"),
    ]
    return values, annotations


def panel_radar(ax: plt.Axes, metrics: dict) -> None:
    values, annotations = _normalize_metrics(metrics)
    N = len(_RADAR_DIMENSIONS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Close the polygon
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    ax.set_facecolor(BG_PANEL)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw reference circles
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(np.linspace(0, 2 * np.pi, 100), [r] * 100,
                color=GRID_COLOR, linewidth=0.5, alpha=0.5)

    # Fill polygon
    ax.fill(angles_closed, values_closed, color="#4ecdc4", alpha=0.25)
    ax.plot(angles_closed, values_closed, color="#4ecdc4", linewidth=2.5,
            marker="o", markersize=6, markerfacecolor="#4ecdc4",
            markeredgecolor="white", markeredgewidth=1)

    # Labels and annotations
    ax.set_xticks(angles)
    ax.set_xticklabels(_RADAR_DIMENSIONS, color=TEXT_COLOR, fontsize=8.5,
                        fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"],
                        color=TEXT_COLOR, fontsize=7, alpha=0.6)
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.3)

    # Annotate raw values near each vertex
    for i, (angle, val, ann) in enumerate(zip(angles, values, annotations)):
        r_offset = max(val, 0.15) + 0.13
        ax.text(angle, r_offset, ann, ha="center", va="center",
                fontsize=7.5, color=TEXT_COLOR, alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.2", fc=BG_PANEL, alpha=0.7,
                          edgecolor="none"))

    ax.set_title("D) Skill Assessment", color=TEXT_COLOR,
                  fontsize=13, fontweight="bold", pad=20)


# ── Main figure assembly ────────────────────────────────────────────────────

def generate_figure(session_dir: Path, output: Path, frame_idx: int | None,
                    dpi: int, title: str | None) -> None:
    results_dir = session_dir / "results"
    if not results_dir.exists():
        sys.exit(f"Error: results directory not found: {results_dir}")

    poses = load_poses(results_dir)
    metrics = load_metrics(results_dir)

    if title is None:
        title = f"Lap-Trackr Session: {session_dir.name}"

    # Set up dark theme
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_PANEL,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
    })

    fig = plt.figure(figsize=(16, 12), facecolor=BG_DARK)

    # Panel A — top-left
    ax_a = fig.add_subplot(2, 2, 1)
    panel_video_overlay(ax_a, session_dir, results_dir, frame_idx)

    # Panel B — top-right (3D)
    ax_b = fig.add_subplot(2, 2, 2, projection="3d")
    ax_b.set_facecolor(BG_PANEL)
    panel_3d_trajectory(ax_b, poses)

    # Panel C — bottom-left
    ax_c = fig.add_subplot(2, 2, 3)
    panel_velocity(ax_c, poses)

    # Panel D — bottom-right (polar)
    ax_d = fig.add_subplot(2, 2, 4, polar=True)
    panel_radar(ax_d, metrics)

    # Figure title
    fig.suptitle(title, fontsize=18, fontweight="bold", color="white",
                 y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, dpi=dpi, facecolor=BG_DARK, edgecolor="none",
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved poster figure: {output}  ({dpi} DPI)")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 4-panel SAGES poster figure from a graded session."
    )
    parser.add_argument("session_dir", type=Path,
                        help="Path to session directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("poster_figure.png"),
                        help="Output image path (default: poster_figure.png)")
    parser.add_argument("--frame", type=int, default=None,
                        help="Video frame index for panel A (default: mid-video)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI (default: 300)")
    parser.add_argument("--title", type=str, default=None,
                        help="Figure title (default: session timestamp)")
    args = parser.parse_args()

    if not args.session_dir.exists():
        sys.exit(f"Error: session directory not found: {args.session_dir}")

    generate_figure(args.session_dir, args.output, args.frame, args.dpi,
                    args.title)


if __name__ == "__main__":
    main()
