#!/usr/bin/env python3
"""Generate a publication-quality 4-panel composite figure for SAGES poster.

Panels:
  A) Video frame with 2D tip-track overlay
  B) 3D trajectory with convex-hull wireframe
  C) Velocity profile over time
  D) Per-instrument metric comparison

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


# ── Panel D: Dual-instrument radar chart ───────────────────────────────────

# Metrics shown on the radar, in clockwise order.
# Each entry: (json_key, display_label, unit, invert)
# invert=True means *lower* raw values are better (map to outer ring).
_RADAR_METRICS = [
    ("workspace_volume", "Workspace\nVolume", "cm³", False),
    ("avg_speed", "Avg\nSpeed", "mm/s", False),
    ("path_length", "Path\nLength", "mm", False),
    ("economy_of_motion", "Economy of\nMotion", "", False),
    ("max_jerk", "Smoothness\n(1/Jerk)", "mm/s³", True),
]


def _fmt_value(val: float, unit: str) -> str:
    """Compact human-readable value string."""
    if val >= 10000:
        return f"{val:,.0f} {unit}".strip()
    if val >= 100:
        return f"{val:.1f} {unit}".strip()
    if val >= 1:
        return f"{val:.2f} {unit}".strip()
    if val >= 0.001:
        return f"{val:.4f} {unit}".strip()
    return f"{val:.2e} {unit}".strip()


def panel_radar(ax: plt.Axes, metrics: dict) -> None:
    """Dual-polygon radar chart comparing green_tip vs pink_tip.

    Each axis is normalized to the max of the two instruments, so the chart
    shows relative head-to-head comparison without arbitrary benchmarks.
    """
    ax.set_facecolor(BG_PANEL)

    per_inst = metrics.get("per_instrument", {})
    green = per_inst.get("green_tip", {})
    pink = per_inst.get("pink_tip", {})

    if not green and not pink:
        ax.text(0.5, 0.5, "No per-instrument data",
                transform=ax.transAxes, ha="center", va="center",
                color=TEXT_COLOR, fontsize=14)
        ax.set_title("D) Instrument Comparison", color=TEXT_COLOR,
                      fontsize=13, fontweight="bold", pad=10)
        return

    N = len(_RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    g_values = []
    p_values = []
    g_annotations = []
    p_annotations = []

    for key, label, unit, invert in _RADAR_METRICS:
        g_raw = green.get(key, 0)
        p_raw = pink.get(key, 0)
        max_val = max(g_raw, p_raw, 1e-10)

        # Normalize to [0, 1] relative to max of the two instruments
        g_norm = g_raw / max_val
        p_norm = p_raw / max_val

        # For inverted metrics (like jerk), lower is better → flip
        if invert:
            g_norm = 1.0 - g_norm + 0.15 if g_norm < 1.0 else 0.15
            p_norm = 1.0 - p_norm + 0.15 if p_norm < 1.0 else 0.15

        # Ensure a minimum visible radius
        g_norm = max(g_norm, 0.08)
        p_norm = max(p_norm, 0.08)

        g_values.append(g_norm)
        p_values.append(p_norm)
        g_annotations.append(_fmt_value(g_raw, unit))
        p_annotations.append(_fmt_value(p_raw, unit))

    g_closed = g_values + [g_values[0]]
    p_closed = p_values + [p_values[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Reference circles
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(np.linspace(0, 2 * np.pi, 100), [r] * 100,
                color=GRID_COLOR, linewidth=0.5, alpha=0.4)

    # Green instrument polygon
    ax.fill(angles_closed, g_closed, color=GREEN, alpha=0.15)
    ax.plot(angles_closed, g_closed, color=GREEN, linewidth=2.0,
            marker="o", markersize=5, markerfacecolor=GREEN,
            markeredgecolor="white", markeredgewidth=0.8, label="Green Tip")

    # Pink instrument polygon
    ax.fill(angles_closed, p_closed, color=PINK, alpha=0.15)
    ax.plot(angles_closed, p_closed, color=PINK, linewidth=2.0,
            marker="o", markersize=5, markerfacecolor=PINK,
            markeredgecolor="white", markeredgewidth=0.8, label="Pink Tip")

    # Axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels([m[1] for m in _RADAR_METRICS],
                        color=TEXT_COLOR, fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([])  # hide radial ticks — values are annotated directly
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.25)

    # Annotate raw values: single combined label per vertex
    for i, angle in enumerate(angles):
        outer_r = max(g_values[i], p_values[i]) + 0.18
        combined = f"{g_annotations[i]}  /  {p_annotations[i]}"
        ax.text(angle, outer_r, combined,
                ha="center", va="center", fontsize=6.5, color=TEXT_COLOR,
                bbox=dict(boxstyle="round,pad=0.2", fc=BG_PANEL, alpha=0.85,
                          edgecolor=GRID_COLOR, linewidth=0.5))

    ax.legend(loc="lower right", fontsize=8, framealpha=0.6,
              labelcolor=TEXT_COLOR, facecolor=BG_PANEL, edgecolor=GRID_COLOR,
              bbox_to_anchor=(1.15, -0.05))

    # Session summary
    total_time = metrics.get("total_time", 0)
    ws_vol = metrics.get("workspace_volume", 0)
    ax.text(0.5, -0.12,
            f"Session: {total_time:.1f}s  |  Combined Workspace: {ws_vol:.0f} cm³",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color=TEXT_COLOR, alpha=0.7, style="italic")

    ax.set_title("D) Instrument Comparison", color=TEXT_COLOR,
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

    # Panel D — bottom-right (polar radar)
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
