"""Standalone offline grading CLI.

Run the v2 pipeline on a session directory without Redis, PostgreSQL, or Docker.
Designed for processing on a workstation with a proper GPU.

Usage:
    python -m app.grade_offline /path/to/session_dir [--sample-interval N] [--device DEVICE] \
        [--sam2-model PATH] [--sam2-config NAME] [--cotracker-model PATH] \
        [--segmentation-backend sam2|sam3]

The session directory should contain:
    - on_axis_left.mp4, off_axis_left.mp4  (exported video)
    - on_axis_depth.npz, off_axis_depth.npz  (depth maps)
    - tip_init.json or tip_detections.json  (instrument tip positions)
    - calibration_on_axis.json, calibration_off_axis.json  (optional)
    - stereo_calibration.json  (optional)

Results are written to <session_dir>/results/:
    - metrics.json, poses.json, timings.json
    - tracking_on_axis.mp4, tracking_off_axis.mp4
    - detections_on_axis.csv, detections_off_axis.csv
    - tracked_positions_world.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.tree import Tree

# ---------------------------------------------------------------------------
# Rich console and logging setup
# ---------------------------------------------------------------------------

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("grader.offline")


def _detect_device() -> str:
    """Auto-detect the best available compute device (CUDA > MPS > CPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("CUDA device detected: %s", name)
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device detected")
            return "mps"
    except ImportError:
        pass
    logger.info("No GPU detected, using CPU")
    return "cpu"


def _build_job(session_dir: Path) -> dict[str, Any]:
    """Build a pipeline job dict from files in the session directory."""
    on_axis_svo = session_dir / "on_axis.svo2"
    off_axis_svo = session_dir / "off_axis.svo2"

    job: dict[str, Any] = {
        "session_id": session_dir.name,
        "on_axis_path": str(on_axis_svo),
        "off_axis_path": str(off_axis_svo),
    }

    calib_on = session_dir / "calibration_on_axis.json"
    if calib_on.exists():
        job["calibration_path"] = str(calib_on)

    stereo_calib = session_dir / "stereo_calibration.json"
    if stereo_calib.exists():
        job["stereo_calibration_path"] = str(stereo_calib)

    metadata_path = session_dir / "session_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            camera_config = metadata.get("camera_config")
            if isinstance(camera_config, dict):
                job["camera_config"] = camera_config
        except Exception as exc:
            logger.warning("Failed to read session_metadata.json: %s", exc)

    return job


def _check_session_files(session_dir: Path) -> Tree:
    """Build a rich Tree showing which session files are present."""
    tree = Tree(f"[bold]{session_dir.name}")

    expected_files = [
        ("on_axis_left.mp4", True),
        ("off_axis_left.mp4", True),
        ("on_axis_depth.npz", True),
        ("off_axis_depth.npz", True),
        ("tip_init.json", False),
        ("tip_detections.json", False),
        ("calibration_on_axis.json", False),
        ("calibration_off_axis.json", False),
        ("stereo_calibration.json", False),
        ("session_metadata.json", False),
    ]

    for filename, required in expected_files:
        exists = (session_dir / filename).exists()
        if exists:
            size_mb = (session_dir / filename).stat().st_size / (1024 * 1024)
            tree.add(f"[green]{filename}[/] ({size_mb:.1f} MB)")
        elif required:
            tree.add(f"[red]{filename}[/] (MISSING)")
        else:
            tree.add(f"[dim]{filename}[/] (not found)")

    return tree


# ---------------------------------------------------------------------------
# Rich progress tracking
# ---------------------------------------------------------------------------

class RichProgressTracker:
    """Wraps rich.Progress to work with the pipeline's callback interface."""

    def __init__(self) -> None:
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[stage]}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[detail]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=False,
        )
        self._tasks: dict[str, TaskID] = {}

    def __enter__(self) -> RichProgressTracker:
        self.progress.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self.progress.__exit__(*args)

    def callback(self, stage: str, current: int, total: int, detail: str = "") -> None:
        """Pipeline progress callback — creates or updates a rich task per stage."""
        if stage not in self._tasks:
            task_id = self.progress.add_task(
                stage,
                total=max(total, 1),
                stage=stage,
                detail=detail,
            )
            self._tasks[stage] = task_id
        else:
            task_id = self._tasks[stage]
            self.progress.update(
                task_id,
                completed=current,
                total=max(total, 1),
                detail=detail,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the v2 grading pipeline offline on a session directory.",
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Path to the session directory containing MP4, NPZ, and calibration files.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=1,
        help="Keep every Nth frame (default: 1 = all frames).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Compute device (default: auto-detect CUDA > MPS > CPU).",
    )
    parser.add_argument(
        "--segmentation-backend",
        type=str,
        default="sam2",
        choices=["sam2", "sam3"],
        help="Segmentation model for Pass 1 (default: sam2).",
    )
    parser.add_argument(
        "--sam2-model",
        type=str,
        default=None,
        help="Path to SAM2 checkpoint (e.g. sam2.1_hiera_large.pt).",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default=None,
        help="SAM2 Hydra config name (default: configs/sam2.1/sam2.1_hiera_l.yaml).",
    )
    parser.add_argument(
        "--sam3-model",
        type=str,
        default=None,
        help="Path to SAM3 checkpoint. If omitted, auto-downloads from HuggingFace.",
    )
    parser.add_argument(
        "--cotracker-model",
        type=str,
        default=None,
        help="Path to CoTracker checkpoint (.pth). If omitted, Pass 2 is skipped.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Render debug videos for each pipeline stage (segmentation masks, CoTracker points).",
    )
    args = parser.parse_args()

    session_dir: Path = args.session_dir.resolve()
    if not session_dir.is_dir():
        console.print(f"[red bold]Error:[/] {session_dir} is not a directory")
        sys.exit(1)

    # Check for required files
    has_on = (session_dir / "on_axis_left.mp4").exists() or (session_dir / "on_axis.svo2").exists()
    has_off = (session_dir / "off_axis_left.mp4").exists() or (session_dir / "off_axis.svo2").exists()
    if not has_on or not has_off:
        console.print(
            "[red bold]Error:[/] session directory must contain on-axis and off-axis video files\n"
            "  Expected: on_axis_left.mp4 + off_axis_left.mp4 (or .svo2 files)",
        )
        sys.exit(1)

    has_tips = (session_dir / "tip_init.json").exists() or (session_dir / "tip_detections.json").exists()
    seg_backend = args.segmentation_backend

    if not has_tips and seg_backend != "sam3":
        console.print(
            "[red bold]Error:[/] session directory must contain tip_init.json or tip_detections.json\n"
            "  (or use [cyan]--segmentation-backend sam3[/] for text-based prompting)",
        )
        sys.exit(1)

    # Configure environment before importing pipeline modules
    device = args.device or _detect_device()
    os.environ.setdefault("FRAME_SAMPLE_INTERVAL", str(args.sample_interval))
    os.environ.setdefault("PIPELINE_MODE", "v2")

    os.environ["_GRADER_DEVICE"] = device
    os.environ["_SEGMENTATION_BACKEND"] = seg_backend

    if args.sam2_model:
        os.environ["SAM2_MODEL_PATH"] = str(Path(args.sam2_model).resolve())
    if args.sam2_config:
        os.environ["SAM2_CONFIG_PATH"] = args.sam2_config
    if args.sam3_model:
        os.environ["SAM3_MODEL_PATH"] = str(Path(args.sam3_model).resolve())
    if args.cotracker_model:
        os.environ["_COTRACKER_MODEL_PATH"] = str(Path(args.cotracker_model).resolve())
    if args.debug:
        os.environ["_DEBUG_RENDER"] = "1"

    # Display session info
    console.print()
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column(style="bold")
    info_table.add_column()
    info_table.add_row("Session", str(session_dir))
    info_table.add_row("Device", f"[cyan]{device}[/]")
    info_table.add_row("Segmentation", f"[cyan]{seg_backend.upper()}[/]")
    info_table.add_row("Sampling", f"every {args.sample_interval} frame(s)")
    if args.debug:
        info_table.add_row("Debug", "[yellow]ON — rendering per-stage debug videos[/]")
    if args.sam2_model:
        info_table.add_row("SAM2 Model", args.sam2_model)
    if args.sam3_model:
        info_table.add_row("SAM3 Model", args.sam3_model)
    if args.cotracker_model:
        info_table.add_row("CoTracker", args.cotracker_model)

    console.print(Panel(info_table, title="[bold]Offline Grader", border_style="blue"))
    console.print()

    # Show session file tree
    file_tree = _check_session_files(session_dir)
    console.print(file_tree)
    console.print()

    # Import pipeline after env is configured
    from app.pipeline import run_v2_pipeline

    job = _build_job(session_dir)
    logger.info("Job: %s", {k: v for k, v in job.items() if k != "camera_config"})

    # Run pipeline with rich progress
    t0 = time.monotonic()
    with RichProgressTracker() as tracker:
        results = run_v2_pipeline(job, on_progress=tracker.callback)
    elapsed = time.monotonic() - t0

    # Write results to session_dir/results/
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = results.get("metrics", {})
    poses = results.get("poses", [])
    timings = results.get("timings", {})
    warnings_list = results.get("warnings", [])

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "poses.json", "w") as f:
        json.dump(poses, f, indent=2)

    with open(results_dir / "timings.json", "w") as f:
        json.dump(
            {
                "pipeline_mode": "v2",
                "segmentation_backend": seg_backend,
                "timings": timings,
                "device": device,
            },
            f,
            indent=2,
        )

    # Summary panel
    console.print()

    # Timings table
    timing_table = Table(title="Pipeline Timings", show_lines=False)
    timing_table.add_column("Stage", style="bold")
    timing_table.add_column("Time", justify="right")
    timing_table.add_column("", justify="left")

    total_time = sum(timings.values()) if timings else elapsed
    for stage, t in timings.items():
        pct = (t / total_time * 100) if total_time > 0 else 0
        bar_len = int(pct / 100 * 20)
        bar = "[green]" + "█" * bar_len + "[/]" + "░" * (20 - bar_len)
        timing_table.add_row(stage, f"{t:.1f}s", bar)
    timing_table.add_row("[bold]Total", f"[bold]{elapsed:.1f}s", "")
    console.print(timing_table)
    console.print()

    # Metrics table
    metrics_table = Table(title="Grading Metrics", show_lines=False)
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")

    for key, value in metrics.items():
        if key == "per_instrument":
            continue
        if isinstance(value, float):
            metrics_table.add_row(key, f"{value:.4f}")
        else:
            metrics_table.add_row(key, str(value))
    console.print(metrics_table)

    # Warnings
    if warnings_list:
        console.print()
        console.print("[yellow bold]Warnings:[/]")
        for w in warnings_list:
            console.print(f"  [yellow]- {w}[/]")

    # Output files
    console.print()
    output_tree = Tree(f"[bold green]Results written to: {results_dir}")
    for f in sorted(results_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        output_tree.add(f"{f.name} [dim]({size_mb:.1f} MB)[/]")
    console.print(output_tree)
    console.print()


if __name__ == "__main__":
    main()
