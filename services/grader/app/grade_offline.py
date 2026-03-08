"""Standalone offline grading CLI.

Run the v2 pipeline on a session directory without Redis, PostgreSQL, or Docker.
Designed for processing on a workstation with a proper GPU.

Usage:
    python -m app.grade_offline /path/to/session_dir [--sample-interval N] [--device DEVICE] \
        [--sam2-model PATH] [--sam2-config NAME] [--cotracker-model PATH]

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("grader.offline")


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
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
    # Find on_axis path — prefer SVO2, fall back to sentinel
    on_axis_svo = session_dir / "on_axis.svo2"
    off_axis_svo = session_dir / "off_axis.svo2"

    # The pipeline uses on_axis_path to derive the session directory.
    # Even if the SVO2 doesn't exist, the MP4+NPZ loader uses the parent dir.
    job: dict[str, Any] = {
        "session_id": session_dir.name,
        "on_axis_path": str(on_axis_svo),
        "off_axis_path": str(off_axis_svo),
    }

    # Calibration paths
    calib_on = session_dir / "calibration_on_axis.json"
    if calib_on.exists():
        job["calibration_path"] = str(calib_on)

    stereo_calib = session_dir / "stereo_calibration.json"
    if stereo_calib.exists():
        job["stereo_calibration_path"] = str(stereo_calib)

    # Camera config from session metadata
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


def _print_progress(stage: str, current: int, total: int, detail: str = "") -> None:
    """Print progress to stdout."""
    if total > 0:
        pct = current / total * 100
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  {stage}: {detail}", end="", flush=True)
        if current >= total:
            print()  # newline when stage completes
    else:
        print(f"  {stage}: {detail}", flush=True)


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
        help="Compute device (default: auto-detect).",
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
        "--cotracker-model",
        type=str,
        default=None,
        help="Path to CoTracker checkpoint (.pth). If omitted, Pass 2 is skipped in offline mode.",
    )
    args = parser.parse_args()

    session_dir: Path = args.session_dir.resolve()
    if not session_dir.is_dir():
        print(f"Error: {session_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Check for required files
    has_on = (session_dir / "on_axis_left.mp4").exists() or (session_dir / "on_axis.svo2").exists()
    has_off = (session_dir / "off_axis_left.mp4").exists() or (session_dir / "off_axis.svo2").exists()
    if not has_on or not has_off:
        print(
            "Error: session directory must contain on-axis and off-axis video files\n"
            "  Expected: on_axis_left.mp4 + off_axis_left.mp4 (or .svo2 files)",
            file=sys.stderr,
        )
        sys.exit(1)

    has_tips = (session_dir / "tip_init.json").exists() or (session_dir / "tip_detections.json").exists()
    if not has_tips:
        print(
            "Error: session directory must contain tip_init.json or tip_detections.json",
            file=sys.stderr,
        )
        sys.exit(1)

    # Configure environment before importing pipeline modules
    device = args.device or _detect_device()
    os.environ.setdefault("FRAME_SAMPLE_INTERVAL", str(args.sample_interval))
    os.environ.setdefault("PIPELINE_MODE", "v2")

    # Set device override for all GPU passes
    os.environ["_GRADER_DEVICE"] = device

    # Set model paths if provided
    if args.sam2_model:
        os.environ["SAM2_MODEL_PATH"] = str(Path(args.sam2_model).resolve())
    if args.sam2_config:
        os.environ["SAM2_CONFIG_PATH"] = args.sam2_config
    if args.cotracker_model:
        os.environ["_COTRACKER_MODEL_PATH"] = str(Path(args.cotracker_model).resolve())

    print(f"Session:  {session_dir}")
    print(f"Device:   {device}")
    print(f"Sampling: every {args.sample_interval} frame(s)")
    if args.sam2_model:
        print(f"SAM2:     {args.sam2_model}")
    if args.cotracker_model:
        print(f"CoTracker: {args.cotracker_model}")
    print()

    # Import pipeline after env is configured
    from app.pipeline import run_v2_pipeline

    job = _build_job(session_dir)
    logger.info("Job: %s", {k: v for k, v in job.items() if k != "camera_config"})

    t0 = time.monotonic()
    results = run_v2_pipeline(job, on_progress=_print_progress)
    elapsed = time.monotonic() - t0

    # Write results to session_dir/results/
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = results.get("metrics", {})
    poses = results.get("poses", [])
    timings = results.get("timings", {})
    warnings = results.get("warnings", [])

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "poses.json", "w") as f:
        json.dump(poses, f, indent=2)

    with open(results_dir / "timings.json", "w") as f:
        json.dump(
            {"pipeline_mode": "v2", "timings": timings, "device": device},
            f,
            indent=2,
        )

    # Summary
    print()
    print("=" * 60)
    print("Pipeline complete")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Device:     {device}")
    if timings:
        for stage, t in timings.items():
            print(f"  {stage}: {t:.1f}s")

    print()
    print("Metrics:")
    for key, value in metrics.items():
        if key == "per_instrument":
            continue
        print(f"  {key}: {value}")

    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    print()
    print(f"Results written to: {results_dir}")

    # List generated files
    for f in sorted(results_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
