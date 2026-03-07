from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.calibrator import ChArUcoCalibrator
from app.config import config

# Auto-detect whether ZED SDK is available; fall back to mock if not.
try:
    import pyzed.sl  # noqa: F401
    from app.camera_manager import CameraManager
    manager = CameraManager()
except ImportError:
    from app.camera_manager_mock import MockCameraManager  # type: ignore[assignment]
    manager = MockCameraManager()
    print("[camera] ZED SDK not available -- running in MOCK mode")

# Per-camera calibrators, created on first use
_calibrators: dict[str, ChArUcoCalibrator] = {}


def _get_calibrator(camera_name: str) -> ChArUcoCalibrator:
    """Get or create a calibrator for the given camera."""
    if camera_name not in _calibrators:
        intrinsics = manager.get_intrinsics(camera_name)
        if intrinsics is None:
            raise HTTPException(
                status_code=404,
                detail=f"No intrinsics available for '{camera_name}'. Camera may not be open.",
            )
        _calibrators[camera_name] = ChArUcoCalibrator(intrinsics)
    return _calibrators[camera_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.open_cameras()
    yield
    manager.close()


app = FastAPI(
    title="lap-trackr camera service",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CameraConfigApply(BaseModel):
    on_axis_serial: str = ""
    off_axis_serial: str = ""
    on_axis_swap_eyes: bool = False
    off_axis_swap_eyes: bool = False
    on_axis_flip: bool = False
    off_axis_flip: bool = False


class RecordStartRequest(BaseModel):
    session_dir: str


class RecordStartResponse(BaseModel):
    on_axis_path: Optional[str] = None
    off_axis_path: Optional[str] = None


class RecordStopResponse(BaseModel):
    status: str = "stopped"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def get_status():
    """Return recording state and per-camera info."""
    return manager.status()


@app.get("/cameras")
def list_cameras():
    """List all ZED cameras detected on the system."""
    return manager.list_cameras()


@app.post("/record/start", response_model=RecordStartResponse)
def record_start(body: RecordStartRequest):
    """Start SVO2 recording on every open camera."""
    try:
        paths = manager.start_recording(body.session_dir)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return RecordStartResponse(**paths)


@app.post("/record/stop", response_model=RecordStopResponse)
def record_stop():
    """Stop any active recording."""
    if not manager.recording:
        raise HTTPException(status_code=409, detail="Not recording")
    manager.stop_recording()
    return RecordStopResponse()


@app.get("/stream/{camera_name}")
def stream(camera_name: str, eye: str = "left"):
    """MJPEG stream for the requested camera."""
    if camera_name not in manager.cameras:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_name}' not found. "
                   f"Available: {list(manager.cameras.keys())}",
        )

    if eye not in ("left", "right"):
        raise HTTPException(status_code=400, detail="eye must be 'left' or 'right'")

    def generate():
        while True:
            frame = manager.get_frame(camera_name, eye=eye)
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Intrinsics / camera info
# ---------------------------------------------------------------------------

@app.get("/intrinsics/{camera_name}")
def get_intrinsics(camera_name: str):
    """Return ZED SDK factory intrinsics for a camera."""
    intrinsics = manager.get_intrinsics(camera_name)
    if intrinsics is None:
        raise HTTPException(
            status_code=404,
            detail=f"No intrinsics for '{camera_name}'. Camera may not be open.",
        )
    return intrinsics


@app.get("/camera-info")
def get_camera_info():
    """Return per-camera info (serial, resolution, fps) and SDK version."""
    return manager.get_camera_info()


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------

@app.post("/config/apply")
def apply_config(body: CameraConfigApply):
    """Apply camera configuration (eye swap, flip, serial assignment)."""
    config_dict = body.model_dump()
    manager.apply_config(config_dict)
    return {"status": "applied", "config": config_dict}


# ---------------------------------------------------------------------------
# Calibration endpoints
# ---------------------------------------------------------------------------

@app.post("/calibration/capture/{camera_name}")
def calibration_capture(camera_name: str):
    """Capture a frame, detect ChArUco corners, and accumulate."""
    if camera_name not in manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")

    jpeg_bytes, bgr = manager.capture_calibration_frame(camera_name)
    if bgr is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame")

    calibrator = _get_calibrator(camera_name)
    result = calibrator.detect(bgr)

    # Convert preview JPEG to base64 for JSON transport
    preview_b64 = None
    if result.get("preview_jpeg"):
        preview_b64 = base64.b64encode(result["preview_jpeg"]).decode("ascii")

    return {
        "success": result["success"],
        "markers_detected": result["markers_detected"],
        "charuco_corners": result["charuco_corners"],
        "coverage_pct": result["coverage_pct"],
        "total_captures": result["total_captures"],
        "preview_jpeg_b64": preview_b64,
    }


@app.post("/calibration/compute/{camera_name}")
def calibration_compute(camera_name: str):
    """Compute extrinsic calibration from accumulated captures."""
    if camera_name not in manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")

    calibrator = _get_calibrator(camera_name)
    intrinsics = manager.get_intrinsics(camera_name)
    result = calibrator.compute()

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result.get("error", "Calibration failed"))

    # Build the full calibration JSON structure
    calibration = {
        "version": 1,
        "is_global": False,
        "camera_name": camera_name,
        "intrinsics": intrinsics,
        "extrinsic_matrix": result["extrinsic_matrix"],
        "board_config": calibrator.get_board_config(),
        "quality": {
            "reprojection_error": result["reprojection_error"],
            "num_frames_used": result["num_frames_used"],
        },
    }

    return calibration


@app.post("/calibration/reset/{camera_name}")
def calibration_reset(camera_name: str):
    """Reset accumulated calibration captures for a camera."""
    if camera_name in _calibrators:
        _calibrators[camera_name].reset()
    return {"status": "reset", "camera": camera_name}


@app.get("/calibration/status")
def calibration_status():
    """Return capture counts and board config for all cameras."""
    status = {}
    for name in manager.cameras:
        if name in _calibrators:
            cal = _calibrators[name]
            status[name] = {
                "total_captures": cal.num_captures,
                "board_config": cal.get_board_config(),
            }
        else:
            # Not yet initialized
            status[name] = {
                "total_captures": 0,
                "board_config": {
                    "rows": config.CHARUCO_ROWS,
                    "cols": config.CHARUCO_COLS,
                    "square_size_mm": config.CHARUCO_SQUARE_SIZE_MM,
                    "marker_size_mm": config.CHARUCO_MARKER_SIZE_MM,
                    "aruco_dict": config.CHARUCO_DICT,
                },
            }
    return status
