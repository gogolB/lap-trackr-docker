import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.camera_manager import CameraManager


# ---------------------------------------------------------------------------
# Application lifespan -- open cameras on startup, close on shutdown
# ---------------------------------------------------------------------------

manager = CameraManager()


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

class RecordStartRequest(BaseModel):
    session_dir: str


class RecordStartResponse(BaseModel):
    on_axis_path: str | None = None
    off_axis_path: str | None = None


class RecordStopResponse(BaseModel):
    status: str = "stopped"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

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
def stream(camera_name: str):
    """MJPEG stream of the left eye for the requested camera.

    Usage from HTML::

        <img src="http://host:8001/stream/on_axis">
    """
    if camera_name not in manager.cameras:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_name}' not found. "
                   f"Available: {list(manager.cameras.keys())}",
        )

    def generate():
        while True:
            frame = manager.get_frame(camera_name)
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            # Cap at roughly 30 fps.
            time.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
