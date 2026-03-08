# Calibration Guide

Camera calibration establishes the spatial relationship between each camera and the workspace, enabling accurate 3D measurements. The system uses ChArUco boards (a combination of checkerboard and ArUco markers) for robust detection even at oblique viewing angles.

## What Gets Calibrated

1. **Extrinsic calibration** (per camera): 4x4 transform from camera frame to board (workspace) frame
2. **Stereo calibration**: 4x4 transform between the two cameras (`T_on_to_off`)
3. **Intrinsics** are read from the ZED SDK factory calibration (not re-calibrated)

## ChArUco Board

### Default Configuration

Set via environment variables in `.env`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHARUCO_ROWS` | 9 | Board rows |
| `CHARUCO_COLS` | 14 | Board columns |
| `CHARUCO_SQUARE_SIZE_MM` | 20.0 | Checkerboard square edge length |
| `CHARUCO_MARKER_SIZE_MM` | 15.0 | ArUco marker edge length |
| `CHARUCO_DICT` | DICT_5X5_100 | ArUco dictionary |

### Generating the Board Image

```python
python3 -c "
import cv2
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
b = cv2.aruco.CharucoBoard((9, 14), 0.020, 0.015, d)
img = b.generateImage((900, 1400))
cv2.imwrite('charuco_9x14_dict5x5_100.png', img)
"
```

### Printing

- Print at **actual size** (no scaling/fit-to-page)
- Verify dimensions with a ruler: squares should measure exactly 20mm
- Mount on a rigid, flat surface (foam board, acrylic, etc.)
- Matte finish preferred to reduce reflections

## Calibration Workflow

### Prerequisites

- Cameras mounted in their final operating positions
- ChArUco board printed and mounted
- System running (`docker compose up -d`)

### Single Camera Calibration

1. Open the web UI and navigate to **Live View**
2. Expand the **Calibration** panel
3. Hold the ChArUco board in view of the target camera
4. Click **Capture** -- the system detects markers and corners
5. Move the board to a different position/angle and capture again
6. Repeat for at least **5 captures** at varied positions and angles
   - Cover different areas of the frame
   - Include some tilted views (30-45 degrees)
   - Ensure the board is fully visible in each capture
7. Click **Compute** for the camera
8. Check the **reprojection error** -- should be < 1.0 px (ideally < 0.5 px)
9. If error is too high, click **Reset** and start over with better captures

### Stereo Calibration

Stereo calibration computes the transform between the two cameras. The board must be visible to **both cameras simultaneously**.

1. Position the board so both cameras can see it
2. Click **Capture Stereo** -- detects the board in both camera views
3. Capture 5+ frames with the board at varied positions
4. Click **Compute Stereo**
5. The system computes per-camera extrinsics and derives `T_on_to_off`

### Saving as Default

Calibrations computed through the UI are automatically saved as global defaults:
- `{CALIBRATION_DIR}/default/on_axis.json`
- `{CALIBRATION_DIR}/default/off_axis.json`
- `{CALIBRATION_DIR}/default/stereo_calibration.json`

These defaults are automatically copied into each new session directory when recording starts.

## How Calibration Works Internally

### Detection

The `ChArUcoCalibrator` class in `services/camera/app/calibrator.py`:

1. Detects ArUco markers in the image using optimized detector parameters (tuned for oblique angles)
2. Interpolates ChArUco corners from detected markers
3. Accumulates corners across multiple captures
4. Requires at least 4 corners per capture to be useful

### Computation

Two strategies based on accumulated data:

**Single-frame strategy** (< 3 captures or < 12 best corners):
- Uses `cv2.solvePnP()` on the best single frame
- Less accurate but works with minimal data

**Multi-frame strategy** (>= 3 captures with >= 12 corners):
- Uses `cv2.calibrateCamera()` with `CALIB_USE_INTRINSIC_GUESS`
- Refines across all accumulated frames
- Falls back to single-frame if reprojection error is > 2x worse

### Camera Matrix Adjustment

The calibrator adjusts intrinsics for any applied image transforms (rotation, flip) before detection:
- **Rotation 90/270**: Swaps fx/fy, transforms cx/cy, swaps width/height
- **Rotation 180**: Flips cx and cy
- **Flip H**: Flips cx
- **Flip V**: Flips cy
- Distortion is always zero (ZED SDK provides rectified images)

### Stereo Transform

The inter-camera transform is derived from individual calibrations:

```
T_on_to_off = T_off_to_board @ inv(T_on_to_board)
```

This transform is used during dual-camera fusion in the grading pipeline to triangulate 3D positions from both camera views.

## Tips for Good Calibration

1. **Board quality matters**: Ensure the board is flat and printed at exact size
2. **Lighting**: Even illumination, avoid shadows on the board
3. **Coverage**: Spread captures across the full field of view
4. **Angles**: Include tilted views, not just head-on
5. **Stability**: Hold the board steady during capture (motion blur degrades detection)
6. **Distance**: Capture at the typical working distance (where instruments will be)
7. **Recalibrate** after moving cameras or changing their mounting

## Calibration Data in Grading

During grading, calibration data is used to:

1. **Back-project 2D detections to 3D**: Using intrinsics (fx, fy, cx, cy) + depth
2. **Transform to world frame**: Using the extrinsic matrix (camera-to-board transform)
3. **Fuse dual cameras**: Using the stereo transform to triangulate and cross-reference

Without calibration, the grader falls back to default intrinsic values and operates in the camera frame (no world-frame transformation). Metrics will still be computed but may be less accurate.
