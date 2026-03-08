"""ChArUco board detection and camera extrinsic calibration."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import config

# Map config string names to OpenCV dictionary IDs
_ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


class ChArUcoCalibrator:
    """Accumulates ChArUco board detections and computes extrinsic pose."""

    _MIN_CORNERS_FOR_MULTI = 12

    def __init__(
        self,
        intrinsics: dict,
        rotation: int = 0,
        flip_h: bool = False,
        flip_v: bool = False,
    ) -> None:
        self._intrinsics = intrinsics

        # Board parameters from config
        self.rows = config.CHARUCO_ROWS
        self.cols = config.CHARUCO_COLS
        self.square_size_m = config.CHARUCO_SQUARE_SIZE_MM / 1000.0
        self.marker_size_m = config.CHARUCO_MARKER_SIZE_MM / 1000.0
        self.aruco_dict_name = config.CHARUCO_DICT

        # Build camera matrix adjusted for rotation/flip transforms.
        # ZED SDK retrieve_image(VIEW.LEFT) returns rectified images,
        # so distortion is zero.  The transforms applied by
        # _apply_transforms (rotation → flip_h → flip_v) shift the
        # principal point and may swap fx/fy.
        self._camera_matrix = self._build_adjusted_matrix(
            intrinsics, rotation, flip_h, flip_v
        )
        self._zero_dist = np.zeros(5, dtype=np.float64)

        # Build the ArUco dictionary and CharucoBoard
        dict_id = _ARUCO_DICTS.get(self.aruco_dict_name, cv2.aruco.DICT_4X4_50)
        self._dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self._board = cv2.aruco.CharucoBoard(
            (self.cols, self.rows),
            self.square_size_m,
            self.marker_size_m,
            self._dictionary,
        )

        # Detector parameters tuned for oblique viewing angles
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.adaptiveThreshWinSizeMin = 3
        detector_params.perspectiveRemovePixelPerCell = 4
        detector_params.minMarkerPerimeterRate = 0.01
        detector_params.minCornerDistanceRate = 0.01
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector_params.cornerRefinementWinSize = 7
        detector_params.adaptiveThreshWinSizeMax = 53
        detector_params.adaptiveThreshWinSizeStep = 5
        detector_params.errorCorrectionRate = 0.8
        if hasattr(detector_params, "useAruco3Detection"):
            detector_params.useAruco3Detection = True

        # Use homography-based corner interpolation (no camera matrix).
        # Camera matrix-based interpolation is too sensitive to intrinsic
        # errors (e.g. swap_eyes using right eye with left eye intrinsics).
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.minMarkers = 1
        board_marker_count = (self.rows * self.cols) // 2
        dictionary_size = int(self._dictionary.bytesList.shape[0])
        # Board-guided refinement only works if the dictionary can cover every
        # marker ID on the board. DICT_5X5_100 can for a 9x14 board; 5X5_50 cannot.
        charuco_params.tryRefineMarkers = dictionary_size >= board_marker_count

        self._charuco_detector = cv2.aruco.CharucoDetector(
            self._board, charuco_params, detector_params
        )

        # Accumulated detections across captures
        self._all_charuco_corners: list[np.ndarray] = []
        self._all_charuco_ids: list[np.ndarray] = []
        self._image_size: tuple[int, int] | None = None

    @property
    def num_captures(self) -> int:
        return len(self._all_charuco_corners)

    @property
    def max_corners(self) -> int:
        """Maximum possible charuco corners for this board."""
        return (self.rows - 1) * (self.cols - 1)

    def detect(self, bgr_image: np.ndarray, camera_name: str = "") -> dict:
        """Detect ChArUco corners in a BGR image and accumulate them.

        Returns a dict with detection results and a preview JPEG.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        self._image_size = (w, h)

        result: dict = {
            "success": False,
            "markers_detected": 0,
            "charuco_corners": 0,
            "coverage_pct": 0.0,
            "total_captures": self.num_captures,
            "preview_jpeg": None,
        }

        # detectBoard returns (charuco_corners, charuco_ids, marker_corners, marker_ids)
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            self._charuco_detector.detectBoard(gray)
        )

        num_markers = 0 if marker_ids is None else len(marker_ids)
        num_corners = 0 if charuco_corners is None else len(charuco_corners)
        marker_id_list = [] if marker_ids is None else sorted(marker_ids.flatten().tolist())
        print(
            f"[calibrator] {camera_name} image={w}x{h} markers={num_markers} corners={num_corners} "
            f"fx={self._camera_matrix[0,0]:.1f} fy={self._camera_matrix[1,1]:.1f} "
            f"cx={self._camera_matrix[0,2]:.1f} cy={self._camera_matrix[1,2]:.1f} "
            f"ids={marker_id_list}"
        )

        if marker_ids is None or len(marker_ids) == 0:
            preview = bgr_image.copy()
            result["preview_jpeg"] = self._encode_preview(preview)
            return result

        result["markers_detected"] = len(marker_ids)

        if num_corners < 4:
            preview = cv2.aruco.drawDetectedMarkers(bgr_image.copy(), marker_corners, marker_ids)
            result["preview_jpeg"] = self._encode_preview(preview)
            return result

        # Accumulate this detection
        self._all_charuco_corners.append(charuco_corners)
        self._all_charuco_ids.append(charuco_ids)

        result["success"] = True
        result["charuco_corners"] = int(num_corners)
        result["coverage_pct"] = round(num_corners / self.max_corners * 100, 1)
        result["total_captures"] = self.num_captures

        # Draw preview with detected corners
        preview = cv2.aruco.drawDetectedMarkers(bgr_image.copy(), marker_corners, marker_ids)
        cv2.aruco.drawDetectedCornersCharuco(preview, charuco_corners, charuco_ids)
        result["preview_jpeg"] = self._encode_preview(preview)

        return result

    def compute(self) -> dict:
        """Compute extrinsic calibration from accumulated detections.

        Returns a dict with the calibration data or an error.
        """
        if self.num_captures < 1:
            return {"success": False, "error": "No captures accumulated"}

        if self._image_size is None:
            return {"success": False, "error": "No image size available"}

        best_corner_count = max(
            (len(ids) for ids in self._all_charuco_ids),
            default=0,
        )

        # Sparse captures are better handled as a single best-frame pose estimate.
        # Multi-frame calibration helps when the board is well-constrained across
        # several frames; it performs poorly when each frame only has a few corners.
        if self.num_captures < 3 or best_corner_count < self._MIN_CORNERS_FOR_MULTI:
            return self._compute_single_frame()

        multi_result = self._compute_multi_frame()
        if not multi_result["success"]:
            return self._compute_single_frame()

        single_result = self._compute_single_frame()
        if single_result["success"] and multi_result["reprojection_error"] > max(
            2.0,
            single_result["reprojection_error"] * 2.0,
        ):
            return single_result

        return multi_result

    def reset(self) -> None:
        """Clear all accumulated detections."""
        self._all_charuco_corners.clear()
        self._all_charuco_ids.clear()
        self._image_size = None

    def get_board_config(self) -> dict:
        """Return the board configuration."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "square_size_mm": config.CHARUCO_SQUARE_SIZE_MM,
            "marker_size_mm": config.CHARUCO_MARKER_SIZE_MM,
            "aruco_dict": self.aruco_dict_name,
        }

    def _compute_single_frame(self) -> dict:
        """Compute extrinsic from the best single frame using solvePnP."""
        best_idx = max(
            range(self.num_captures),
            key=lambda i: len(self._all_charuco_ids[i]),
        )
        corners = self._all_charuco_corners[best_idx]
        ids = self._all_charuco_ids[best_idx]

        # Get 3D object points for the detected charuco corner IDs
        obj_points = self._board.getChessboardCorners()
        obj_pts = np.array([obj_points[i] for i in ids.flatten()], dtype=np.float64)
        img_pts = corners.reshape(-1, 1, 2).astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, self._camera_matrix, self._zero_dist
        )
        if not success:
            return {"success": False, "error": "Pose estimation failed"}

        T = self._build_extrinsic(rvec, tvec)
        reproj_error = self._reprojection_error(obj_pts, img_pts, rvec, tvec)

        return {
            "success": True,
            "extrinsic_matrix": T.tolist(),
            "reprojection_error": round(float(reproj_error), 4),
            "num_frames_used": 1,
        }

    def _compute_multi_frame(self) -> dict:
        """Compute extrinsic using multi-frame calibrateCamera with charuco points."""
        # Build per-frame object/image point arrays from accumulated charuco detections
        obj_points_list = []
        img_points_list = []
        board_corners = self._board.getChessboardCorners()

        for corners, ids in zip(self._all_charuco_corners, self._all_charuco_ids):
            obj_pts = np.array(
                [board_corners[i] for i in ids.flatten()], dtype=np.float32
            )
            img_pts = corners.reshape(-1, 1, 2).astype(np.float32)
            obj_points_list.append(obj_pts.reshape(-1, 1, 3))
            img_points_list.append(img_pts)

        try:
            reproj_error, _, _, rvecs, tvecs = cv2.calibrateCamera(
                obj_points_list,
                img_points_list,
                self._image_size,
                self._camera_matrix.copy(),
                self._zero_dist.copy(),
                flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_INTRINSIC,
            )
        except cv2.error as exc:
            return {"success": False, "error": f"OpenCV calibration failed: {exc}"}

        T = self._average_extrinsics(rvecs, tvecs)

        return {
            "success": True,
            "extrinsic_matrix": T.tolist(),
            "reprojection_error": round(float(reproj_error), 4),
            "num_frames_used": len(rvecs),
        }

    def _build_extrinsic(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Build a 4x4 board-to-camera transform, then invert to get camera-to-board."""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        # Invert: camera-to-board transform (transforms points FROM camera TO board frame)
        return np.linalg.inv(T)

    def _average_extrinsics(
        self, rvecs: list[np.ndarray], tvecs: list[np.ndarray]
    ) -> np.ndarray:
        """Average multiple extrinsic estimates via SVD-orthogonalized rotation."""
        rotations = []
        translations = []
        for rvec, tvec in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rvec)
            T_board_to_cam = np.eye(4)
            T_board_to_cam[:3, :3] = R
            T_board_to_cam[:3, 3] = tvec.flatten()
            T_cam_to_board = np.linalg.inv(T_board_to_cam)
            rotations.append(T_cam_to_board[:3, :3])
            translations.append(T_cam_to_board[:3, 3])

        # Average rotation via SVD orthogonalization of the mean
        R_sum = sum(rotations)
        U, _, Vt = np.linalg.svd(R_sum)
        R_avg = U @ Vt
        # Ensure proper rotation (det=+1)
        if np.linalg.det(R_avg) < 0:
            U[:, -1] *= -1
            R_avg = U @ Vt

        t_avg = np.mean(translations, axis=0)

        T = np.eye(4)
        T[:3, :3] = R_avg
        T[:3, 3] = t_avg
        return T

    def _reprojection_error(
        self,
        obj_pts: np.ndarray,
        img_pts: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> float:
        """Compute RMS reprojection error for a single pose estimate."""
        proj_pts, _ = cv2.projectPoints(
            obj_pts,
            rvec,
            tvec,
            self._camera_matrix,
            self._zero_dist,
        )
        diff = proj_pts.reshape(-1, 2) - img_pts.reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

    @staticmethod
    def _build_adjusted_matrix(
        intrinsics: dict, rotation: int, flip_h: bool, flip_v: bool
    ) -> np.ndarray:
        """Build a camera matrix adjusted for image rotation and flips.

        Applies the same transform order as CameraManager._apply_transforms:
        rotation first, then horizontal flip, then vertical flip.
        """
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        W = int(intrinsics["image_width"])
        H = int(intrinsics["image_height"])

        if rotation == 90:
            fx, fy = fy, fx
            cx, cy = H - 1 - cy, cx
            W, H = H, W
        elif rotation == 180:
            cx = W - 1 - cx
            cy = H - 1 - cy
        elif rotation == 270:
            fx, fy = fy, fx
            cx, cy = cy, W - 1 - cx
            W, H = H, W

        if flip_h:
            cx = W - 1 - cx
        if flip_v:
            cy = H - 1 - cy

        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float64)

    def _encode_preview(self, image: np.ndarray) -> bytes:
        """Encode a BGR image as JPEG bytes."""
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return b""
        return jpeg.tobytes()
