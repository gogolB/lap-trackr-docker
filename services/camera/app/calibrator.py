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

    def __init__(self, intrinsics: dict) -> None:
        self._intrinsics = intrinsics

        # Board parameters from config
        self.rows = config.CHARUCO_ROWS
        self.cols = config.CHARUCO_COLS
        self.square_size_m = config.CHARUCO_SQUARE_SIZE_MM / 1000.0
        self.marker_size_m = config.CHARUCO_MARKER_SIZE_MM / 1000.0
        self.aruco_dict_name = config.CHARUCO_DICT

        # Build the ArUco dictionary and CharucoBoard
        dict_id = _ARUCO_DICTS.get(self.aruco_dict_name, cv2.aruco.DICT_4X4_50)
        self._dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self._board = cv2.aruco.CharucoBoard(
            (self.cols, self.rows),
            self.square_size_m,
            self.marker_size_m,
            self._dictionary,
        )
        self._detector = cv2.aruco.ArucoDetector(
            self._dictionary, cv2.aruco.DetectorParameters()
        )

        # Build camera matrix and distortion from intrinsics
        self._camera_matrix = np.array([
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1],
        ], dtype=np.float64)

        disto = intrinsics.get("distortion", [0, 0, 0, 0, 0])
        self._dist_coeffs = np.array(disto[:5], dtype=np.float64)

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

    def detect(self, bgr_image: np.ndarray) -> dict:
        """Detect ChArUco corners in a BGR image and accumulate them.

        Returns a dict with detection results and a preview JPEG.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        self._image_size = (w, h)

        # Detect ArUco markers first
        marker_corners, marker_ids, _ = self._detector.detectMarkers(gray)

        result: dict = {
            "success": False,
            "markers_detected": 0,
            "charuco_corners": 0,
            "coverage_pct": 0.0,
            "total_captures": self.num_captures,
            "preview_jpeg": None,
        }

        if marker_ids is None or len(marker_ids) == 0:
            # Draw on preview even if no markers found
            preview = bgr_image.copy()
            result["preview_jpeg"] = self._encode_preview(preview)
            return result

        result["markers_detected"] = len(marker_ids)

        # Interpolate ChArUco corners
        num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self._board
        )

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

        # If we have enough frames, use calibrateCameraCharuco for better accuracy,
        # then extract the average extrinsic. Otherwise use single-frame estimatePose.
        if self.num_captures >= 3:
            return self._compute_multi_frame()
        else:
            return self._compute_single_frame()

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
        """Compute extrinsic from the best single frame."""
        # Use the frame with most corners
        best_idx = max(
            range(self.num_captures),
            key=lambda i: len(self._all_charuco_ids[i]),
        )
        corners = self._all_charuco_corners[best_idx]
        ids = self._all_charuco_ids[best_idx]

        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            corners, ids, self._board, self._camera_matrix, self._dist_coeffs,
            None, None,
        )
        if not success:
            return {"success": False, "error": "Pose estimation failed"}

        T = self._build_extrinsic(rvec, tvec)

        return {
            "success": True,
            "extrinsic_matrix": T.tolist(),
            "reprojection_error": 0.0,  # Not available for single frame
            "num_frames_used": 1,
        }

    def _compute_multi_frame(self) -> dict:
        """Compute extrinsic using multi-frame calibrateCameraCharuco."""
        # calibrateCameraCharuco refines the camera matrix, but we trust ZED's
        # factory calibration, so we fix it and only extract the extrinsics.
        try:
            reproj_error, _, _, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                self._all_charuco_corners,
                self._all_charuco_ids,
                self._board,
                self._image_size,
                self._camera_matrix.copy(),
                self._dist_coeffs.copy(),
                flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_INTRINSIC,
            )
        except cv2.error as exc:
            return {"success": False, "error": f"OpenCV calibration failed: {exc}"}

        # Average the extrinsic transforms
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

    def _encode_preview(self, image: np.ndarray) -> bytes:
        """Encode a BGR image as JPEG bytes."""
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return b""
        return jpeg.tobytes()
