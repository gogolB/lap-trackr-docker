import React, { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  startSession,
  stopSession,
  captureStereoCalibrationFrame,
  computeStereoCalibration,
  resetStereoCalibration,
  getCalibrationStatus,
  getDefaultCalibrations,
  type Session,
  type StereoCaptureResult,
  type StereoCalibrationResult,
  type CalibrationStatus,
  type CalibrationDefault,
  type BoardConfig,
} from "../api/client";

type ViewStatus = "idle" | "recording" | "stopping";

type CameraFeed = "on_axis" | "off_axis";
type EyeFeed = "left" | "right";

interface VisibleFeed {
  camera: CameraFeed;
  eye: EyeFeed;
}

const ALL_FEEDS: VisibleFeed[] = [
  { camera: "on_axis", eye: "left" },
  { camera: "on_axis", eye: "right" },
  { camera: "off_axis", eye: "left" },
  { camera: "off_axis", eye: "right" },
];

export default function LiveView() {
  const [status, setStatus] = useState<ViewStatus>("idle");
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const navigate = useNavigate();

  // Pre-record settings
  const [sessionName, setSessionName] = useState("");
  const [visibleCameras, setVisibleCameras] = useState<Set<CameraFeed>>(
    new Set(["on_axis", "off_axis"])
  );
  const [visibleEyes, setVisibleEyes] = useState<Set<EyeFeed>>(
    new Set(["left"])
  );

  const visibleFeeds = ALL_FEEDS.filter(
    (f) => visibleCameras.has(f.camera) && visibleEyes.has(f.eye)
  );
  const gridCols = visibleFeeds.length <= 1 ? 1 : 2;

  // Calibration state
  const [calibOpen, setCalibOpen] = useState(false);
  const [captures, setCaptures] = useState(0);
  const [lastCapture, setLastCapture] = useState<StereoCaptureResult | null>(null);
  const [calibResult, setCalibResult] = useState<StereoCalibrationResult | null>(null);
  const [calibError, setCalibError] = useState("");
  const [calibLoading, setCalibLoading] = useState(false);
  const [calibStatus, setCalibStatus] = useState<CalibrationStatus | null>(null);
  const [defaults, setDefaults] = useState<CalibrationDefault[]>([]);
  const [boardConfig, setBoardConfig] = useState<BoardConfig | null>(null);

  // Timer management
  useEffect(() => {
    if (status === "recording") {
      timerRef.current = setInterval(() => {
        setElapsed((prev) => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [status]);

  // Fetch calibration status when panel opens
  useEffect(() => {
    if (calibOpen) {
      refreshCalibStatus();
    }
  }, [calibOpen]);

  const refreshCalibStatus = async () => {
    try {
      const [statusData, defaultsData] = await Promise.all([
        getCalibrationStatus(),
        getDefaultCalibrations(),
      ]);
      setCalibStatus(statusData);
      setDefaults(defaultsData);
      const camStatus = statusData["on_axis"];
      if (camStatus) {
        setCaptures(camStatus.total_captures);
        setBoardConfig(camStatus.board_config);
      }
    } catch {
      // Ignore - status is optional
    }
  };

  const formatElapsed = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  const handleStart = useCallback(async () => {
    setError("");
    try {
      const session = await startSession(sessionName);
      setActiveSession(session);
      setStatus("recording");
      setElapsed(0);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start recording."
      );
    }
  }, [sessionName]);

  const handleStop = useCallback(async () => {
    if (!activeSession) return;
    setError("");
    setStatus("stopping");
    try {
      await stopSession(activeSession.id);
      navigate(`/sessions/${activeSession.id}`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to stop recording."
      );
      setStatus("recording");
    }
  }, [activeSession, navigate]);

  const handleCapture = async () => {
    setCalibError("");
    setCalibLoading(true);
    try {
      const result = await captureStereoCalibrationFrame();
      setLastCapture(result);
      // Use on_axis capture count as representative
      setCaptures(result.on_axis.total_captures);
      if (!result.on_axis.success && !result.off_axis.success) {
        setCalibError("No ChArUco corners detected on either camera. Adjust the board position.");
      } else if (!result.on_axis.success) {
        setCalibError("No corners detected on on-axis camera.");
      } else if (!result.off_axis.success) {
        setCalibError("No corners detected on off-axis camera.");
      }
    } catch (err) {
      setCalibError(
        err instanceof Error ? err.message : "Capture failed"
      );
    } finally {
      setCalibLoading(false);
    }
  };

  const handleCompute = async () => {
    setCalibError("");
    setCalibLoading(true);
    try {
      const result = await computeStereoCalibration(true);
      setCalibResult(result);
      await refreshCalibStatus();
    } catch (err) {
      setCalibError(
        err instanceof Error ? err.message : "Compute failed"
      );
    } finally {
      setCalibLoading(false);
    }
  };

  const handleReset = async () => {
    setCalibError("");
    try {
      await resetStereoCalibration();
      setCaptures(0);
      setLastCapture(null);
      setCalibResult(null);
    } catch (err) {
      setCalibError(
        err instanceof Error ? err.message : "Reset failed"
      );
    }
  };

  const hasDefault = (cam: string) =>
    defaults.some((d) => d.camera_name === cam);

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Live View</h1>
          <p className="mt-1 text-sm text-slate-400">
            Monitor and record laparoscopic training sessions
          </p>
        </div>

        {/* Status indicator */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span
              className={`inline-block h-2.5 w-2.5 rounded-full ${
                status === "recording"
                  ? "animate-pulse bg-red-500"
                  : status === "stopping"
                  ? "animate-pulse bg-yellow-500"
                  : "bg-emerald-500"
              }`}
            />
            <span className="text-sm font-medium text-slate-300">
              {status === "recording"
                ? "Recording"
                : status === "stopping"
                ? "Stopping..."
                : "Connected"}
            </span>
          </div>

          {status === "recording" && (
            <div className="rounded-lg bg-slate-800 px-3 py-1.5 font-mono text-lg font-bold text-white tabular-nums">
              {formatElapsed(elapsed)}
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-lg bg-red-400/10 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Camera grid — filtered by visibility settings */}
      <div className={`grid gap-3 ${gridCols === 1 ? "grid-cols-1" : "grid-cols-2"}`}>
        {visibleFeeds.map(({ camera: cam, eye: e }) => (
          <div key={`${cam}-${e}`} className="card overflow-hidden p-0">
            <div className="relative aspect-video w-full bg-black">
              <img
                src={`/ws/camera/stream/${cam}?eye=${e}`}
                alt={`${cam} ${e}`}
                className="h-full w-full object-contain"
                onError={(ev) => {
                  (ev.target as HTMLImageElement).style.display = "none";
                  const fallback = (ev.target as HTMLImageElement).nextElementSibling as HTMLElement | null;
                  if (fallback?.dataset.fallback) fallback.style.display = "flex";
                }}
              />
              <div
                data-fallback="true"
                className="absolute inset-0 items-center justify-center text-sm text-slate-500"
                style={{ display: "none" }}
              >
                Stream unavailable
              </div>
              {/* Label overlay */}
              <div className="absolute left-2 top-2 rounded bg-black/60 px-2 py-1 text-xs font-medium text-white backdrop-blur-sm">
                {cam.replace("_", " ")} / {e}
              </div>
              {/* Recording indicator on first visible feed */}
              {status === "recording" && visibleFeeds[0]?.camera === cam && visibleFeeds[0]?.eye === e && (
                <div className="absolute right-2 top-2 flex items-center gap-1.5 rounded bg-black/60 px-2 py-1 backdrop-blur-sm">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
                  <span className="font-mono text-xs font-bold text-white">
                    REC {formatElapsed(elapsed)}
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Controls row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {/* Recording controls */}
        <div className="card">
          <h3 className="mb-3 text-sm font-semibold text-slate-300">
            Recording
          </h3>

          {status === "idle" && (
            <div className="mb-3">
              <label className="mb-1 block text-xs font-medium text-slate-400">
                Session Name
              </label>
              <input
                type="text"
                value={sessionName}
                onChange={(ev) => setSessionName(ev.target.value)}
                placeholder="e.g. Peg Transfer Practice"
                className="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500"
              />
            </div>
          )}

          {status === "idle" ? (
            <button
              onClick={handleStart}
              className="btn-primary w-full gap-2"
            >
              <svg
                className="h-4 w-4"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <circle cx="12" cy="12" r="8" />
              </svg>
              Start Recording
            </button>
          ) : (
            <button
              onClick={handleStop}
              disabled={status === "stopping"}
              className="btn-danger w-full gap-2"
            >
              <svg
                className="h-4 w-4"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
              {status === "stopping" ? "Stopping..." : "Stop Recording"}
            </button>
          )}

          {status === "recording" && (
            <div className="mt-3 text-center">
              <p className="font-mono text-2xl font-bold text-white tabular-nums">
                {formatElapsed(elapsed)}
              </p>
              {activeSession && (
                <p className="mt-1 truncate text-xs text-slate-400">
                  {activeSession.name}
                </p>
              )}
            </div>
          )}
        </div>

        {/* View controls */}
        <div className="card">
          <h3 className="mb-3 text-sm font-semibold text-slate-300">
            Camera Views
          </h3>

          <div className="mb-3">
            <label className="mb-1.5 block text-xs font-medium text-slate-400">
              Cameras
            </label>
            <div className="flex gap-2">
              {(["on_axis", "off_axis"] as const).map((cam) => (
                <button
                  key={cam}
                  onClick={() => {
                    setVisibleCameras((prev) => {
                      const next = new Set(prev);
                      if (next.has(cam)) {
                        if (next.size > 1) next.delete(cam);
                      } else {
                        next.add(cam);
                      }
                      return next;
                    });
                  }}
                  className={`flex-1 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                    visibleCameras.has(cam)
                      ? "bg-teal-600 text-white"
                      : "bg-slate-700/50 text-slate-400 hover:bg-slate-700"
                  }`}
                >
                  {cam.replace("_", " ")}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="mb-1.5 block text-xs font-medium text-slate-400">
              Eyes
            </label>
            <div className="flex gap-2">
              {(["left", "right"] as const).map((eye) => (
                <button
                  key={eye}
                  onClick={() => {
                    setVisibleEyes((prev) => {
                      const next = new Set(prev);
                      if (next.has(eye)) {
                        if (next.size > 1) next.delete(eye);
                      } else {
                        next.add(eye);
                      }
                      return next;
                    });
                  }}
                  className={`flex-1 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                    visibleEyes.has(eye)
                      ? "bg-teal-600 text-white"
                      : "bg-slate-700/50 text-slate-400 hover:bg-slate-700"
                  }`}
                >
                  {eye}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Calibration status */}
        <div className="card">
          <h3 className="mb-3 text-sm font-semibold text-slate-300">
            Calibration
          </h3>
          <div className="mb-3 flex flex-wrap gap-2">
            {(["on_axis", "off_axis"] as const).map((cam) => (
              <span
                key={cam}
                className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                  hasDefault(cam)
                    ? "bg-emerald-500/10 text-emerald-400"
                    : "bg-yellow-500/10 text-yellow-400"
                }`}
              >
                {cam.replace("_", " ")}:{" "}
                {hasDefault(cam) ? "Calibrated" : "Not calibrated"}
              </span>
            ))}
          </div>
          <button
            onClick={() => setCalibOpen(!calibOpen)}
            className="w-full rounded-lg bg-slate-700/50 px-3 py-2 text-sm font-medium text-slate-300 transition-colors hover:bg-slate-700"
          >
            {calibOpen ? "Hide Calibration Panel" : "Open Calibration Panel"}
          </button>
        </div>
      </div>

      {/* Stereo Calibration panel (collapsible) */}
      {calibOpen && (
        <div className="card">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">
              Stereo ChArUco Calibration
            </h2>
            <button
              onClick={() => setCalibOpen(false)}
              className="text-sm text-slate-400 hover:text-white"
            >
              Close
            </button>
          </div>

          <p className="mb-4 text-sm text-slate-400">
            Hold the ChArUco board visible to both cameras simultaneously. Each capture detects corners on both cameras at once.
          </p>

          {/* Board config display */}
          {boardConfig && (
            <div className="mb-4 rounded-lg bg-slate-800/50 p-3">
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                Board Configuration
              </h4>
              <div className="grid grid-cols-5 gap-3 text-sm">
                <div>
                  <span className="text-slate-500">Rows:</span>{" "}
                  <span className="text-slate-200">{boardConfig.rows}</span>
                </div>
                <div>
                  <span className="text-slate-500">Cols:</span>{" "}
                  <span className="text-slate-200">{boardConfig.cols}</span>
                </div>
                <div>
                  <span className="text-slate-500">Square:</span>{" "}
                  <span className="text-slate-200">
                    {boardConfig.square_size_mm}mm
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">Marker:</span>{" "}
                  <span className="text-slate-200">
                    {boardConfig.marker_size_mm}mm
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">Dict:</span>{" "}
                  <span className="text-slate-200">
                    {boardConfig.aruco_dict}
                  </span>
                </div>
              </div>
            </div>
          )}

          {calibError && (
            <div className="mb-4 rounded-lg bg-red-400/10 px-4 py-3 text-sm text-red-400">
              {calibError}
            </div>
          )}

          {/* Capture controls */}
          <div className="mb-4 flex items-center gap-4">
            <button
              onClick={handleCapture}
              disabled={calibLoading}
              className="btn-primary gap-2"
            >
              {calibLoading ? (
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              ) : (
                <svg
                  className="h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={2}
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z"
                  />
                </svg>
              )}
              Capture Both Cameras
            </button>

            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-400">Progress:</span>
              <span
                className={`rounded-full px-2.5 py-1 text-xs font-bold ${
                  captures >= 5
                    ? "bg-emerald-500/10 text-emerald-400"
                    : captures > 0
                    ? "bg-yellow-500/10 text-yellow-400"
                    : "bg-slate-700/50 text-slate-400"
                }`}
              >
                {captures}/5 frames
              </span>
            </div>

            <button
              onClick={handleCompute}
              disabled={captures < 1 || calibLoading}
              className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                captures >= 1 && !calibLoading
                  ? "bg-emerald-600 text-white hover:bg-emerald-500"
                  : "cursor-not-allowed bg-slate-700/30 text-slate-500"
              }`}
            >
              Compute Stereo Calibration
            </button>

            <button
              onClick={handleReset}
              className="rounded-lg bg-slate-700/50 px-4 py-2 text-sm font-medium text-slate-300 transition-colors hover:bg-slate-700"
            >
              Reset
            </button>
          </div>

          {/* Side-by-side last capture previews */}
          {lastCapture && (
            <div className="mb-4 grid grid-cols-2 gap-4">
              {(["on_axis", "off_axis"] as const).map((cam) => {
                const cap = lastCapture[cam];
                return (
                  <div key={cam}>
                    <h4 className="mb-2 text-sm font-medium text-slate-400">
                      {cam.replace("_", " ")}
                      {cap.success ? (
                        <span className="ml-2 text-emerald-400">
                          {cap.charuco_corners} corners ({cap.coverage_pct}%)
                        </span>
                      ) : (
                        <span className="ml-2 text-red-400">No corners</span>
                      )}
                    </h4>
                    {cap.preview_jpeg_b64 && (
                      <img
                        src={`data:image/jpeg;base64,${cap.preview_jpeg_b64}`}
                        alt={`${cam} calibration preview`}
                        className="max-h-48 rounded-lg border border-slate-700"
                      />
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Stereo calibration result */}
          {calibResult && (
            <div className="rounded-lg border border-emerald-700/50 bg-emerald-900/10 p-4">
              <h4 className="mb-2 text-sm font-semibold text-emerald-400">
                Stereo Calibration Computed Successfully
              </h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                {(["on_axis", "off_axis"] as const).map((cam) => {
                  const camResult = calibResult[cam];
                  const err = calibResult.stereo[
                    `${cam}_reprojection_error` as keyof typeof calibResult.stereo
                  ] as number;
                  return (
                    <div key={cam} className="space-y-1">
                      <p className="font-medium text-slate-200">
                        {cam.replace("_", " ")}
                      </p>
                      <div className="flex items-center gap-2">
                        <span className="text-slate-400">Reproj error:</span>
                        <span
                          className={`font-mono ${
                            err < 0.5
                              ? "text-emerald-400"
                              : err < 1.0
                              ? "text-amber-400"
                              : "text-red-400"
                          }`}
                        >
                          {err.toFixed(4)} px
                        </span>
                        <span
                          className={`h-2 w-2 rounded-full ${
                            err < 0.5
                              ? "bg-emerald-400"
                              : err < 1.0
                              ? "bg-amber-400"
                              : "bg-red-400"
                          }`}
                        />
                      </div>
                      <div>
                        <span className="text-slate-400">Frames:</span>{" "}
                        <span className="font-mono text-white">
                          {camResult.quality.num_frames_used}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Stereo transform and per-camera calibrations saved as defaults.
              </p>
            </div>
          )}

          {/* Existing defaults */}
          {defaults.length > 0 && (
            <div className="mt-4">
              <h4 className="mb-2 text-sm font-medium text-slate-400">
                Saved Default Calibrations
              </h4>
              <div className="space-y-2">
                {defaults.map((d) => (
                  <div
                    key={d.id}
                    className="flex items-center justify-between rounded-lg bg-slate-800/50 px-4 py-2 text-sm"
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-medium text-slate-200">
                        {d.camera_name.replace("_", " ")}
                      </span>
                      <span className="font-mono text-slate-400">
                        fx={d.fx.toFixed(1)} fy={d.fy.toFixed(1)}
                      </span>
                      {d.reprojection_error != null && (
                        <span className="text-slate-500">
                          err={d.reprojection_error.toFixed(4)}px
                        </span>
                      )}
                    </div>
                    <span className="text-xs text-slate-500">
                      {new Date(d.created_at).toLocaleDateString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
