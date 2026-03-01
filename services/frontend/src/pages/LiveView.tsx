import React, { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { startSession, stopSession, type Session } from "../api/client";

type CameraName = "on_axis" | "off_axis";
type ViewStatus = "idle" | "recording" | "stopping";

export default function LiveView() {
  const [camera, setCamera] = useState<CameraName>("on_axis");
  const [status, setStatus] = useState<ViewStatus>("idle");
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const navigate = useNavigate();

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
      const session = await startSession();
      setActiveSession(session);
      setStatus("recording");
      setElapsed(0);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start recording."
      );
    }
  }, []);

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

  const streamUrl = `/ws/camera/stream/${camera}`;

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

      {/* Main content grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
        {/* Video stream */}
        <div className="lg:col-span-3">
          <div className="card overflow-hidden p-0">
            <div className="relative aspect-video w-full bg-black">
              <img
                src={streamUrl}
                alt={`${camera} camera stream`}
                className="h-full w-full object-contain"
              />
              {/* Recording overlay */}
              {status === "recording" && (
                <div className="absolute left-4 top-4 flex items-center gap-2 rounded-lg bg-black/60 px-3 py-1.5 backdrop-blur-sm">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
                  <span className="font-mono text-sm font-bold text-white">
                    REC {formatElapsed(elapsed)}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Controls panel */}
        <div className="space-y-4">
          {/* Camera selector */}
          <div className="card">
            <h3 className="mb-3 text-sm font-semibold text-slate-300">
              Camera
            </h3>
            <select
              value={camera}
              onChange={(e) => setCamera(e.target.value as CameraName)}
              className="input-field"
            >
              <option value="on_axis">On-Axis</option>
              <option value="off_axis">Off-Axis</option>
            </select>
          </div>

          {/* Recording controls */}
          <div className="card">
            <h3 className="mb-3 text-sm font-semibold text-slate-300">
              Recording
            </h3>

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
          </div>

          {/* Timer display */}
          {status === "recording" && (
            <div className="card text-center">
              <h3 className="mb-2 text-sm font-semibold text-slate-300">
                Elapsed Time
              </h3>
              <p className="font-mono text-4xl font-bold text-white tabular-nums">
                {formatElapsed(elapsed)}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
