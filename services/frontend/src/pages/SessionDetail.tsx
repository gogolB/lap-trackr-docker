import React, { useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  getSession,
  getMetrics,
  gradeSession,
  retrySession,
  reExportSession,
  deleteSession,
  downloadSession,
  getSessionDuration,
  getSessionProgress,
  type MetricsData,
  type JobProgress,
  type SessionDetail as SessionDetailType,
} from "../api/client";
import StatusBadge from "../components/StatusBadge";

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return "--:--";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}m ${secs}s`;
}

function formatMetricValue(value: number | null | undefined): string {
  if (value === undefined || value === null) return "N/A";
  return Number.isFinite(value) ? value.toFixed(2) : "N/A";
}

function formatInstrumentName(label: string): string {
  return label
    .replace(/_tip$/i, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function instrumentDotClass(label: string): string {
  if (label.includes("green") || label.includes("left")) return "bg-emerald-400";
  if (label.includes("pink") || label.includes("right")) return "bg-pink-400";
  return "bg-cyan-400";
}

const METRIC_LABELS: Record<string, { label: string; unit: string }> = {
  workspace_volume: { label: "Workspace Volume", unit: "cm\u00B3" },
  avg_speed: { label: "Average Speed", unit: "cm/s" },
  max_jerk: { label: "Max Jerk", unit: "cm/s\u00B3" },
  path_length: { label: "Path Length", unit: "cm" },
  economy_of_motion: { label: "Economy of Motion", unit: "%" },
  total_time: { label: "Total Time", unit: "s" },
};

const CHART_COLORS = [
  "#14b8a6", // teal-500
  "#06b6d4", // cyan-500
  "#8b5cf6", // violet-500
  "#f59e0b", // amber-500
  "#10b981", // emerald-500
  "#3b82f6", // blue-500
];

type StageDefinition = {
  key: string;
  label: string;
};

const EXPORT_STAGES: StageDefinition[] = [
  { key: "export_on_axis", label: "Export On-Axis Cameras" },
  { key: "export_off_axis", label: "Export Off-Axis Cameras" },
  { key: "detect_tips", label: "Detect Initial Tips" },
];

const GRADING_STAGES: StageDefinition[] = [
  { key: "load_on_axis", label: "Load On-Axis Data" },
  { key: "load_off_axis", label: "Load Off-Axis Data" },
  { key: "detect_on_axis", label: "Detect Tips On-Axis" },
  { key: "detect_off_axis", label: "Detect Tips Off-Axis" },
  { key: "render_on_axis", label: "Render On-Axis Overlay" },
  { key: "render_off_axis", label: "Render Off-Axis Overlay" },
  { key: "estimate_poses", label: "Calculate Fused Positions" },
  { key: "calculate_metrics", label: "Calculate Metrics" },
];

const GRADING_STAGES_V2: StageDefinition[] = [
  { key: "load_frames", label: "Load Camera Frames" },
  { key: "pass1_sam2", label: "Segmentation (SAM2)" },
  { key: "pass2_cotracker", label: "Point Tracking (CoTracker)" },
  { key: "pass3_color", label: "Color Gap Filling" },
  { key: "pass4_triangulation", label: "Stereo Triangulation" },
  { key: "pass5_smoothing", label: "Trajectory Smoothing" },
  { key: "pass6_identity", label: "Identity Verification" },
  { key: "render_tracking", label: "Render Tracking Videos" },
  { key: "calculate_metrics", label: "Calculate Metrics" },
];

const V2_ONLY_KEYS = new Set([
  "load_frames",
  "pass1_sam2",
  "pass2_cotracker",
  "pass3_color",
  "pass4_triangulation",
  "pass5_smoothing",
  "pass6_identity",
  "render_tracking",
]);

function resolveGradingStages(progress?: JobProgress): StageDefinition[] {
  const stageKeys = progress?.stages ? Object.keys(progress.stages) : [];
  const isV2 = stageKeys.some((key) => V2_ONLY_KEYS.has(key));
  return isV2 ? GRADING_STAGES_V2 : GRADING_STAGES;
}

function formatEta(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "--";
  const rounded = Math.max(0, Math.ceil(seconds));
  const mins = Math.floor(rounded / 60);
  const secs = rounded % 60;
  if (mins === 0) return `${secs}s`;
  const hours = Math.floor(mins / 60);
  if (hours === 0) return `${mins}m ${secs}s`;
  return `${hours}h ${mins % 60}m`;
}

type StageView = StageDefinition & {
  active: boolean;
  done: boolean;
  current: number;
  total: number;
  percent: number;
  detail: string;
  startedAt: number | null;
};

function estimateEtaSeconds(stage: StageView): number | null {
  if (
    !stage.active ||
    !stage.startedAt ||
    stage.total <= 0 ||
    stage.current <= 0
  ) {
    return null;
  }

  const elapsed = Date.now() / 1000 - stage.startedAt;
  if (elapsed <= 0) return null;

  const remaining = stage.total - stage.current;
  if (remaining <= 0) return 0;

  const rate = stage.current / elapsed;
  if (rate <= 0) return null;
  return remaining / rate;
}

function buildStageProgress(stages: StageDefinition[], progress?: JobProgress): StageView[] {
  const explicitStages = progress?.stages;
  if (explicitStages && Object.keys(explicitStages).length > 0) {
    return stages.map((stage) => {
      const live = explicitStages[stage.key];
      const active = live?.status === "running";
      const done = live?.status === "completed";
      return {
        ...stage,
        active,
        done,
        current: live?.current ?? 0,
        total: live?.total ?? 0,
        percent: done ? 100 : Math.min(live?.percent ?? 0, 100),
        detail: live?.detail ?? (done ? "Done" : "Pending"),
        startedAt: live?.started_at ?? null,
      };
    });
  }

  const isComplete = progress?.stage === "complete";
  const activeIndex = isComplete
    ? stages.length
    : stages.findIndex((stage) => stage.key === progress?.stage);

  return stages.map((stage, index) => {
    const done = isComplete || (activeIndex >= 0 && index < activeIndex);
    const active = !isComplete && activeIndex === index;
    const current = active ? progress?.current ?? 0 : done ? 1 : 0;
    const total = active ? progress?.total ?? 0 : done ? 1 : 0;
    const percent = active ? Math.min(progress?.percent ?? 0, 100) : done ? 100 : 0;

    return {
      ...stage,
      active,
      done,
      current,
      total,
      percent,
      detail: active ? progress?.detail ?? "" : done ? "Done" : "Pending",
      startedAt: active ? progress?.stage_started_at ?? null : null,
    };
  });
}

function formatStageCount(stage: {
  active: boolean;
  done: boolean;
  current: number;
  total: number;
}): string {
  if (stage.active && stage.total > 0) {
    return `${stage.current.toLocaleString()} / ${stage.total.toLocaleString()}`;
  }
  if (stage.done) {
    return "Done";
  }
  return "Pending";
}

export default function SessionDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [isGrading, setIsGrading] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);
  const [isReExporting, setIsReExporting] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [actionError, setActionError] = useState("");

  // Fetch session - poll every 3s while grading
  const {
    data: session,
    isLoading: sessionLoading,
    error: sessionError,
  } = useQuery({
    queryKey: ["session", id],
    queryFn: () => getSession(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.status === "grading" || data?.status === "exporting" || data?.status === "awaiting_init"
        ? 3000
        : false;
    },
  });

  // Fetch metrics when graded
  const { data: metrics } = useQuery({
    queryKey: ["metrics", id],
    queryFn: () => getMetrics(id!),
    enabled: !!id && session?.status === "graded",
  });

  // Fetch live progress while exporting or grading
  const isProcessing = session?.status === "exporting" || session?.status === "grading";
  const { data: progress } = useQuery({
    queryKey: ["progress", id],
    queryFn: () => getSessionProgress(id!),
    enabled: !!id && isProcessing,
    refetchInterval: isProcessing ? 2000 : false,
  });

  const handleGrade = async () => {
    if (!id) return;
    setActionError("");
    setIsGrading(true);
    try {
      await gradeSession(id);
      queryClient.invalidateQueries({ queryKey: ["session", id] });
      queryClient.invalidateQueries({ queryKey: ["progress", id] });
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : "Failed to submit for grading."
      );
    } finally {
      setIsGrading(false);
    }
  };

  const handleRetry = async () => {
    if (!id) return;
    setActionError("");
    setIsRetrying(true);
    try {
      await retrySession(id);
      queryClient.invalidateQueries({ queryKey: ["session", id] });
      queryClient.invalidateQueries({ queryKey: ["progress", id] });
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : "Failed to retry session."
      );
    } finally {
      setIsRetrying(false);
    }
  };

  const handleReExport = async () => {
    if (!id) return;
    setActionError("");
    setIsReExporting(true);
    try {
      await reExportSession(id);
      queryClient.invalidateQueries({ queryKey: ["session", id] });
      queryClient.invalidateQueries({ queryKey: ["progress", id] });
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : "Failed to restart export."
      );
    } finally {
      setIsReExporting(false);
    }
  };

  const handleDownload = async () => {
    if (!id) return;
    setActionError("");
    try {
      await downloadSession(id);
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : "Failed to download session."
      );
    }
  };

  const handleDelete = async () => {
    if (!id) return;
    setActionError("");
    setIsDeleting(true);
    try {
      await deleteSession(id);
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      navigate("/sessions");
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : "Failed to delete session."
      );
      setIsDeleting(false);
    }
  };

  if (sessionLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="flex flex-col items-center gap-4">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-600 border-t-teal-500" />
          <p className="text-sm text-slate-400">Loading session...</p>
        </div>
      </div>
    );
  }

  if (sessionError || !session) {
    return (
      <div className="space-y-4">
        <div className="rounded-lg bg-red-400/10 px-4 py-3 text-sm text-red-400">
          Failed to load session details.
        </div>
        <Link to="/sessions" className="btn-secondary">
          Back to Sessions
        </Link>
      </div>
    );
  }

  // Prepare chart data
  const chartData = metrics
    ? Object.entries(metrics)
        .filter(([key]) => key in METRIC_LABELS)
        .map(([key, value]) => ({
          name: METRIC_LABELS[key]?.label ?? key,
          value: Number(value),
          unit: METRIC_LABELS[key]?.unit ?? "",
        }))
    : [];
  const perInstrumentMetrics = metrics?.per_instrument
    ? Object.entries(metrics.per_instrument)
    : [];

  const canReExport =
    (!!session.on_axis_path || !!session.off_axis_path) &&
    session.status !== "recording" &&
    session.status !== "grading";
  const canGrade = session.status === "completed" || session.status === "graded";
  const exportStageProgress =
    session.status === "exporting"
      ? buildStageProgress(EXPORT_STAGES, progress)
      : [];
  const gradingStageProgress =
    session.status === "grading"
      ? buildStageProgress(resolveGradingStages(progress), progress)
      : [];

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-slate-400">
        <Link to="/sessions" className="hover:text-teal-400">
          Sessions
        </Link>
        <span>/</span>
        <span className="text-slate-200">{session.name}</span>
      </nav>

      {/* Session header */}
      <div className="card">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold text-white">{session.name}</h1>
              <StatusBadge status={session.status} />
            </div>
            <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-sm text-slate-400">
              <span>Started: {formatDate(session.started_at)}</span>
              {session.stopped_at && (
                <span>Stopped: {formatDate(session.stopped_at)}</span>
              )}
              <span>
                Duration: {formatDuration(getSessionDuration(session))}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Re-export */}
            {canReExport && (
              <button
                onClick={handleReExport}
                disabled={isReExporting}
                className="btn-secondary gap-2"
              >
                {isReExporting ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-400/30 border-t-slate-400" />
                    {session.status === "exporting" ? "Restarting..." : "Re-exporting..."}
                  </>
                ) : (
                  <>
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
                        d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182"
                      />
                    </svg>
                    {session.status === "exporting" ? "Restart Export" : "Re-export"}
                  </>
                )}
              </button>
            )}

            {/* Download */}
            {(session.status === "completed" || session.status === "graded") && (
              <button
                onClick={handleDownload}
                className="btn-secondary gap-2"
              >
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
                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"
                  />
                </svg>
                Download
              </button>
            )}

            {/* Initialize tips */}
            {session.status === "awaiting_init" && (
              <button
                onClick={() => navigate(`/sessions/${id}/init`)}
                className="btn-primary gap-2"
              >
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
                    d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z"
                  />
                </svg>
                Initialize Tips
              </button>
            )}

            {/* Submit for grading */}
            {canGrade && (
              <button
                onClick={handleGrade}
                disabled={isGrading}
                className="btn-primary gap-2"
              >
                {isGrading ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    {session.status === "graded" ? "Re-submitting..." : "Submitting..."}
                  </>
                ) : (
                  <>
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
                        d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    {session.status === "graded" ? "Re-grade" : "Submit for Grading"}
                  </>
                )}
              </button>
            )}

            {/* Retry failed sessions */}
            {(session.status === "failed" || session.status === "export_failed") && (
              <button
                onClick={handleRetry}
                disabled={isRetrying}
                className="btn-secondary gap-2"
              >
                {isRetrying ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-400/30 border-t-slate-400" />
                    Retrying...
                  </>
                ) : (
                  <>
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
                        d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182"
                      />
                    </svg>
                    Retry
                  </>
                )}
              </button>
            )}

            {/* Delete button */}
            {!showDeleteConfirm ? (
              <button
                onClick={() => setShowDeleteConfirm(true)}
                className="btn-secondary gap-2 text-red-400 hover:bg-red-600 hover:text-white"
              >
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
                    d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"
                  />
                </svg>
                Delete
              </button>
            ) : (
              <div className="flex items-center gap-2">
                <span className="text-sm text-slate-400">Are you sure?</span>
                <button
                  onClick={handleDelete}
                  disabled={isDeleting}
                  className="btn-danger"
                >
                  {isDeleting ? "Deleting..." : "Confirm Delete"}
                </button>
                <button
                  onClick={() => setShowDeleteConfirm(false)}
                  className="btn-secondary"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {actionError && (
        <div className="rounded-lg bg-red-400/10 px-4 py-3 text-sm text-red-400">
          {actionError}
        </div>
      )}

      {/* Exporting in progress */}
      {session.status === "exporting" && (
        <div className="card">
          <div className="space-y-4 py-4">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-3 border-orange-400/30 border-t-orange-400" />
              <div>
                <p className="text-lg font-semibold text-white">
                  Exporting session files...
                </p>
                <p className="text-sm text-slate-400">
                  {progress?.detail || "Converting SVO2 recordings to MP4 + depth data"}
                </p>
              </div>
            </div>
            <div className="space-y-3">
              {exportStageProgress.map((stage) => (
                <div key={stage.key} className="space-y-1.5">
                  <div className="flex items-center justify-between text-sm">
                    <span className={stage.active ? "text-white" : "text-slate-400"}>
                      {stage.label}
                    </span>
                    <span className="font-mono text-slate-300">
                      {formatStageCount(stage)}
                    </span>
                  </div>
                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-700">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        stage.done
                          ? "bg-emerald-400"
                          : stage.active
                            ? "bg-orange-400"
                            : "bg-slate-600"
                      }`}
                      style={{ width: `${stage.percent}%` }}
                    />
                  </div>
                  {stage.active && (
                    <div className="flex items-center justify-between text-xs text-slate-500">
                      <span>{stage.detail || "Processing..."}</span>
                      <span>
                        {estimateEtaSeconds(stage) !== null
                          ? `ETA ${formatEta(estimateEtaSeconds(stage))}`
                          : "ETA calculating..."}
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
            {!progress && (
              <p className="text-sm text-slate-500">
                Waiting for progress data...
              </p>
            )}
          </div>
        </div>
      )}

      {/* Export failed */}
      {session.status === "export_failed" && (
        <div className="card border-red-500/30">
          <div className="flex items-start gap-3">
            <svg
              className="mt-0.5 h-5 w-5 flex-shrink-0 text-red-400"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
              />
            </svg>
            <div>
              <p className="font-semibold text-red-400">Export Failed</p>
              <p className="mt-1 text-sm text-slate-400">
                Failed to export SVO2 files. The original recordings are still
                available on the device.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Awaiting tip initialization */}
      {session.status === "awaiting_init" && (
        <div className="card">
          <div className="flex flex-col items-center gap-4 py-8">
            <svg
              className="h-10 w-10 text-amber-400"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z"
              />
            </svg>
            <div className="text-center">
              <p className="text-lg font-semibold text-white">
                Tip Initialization Required
              </p>
              <p className="mt-1 text-sm text-slate-400">
                Instrument tips have been auto-detected. Review and confirm their positions before grading.
              </p>
            </div>
            <button
              onClick={() => navigate(`/sessions/${id}/init`)}
              className="btn-primary gap-2"
            >
              Initialize Tips
            </button>
          </div>
        </div>
      )}

      {/* Grading in progress */}
      {session.status === "grading" && (
        <div className="card">
          <div className="space-y-4 py-4">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-3 border-purple-400/30 border-t-purple-400" />
              <div>
                <p className="text-lg font-semibold text-white">
                  Grading in progress...
                </p>
                <p className="text-sm text-slate-400">
                  {progress?.detail || "Analyzing instrument movements and calculating metrics"}
                </p>
              </div>
            </div>
            <div className="space-y-3">
              {gradingStageProgress.map((stage) => (
                <div key={stage.key} className="space-y-1.5">
                  <div className="flex items-center justify-between text-sm">
                    <span className={stage.active ? "text-white" : "text-slate-400"}>
                      {stage.label}
                    </span>
                    <span className="font-mono text-slate-300">
                      {formatStageCount(stage)}
                    </span>
                  </div>
                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-700">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        stage.done
                          ? "bg-emerald-400"
                          : stage.active
                            ? "bg-purple-400"
                            : "bg-slate-600"
                      }`}
                      style={{ width: `${stage.percent}%` }}
                    />
                  </div>
                  {stage.active && (
                    <div className="flex items-center justify-between text-xs text-slate-500">
                      <span>{stage.detail || "Processing..."}</span>
                      <span>
                        {estimateEtaSeconds(stage) !== null
                          ? `ETA ${formatEta(estimateEtaSeconds(stage))}`
                          : "ETA calculating..."}
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
            {!progress && (
              <p className="text-sm text-slate-500">
                Waiting for progress data...
              </p>
            )}
          </div>
        </div>
      )}

      {/* Failed status */}
      {session.status === "failed" && (
        <div className="card border-red-500/30">
          <div className="flex items-start gap-3">
            <svg
              className="mt-0.5 h-5 w-5 flex-shrink-0 text-red-400"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
              />
            </svg>
            <div>
              <p className="font-semibold text-red-400">Session Failed</p>
              <p className="mt-1 text-sm text-slate-400">
                {session.grading_result?.error ||
                  "An error occurred during processing. Please try again with a new session."}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Warnings from grading */}
      {session.status === "graded" &&
        session.grading_result?.warnings &&
        session.grading_result.warnings.length > 0 && (
        <div className="card border-amber-500/30">
          <div className="flex items-start gap-3">
            <svg
              className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-400"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
              />
            </svg>
            <div>
              <p className="font-semibold text-amber-400">Grading Warnings</p>
              <ul className="mt-1 space-y-1 text-sm text-slate-400">
                {session.grading_result.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Graded results */}
      {session.status === "graded" && metrics && (
        <>
          {/* Metric cards */}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
            {Object.entries(METRIC_LABELS).map(([key, meta]) => {
              const value = metrics[key as keyof MetricsData];
              return (
                <div key={key} className="card text-center">
                  <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
                    {meta.label}
                  </p>
                  <p className="mt-2 text-2xl font-bold text-white">
                    {formatMetricValue(
                      typeof value === "number" ? value : undefined
                    )}
                  </p>
                  <p className="mt-0.5 text-xs text-slate-500">{meta.unit}</p>
                </div>
              );
            })}
          </div>

          {perInstrumentMetrics.length > 0 && (
            <div className="grid gap-4 lg:grid-cols-2">
              {perInstrumentMetrics.map(([label, instrumentMetrics]) => (
                <div key={label} className="card">
                  <div className="mb-4 flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-white">
                      {formatInstrumentName(label)}
                    </h2>
                    <span
                      className={`inline-flex h-2.5 w-2.5 rounded-full ${instrumentDotClass(label)}`}
                    />
                  </div>
                  <div className="space-y-2">
                    {Object.entries(METRIC_LABELS).map(([key, meta]) => (
                      <div
                        key={`${label}-${key}`}
                        className="flex items-center justify-between text-sm"
                      >
                        <span className="text-slate-400">{meta.label}</span>
                        <span className="font-medium text-white">
                          {formatMetricValue(instrumentMetrics[key as keyof typeof instrumentMetrics])} {meta.unit}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Bar chart */}
          <div className="card">
            <h2 className="mb-4 text-lg font-semibold text-white">
              Performance Metrics
            </h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  margin={{ top: 10, right: 30, left: 20, bottom: 60 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#334155"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    angle={-30}
                    textAnchor="end"
                    height={80}
                    axisLine={{ stroke: "#475569" }}
                    tickLine={{ stroke: "#475569" }}
                  />
                  <YAxis
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    axisLine={{ stroke: "#475569" }}
                    tickLine={{ stroke: "#475569" }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1e293b",
                      border: "1px solid #334155",
                      borderRadius: "0.5rem",
                      color: "#f1f5f9",
                      fontSize: "0.875rem",
                    }}
                    formatter={(value: number, _name: string, props: { payload?: { unit?: string } }) => [
                      `${value.toFixed(2)} ${props.payload?.unit ?? ""}`,
                      "Value",
                    ]}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {chartData.map((_entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={CHART_COLORS[index % CHART_COLORS.length]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
