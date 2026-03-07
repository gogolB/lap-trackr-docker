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

export default function SessionDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [isGrading, setIsGrading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
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
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : "Failed to submit for grading."
      );
    } finally {
      setIsGrading(false);
    }
  };

  const handleDownload = async () => {
    if (!id) return;
    setActionError("");
    setIsDownloading(true);
    try {
      await downloadSession(id);
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : "Failed to download session."
      );
    } finally {
      setIsDownloading(false);
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
            {/* Download */}
            {(session.status === "completed" || session.status === "graded") && (
              <button
                onClick={handleDownload}
                disabled={isDownloading}
                className="btn-secondary gap-2"
              >
                {isDownloading ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-400/30 border-t-slate-400" />
                    Downloading...
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
                        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"
                      />
                    </svg>
                    Download
                  </>
                )}
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
            {session.status === "completed" && (
              <button
                onClick={handleGrade}
                disabled={isGrading}
                className="btn-primary gap-2"
              >
                {isGrading ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    Submitting...
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
                    Submit for Grading
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
            {progress && progress.total > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">
                    {progress.stage || "Processing"}
                  </span>
                  <span className="font-mono text-slate-300">
                    {progress.current.toLocaleString()} / {progress.total.toLocaleString()} frames
                  </span>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-700">
                  <div
                    className="h-full rounded-full bg-orange-400 transition-all duration-300"
                    style={{ width: `${Math.min(progress.percent, 100)}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-slate-500">
                  <span>{progress.percent.toFixed(1)}% complete</span>
                  <span>This page updates automatically</span>
                </div>
              </div>
            )}
            {(!progress || progress.total === 0) && (
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
            {progress && progress.total > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">
                    {progress.stage || "Processing"}
                  </span>
                  <span className="font-mono text-slate-300">
                    Step {progress.current} / {progress.total}
                  </span>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-700">
                  <div
                    className="h-full rounded-full bg-purple-400 transition-all duration-300"
                    style={{ width: `${Math.min(progress.percent, 100)}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-slate-500">
                  <span>{progress.percent.toFixed(1)}% complete</span>
                  <span>This page updates automatically</span>
                </div>
              </div>
            )}
            {(!progress || progress.total === 0) && (
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
                    {value !== undefined && value !== null
                      ? typeof value === "number"
                        ? value.toFixed(2)
                        : value
                      : "N/A"}
                  </p>
                  <p className="mt-0.5 text-xs text-slate-500">{meta.unit}</p>
                </div>
              );
            })}
          </div>

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
