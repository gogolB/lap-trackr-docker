import React from "react";

type SessionStatus = "recording" | "completed" | "grading" | "graded" | "failed";

interface StatusBadgeProps {
  status: SessionStatus;
  className?: string;
}

const statusConfig: Record<
  SessionStatus,
  { label: string; bgClass: string; textClass: string; dotClass: string }
> = {
  recording: {
    label: "Recording",
    bgClass: "bg-yellow-400/10",
    textClass: "text-yellow-400",
    dotClass: "bg-yellow-400",
  },
  completed: {
    label: "Completed",
    bgClass: "bg-blue-400/10",
    textClass: "text-blue-400",
    dotClass: "bg-blue-400",
  },
  grading: {
    label: "Grading",
    bgClass: "bg-purple-400/10",
    textClass: "text-purple-400",
    dotClass: "bg-purple-400",
  },
  graded: {
    label: "Graded",
    bgClass: "bg-emerald-400/10",
    textClass: "text-emerald-400",
    dotClass: "bg-emerald-400",
  },
  failed: {
    label: "Failed",
    bgClass: "bg-red-400/10",
    textClass: "text-red-400",
    dotClass: "bg-red-400",
  },
};

export default function StatusBadge({ status, className = "" }: StatusBadgeProps) {
  const config = statusConfig[status] ?? statusConfig.completed;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${config.bgClass} ${config.textClass} ${className}`}
    >
      <span
        className={`inline-block h-1.5 w-1.5 rounded-full ${config.dotClass} ${
          status === "recording" || status === "grading" ? "animate-pulse" : ""
        }`}
      />
      {config.label}
    </span>
  );
}
