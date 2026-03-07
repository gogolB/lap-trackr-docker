import React from "react";
import type { SessionStatus } from "../api/client";

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
  exporting: {
    label: "Exporting",
    bgClass: "bg-orange-400/10",
    textClass: "text-orange-400",
    dotClass: "bg-orange-400",
  },
  export_failed: {
    label: "Export Failed",
    bgClass: "bg-red-400/10",
    textClass: "text-red-400",
    dotClass: "bg-red-400",
  },
  awaiting_init: {
    label: "Awaiting Init",
    bgClass: "bg-amber-400/10",
    textClass: "text-amber-400",
    dotClass: "bg-amber-400",
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

const animatedStatuses: SessionStatus[] = ["recording", "grading", "exporting"];

export default function StatusBadge({ status, className = "" }: StatusBadgeProps) {
  const config = statusConfig[status] ?? statusConfig.completed;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${config.bgClass} ${config.textClass} ${className}`}
    >
      <span
        className={`inline-block h-1.5 w-1.5 rounded-full ${config.dotClass} ${
          animatedStatuses.includes(status) ? "animate-pulse" : ""
        }`}
      />
      {config.label}
    </span>
  );
}
