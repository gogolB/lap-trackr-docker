import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getSessions, getSessionDuration, type Session } from "../api/client";
import StatusBadge from "../components/StatusBadge";

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
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
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export default function Dashboard() {
  const navigate = useNavigate();

  const {
    data: sessions,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["sessions"],
    queryFn: getSessions,
  });

  const recentSessions = sessions?.slice(0, 5) ?? [];

  const totalSessions = sessions?.length ?? 0;
  const gradedSessions =
    sessions?.filter((s) => s.status === "graded").length ?? 0;
  const activeSessions =
    sessions?.filter((s) => s.status === "recording").length ?? 0;

  return (
    <div className="space-y-8">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="mt-1 text-sm text-slate-400">
            Overview of your laparoscopic training sessions
          </p>
        </div>
        <button
          onClick={() => navigate("/live")}
          className="btn-primary gap-2 px-6 py-3 text-base"
        >
          <svg
            className="h-5 w-5"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <circle cx="12" cy="12" r="5" />
          </svg>
          Start Recording
        </button>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="card">
          <p className="text-sm font-medium text-slate-400">Total Sessions</p>
          <p className="mt-1 text-3xl font-bold text-white">{totalSessions}</p>
        </div>
        <div className="card">
          <p className="text-sm font-medium text-slate-400">Graded Sessions</p>
          <p className="mt-1 text-3xl font-bold text-emerald-400">
            {gradedSessions}
          </p>
        </div>
        <div className="card">
          <p className="text-sm font-medium text-slate-400">
            Active Recordings
          </p>
          <p className="mt-1 text-3xl font-bold text-yellow-400">
            {activeSessions}
          </p>
        </div>
      </div>

      {/* Recent sessions */}
      <div className="card">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Recent Sessions</h2>
          <Link
            to="/sessions"
            className="text-sm font-medium text-teal-400 transition-colors hover:text-teal-300"
          >
            View all
          </Link>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-600 border-t-teal-500" />
          </div>
        ) : error ? (
          <div className="rounded-lg bg-red-400/10 px-4 py-3 text-sm text-red-400">
            Failed to load sessions.
          </div>
        ) : recentSessions.length === 0 ? (
          <div className="py-12 text-center">
            <svg
              className="mx-auto mb-3 h-10 w-10 text-slate-600"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
              />
            </svg>
            <p className="text-sm text-slate-400">
              No sessions yet. Start your first recording!
            </p>
          </div>
        ) : (
          <div className="divide-y divide-slate-700/50">
            {recentSessions.map((session) => (
              <Link
                key={session.id}
                to={`/sessions/${session.id}`}
                className="flex items-center justify-between px-1 py-3.5 transition-colors hover:bg-slate-700/20 rounded-lg px-3 -mx-1"
              >
                <div className="flex items-center gap-4">
                  <div>
                    <p className="text-sm font-medium text-white">
                      {formatDate(session.started_at)}
                    </p>
                    <p className="mt-0.5 text-xs text-slate-400">
                      Duration: {formatDuration(getSessionDuration(session))}
                    </p>
                  </div>
                </div>
                <StatusBadge status={session.status} />
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
