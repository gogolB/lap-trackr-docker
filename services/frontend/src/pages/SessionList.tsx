import React from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getSessions, getSessionDuration } from "../api/client";
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

export default function SessionList() {
  const {
    data: sessions,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["sessions"],
    queryFn: getSessions,
    refetchInterval: 10000,
  });

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Sessions</h1>
          <p className="mt-1 text-sm text-slate-400">
            All your recorded training sessions
          </p>
        </div>
        <Link to="/live" className="btn-primary gap-2">
          <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="5" />
          </svg>
          New Session
        </Link>
      </div>

      {/* Table */}
      <div className="card overflow-hidden p-0">
        {isLoading ? (
          <div className="flex items-center justify-center py-16">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-600 border-t-teal-500" />
          </div>
        ) : error ? (
          <div className="px-6 py-8">
            <div className="rounded-lg bg-red-400/10 px-4 py-3 text-sm text-red-400">
              Failed to load sessions. Please try again.
            </div>
          </div>
        ) : !sessions || sessions.length === 0 ? (
          <div className="py-16 text-center">
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
              No sessions found. Start your first recording!
            </p>
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700/50 bg-slate-800/50">
                <th className="px-6 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Date
                </th>
                <th className="px-6 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Duration
                </th>
                <th className="px-6 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Status
                </th>
                <th className="px-6 py-3.5 text-right text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/30">
              {sessions.map((session) => (
                <tr
                  key={session.id}
                  className="transition-colors hover:bg-slate-700/20"
                >
                  <td className="px-6 py-4">
                    <Link
                      to={`/sessions/${session.id}`}
                      className="text-sm font-medium text-white hover:text-teal-400"
                    >
                      {formatDate(session.started_at)}
                    </Link>
                  </td>
                  <td className="px-6 py-4 font-mono text-sm text-slate-300">
                    {formatDuration(getSessionDuration(session))}
                  </td>
                  <td className="px-6 py-4">
                    <StatusBadge status={session.status} />
                  </td>
                  <td className="px-6 py-4 text-right">
                    <Link
                      to={`/sessions/${session.id}`}
                      className="text-sm font-medium text-teal-400 transition-colors hover:text-teal-300"
                    >
                      View
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
