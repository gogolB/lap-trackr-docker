import React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  getSystemHealth,
  getDefaultCalibrations,
  type SystemHealth,
  type ServiceHealth,
  type CalibrationDefault,
} from "../api/client";

function StatusDot({ status }: { status: string }) {
  const ok = status === "ok";
  return (
    <span className="relative flex h-3 w-3">
      {ok && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
      )}
      <span
        className={`relative inline-flex h-3 w-3 rounded-full ${
          ok ? "bg-emerald-500" : "bg-red-500"
        }`}
      />
    </span>
  );
}

function ServiceCard({
  name,
  info,
}: {
  name: string;
  info: ServiceHealth;
}) {
  const ok = info.status === "ok";

  // Build detail rows from the info object
  const details: { label: string; value: string }[] = [];
  if (info.latency_ms != null) {
    details.push({ label: "Latency", value: `${info.latency_ms}ms` });
  }
  if (info.error) {
    details.push({ label: "Error", value: info.error });
  }

  // Service-specific fields
  if (name === "database" && info.version) {
    const short = String(info.version).split(" on ")[0];
    details.push({ label: "Version", value: short });
  }
  if (name === "redis" && info.used_memory_human) {
    details.push({ label: "Memory", value: String(info.used_memory_human) });
  }
  if (name === "camera") {
    const cameras = info.cameras as
      | Record<string, { serial: string | null; opened: boolean }>
      | undefined;
    if (cameras) {
      for (const [cam, state] of Object.entries(cameras)) {
        const label = cam.replace("_", " ");
        details.push({
          label: label,
          value: state.opened
            ? `Connected (${state.serial})`
            : state.serial
              ? `Offline (${state.serial})`
              : "Not configured",
        });
      }
    }
    if (info.recording != null) {
      details.push({
        label: "Recording",
        value: info.recording ? "Active" : "Idle",
      });
    }
  }
  if (name === "grader" && info.pending_jobs != null) {
    details.push({
      label: "Pending jobs",
      value: String(info.pending_jobs),
    });
  }

  const displayName: Record<string, string> = {
    api: "API Server",
    database: "PostgreSQL",
    redis: "Redis",
    camera: "Camera Service",
    grader: "Grading Worker",
  };

  return (
    <div
      className={`rounded-xl border p-5 ${
        ok
          ? "border-slate-700/50 bg-slate-800/50"
          : "border-red-700/50 bg-red-900/20"
      }`}
    >
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <StatusDot status={info.status} />
          <h3 className="text-lg font-semibold text-white">
            {displayName[name] ?? name}
          </h3>
        </div>
        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
            ok
              ? "bg-emerald-500/10 text-emerald-400"
              : "bg-red-500/10 text-red-400"
          }`}
        >
          {ok ? "Healthy" : "Error"}
        </span>
      </div>

      {details.length > 0 && (
        <dl className="space-y-2">
          {details.map((d) => (
            <div key={d.label} className="flex justify-between text-sm">
              <dt className="text-slate-400">{d.label}</dt>
              <dd
                className={`font-mono ${
                  d.label === "Error" ? "text-red-400" : "text-slate-200"
                } max-w-[60%] truncate text-right`}
                title={d.value}
              >
                {d.value}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

export default function HealthPage() {
  const {
    data: health,
    isLoading,
    error,
    dataUpdatedAt,
  } = useQuery<SystemHealth>({
    queryKey: ["system-health"],
    queryFn: getSystemHealth,
    refetchInterval: 5000,
  });

  const { data: calibrations } = useQuery<CalibrationDefault[]>({
    queryKey: ["calibration-defaults"],
    queryFn: getDefaultCalibrations,
    refetchInterval: 10000,
  });

  const lastChecked = dataUpdatedAt
    ? new Date(dataUpdatedAt).toLocaleTimeString()
    : null;

  return (
    <div>
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">System Health</h1>
          <p className="mt-1 text-sm text-slate-400">
            Real-time status of all services — refreshes every 5s
          </p>
        </div>
        {health && (
          <div className="flex items-center gap-3">
            {lastChecked && (
              <span className="text-xs text-slate-500">
                Last check: {lastChecked}
              </span>
            )}
            <span
              className={`rounded-full px-4 py-1.5 text-sm font-semibold ${
                health.overall === "healthy"
                  ? "bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20"
                  : "bg-red-500/10 text-red-400 ring-1 ring-red-500/20"
              }`}
            >
              {health.overall === "healthy"
                ? "All Systems Operational"
                : "System Degraded"}
            </span>
          </div>
        )}
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center justify-center py-20">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-teal-500 border-t-transparent" />
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-xl border border-red-700/50 bg-red-900/20 p-6">
          <h3 className="font-semibold text-red-400">
            Failed to fetch system health
          </h3>
          <p className="mt-1 text-sm text-red-300/70">
            {(error as Error).message}
          </p>
          <p className="mt-2 text-xs text-slate-500">
            The API service may be down. Check Docker logs.
          </p>
        </div>
      )}

      {/* Service cards */}
      {health && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(health.services).map(([name, info]) => (
            <ServiceCard key={name} name={name} info={info} />
          ))}
        </div>
      )}

      {/* Calibration status */}
      {health && (
        <div className="mt-8">
          <h2 className="mb-4 text-lg font-semibold text-white">
            Camera Calibration
          </h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {(["on_axis", "off_axis"] as const).map((cam) => {
              const cal = calibrations?.find(
                (c) => c.camera_name === cam
              );
              return (
                <div
                  key={cam}
                  className={`rounded-xl border p-5 ${
                    cal
                      ? "border-emerald-700/30 bg-emerald-900/10"
                      : "border-yellow-700/30 bg-yellow-900/10"
                  }`}
                >
                  <div className="mb-3 flex items-center justify-between">
                    <h3 className="font-medium text-white">
                      {cam.replace("_", " ")}
                    </h3>
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                        cal
                          ? "bg-emerald-500/10 text-emerald-400"
                          : "bg-yellow-500/10 text-yellow-400"
                      }`}
                    >
                      {cal ? "Calibrated" : "Not calibrated"}
                    </span>
                  </div>
                  {cal ? (
                    <dl className="space-y-1.5 text-sm">
                      <div className="flex justify-between">
                        <dt className="text-slate-400">Intrinsics</dt>
                        <dd className="font-mono text-slate-200">
                          fx={cal.fx.toFixed(1)} fy={cal.fy.toFixed(1)}
                        </dd>
                      </div>
                      <div className="flex justify-between">
                        <dt className="text-slate-400">Resolution</dt>
                        <dd className="font-mono text-slate-200">
                          {cal.image_width}x{cal.image_height}
                        </dd>
                      </div>
                      {cal.reprojection_error != null && (
                        <div className="flex justify-between">
                          <dt className="text-slate-400">Reproj. error</dt>
                          <dd className="font-mono text-slate-200">
                            {cal.reprojection_error.toFixed(4)} px
                          </dd>
                        </div>
                      )}
                      <div className="flex justify-between">
                        <dt className="text-slate-400">Calibrated</dt>
                        <dd className="text-slate-200">
                          {new Date(cal.created_at).toLocaleDateString()}
                        </dd>
                      </div>
                    </dl>
                  ) : (
                    <p className="text-sm text-slate-400">
                      No default calibration. Go to Live View to calibrate.
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Troubleshooting tips when degraded */}
      {health && health.overall === "degraded" && (
        <div className="mt-8 rounded-xl border border-yellow-700/50 bg-yellow-900/10 p-6">
          <h3 className="font-semibold text-yellow-400">
            Troubleshooting Tips
          </h3>
          <ul className="mt-3 space-y-2 text-sm text-slate-300">
            {health.services.camera?.status === "error" && (
              <>
                <li>
                  <span className="font-medium text-yellow-400">Camera:</span>{" "}
                  Restart host daemons:{" "}
                  <code className="rounded bg-slate-700/50 px-1.5 py-0.5 text-xs">
                    sudo systemctl restart zed_x_daemon nvargus-daemon
                  </code>
                </li>
                <li>
                  <span className="font-medium text-yellow-400">Camera:</span>{" "}
                  Then restart the container:{" "}
                  <code className="rounded bg-slate-700/50 px-1.5 py-0.5 text-xs">
                    docker compose restart camera
                  </code>
                </li>
              </>
            )}
            {health.services.database?.status === "error" && (
              <li>
                <span className="font-medium text-yellow-400">Database:</span>{" "}
                Check if PostgreSQL is running:{" "}
                <code className="rounded bg-slate-700/50 px-1.5 py-0.5 text-xs">
                  docker compose logs db
                </code>
              </li>
            )}
            {health.services.redis?.status === "error" && (
              <li>
                <span className="font-medium text-yellow-400">Redis:</span>{" "}
                Check Redis logs:{" "}
                <code className="rounded bg-slate-700/50 px-1.5 py-0.5 text-xs">
                  docker compose logs redis
                </code>
              </li>
            )}
            {health.services.grader?.status === "error" && (
              <li>
                <span className="font-medium text-yellow-400">Grader:</span>{" "}
                Check grader logs:{" "}
                <code className="rounded bg-slate-700/50 px-1.5 py-0.5 text-xs">
                  docker compose logs grader
                </code>
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
