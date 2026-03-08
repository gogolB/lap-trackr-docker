import React, { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getCameraConfig,
  updateCameraConfig,
  applyCameraConfig,
  type CameraConfig,
} from "../api/client";

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3">
      <span className="text-sm text-slate-300">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-6 w-11 flex-shrink-0 rounded-full border-2 border-transparent transition-colors ${
          checked ? "bg-teal-600" : "bg-slate-600"
        }`}
      >
        <span
          className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
            checked ? "translate-x-5" : "translate-x-0"
          }`}
        />
      </button>
    </label>
  );
}

export default function ConfigPage() {
  const queryClient = useQueryClient();
  const [isSaving, setIsSaving] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  const [form, setForm] = useState<CameraConfig | null>(null);
  const [formInitialized, setFormInitialized] = useState(false);

  const { data: config, isLoading, error } = useQuery({
    queryKey: ["camera-config"],
    queryFn: getCameraConfig,
  });

  useEffect(() => {
    if (config && !formInitialized) {
      setForm(config);
      setFormInitialized(true);
    }
  }, [config, formInitialized]);

  const updateField = <K extends keyof CameraConfig>(
    key: K,
    value: CameraConfig[K]
  ) => {
    if (!form) return;
    setForm({ ...form, [key]: value });
  };

  const handleSwapCameras = () => {
    if (!form) return;
    setForm({
      ...form,
      on_axis_serial: form.off_axis_serial,
      off_axis_serial: form.on_axis_serial,
    });
  };

  const handleSave = async () => {
    if (!form) return;
    setIsSaving(true);
    setMessage(null);
    try {
      await updateCameraConfig({
        on_axis_serial: form.on_axis_serial,
        off_axis_serial: form.off_axis_serial,
        on_axis_swap_eyes: form.on_axis_swap_eyes,
        off_axis_swap_eyes: form.off_axis_swap_eyes,
        on_axis_rotation: form.on_axis_rotation,
        off_axis_rotation: form.off_axis_rotation,
        on_axis_flip_h: form.on_axis_flip_h,
        on_axis_flip_v: form.on_axis_flip_v,
        off_axis_flip_h: form.off_axis_flip_h,
        off_axis_flip_v: form.off_axis_flip_v,
      });
      queryClient.invalidateQueries({ queryKey: ["camera-config"] });
      setMessage({ type: "success", text: "Configuration saved." });
    } catch (err) {
      setMessage({
        type: "error",
        text: err instanceof Error ? err.message : "Failed to save.",
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleApply = async () => {
    setIsApplying(true);
    setMessage(null);
    try {
      // Save first, then apply
      if (form) {
        await updateCameraConfig({
          on_axis_serial: form.on_axis_serial,
          off_axis_serial: form.off_axis_serial,
          on_axis_swap_eyes: form.on_axis_swap_eyes,
          off_axis_swap_eyes: form.off_axis_swap_eyes,
          on_axis_rotation: form.on_axis_rotation,
          off_axis_rotation: form.off_axis_rotation,
          on_axis_flip_h: form.on_axis_flip_h,
          on_axis_flip_v: form.on_axis_flip_v,
          off_axis_flip_h: form.off_axis_flip_h,
          off_axis_flip_v: form.off_axis_flip_v,
        });
      }
      await applyCameraConfig();
      queryClient.invalidateQueries({ queryKey: ["camera-config"] });
      setMessage({
        type: "success",
        text: "Configuration saved and applied to camera service.",
      });
    } catch (err) {
      setMessage({
        type: "error",
        text: err instanceof Error ? err.message : "Failed to apply.",
      });
    } finally {
      setIsApplying(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-600 border-t-teal-500" />
      </div>
    );
  }

  if (error || !form) {
    return (
      <div className="rounded-lg bg-red-400/10 px-4 py-3 text-sm text-red-400">
        Failed to load camera configuration.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Camera Configuration</h1>
        <p className="mt-1 text-sm text-slate-400">
          Configure camera serial assignments, eye swap, and rotation settings
        </p>
      </div>

      {message && (
        <div
          className={`rounded-lg px-4 py-3 text-sm ${
            message.type === "success"
              ? "bg-emerald-400/10 text-emerald-400"
              : "bg-red-400/10 text-red-400"
          }`}
        >
          {message.text}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* On-Axis Camera */}
        <div className="card space-y-4">
          <div className="flex items-center gap-2">
            <div className="h-2.5 w-2.5 rounded-full bg-teal-500" />
            <h2 className="text-lg font-semibold text-white">
              On-Axis Camera
            </h2>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-slate-400">
              Serial Number
            </label>
            <input
              type="text"
              value={form.on_axis_serial}
              onChange={(e) => updateField("on_axis_serial", e.target.value)}
              className="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500"
              placeholder="e.g. 38085162"
            />
          </div>
          <div className="space-y-3">
            <Toggle
              label="Swap Left/Right Eyes"
              checked={form.on_axis_swap_eyes}
              onChange={(v) => updateField("on_axis_swap_eyes", v)}
            />
            <label className="flex items-center justify-between gap-3">
              <span className="text-sm text-slate-300">Rotation</span>
              <select
                value={form.on_axis_rotation}
                onChange={(e) => updateField("on_axis_rotation", Number(e.target.value))}
                className="rounded-lg border border-slate-600 bg-slate-800 px-3 py-1.5 text-sm text-white focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500"
              >
                <option value={0}>0° (normal)</option>
                <option value={90}>90° CW</option>
                <option value={180}>180° (upside-down)</option>
                <option value={270}>270° CW</option>
              </select>
            </label>
            <Toggle
              label="Flip Horizontal (mirror left-right)"
              checked={form.on_axis_flip_h}
              onChange={(v) => updateField("on_axis_flip_h", v)}
            />
            <Toggle
              label="Flip Vertical (mirror top-bottom)"
              checked={form.on_axis_flip_v}
              onChange={(v) => updateField("on_axis_flip_v", v)}
            />
          </div>
        </div>

        {/* Off-Axis Camera */}
        <div className="card space-y-4">
          <div className="flex items-center gap-2">
            <div className="h-2.5 w-2.5 rounded-full bg-cyan-500" />
            <h2 className="text-lg font-semibold text-white">
              Off-Axis Camera
            </h2>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-slate-400">
              Serial Number
            </label>
            <input
              type="text"
              value={form.off_axis_serial}
              onChange={(e) => updateField("off_axis_serial", e.target.value)}
              className="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500"
              placeholder="e.g. 38085163"
            />
          </div>
          <div className="space-y-3">
            <Toggle
              label="Swap Left/Right Eyes"
              checked={form.off_axis_swap_eyes}
              onChange={(v) => updateField("off_axis_swap_eyes", v)}
            />
            <label className="flex items-center justify-between gap-3">
              <span className="text-sm text-slate-300">Rotation</span>
              <select
                value={form.off_axis_rotation}
                onChange={(e) => updateField("off_axis_rotation", Number(e.target.value))}
                className="rounded-lg border border-slate-600 bg-slate-800 px-3 py-1.5 text-sm text-white focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500"
              >
                <option value={0}>0° (normal)</option>
                <option value={90}>90° CW</option>
                <option value={180}>180° (upside-down)</option>
                <option value={270}>270° CW</option>
              </select>
            </label>
            <Toggle
              label="Flip Horizontal (mirror left-right)"
              checked={form.off_axis_flip_h}
              onChange={(v) => updateField("off_axis_flip_h", v)}
            />
            <Toggle
              label="Flip Vertical (mirror top-bottom)"
              checked={form.off_axis_flip_v}
              onChange={(v) => updateField("off_axis_flip_v", v)}
            />
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-3">
          <button onClick={handleSwapCameras} className="btn-secondary gap-2">
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
                d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5"
              />
            </svg>
            Swap Cameras
          </button>

          <div className="flex-1" />

          <button
            onClick={handleSave}
            disabled={isSaving}
            className="btn-secondary gap-2"
          >
            {isSaving ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-400/30 border-t-slate-400" />
            ) : null}
            Save
          </button>

          <button
            onClick={handleApply}
            disabled={isApplying}
            className="btn-primary gap-2"
          >
            {isApplying ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
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
                  d="M5.636 5.636a9 9 0 1012.728 0M12 3v9"
                />
              </svg>
            )}
            Save &amp; Apply
          </button>
        </div>
        <p className="mt-3 text-xs text-slate-500">
          "Save" persists the configuration. "Save &amp; Apply" also pushes it to
          the running camera service. If serials changed, streams will briefly
          interrupt.
        </p>
      </div>
    </div>
  );
}
