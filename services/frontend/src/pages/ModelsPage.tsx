import React, { useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getModels,
  downloadModel,
  getModelProgress,
  activateModel,
  deleteModel,
  uploadModel,
  type MLModel,
  type ModelDownloadProgress,
} from "../api/client";

function formatBytes(bytes: number | null): string {
  if (bytes === null || bytes === 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

const TYPE_LABELS: Record<string, string> = {
  cotracker: "Point Tracking",
  yolo: "Detection",
  sam2: "Segmentation",
};

function ProgressPoller({ modelId }: { modelId: string }) {
  const { data } = useQuery<ModelDownloadProgress>({
    queryKey: ["model-progress", modelId],
    queryFn: () => getModelProgress(modelId),
    refetchInterval: (query) => {
      const progress = query.state.data;
      return progress?.status === "downloading" ? 1000 : false;
    },
  });

  if (!data || data.status === "unknown") return null;

  const pct = Math.min(data.percent, 100);

  return (
    <div className="mt-2">
      <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
        <span>
          {formatBytes(data.downloaded_bytes)} / {formatBytes(data.total_bytes)}
        </span>
        <span>{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-slate-700">
        <div
          className="h-2 rounded-full bg-teal-500 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      {data.error && (
        <p className="mt-1 text-xs text-red-400">{data.error}</p>
      )}
    </div>
  );
}

function ModelCard({ model }: { model: MLModel }) {
  const queryClient = useQueryClient();

  const downloadMut = useMutation({
    mutationFn: () => downloadModel(model.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["models"] }),
  });

  const activateMut = useMutation({
    mutationFn: () => activateModel(model.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["models"] }),
  });

  const deleteMut = useMutation({
    mutationFn: () => deleteModel(model.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["models"] }),
  });

  const isDownloading = model.status === "downloading";
  const isReady =
    model.status === "ready" ||
    model.status === "active" ||
    model.status === "custom";
  const canDownload = model.status === "available" || model.status === "failed";

  return (
    <div
      className={`relative rounded-xl border p-5 transition-colors ${
        model.is_active
          ? "border-teal-500/50 bg-slate-800/80"
          : "border-slate-700/50 bg-slate-800/40"
      }`}
    >
      {/* Badges */}
      <div className="flex items-center gap-2 mb-3">
        {model.is_active && (
          <span className="inline-flex items-center rounded-full bg-teal-500/20 px-2.5 py-0.5 text-xs font-medium text-teal-400">
            Active
          </span>
        )}
        {model.is_custom && (
          <span className="inline-flex items-center rounded-full bg-purple-500/20 px-2.5 py-0.5 text-xs font-medium text-purple-400">
            Custom
          </span>
        )}
        {model.status === "failed" && (
          <span className="inline-flex items-center rounded-full bg-red-500/20 px-2.5 py-0.5 text-xs font-medium text-red-400">
            Failed
          </span>
        )}
        <span className="inline-flex items-center rounded-full bg-slate-700 px-2.5 py-0.5 text-xs font-medium text-slate-300">
          {TYPE_LABELS[model.model_type] ?? model.model_type}
        </span>
      </div>

      {/* Name + description */}
      <h3 className="text-base font-semibold text-white">{model.name}</h3>
      {model.description && (
        <p className="mt-1 text-sm text-slate-400 line-clamp-2">
          {model.description}
        </p>
      )}

      {/* Size + version */}
      <div className="mt-3 flex items-center gap-3 text-xs text-slate-500">
        {model.file_size_bytes && (
          <span>~{formatBytes(model.file_size_bytes)}</span>
        )}
        {model.version && <span>v{model.version}</span>}
      </div>

      {/* Progress bar */}
      {isDownloading && <ProgressPoller modelId={model.id} />}

      {/* Actions */}
      <div className="mt-4 flex items-center gap-2">
        {canDownload && model.download_url && (
          <button
            onClick={() => downloadMut.mutate()}
            disabled={downloadMut.isPending}
            className="rounded-lg bg-teal-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-teal-500 disabled:opacity-50"
          >
            {downloadMut.isPending ? "Starting..." : "Download"}
          </button>
        )}

        {isReady && !model.is_active && (
          <button
            onClick={() => activateMut.mutate()}
            disabled={activateMut.isPending}
            className="rounded-lg bg-teal-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-teal-500 disabled:opacity-50"
          >
            {activateMut.isPending ? "Activating..." : "Activate"}
          </button>
        )}

        {(isReady || model.status === "failed") && (
          <button
            onClick={() => {
              if (confirm("Delete this model's files?")) deleteMut.mutate();
            }}
            disabled={deleteMut.isPending}
            className="rounded-lg border border-slate-600 px-3 py-1.5 text-sm font-medium text-slate-300 transition-colors hover:border-red-500/50 hover:text-red-400 disabled:opacity-50"
          >
            Delete
          </button>
        )}
      </div>
    </div>
  );
}

export default function ModelsPage() {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const {
    data: models,
    isLoading,
    error,
  } = useQuery<MLModel[]>({
    queryKey: ["models"],
    queryFn: getModels,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.some((m) => m.status === "downloading") ? 5000 : false;
    },
  });

  const uploadMut = useMutation({
    mutationFn: uploadModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      setUploadError(null);
    },
    onError: (err: Error) => setUploadError(err.message),
  });

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadMut.mutate(file);
    }
    // Reset input so the same file can be re-selected
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const activeModels = models?.filter((m) => m.is_active) ?? [];

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">Models</h1>
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pt"
            className="hidden"
            onChange={handleUpload}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadMut.isPending}
            className="rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-600 disabled:opacity-50"
          >
            {uploadMut.isPending ? "Uploading..." : "Upload Custom YOLO Model"}
          </button>
        </div>
      </div>

      {uploadError && (
        <div className="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
          {uploadError}
        </div>
      )}

      {/* Active model banner */}
      {activeModels.length > 0 && (
        <div className="mb-6 rounded-xl border border-teal-500/30 bg-teal-500/10 p-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-teal-400 font-medium">Active Models</span>
          </div>
          <div className="space-y-2">
            {activeModels.map((model) => (
              <div key={model.id}>
                <div className="flex items-center gap-2">
                  <span className="text-white font-semibold">{model.name}</span>
                  <span className="text-xs text-slate-400">
                    {TYPE_LABELS[model.model_type] ?? model.model_type}
                  </span>
                  {model.is_custom && (
                    <span className="text-xs text-purple-400">(custom)</span>
                  )}
                </div>
                {model.description && (
                  <p className="mt-1 text-sm text-slate-400">
                    {model.description}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Loading / error states */}
      {isLoading && (
        <div className="flex items-center justify-center py-16">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-teal-500 border-t-transparent" />
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-400">
          Failed to load models: {(error as Error).message}
        </div>
      )}

      {/* Model grid */}
      {models && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {models.map((model) => (
            <ModelCard key={model.id} model={model} />
          ))}
        </div>
      )}

      {models && models.length === 0 && (
        <div className="text-center py-16 text-slate-500">
          No models available. The registry will be seeded on next API restart.
        </div>
      )}
    </div>
  );
}
