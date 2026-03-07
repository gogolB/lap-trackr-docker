import React, { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getTipInit,
  updateTipInit,
  getSampleFrameUrl,
  type TipDetection,
} from "../api/client";

const MARKER_RADIUS = 10;
const COLORS: Record<string, string> = {
  green: "#22c55e",
  pink: "#ec4899",
};
const LABEL_COLORS: Record<string, string> = {
  left_tip: "#22c55e",
  right_tip: "#ec4899",
};

interface Marker {
  label: string;
  x: number;
  y: number;
  color: string;
}

export default function TipInitPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const [markers, setMarkers] = useState<Record<string, Marker[]>>({});
  const [selectedFrame, setSelectedFrame] = useState<string>("");
  const [dragging, setDragging] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const { data: tipData, isLoading } = useQuery({
    queryKey: ["tip-init", id],
    queryFn: () => getTipInit(id!),
    enabled: !!id,
  });

  // Initialize markers from detections
  useEffect(() => {
    if (!tipData) return;

    const initial: Record<string, Marker[]> = {};
    for (const [filename, detections] of Object.entries(tipData.detections)) {
      initial[filename] = (detections as TipDetection[]).map((d) => ({
        label: d.label,
        x: d.x,
        y: d.y,
        color: d.color,
      }));
    }
    setMarkers(initial);

    if (tipData.sample_frames.length > 0 && !selectedFrame) {
      setSelectedFrame(tipData.sample_frames[0]);
    }
  }, [tipData]);

  // Draw canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.complete || img.naturalWidth === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    ctx.drawImage(img, 0, 0);

    const frameMarkers = markers[selectedFrame] || [];
    for (const marker of frameMarkers) {
      const color = LABEL_COLORS[marker.label] || COLORS[marker.color] || "#fff";

      // Outer circle
      ctx.beginPath();
      ctx.arc(marker.x, marker.y, MARKER_RADIUS, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.stroke();

      // Inner dot
      ctx.beginPath();
      ctx.arc(marker.x, marker.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Crosshair
      ctx.beginPath();
      ctx.moveTo(marker.x - MARKER_RADIUS - 4, marker.y);
      ctx.lineTo(marker.x - MARKER_RADIUS + 2, marker.y);
      ctx.moveTo(marker.x + MARKER_RADIUS - 2, marker.y);
      ctx.lineTo(marker.x + MARKER_RADIUS + 4, marker.y);
      ctx.moveTo(marker.x, marker.y - MARKER_RADIUS - 4);
      ctx.lineTo(marker.x, marker.y - MARKER_RADIUS + 2);
      ctx.moveTo(marker.x, marker.y + MARKER_RADIUS - 2);
      ctx.lineTo(marker.x, marker.y + MARKER_RADIUS + 4);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.font = "bold 12px sans-serif";
      ctx.fillStyle = color;
      ctx.fillText(
        marker.label.replace("_", " "),
        marker.x + MARKER_RADIUS + 6,
        marker.y + 4
      );
    }
  }, [markers, selectedFrame]);

  useEffect(() => {
    draw();
  }, [draw]);

  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getCanvasCoords(e);
    if (!coords) return;

    const frameMarkers = markers[selectedFrame] || [];

    // Check if clicking on existing marker
    for (let i = 0; i < frameMarkers.length; i++) {
      const dx = coords.x - frameMarkers[i].x;
      const dy = coords.y - frameMarkers[i].y;
      if (Math.sqrt(dx * dx + dy * dy) < MARKER_RADIUS * 2) {
        setDragging(i);
        return;
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (dragging === null) return;
    const coords = getCanvasCoords(e);
    if (!coords) return;

    setMarkers((prev) => {
      const frameMarkers = [...(prev[selectedFrame] || [])];
      if (dragging < frameMarkers.length) {
        frameMarkers[dragging] = { ...frameMarkers[dragging], ...coords };
      }
      return { ...prev, [selectedFrame]: frameMarkers };
    });
  };

  const handleMouseUp = () => {
    setDragging(null);
  };

  const handleContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const coords = getCanvasCoords(e);
    if (!coords) return;

    // Remove closest marker
    const frameMarkers = markers[selectedFrame] || [];
    let closest = -1;
    let minDist = Infinity;
    for (let i = 0; i < frameMarkers.length; i++) {
      const dx = coords.x - frameMarkers[i].x;
      const dy = coords.y - frameMarkers[i].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < MARKER_RADIUS * 3 && dist < minDist) {
        minDist = dist;
        closest = i;
      }
    }
    if (closest >= 0) {
      setMarkers((prev) => {
        const updated = [...(prev[selectedFrame] || [])];
        updated.splice(closest, 1);
        return { ...prev, [selectedFrame]: updated };
      });
    }
  };

  const handleDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getCanvasCoords(e);
    if (!coords) return;

    const frameMarkers = markers[selectedFrame] || [];
    const hasLeft = frameMarkers.some((m) => m.label === "left_tip");
    const hasRight = frameMarkers.some((m) => m.label === "right_tip");

    let label: string;
    let color: string;
    if (!hasLeft) {
      label = "left_tip";
      color = "green";
    } else if (!hasRight) {
      label = "right_tip";
      color = "pink";
    } else {
      return; // Both tips already placed
    }

    setMarkers((prev) => ({
      ...prev,
      [selectedFrame]: [...(prev[selectedFrame] || []), { label, color, ...coords }],
    }));
  };

  const handleConfirm = async () => {
    if (!id) return;
    setError("");
    setSaving(true);
    try {
      // Convert markers to TipDetection format
      const tips: Record<string, TipDetection[]> = {};
      for (const [filename, ms] of Object.entries(markers)) {
        tips[filename] = ms.map((m) => ({
          label: m.label,
          x: m.x,
          y: m.y,
          confidence: 1.0,
          color: m.color,
        }));
      }
      await updateTipInit(id, tips);
      queryClient.invalidateQueries({ queryKey: ["session", id] });
      navigate(`/sessions/${id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save tip init.");
    } finally {
      setSaving(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-600 border-t-teal-500" />
      </div>
    );
  }

  if (!tipData) {
    return (
      <div className="rounded-lg bg-red-400/10 px-4 py-3 text-sm text-red-400">
        Failed to load tip initialization data.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Initialize Instrument Tips</h1>
          <p className="mt-1 text-sm text-slate-400">
            Verify and adjust auto-detected tip positions. Double-click to add, right-click to remove, drag to adjust.
          </p>
        </div>
        <button
          onClick={handleConfirm}
          disabled={saving}
          className="btn-primary gap-2"
        >
          {saving ? (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
          ) : (
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )}
          Confirm & Continue
        </button>
      </div>

      {error && (
        <div className="rounded-lg bg-red-400/10 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Frame selector */}
      <div className="flex flex-wrap gap-2">
        {tipData.sample_frames.map((filename) => (
          <button
            key={filename}
            onClick={() => setSelectedFrame(filename)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              selectedFrame === filename
                ? "bg-teal-600 text-white"
                : "bg-slate-700/50 text-slate-300 hover:bg-slate-700"
            }`}
          >
            {filename.replace(/_/g, " ").replace(".jpg", "")}
          </button>
        ))}
      </div>

      {/* Canvas */}
      {selectedFrame && id && (
        <div className="card overflow-hidden p-0">
          <div className="relative">
            <img
              ref={imgRef}
              src={getSampleFrameUrl(id, selectedFrame)}
              alt={selectedFrame}
              className="hidden"
              crossOrigin="anonymous"
              onLoad={draw}
            />
            <canvas
              ref={canvasRef}
              className="w-full cursor-crosshair"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onContextMenu={handleContextMenu}
              onDoubleClick={handleDoubleClick}
            />
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-6 text-sm text-slate-400">
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded-full bg-green-500" />
          Left tip (green tape)
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded-full bg-pink-500" />
          Right tip (pink tape)
        </div>
      </div>
    </div>
  );
}
