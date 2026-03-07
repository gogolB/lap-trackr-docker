const BASE_URL = "";

function getToken(): string | null {
  return localStorage.getItem("lap_trackr_token");
}

function setToken(token: string): void {
  localStorage.setItem("lap_trackr_token", token);
}

function clearToken(): void {
  localStorage.removeItem("lap_trackr_token");
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(`${BASE_URL}${path}`, {
      ...options,
      headers,
      signal: controller.signal,
    });

    if (response.status === 401) {
      clearToken();
      window.location.href = "/login";
      throw new Error("Unauthorized");
    }

    if (!response.ok) {
      const errorBody = await response.text().catch(() => "");
      throw new Error(
        `API Error ${response.status}: ${errorBody || response.statusText}`
      );
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

// ---- Auth ----

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export interface User {
  id: string;
  username: string;
  created_at: string;
}

export async function login(
  username: string,
  password: string
): Promise<LoginResponse> {
  const res = await request<LoginResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  setToken(res.access_token);
  return res;
}

export async function register(
  username: string,
  password: string
): Promise<User> {
  return request<User>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function getMe(): Promise<User> {
  return request<User>("/api/auth/me");
}

// ---- Sessions ----

export type SessionStatus =
  | "recording"
  | "completed"
  | "exporting"
  | "export_failed"
  | "awaiting_init"
  | "grading"
  | "graded"
  | "failed";

export interface Session {
  id: string;
  user_id: string;
  status: SessionStatus;
  started_at: string;
  stopped_at: string | null;
  on_axis_path: string | null;
  off_axis_path: string | null;
  created_at: string;
}

export interface SessionDetail extends Session {
  grading_result: GradingResult | null;
}

/** Compute duration in seconds from session timestamps */
export function getSessionDuration(session: Session): number | null {
  if (!session.started_at) return null;
  const start = new Date(session.started_at).getTime();
  const end = session.stopped_at
    ? new Date(session.stopped_at).getTime()
    : Date.now();
  return Math.floor((end - start) / 1000);
}

export async function getSessions(): Promise<Session[]> {
  return request<Session[]>("/api/sessions/");
}

export async function getSession(id: string): Promise<SessionDetail> {
  return request<SessionDetail>(`/api/sessions/${id}`);
}

export async function startSession(): Promise<Session> {
  return request<Session>("/api/sessions/start", {
    method: "POST",
  });
}

export async function stopSession(id: string): Promise<Session> {
  return request<Session>(`/api/sessions/${id}/stop`, {
    method: "POST",
  });
}

export async function deleteSession(id: string): Promise<void> {
  return request<void>(`/api/sessions/${id}`, {
    method: "DELETE",
  });
}

export async function downloadSession(id: string): Promise<void> {
  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${BASE_URL}/api/sessions/${id}/download`, {
    headers,
  });

  if (!response.ok) {
    throw new Error(`Download failed: ${response.statusText}`);
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const disposition = response.headers.get("Content-Disposition");
  const match = disposition?.match(/filename="?([^"]+)"?/);
  a.download = match?.[1] ?? `session_${id}.zip`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---- Grading ----

export async function gradeSession(id: string): Promise<Session> {
  return request<Session>(`/api/sessions/${id}/grade`, {
    method: "POST",
  });
}

export interface GradingResult {
  id: string;
  session_id: string;
  workspace_volume: number | null;
  avg_speed: number | null;
  max_jerk: number | null;
  path_length: number | null;
  economy_of_motion: number | null;
  total_time: number | null;
  completed_at: string | null;
  error: string | null;
}

export async function getResults(sessionId: string): Promise<GradingResult> {
  return request<GradingResult>(`/api/results/${sessionId}`);
}

export interface MetricsData {
  workspace_volume: number;
  avg_speed: number;
  max_jerk: number;
  path_length: number;
  economy_of_motion: number;
  total_time: number;
  [key: string]: number;
}

export async function getMetrics(sessionId: string): Promise<MetricsData> {
  return request<MetricsData>(`/api/results/${sessionId}/metrics`);
}

export interface PoseData {
  frame_idx: number;
  timestamp: number;
  left_tip: number[];
  right_tip: number[];
}

export async function getPoses(sessionId: string): Promise<PoseData[]> {
  return request<PoseData[]>(`/api/results/${sessionId}/poses`);
}

// ---- ML Models ----

export type ModelStatus =
  | "available"
  | "downloading"
  | "ready"
  | "active"
  | "custom"
  | "failed";

export interface MLModel {
  id: string;
  slug: string;
  name: string;
  model_type: string;
  description: string | null;
  version: string | null;
  download_url: string | null;
  file_size_bytes: number | null;
  file_path: string | null;
  status: ModelStatus;
  is_active: boolean;
  is_custom: boolean;
  created_at: string;
  updated_at: string;
}

export interface ModelDownloadProgress {
  model_id: string;
  status: string;
  downloaded_bytes: number;
  total_bytes: number;
  percent: number;
  error: string | null;
}

export async function getModels(): Promise<MLModel[]> {
  return request<MLModel[]>("/api/models/");
}

export async function downloadModel(id: string): Promise<MLModel> {
  return request<MLModel>(`/api/models/${id}/download`, { method: "POST" });
}

export async function getModelProgress(
  id: string
): Promise<ModelDownloadProgress> {
  return request<ModelDownloadProgress>(`/api/models/${id}/progress`);
}

export async function activateModel(id: string): Promise<MLModel> {
  return request<MLModel>(`/api/models/${id}/activate`, { method: "POST" });
}

export async function deleteModel(id: string): Promise<void> {
  return request<void>(`/api/models/${id}`, { method: "DELETE" });
}

export async function uploadModel(file: File): Promise<MLModel> {
  const token = getToken();
  const formData = new FormData();
  formData.append("file", file);

  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${BASE_URL}/api/models/upload`, {
    method: "POST",
    headers,
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.text().catch(() => "");
    throw new Error(
      `API Error ${response.status}: ${errorBody || response.statusText}`
    );
  }

  return response.json();
}

// ---- Calibration ----

export interface BoardConfig {
  rows: number;
  cols: number;
  square_size_mm: number;
  marker_size_mm: number;
  aruco_dict: string;
}

export interface CalibrationCaptureResult {
  success: boolean;
  markers_detected: number;
  charuco_corners: number;
  coverage_pct: number;
  total_captures: number;
  preview_jpeg_b64: string | null;
}

export interface CalibrationStatus {
  [camera: string]: {
    total_captures: number;
    board_config: BoardConfig;
  };
}

export interface CalibrationData {
  version: number;
  is_global: boolean;
  camera_name: string;
  intrinsics: {
    fx: number;
    fy: number;
    cx: number;
    cy: number;
    distortion: number[];
    image_width: number;
    image_height: number;
  };
  extrinsic_matrix: number[][] | null;
  board_config: BoardConfig;
  quality: {
    reprojection_error: number;
    num_frames_used: number;
  };
}

export interface CalibrationDefault {
  id: string;
  camera_name: string;
  is_default: boolean;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  image_width: number;
  image_height: number;
  reprojection_error: number | null;
  num_frames_used: number | null;
  created_at: string;
}

export async function captureCalibrationFrame(
  camera: string
): Promise<CalibrationCaptureResult> {
  return request<CalibrationCaptureResult>(
    `/api/calibration/capture/${camera}`,
    { method: "POST" }
  );
}

export async function computeCalibration(
  camera: string,
  saveAsDefault: boolean = true
): Promise<CalibrationData> {
  return request<CalibrationData>(
    `/api/calibration/compute/${camera}?save_as_default=${saveAsDefault}`,
    { method: "POST" }
  );
}

export async function resetCalibration(
  camera: string
): Promise<{ status: string }> {
  return request<{ status: string }>(
    `/api/calibration/reset/${camera}`,
    { method: "POST" }
  );
}

export async function getCalibrationStatus(): Promise<CalibrationStatus> {
  return request<CalibrationStatus>("/api/calibration/status");
}

export async function getDefaultCalibrations(): Promise<CalibrationDefault[]> {
  return request<CalibrationDefault[]>("/api/calibration/defaults");
}

export async function deleteDefaultCalibration(
  camera: string
): Promise<void> {
  return request<void>(`/api/calibration/defaults/${camera}`, {
    method: "DELETE",
  });
}

// ---- Stereo Calibration ----

export interface StereoCaptureResult {
  on_axis: CalibrationCaptureResult;
  off_axis: CalibrationCaptureResult;
}

export interface StereoCalibrationResult {
  on_axis: CalibrationData;
  off_axis: CalibrationData;
  stereo: {
    T_on_to_off: number[][];
    on_axis_reprojection_error: number;
    off_axis_reprojection_error: number;
  };
}

export async function captureStereoCalibrationFrame(): Promise<StereoCaptureResult> {
  return request<StereoCaptureResult>("/api/calibration/capture/stereo", {
    method: "POST",
  });
}

export async function computeStereoCalibration(
  saveAsDefault: boolean = true
): Promise<StereoCalibrationResult> {
  return request<StereoCalibrationResult>(
    `/api/calibration/compute/stereo?save_as_default=${saveAsDefault}`,
    { method: "POST" }
  );
}

export async function resetStereoCalibration(): Promise<{ status: string }> {
  return request<{ status: string }>("/api/calibration/reset/stereo", {
    method: "POST",
  });
}

// ---- Tip Initialization ----

export interface TipDetection {
  label: string;
  x: number;
  y: number;
  confidence: number;
  color: string;
}

export interface TipInitData {
  detections: Record<string, TipDetection[]>;
  sample_frames: string[];
}

export async function getTipInit(id: string): Promise<TipInitData> {
  return request<TipInitData>(`/api/sessions/${id}/tip-init`);
}

export async function updateTipInit(
  id: string,
  tips: Record<string, TipDetection[]>
): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/sessions/${id}/tip-init`, {
    method: "PUT",
    body: JSON.stringify({ tips }),
  });
}

export function getSampleFrameUrl(id: string, filename: string): string {
  const token = getToken();
  return `${BASE_URL}/api/sessions/${id}/sample-frame/${filename}${
    token ? `?token=${token}` : ""
  }`;
}

// ---- System Health ----

export interface ServiceHealth {
  status: "ok" | "error";
  latency_ms?: number;
  error?: string;
  [key: string]: unknown;
}

export interface SystemHealth {
  overall: "healthy" | "degraded";
  services: {
    api: ServiceHealth;
    database: ServiceHealth;
    redis: ServiceHealth;
    camera: ServiceHealth;
    grader: ServiceHealth;
  };
}

export async function getSystemHealth(): Promise<SystemHealth> {
  return request<SystemHealth>("/api/health/system");
}

// ---- Camera Config ----

export interface CameraConfig {
  on_axis_serial: string;
  off_axis_serial: string;
  on_axis_swap_eyes: boolean;
  off_axis_swap_eyes: boolean;
  on_axis_flip: boolean;
  off_axis_flip: boolean;
  updated_at: string | null;
}

export interface CameraConfigUpdate {
  on_axis_serial?: string;
  off_axis_serial?: string;
  on_axis_swap_eyes?: boolean;
  off_axis_swap_eyes?: boolean;
  on_axis_flip?: boolean;
  off_axis_flip?: boolean;
}

export async function getCameraConfig(): Promise<CameraConfig> {
  return request<CameraConfig>("/api/camera-config/");
}

export async function updateCameraConfig(
  data: CameraConfigUpdate
): Promise<CameraConfig> {
  return request<CameraConfig>("/api/camera-config/", {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function applyCameraConfig(): Promise<{ status: string }> {
  return request<{ status: string }>("/api/camera-config/apply", {
    method: "POST",
  });
}

// ---- Token utilities exposed for auth hook ----

export { getToken, setToken, clearToken };
