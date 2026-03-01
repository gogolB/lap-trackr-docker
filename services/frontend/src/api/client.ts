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

  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers,
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

export interface Session {
  id: string;
  user_id: string;
  status: "recording" | "completed" | "grading" | "graded" | "failed";
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

// ---- Token utilities exposed for auth hook ----

export { getToken, setToken, clearToken };
