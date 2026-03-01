import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

type TabMode = "login" | "register";

export default function LoginPage() {
  const [mode, setMode] = useState<TabMode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { login, register } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const from = (location.state as { from?: { pathname: string } })?.from?.pathname || "/";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsSubmitting(true);

    try {
      if (mode === "login") {
        await login(username, password);
      } else {
        await register(username, password);
      }
      navigate(from, { replace: true });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : mode === "login"
          ? "Login failed. Check your credentials."
          : "Registration failed. Try a different username."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-full items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-600 shadow-lg shadow-teal-600/20">
            <svg
              className="h-8 w-8 text-white"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5"
              />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-white">Lap-Trackr</h1>
          <p className="mt-1 text-sm text-slate-400">
            Laparoscopic Surgical Training Tracker
          </p>
        </div>

        {/* Card */}
        <div className="card">
          {/* Tabs */}
          <div className="mb-6 flex rounded-lg bg-slate-700/50 p-1">
            <button
              type="button"
              onClick={() => {
                setMode("login");
                setError("");
              }}
              className={`flex-1 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                mode === "login"
                  ? "bg-slate-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Login
            </button>
            <button
              type="button"
              onClick={() => {
                setMode("register");
                setError("");
              }}
              className={`flex-1 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                mode === "register"
                  ? "bg-slate-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Register
            </button>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label
                htmlFor="username"
                className="mb-1.5 block text-sm font-medium text-slate-300"
              >
                Username
              </label>
              <input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="input-field"
                placeholder="Enter your username"
                required
                autoComplete="username"
              />
            </div>

            <div>
              <label
                htmlFor="password"
                className="mb-1.5 block text-sm font-medium text-slate-300"
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input-field"
                placeholder="Enter your password"
                required
                autoComplete={mode === "login" ? "current-password" : "new-password"}
              />
            </div>

            {error && (
              <div className="rounded-lg bg-red-400/10 px-4 py-3 text-sm text-red-400">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={isSubmitting || !username || !password}
              className="btn-primary w-full"
            >
              {isSubmitting ? (
                <span className="flex items-center gap-2">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                  {mode === "login" ? "Signing in..." : "Creating account..."}
                </span>
              ) : mode === "login" ? (
                "Sign In"
              ) : (
                "Create Account"
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
