/**
 * Typed API client for the Sunset Predictor backend.
 *
 * All functions throw on non-2xx responses so callers can handle errors
 * in a single try/catch.
 */

import type {
  ForecastRequest,
  ForecastResponse,
  GeocodingResult,
  HealthResponse,
  PredictRequest,
  PredictResponse,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function request<T>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // ignore parse errors
    }
    throw new Error(`API error ${res.status}: ${detail}`);
  }

  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Backend endpoints
// ---------------------------------------------------------------------------

/** Predict sunset beauty for a single location and date. */
export async function predict(body: PredictRequest): Promise<PredictResponse> {
  return request<PredictResponse>(`${API_BASE}/predict`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** Fetch multi-day sunset forecast. */
export async function forecast(
  body: ForecastRequest
): Promise<ForecastResponse> {
  return request<ForecastResponse>(`${API_BASE}/forecast`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** Health check. */
export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>(`${API_BASE}/health`);
}

/** ML model metadata. */
export async function getModelInfo(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(`${API_BASE}/model/info`);
}

// ---------------------------------------------------------------------------
// Open-Meteo Geocoding (called directly from the browser)
// ---------------------------------------------------------------------------

/** Search for place names and return lat/lon results (proxied via backend to avoid CORS). */
export async function geocode(query: string): Promise<GeocodingResult[]> {
  if (!query.trim()) return [];

  const url = `${API_BASE}/geocode?name=${encodeURIComponent(query)}&count=8`;

  try {
    const data = await request<{ results?: GeocodingResult[] }>(url);
    return data.results ?? [];
  } catch {
    return [];
  }
}
