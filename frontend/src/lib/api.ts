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
  HeatmapResponse,
  HealthResponse,
  PredictRequest,
  PredictResponse,
  SubmitPhotoResponse,
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

/** Submit a sunset photo with date and location — emails it to the developer. */
export async function submitPhoto(params: {
  photo: File;
  latitude: number;
  longitude: number;
  photoDate: string;       // "YYYY-MM-DD"
  locationName: string;
  userMessage: string;
}): Promise<SubmitPhotoResponse> {
  const form = new FormData();
  form.append("photo", params.photo);
  form.append("latitude", String(params.latitude));
  form.append("longitude", String(params.longitude));
  form.append("photo_date", params.photoDate);
  form.append("location_name", params.locationName);
  form.append("user_message", params.userMessage);

  const res = await fetch(`${API_BASE}/submit-photo`, {
    method: "POST",
    body: form,
    // No Content-Type header — browser sets it with the correct boundary for multipart
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }

  return res.json() as Promise<SubmitPhotoResponse>;
}

/** Fetch historical sunset score heatmap for a location. */
export async function heatmap(params: {
  lat: number;
  lon: number;
  months?: number;
}): Promise<HeatmapResponse> {
  const url = `${API_BASE}/heatmap?lat=${params.lat}&lon=${params.lon}&months=${params.months ?? 12}`;
  return request<HeatmapResponse>(url);
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
