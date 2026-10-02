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
  LocationState,
  PredictRequest,
  PredictResponse,
  RatingRequest,
  RatingResponse,
  RatingStats,
  SubmitPhotoResponse,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** A non-2xx response, with its HTTP status kept so callers can tell a
 *  temporarily busy weather service apart from a real failure. */
export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * True when the backend couldn't get weather data right now — the provider is
 * rate-limited or out of its daily quota (503), or this client asked for too
 * many new places (429). Nothing is broken; it passes on its own, so the UI
 * says so calmly instead of showing a red error.
 */
export function isServiceBusy(err: unknown): boolean {
  return err instanceof ApiError && (err.status === 503 || err.status === 429);
}

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
    throw new ApiError(res.status, `API error ${res.status}: ${detail}`);
  }

  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Backend endpoints
// ---------------------------------------------------------------------------

/** Only Open-Meteo, over HTTPS, may be fetched on the server's behalf. */
function isOpenMeteoUrl(url: unknown): url is string {
  if (typeof url !== "string") return false;
  try {
    const u = new URL(url);
    return u.protocol === "https:" && (u.hostname === "open-meteo.com" || u.hostname.endsWith(".open-meteo.com"));
  } catch {
    return false;
  }
}

/**
 * Predict sunset beauty for a single location and date.
 *
 * Proof of concept (backend/app/utils/client_fetch.py): Open-Meteo limits
 * calls per IP address, and the server's address is shared with other apps.
 * When Open-Meteo refuses the server, tonight's prediction answers 503 with
 * `client_fetch` — the URL it needed. The browser, which has its own limit,
 * fetches it and asks again with everything fetched so far; the server still
 * does all the scoring. A few rounds at most.
 */
export async function predict(body: PredictRequest): Promise<PredictResponse> {
  const clientData: Record<string, unknown> = {};
  for (let round = 0; round < 8; round++) {
    const payload = Object.keys(clientData).length ? { ...body, client_data: clientData } : body;
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.ok) return res.json() as Promise<PredictResponse>;

    let detail: unknown = res.statusText;
    let fetchUrl: unknown;
    try {
      const err = await res.json();
      detail = err?.detail ?? detail;
      fetchUrl = err?.client_fetch;
    } catch {
      // ignore parse errors
    }
    if (res.status !== 503 || !isOpenMeteoUrl(fetchUrl) || fetchUrl in clientData) {
      throw new ApiError(res.status, `API error ${res.status}: ${detail}`);
    }
    const weather = await fetch(fetchUrl);
    if (!weather.ok) throw new ApiError(503, `API error 503: weather provider unavailable`);
    clientData[fetchUrl] = await weather.json();
  }
  throw new ApiError(503, "API error 503: weather data incomplete");
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
  const url = `${API_BASE}/heatmap?lat=${params.lat}&lon=${params.lon}&months=${params.months ?? 6}`;
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

// ── Ratings ──────────────────────────────────────────────────────────────────

/**
 * Record how a sunset actually looked.
 *
 * These are the training labels the scoring engine is measured against — see
 * docs/scoring-v2-plan.md. Rating the DULL evenings matters as much as the good
 * ones: a model with no negative examples cannot learn to say "not tonight".
 */
export async function rateSunset(body: RatingRequest): Promise<RatingResponse> {
  return request<RatingResponse>(`${API_BASE}/rate`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** Aggregate stats over collected ratings, including rank correlation vs. the model. */
export async function getRatingStats(): Promise<RatingStats> {
  return request<RatingStats>(`${API_BASE}/ratings/stats`);
}

// ---------------------------------------------------------------------------
// Push alerts
// ---------------------------------------------------------------------------

async function send(url: string, options: RequestInit): Promise<void> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
}

/** Public VAPID key; throws (503) when alerts aren't configured on the server. */
export async function getVapidKey(): Promise<string> {
  const { public_key } = await request<{ public_key: string }>(`${API_BASE}/push/vapid-key`);
  return public_key;
}

export async function subscribePush(body: {
  subscription: PushSubscriptionJSON;
  places: LocationState[];
  tz: string;
}): Promise<void> {
  await send(`${API_BASE}/push/subscribe`, {
    method: "POST",
    body: JSON.stringify({
      subscription: body.subscription,
      places: body.places.map(({ latitude, longitude, name }) => ({ latitude, longitude, name })),
      tz: body.tz,
    }),
  });
}

export async function unsubscribePush(endpoint: string): Promise<void> {
  await send(`${API_BASE}/push/subscribe`, {
    method: "DELETE",
    body: JSON.stringify({ endpoint }),
  });
}
