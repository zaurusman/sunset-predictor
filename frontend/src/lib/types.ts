/**
 * TypeScript types mirroring the backend Pydantic schemas.
 * Keep in sync with backend/app/schemas/*.py
 */

// ── Weather ─────────────────────────────────────────────────────────────────

export interface WeatherOverride {
  cloud_low?: number;
  cloud_mid?: number;
  cloud_high?: number;
  cloud_total?: number;
  visibility_m?: number;
  relative_humidity?: number;
  dewpoint_c?: number;
  temperature_c?: number;
  precipitation_mm?: number;
  wind_speed_kmh?: number;
  pressure_hpa?: number;
  aerosol_optical_depth?: number;
}

export interface WeatherSummary {
  cloud_low_pct: number;
  cloud_mid_pct: number;
  cloud_high_pct: number;
  cloud_total_pct: number;
  visibility_km: number;
  precipitation_mm: number;
  aerosol_optical_depth: number | null;
  aerosol_is_estimated: boolean;
  temperature_c: number;
  humidity_pct: number;
  wind_speed_kmh: number;
}

// ── Prediction ───────────────────────────────────────────────────────────────

export type SunsetCategory = "Poor" | "Decent" | "Good" | "Great" | "Epic";

export interface PhysicsBreakdown {
  cloud_quality_score: number;
  atmosphere_score: number;
  moisture_score: number;
  horizon_score: number;
  weighted_physics_score: number;
  component_weights: Record<string, number>;
}

export interface PredictRequest {
  latitude: number;
  longitude: number;
  target_date?: string; // ISO date string "YYYY-MM-DD"
  horizon_obstruction_deg?: number;
  weather_override?: WeatherOverride;
}

export interface PredictResponse {
  beauty_score_0_100: number;
  category: SunsetCategory;
  confidence_0_100: number;
  reasons: string[];
  sunset_time: string; // ISO datetime
  best_viewing_window_start: string;
  best_viewing_window_end: string;
  algorithm_version: string;
  ml_model_used: boolean;
  ml_adjustment: number | null;
  physics_component_breakdown: PhysicsBreakdown;
  weather_summary: WeatherSummary;
  location: { latitude: number; longitude: number };
  requested_at: string;
}

// ── Forecast ─────────────────────────────────────────────────────────────────

export interface ForecastRequest {
  latitude: number;
  longitude: number;
  days?: number;
  horizon_obstruction_deg?: number;
}

export interface DayForecast {
  date: string; // "YYYY-MM-DD"
  beauty_score_0_100: number;
  category: SunsetCategory;
  confidence_0_100: number;
  sunset_time: string;
  best_viewing_window_start: string;
  best_viewing_window_end: string;
  reasons: string[];
  physics_component_breakdown: PhysicsBreakdown;
  ml_model_used: boolean;
}

export interface ForecastResponse {
  days: DayForecast[];
  location: { latitude: number; longitude: number };
  algorithm_version: string;
  generated_at: string;
}

// ── Health ────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  algorithm_version: string;
  environment: string;
  ml_model_loaded: boolean;
  model_metadata: Record<string, unknown>;
  timestamp: string;
}

// ── Geocoding ─────────────────────────────────────────────────────────────────

export interface GeocodingResult {
  id: number;
  name: string;
  latitude: number;
  longitude: number;
  country: string;
  country_code: string;
  admin1?: string; // state / province
  timezone: string;
}

// ── App state ─────────────────────────────────────────────────────────────────

export interface LocationState {
  latitude: number;
  longitude: number;
  name: string;
}
