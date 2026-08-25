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
  /** Afterglow potential; non-null only at window points after sunset. */
  afterglow_score: number | null;

  // ── Gates: multiplicative, applied after the weighted average (1.0 = no effect) ──
  /** Fraction of sunset light reaching the clouds overhead, sampled 60-400 km
   *  upstream along the sunset azimuth. Null when corridor data was unavailable. */
  light_corridor_factor: number | null;
  /** Fraction of the score surviving active rain (1.0 = dry). */
  precipitation_gate: number;
  /** Fraction of the score surviving horizon obstruction (1.0 = open). */
  horizon_gate: number;
}

/** The four sampled moments around sunset, in chronological order. */
export const WINDOW_POINTS = ["-15m", "sunset", "+15m", "+30m"] as const;
export type WindowPoint = (typeof WINDOW_POINTS)[number];

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
  /** Highest-scoring moment in the window: "-15m" | "sunset" | "+15m" | "+30m". */
  best_window_point: string;
  /** Physics score at each sampled moment, keyed by window point. */
  window_scores: Record<string, number>;
  /** True when conditions are worth going outside for. */
  go_outside_recommendation: boolean;
  algorithm_version: string;
  ml_model_used: boolean;
  ml_adjustment: number | null;
  /** Uncalibrated physics score, before percentile mapping. */
  raw_physics_score: number | null;
  /** Fraction of evenings at this location that score lower than tonight. */
  climatology_percentile: number | null;
  /** False while the local climatology is still warming — the score is ranked
   *  against a global reference curve and will shift once local history lands. */
  climatology_is_local: boolean;

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
  best_window_point: string;
  window_scores: Record<string, number>;
  go_outside_recommendation: boolean;
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

// ── Photo submission ─────────────────────────────────────────────────────────

export interface SubmitPhotoResponse {
  success: boolean;
  message: string;
}

// ── Heatmap ───────────────────────────────────────────────────────────────────

export interface HeatmapDay {
  date: string; // "YYYY-MM-DD"
  score: number;
  category: SunsetCategory;
}

export interface HeatmapResponse {
  days: HeatmapDay[];
  location: { latitude: number; longitude: number };
  generated_at: string;
}

// ── App state ─────────────────────────────────────────────────────────────────

export interface LocationState {
  latitude: number;
  longitude: number;
  name: string;
}

// ── Sunset ratings (ML training labels) ──────────────────────────────────────

export interface RatingRequest {
  latitude: number;
  longitude: number;
  /** 1 = nothing, 2 = dull, 3 = pleasant, 4 = very good, 5 = exceptional. */
  rating: number;
  target_date?: string; // "YYYY-MM-DD"; defaults to the local sunset date
  location_name?: string;
  notes?: string;
}

export interface RatingResponse {
  success: boolean;
  message: string;
  rated_date: string;
  predicted_score: number | null;
  total_ratings: number;
}

export interface RatingStats {
  total_ratings: number;
  distinct_locations: number;
  distinct_dates: number;
  rating_histogram: Record<string, number>;
  mean_rating: number | null;
  spearman_vs_model: number | null;
  spearman_sample_size: number;
  note: string;
}
