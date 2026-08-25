/**
 * Local persistence for Afterglow.
 *
 * Afterglow is checked daily, like a weather app, so a returning visitor must
 * never be sent back through a location gate. We remember where they were and
 * what we last told them, which lets the app paint a true answer before the
 * network is consulted.
 *
 * Every read is defensive: storage can be unavailable (private mode, disabled
 * cookies) or hold data written by an older version of the app.
 */

import type { LocationState, PredictResponse } from "./types";

const LOCATION_KEY = "afterglow:location";
const PLACES_KEY = "afterglow:places";
const PREDICTION_KEY = "afterglow:lastPrediction";

/** How many saved places the location sheet will keep. */
export const MAX_SAVED_PLACES = 5;

/** Cached readings older than this are ignored — the sky has moved on. */
const MAX_CACHE_AGE_MS = 24 * 60 * 60 * 1000;

export interface CachedPrediction {
  prediction: PredictResponse;
  location: LocationState;
  /** The date the prediction was requested for, "YYYY-MM-DD". */
  targetDate: string;
  /** ISO timestamp of when this reading was fetched. */
  cachedAt: string;
}

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

function readJson<T>(key: string): T | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function writeJson(key: string, value: unknown): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Quota exceeded or storage disabled — persistence is an enhancement here,
    // never a requirement, so failing to write is not an error worth raising.
  }
}

function isLocation(value: unknown): value is LocationState {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.latitude === "number" &&
    typeof v.longitude === "number" &&
    Number.isFinite(v.latitude) &&
    Number.isFinite(v.longitude) &&
    typeof v.name === "string"
  );
}

/** Two locations are the same place if they agree to ~100 m. */
export function sameLocation(a: LocationState | null, b: LocationState | null): boolean {
  if (!a || !b) return false;
  return (
    Math.abs(a.latitude - b.latitude) < 0.001 &&
    Math.abs(a.longitude - b.longitude) < 0.001
  );
}

// ---------------------------------------------------------------------------
// Current location
// ---------------------------------------------------------------------------

export function loadLocation(): LocationState | null {
  const stored = readJson<unknown>(LOCATION_KEY);
  return isLocation(stored) ? stored : null;
}

export function saveLocation(location: LocationState): void {
  writeJson(LOCATION_KEY, location);
}

// ---------------------------------------------------------------------------
// Saved places
// ---------------------------------------------------------------------------

export function loadPlaces(): LocationState[] {
  const stored = readJson<unknown>(PLACES_KEY);
  if (!Array.isArray(stored)) return [];
  return stored.filter(isLocation).slice(0, MAX_SAVED_PLACES);
}

/**
 * Move a place to the front of the saved list, de-duplicating by coordinates
 * so the same spot never appears twice under two spellings.
 */
export function rememberPlace(location: LocationState): LocationState[] {
  const existing = loadPlaces().filter((p) => !sameLocation(p, location));
  const next = [location, ...existing].slice(0, MAX_SAVED_PLACES);
  writeJson(PLACES_KEY, next);
  return next;
}

export function forgetPlace(location: LocationState): LocationState[] {
  const next = loadPlaces().filter((p) => !sameLocation(p, location));
  writeJson(PLACES_KEY, next);
  return next;
}

// ---------------------------------------------------------------------------
// Last reading
// ---------------------------------------------------------------------------

/**
 * The most recent reading, if it is for this location and date and is still
 * fresh enough to be worth showing while a new one loads.
 */
export function loadCachedPrediction(
  location: LocationState | null,
  targetDate: string
): CachedPrediction | null {
  const stored = readJson<CachedPrediction>(PREDICTION_KEY);
  if (!stored?.prediction || !isLocation(stored.location)) return null;
  if (stored.targetDate !== targetDate) return null;
  if (location && !sameLocation(stored.location, location)) return null;

  const age = Date.now() - new Date(stored.cachedAt).getTime();
  if (!Number.isFinite(age) || age > MAX_CACHE_AGE_MS) return null;

  return stored;
}

export function saveCachedPrediction(
  location: LocationState,
  targetDate: string,
  prediction: PredictResponse
): void {
  const entry: CachedPrediction = {
    prediction,
    location,
    targetDate,
    cachedAt: new Date().toISOString(),
  };
  writeJson(PREDICTION_KEY, entry);
}
