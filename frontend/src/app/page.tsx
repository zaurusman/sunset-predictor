"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { CalendarDays, Info } from "lucide-react";
import { predict } from "@/lib/api";
import type { LocationState, PredictResponse } from "@/lib/types";
import { formatTime, scoreToEmoji } from "@/lib/utils";

import ScoreDial from "@/components/ScoreDial";
import LocationSearch from "@/components/LocationSearch";
import DatePicker from "@/components/DatePicker";
import ReasonsList from "@/components/ReasonsList";
import ComponentBreakdown from "@/components/ComponentBreakdown";
import ViewingWindow from "@/components/ViewingWindow";
import ModelInfoPanel from "@/components/ModelInfoPanel";
import LoadingState from "@/components/LoadingState";
import ErrorAlert from "@/components/ErrorAlert";

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

export default function HomePage() {
  const [location, setLocation] = useState<LocationState | null>(null);
  const [selectedDate, setSelectedDate] = useState<string>(todayIso());
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDebug, setShowDebug] = useState(false);

  const fetchPrediction = useCallback(async (loc: LocationState, date: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await predict({
        latitude: loc.latitude,
        longitude: loc.longitude,
        target_date: date,
      });
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch prediction.");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleLocationSelect = useCallback(
    (loc: LocationState) => {
      setLocation(loc);
      fetchPrediction(loc, selectedDate);
    },
    [fetchPrediction, selectedDate]
  );

  const handleDateChange = useCallback(
    (date: string) => {
      setSelectedDate(date);
      if (location) fetchPrediction(location, date);
    },
    [fetchPrediction, location]
  );

  // Try browser geolocation on first load
  useEffect(() => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const loc: LocationState = {
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
          name: `${pos.coords.latitude.toFixed(3)}, ${pos.coords.longitude.toFixed(3)}`,
        };
        setLocation(loc);
        fetchPrediction(loc, todayIso());
      },
      () => {} // silently ignore — user can search manually
    );
  }, [fetchPrediction]);

  return (
    <main className="min-h-screen px-4 py-8 max-w-2xl mx-auto">
      {/* Header */}
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold gradient-text mb-1">Sunset Predictor</h1>
        <p className="text-slate-500 text-sm">
          {selectedDate === todayIso()
            ? "How beautiful will today\u2019s sunset be?"
            : "How beautiful was that sunset?"}
        </p>
      </header>

      {/* Location search + date picker */}
      <div className="flex flex-col items-center gap-3 mb-8">
        <LocationSearch
          onLocationSelect={handleLocationSelect}
          currentLocation={location}
          disabled={loading}
        />
        <DatePicker
          value={selectedDate}
          onChange={handleDateChange}
          disabled={loading}
        />
      </div>

      {/* Error */}
      {error && (
        <div className="mb-6">
          <ErrorAlert
            message={error}
            onRetry={location ? () => fetchPrediction(location, selectedDate) : undefined}
          />
        </div>
      )}

      {/* Loading */}
      {loading && (
        <LoadingState
          message={
            selectedDate === todayIso()
              ? "Predicting tonight\u2019s sunset\u2026"
              : `Looking up the sunset for ${new Date(selectedDate + "T12:00:00").toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}\u2026`
          }
        />
      )}

      {/* Prediction result */}
      {!loading && prediction && (
        <div className="space-y-6 animate-fade-in">
          {/* Location name + date */}
          {location && (
            <div className="text-center text-slate-400 text-sm">
              {location.name}
            </div>
          )}

          {/* Score dial — centred hero element */}
          <div className="flex justify-center">
            <ScoreDial
              score={prediction.beauty_score_0_100}
              category={prediction.category}
              confidence={prediction.confidence_0_100}
              size={220}
            />
          </div>

          {/* Sunset time */}
          <div className="flex justify-center gap-6 text-sm">
            <div className="text-center">
              <div className="text-slate-500 text-xs mb-0.5">Sunset</div>
              <div className="text-white font-semibold tabular-nums">
                {formatTime(prediction.sunset_time)}
              </div>
            </div>
            <div className="w-px bg-slate-700/50" />
            <div className="text-center">
              <div className="text-slate-500 text-xs mb-0.5">Best window</div>
              <div className="text-white font-semibold tabular-nums">
                {formatTime(prediction.best_viewing_window_start)}–{formatTime(prediction.best_viewing_window_end)}
              </div>
            </div>
            <div className="w-px bg-slate-700/50" />
            <div className="text-center">
              <div className="text-slate-500 text-xs mb-0.5">Score</div>
              <div className="text-white font-semibold">
                {scoreToEmoji(prediction.beauty_score_0_100)} {Math.round(prediction.beauty_score_0_100)}
              </div>
            </div>
          </div>

          {/* Viewing window */}
          <ViewingWindow
            sunsetTime={prediction.sunset_time}
            windowStart={prediction.best_viewing_window_start}
            windowEnd={prediction.best_viewing_window_end}
          />

          {/* Reasons */}
          <section>
            <h2 className="text-slate-400 text-xs uppercase tracking-wider mb-3">Why</h2>
            <ReasonsList reasons={prediction.reasons} />
          </section>

          {/* Component breakdown */}
          <section className="bg-slate-900/60 rounded-2xl border border-slate-700/40 p-5">
            <h2 className="text-slate-400 text-xs uppercase tracking-wider mb-4">Score Breakdown</h2>
            <ComponentBreakdown breakdown={prediction.physics_component_breakdown} />
            {prediction.ml_adjustment !== null && (
              <p className="text-slate-500 text-xs mt-3">
                ML adjustment: {prediction.ml_adjustment > 0 ? "+" : ""}{prediction.ml_adjustment} pts
              </p>
            )}
          </section>

          {/* Weather summary */}
          <section className="bg-slate-900/60 rounded-2xl border border-slate-700/40 p-5">
            <h2 className="text-slate-400 text-xs uppercase tracking-wider mb-4">Weather at Sunset</h2>
            <div className="grid grid-cols-3 gap-3 text-sm">
              <Stat label="Cloud (total)" value={`${Math.round(prediction.weather_summary.cloud_total_pct)}%`} />
              <Stat label="Cloud (high)" value={`${Math.round(prediction.weather_summary.cloud_high_pct)}%`} />
              <Stat label="Cloud (low)" value={`${Math.round(prediction.weather_summary.cloud_low_pct)}%`} />
              <Stat label="Visibility" value={`${prediction.weather_summary.visibility_km} km`} />
              <Stat label="Humidity" value={`${Math.round(prediction.weather_summary.humidity_pct)}%`} />
              <Stat label="Rain" value={`${prediction.weather_summary.precipitation_mm} mm`} />
              {prediction.weather_summary.aerosol_optical_depth !== null && (
                <Stat
                  label={`AOD${prediction.weather_summary.aerosol_is_estimated ? " (est.)" : ""}`}
                  value={prediction.weather_summary.aerosol_optical_depth.toFixed(3)}
                />
              )}
              <Stat label="Temp." value={`${prediction.weather_summary.temperature_c}°C`} />
              <Stat label="Wind" value={`${prediction.weather_summary.wind_speed_kmh} km/h`} />
            </div>
          </section>

          {/* Forecast link */}
          {location && (
            <Link
              href={`/forecast?lat=${location.latitude}&lon=${location.longitude}&name=${encodeURIComponent(location.name)}`}
              className="flex items-center justify-center gap-2 w-full py-3 rounded-xl bg-slate-800/60 border border-slate-700/40 text-slate-300 hover:text-orange-400 hover:border-orange-500/30 transition-colors text-sm font-medium"
            >
              <CalendarDays size={16} />
              View 7-day forecast
            </Link>
          )}

          {/* Debug / info section */}
          <button
            onClick={() => setShowDebug((v) => !v)}
            className="flex items-center gap-1.5 text-slate-600 hover:text-slate-400 text-xs transition-colors mx-auto"
          >
            <Info size={12} />
            {showDebug ? "Hide" : "Show"} model info
          </button>
          {showDebug && <ModelInfoPanel />}
        </div>
      )}

      {/* Empty state — no location yet */}
      {!loading && !prediction && !error && (
        <div className="text-center py-20 text-slate-600">
          <div className="text-5xl mb-4">🌅</div>
          <p className="text-lg font-medium text-slate-500">Search for a location to get started</p>
          <p className="text-sm mt-2">or allow location access for your current area</p>
        </div>
      )}
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-800/40 rounded-lg p-2.5">
      <div className="text-slate-500 text-xs mb-0.5">{label}</div>
      <div className="text-white font-medium text-sm">{value}</div>
    </div>
  );
}
