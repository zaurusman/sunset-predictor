"use client";

import { useCallback, useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Info } from "lucide-react";
import { forecast, isServiceBusy } from "@/lib/api";
import type { DayForecast, ForecastResponse, LocationState } from "@/lib/types";
import { loadCachedForecast, loadLocation, saveCachedForecast } from "@/lib/storage";
import { freshnessLabel } from "@/lib/utils";

import AppNav from "@/components/AppNav";
import SupportFooter from "@/components/SupportFooter";
import SunsetCard from "@/components/SunsetCard";
import ForecastChart from "@/components/ForecastChart";
import LoadingState from "@/components/LoadingState";
import ErrorAlert from "@/components/ErrorAlert";

function ForecastContent() {
  const params = useSearchParams();

  const [location, setLocation] = useState<LocationState | null>(null);
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** The last load failed only because the weather service is busy. */
  const [busy, setBusy] = useState(false);
  const [cachedAt, setCachedAt] = useState<string | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);

  const load = useCallback(async (loc: LocationState) => {
    // Paint the last forecast for this place first (as the Tonight tab does),
    // so a busy weather service leaves the week on screen, not an error.
    const cached = loadCachedForecast(loc, new Date().toISOString().slice(0, 10));
    if (cached) {
      setData(cached.forecast);
      setCachedAt(cached.cachedAt);
      setSelectedDate((d) => d ?? cached.forecast.days[0].date);
    }
    setLoading(true);
    setError(null);
    setBusy(false);
    try {
      const result = await forecast({
        latitude: loc.latitude,
        longitude: loc.longitude,
        days: 7,
      });
      setData(result);
      setCachedAt(new Date().toISOString());
      saveCachedForecast(loc, result);
      if (result.days.length > 0) setSelectedDate(result.days[0].date);
    } catch (err) {
      setBusy(isServiceBusy(err));
      setError(err instanceof Error ? err.message : "Failed to load forecast.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const lat = Number(params.get("lat"));
    const lon = Number(params.get("lon"));
    const name = params.get("name");

    // `useSearchParams` already decodes — decoding again here corrupts any
    // place name containing a literal percent sign.
    const fromUrl =
      Number.isFinite(lat) && Number.isFinite(lon) && (lat !== 0 || lon !== 0)
        ? {
            latitude: lat,
            longitude: lon,
            name: name || `${lat.toFixed(3)}, ${lon.toFixed(3)}`,
          }
        : null;

    const loc = fromUrl ?? loadLocation();
    setLocation(loc);
    if (loc) void load(loc);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDayClick = (day: DayForecast) => setSelectedDate(day.date);

  return (
    <>
      <AppNav location={location} active="forecast" />

      {!location && (
        <p className="text-center py-20 text-gray-600 dark:text-slate-400">
          Pick a location on the Tonight tab first.
        </p>
      )}

      {error && (
        <div className="mb-5">
          <ErrorAlert
            variant={busy ? "busy" : "error"}
            message={
              busy
                ? data
                  ? `The weather service is busy right now, so this is your last forecast${cachedAt ? ` (updated ${freshnessLabel(cachedAt)})` : ""}. Try again in a little while.`
                  : "The weather service is busy right now, so the 7-day forecast can't load yet. Try again in a little while."
                : error
            }
            onRetry={location ? () => load(location) : undefined}
          />
        </div>
      )}

      {loading && !data && <LoadingState message="Loading 7-day forecast…" />}

      {data && (
        <div className="flex flex-col gap-5 m-fade">
          <div className="flex items-start gap-3 px-4 py-3 rounded-xl bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-200 dark:border-indigo-500/20 text-indigo-800 dark:text-indigo-300 text-sm">
            <Info size={15} className="flex-shrink-0 mt-0.5" />
            <span className="text-pretty">
              Forecasts are updated daily and cloud cover can change significantly. For the
              best accuracy, check back on the day of each sunset.
            </span>
          </div>

          <section className="bg-white dark:bg-slate-900/60 rounded-2xl border border-gray-200 dark:border-slate-700/40 p-5">
            <h2 className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold mb-4">
              Score overview
            </h2>
            <ForecastChart
              days={data.days}
              onDayClick={handleDayClick}
              selectedDate={selectedDate ?? undefined}
            />
          </section>

          <section className="flex flex-col gap-3">
            {data.days.map((day) => (
              <SunsetCard
                key={day.date}
                day={day}
                defaultExpanded={day.date === selectedDate && day.date === data.days[0]?.date}
              />
            ))}
          </section>

          <p className="text-gray-500 dark:text-slate-500 text-xs text-center">
            Algorithm v{data.algorithm_version}
          </p>
        </div>
      )}
    </>
  );
}

export default function ForecastPage() {
  return (
    <main className="min-h-screen bg-gray-50 dark:bg-slate-950 text-gray-900 dark:text-white px-4 py-6 max-w-2xl mx-auto">
      <Suspense fallback={<LoadingState message="Loading forecast…" />}>
        <ForecastContent />
      </Suspense>
      <SupportFooter />
    </main>
  );
}
