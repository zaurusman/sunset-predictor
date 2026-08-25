"use client";

import { useCallback, useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Info } from "lucide-react";
import { forecast } from "@/lib/api";
import type { DayForecast, ForecastResponse, LocationState } from "@/lib/types";
import { loadLocation } from "@/lib/storage";

import AppNav from "@/components/AppNav";
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
  const [selectedDate, setSelectedDate] = useState<string | null>(null);

  const load = useCallback(async (loc: LocationState) => {
    setLoading(true);
    setError(null);
    try {
      const result = await forecast({
        latitude: loc.latitude,
        longitude: loc.longitude,
        days: 7,
      });
      setData(result);
      if (result.days.length > 0) setSelectedDate(result.days[0].date);
    } catch (err) {
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
          <ErrorAlert message={error} onRetry={location ? () => load(location) : undefined} />
        </div>
      )}

      {loading && <LoadingState message="Loading 7-day forecast…" />}

      {!loading && data && (
        <div className="flex flex-col gap-5 animate-fade-in">
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
    </main>
  );
}
