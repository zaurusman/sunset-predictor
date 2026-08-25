"use client";

import { useCallback, useEffect, useRef, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { heatmap as fetchHeatmap } from "@/lib/api";
import type { HeatmapDay, HeatmapResponse, LocationState } from "@/lib/types";
import { loadLocation } from "@/lib/storage";

import AppNav from "@/components/AppNav";
import HeatmapGrid from "@/components/HeatmapGrid";
import LoadingState from "@/components/LoadingState";
import ErrorAlert from "@/components/ErrorAlert";

const MONTHS_OPTIONS = [6, 12, 24] as const;
type MonthsOption = (typeof MONTHS_OPTIONS)[number];

function computeBestMonths(days: HeatmapDay[]): { month: string; avg: number }[] {
  const byMonth: Record<string, number[]> = {};
  for (const day of days) {
    const key = day.date.slice(0, 7); // "YYYY-MM"
    if (!byMonth[key]) byMonth[key] = [];
    byMonth[key].push(day.score);
  }

  return Object.entries(byMonth)
    .map(([key, scores]) => ({
      month: new Date(key + "-15T12:00:00").toLocaleDateString("en-US", {
        month: "long",
        year: "numeric",
      }),
      avg: scores.reduce((a, b) => a + b, 0) / scores.length,
    }))
    .sort((a, b) => b.avg - a.avg)
    .slice(0, 3);
}

// Animated progress bar that fills to ~85% while loading, then snaps to 100% on done.
function ProgressBar({ loading, months }: { loading: boolean; months: number }) {
  const [width, setWidth] = useState(0);
  const [visible, setVisible] = useState(false);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (loading) {
      setWidth(0);
      setVisible(true);
      if (hideTimer.current) clearTimeout(hideTimer.current);
      const t = setTimeout(() => setWidth(85), 60);
      return () => clearTimeout(t);
    } else if (visible) {
      setWidth(100);
      hideTimer.current = setTimeout(() => setVisible(false), 500);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading]);

  if (!visible) return null;

  return (
    <div className="mb-5">
      <div className="h-1.5 bg-gray-200 dark:bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-orange-500 rounded-full"
          style={{
            width: `${width}%`,
            transition:
              width === 0
                ? "none"
                : width === 100
                ? "width 0.4s ease"
                : "width 25s cubic-bezier(0.1, 0.4, 0.3, 1)",
          }}
        />
      </div>
      <p className="mt-2 text-xs text-gray-600 dark:text-slate-400 text-center">
        Computing {months} months of sunset scores — this takes a few seconds…
      </p>
    </div>
  );
}

function HeatmapContent() {
  const params = useSearchParams();

  const dataCache = useRef<Map<MonthsOption, HeatmapResponse>>(new Map());

  const [location, setLocation] = useState<LocationState | null>(null);
  const [months, setMonths] = useState<MonthsOption>(6);
  const [data, setData] = useState<HeatmapResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (loc: LocationState, m: MonthsOption, force = false) => {
      if (!force) {
        const cached = dataCache.current.get(m);
        if (cached) {
          setData(cached);
          setError(null);
          return;
        }
      }

      setLoading(true);
      setError(null);
      try {
        const result = await fetchHeatmap({
          lat: loc.latitude,
          lon: loc.longitude,
          months: m,
        });
        dataCache.current.set(m, result);
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load sunset history.");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    const lat = Number(params.get("lat"));
    const lon = Number(params.get("lon"));
    const name = params.get("name");

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
    if (loc) void load(loc, 6);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleMonthsChange = (m: MonthsOption) => {
    setMonths(m);
    if (location) void load(location, m);
  };

  const bestMonths = data ? computeBestMonths(data.days) : [];

  return (
    <>
      <AppNav location={location} active="heatmap" />

      {!location && (
        <p className="text-center py-20 text-gray-600 dark:text-slate-400">
          Pick a location on the Tonight tab first.
        </p>
      )}

      {location && (
        <div className="flex gap-2 mb-5">
          {MONTHS_OPTIONS.map((m) => (
            <button
              key={m}
              onClick={() => handleMonthsChange(m)}
              disabled={loading}
              aria-pressed={months === m}
              className={`px-4 py-2 rounded-xl text-sm font-semibold border transition-colors disabled:opacity-50 ${
                months === m
                  ? "bg-orange-600 border-orange-600 text-white"
                  : "bg-white dark:bg-slate-800/60 border-gray-200 dark:border-slate-700/40 text-gray-700 dark:text-slate-300 hover:border-orange-500/40"
              }`}
            >
              {m}m
            </button>
          ))}
        </div>
      )}

      <ProgressBar loading={loading} months={months} />

      {error && (
        <div className="mb-5">
          <ErrorAlert
            message={error}
            onRetry={location ? () => load(location, months, true) : undefined}
          />
        </div>
      )}

      {data && (
        <div
          className={`flex flex-col gap-5 ${loading ? "opacity-40 pointer-events-none" : "animate-fade-in"}`}
        >
          <section className="bg-white dark:bg-slate-900/60 rounded-2xl border border-gray-200 dark:border-slate-700/40 p-5">
            <h2 className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold mb-4">
              Daily sunset scores
            </h2>
            <HeatmapGrid days={data.days} />
          </section>

          {bestMonths.length > 0 && (
            <section className="bg-white dark:bg-slate-900/60 rounded-2xl border border-gray-200 dark:border-slate-700/40 p-5">
              <h2 className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold mb-4">
                Best months
              </h2>
              <div className="flex flex-col gap-3">
                {bestMonths.map(({ month, avg }, rank) => (
                  <div key={month} className="flex items-center gap-3">
                    <span className="text-xs text-gray-600 dark:text-slate-400 w-4 text-right tabular-nums">
                      {rank + 1}
                    </span>
                    <div className="flex-1">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm text-gray-800 dark:text-slate-200">{month}</span>
                        <span className="text-sm font-semibold text-gray-900 dark:text-white tabular-nums">
                          {avg.toFixed(0)}
                        </span>
                      </div>
                      <div className="h-1.5 rounded-full bg-gray-200 dark:bg-slate-700">
                        <div
                          className="h-full rounded-full bg-orange-500"
                          style={{ width: `${avg}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          <p className="text-gray-500 dark:text-slate-500 text-xs text-center">
            {data.days.length} days
          </p>
        </div>
      )}
    </>
  );
}

export default function HeatmapPage() {
  return (
    <main className="min-h-screen bg-gray-50 dark:bg-slate-950 text-gray-900 dark:text-white px-4 py-6 max-w-3xl mx-auto">
      <Suspense fallback={<LoadingState message="Loading sunset history…" />}>
        <HeatmapContent />
      </Suspense>
    </main>
  );
}
