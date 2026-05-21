"use client";

import { useEffect, useRef, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { heatmap as fetchHeatmap } from "@/lib/api";
import type { HeatmapDay, HeatmapResponse } from "@/lib/types";

import HeatmapGrid from "@/components/HeatmapGrid";
import LoadingState from "@/components/LoadingState";
import ErrorAlert from "@/components/ErrorAlert";
import ThemeToggle from "@/components/ThemeToggle";

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
      // Reset to 0 (no transition), then animate to 85% slowly
      setWidth(0);
      setVisible(true);
      if (hideTimer.current) clearTimeout(hideTimer.current);
      const t = setTimeout(() => setWidth(85), 60);
      return () => clearTimeout(t);
    } else if (visible) {
      // Snap to 100%, then fade out
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
          className="h-full bg-orange-400 rounded-full"
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
      <p className="mt-2 text-xs text-gray-400 dark:text-slate-500 text-center">
        Computing {months} months of sunset scores — this takes a few seconds…
      </p>
    </div>
  );
}

function HeatmapContent() {
  const params = useSearchParams();
  const lat = parseFloat(params.get("lat") ?? "0");
  const lon = parseFloat(params.get("lon") ?? "0");
  const name = params.get("name") ?? `${lat.toFixed(3)}, ${lon.toFixed(3)}`;

  // Client-side cache: avoid re-fetching months we already have
  const dataCache = useRef<Map<MonthsOption, HeatmapResponse>>(new Map());

  const [months, setMonths] = useState<MonthsOption>(6);
  const [data, setData] = useState<HeatmapResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async (m: MonthsOption, force = false) => {
    if (!lat || !lon) return;

    // Serve from cache if available (unless forced refresh)
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
      const result = await fetchHeatmap({ lat, lon, months: m });
      dataCache.current.set(m, result);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sunset history.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load(months);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lat, lon]);

  const handleMonthsChange = (m: MonthsOption) => {
    setMonths(m);
    load(m);
  };

  const bestMonths = data ? computeBestMonths(data.days) : [];

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-slate-950 text-gray-900 dark:text-white px-4 py-8 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Link
          href={lat && lon ? `/?lat=${lat}&lon=${lon}&name=${encodeURIComponent(name)}` : "/"}
          className="w-9 h-9 rounded-xl bg-gray-100/60 dark:bg-slate-800/60 border border-gray-200/40 dark:border-slate-700/40 flex items-center justify-center text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white transition-colors"
        >
          <ArrowLeft size={16} />
        </Link>
        <div className="flex-1">
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">Sunset History</h1>
          <p className="text-gray-400 dark:text-slate-500 text-sm">{decodeURIComponent(name)}</p>
        </div>
        <ThemeToggle />
      </div>

      {/* Months selector */}
      <div className="flex gap-2 mb-6">
        {MONTHS_OPTIONS.map((m) => (
          <button
            key={m}
            onClick={() => handleMonthsChange(m)}
            disabled={loading}
            className={`px-4 py-2 rounded-xl text-sm font-medium border transition-colors disabled:opacity-50 ${
              months === m
                ? "bg-orange-500 border-orange-500 text-white"
                : "bg-gray-100/60 dark:bg-slate-800/60 border-gray-200/40 dark:border-slate-700/40 text-gray-600 dark:text-slate-400 hover:text-orange-500 dark:hover:text-orange-400 hover:border-orange-500/30"
            }`}
          >
            {m}m
          </button>
        ))}
      </div>

      {/* Progress bar — shown above content while loading */}
      <ProgressBar loading={loading} months={months} />

      {error && (
        <div className="mb-6">
          <ErrorAlert message={error} onRetry={() => load(months, true)} />
        </div>
      )}

      {!loading && !data && !error && (
        <div className="text-center py-20 text-gray-300 dark:text-slate-600">
          <p>No history data. Check your location.</p>
        </div>
      )}

      {data && (
        <div className={`space-y-6 ${loading ? "opacity-40 pointer-events-none" : "animate-fade-in"}`}>
          {/* Heatmap grid */}
          <section className="bg-gray-100/60 dark:bg-slate-900/60 rounded-2xl border border-gray-200/40 dark:border-slate-700/40 p-5">
            <h2 className="text-gray-400 dark:text-slate-400 text-xs uppercase tracking-wider mb-4">
              Daily Sunset Scores
            </h2>
            <HeatmapGrid days={data.days} />
          </section>

          {/* Best months */}
          {bestMonths.length > 0 && (
            <section className="bg-gray-100/60 dark:bg-slate-900/60 rounded-2xl border border-gray-200/40 dark:border-slate-700/40 p-5">
              <h2 className="text-gray-400 dark:text-slate-400 text-xs uppercase tracking-wider mb-4">
                Best Months
              </h2>
              <div className="space-y-3">
                {bestMonths.map(({ month, avg }, rank) => (
                  <div key={month} className="flex items-center gap-3">
                    <span className="text-xs text-gray-400 dark:text-slate-500 w-4 text-right">
                      {rank + 1}
                    </span>
                    <div className="flex-1">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm text-gray-700 dark:text-slate-300">{month}</span>
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          {avg.toFixed(0)}
                        </span>
                      </div>
                      <div className="h-1.5 rounded-full bg-gray-200 dark:bg-slate-700">
                        <div
                          className="h-full rounded-full bg-orange-400"
                          style={{ width: `${avg}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          <p className="text-gray-300 dark:text-slate-600 text-xs text-center">
            {data.days.length} days · Generated{" "}
            {new Date(data.generated_at).toLocaleTimeString()}
          </p>
        </div>
      )}
    </main>
  );
}

export default function HeatmapPage() {
  return (
    <Suspense fallback={<LoadingState message="Loading sunset history…" />}>
      <HeatmapContent />
    </Suspense>
  );
}
