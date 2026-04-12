"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { forecast } from "@/lib/api";
import type { DayForecast, ForecastResponse } from "@/lib/types";

import SunsetCard from "@/components/SunsetCard";
import ForecastChart from "@/components/ForecastChart";
import LoadingState from "@/components/LoadingState";
import ErrorAlert from "@/components/ErrorAlert";

function ForecastContent() {
  const params = useSearchParams();
  const lat = parseFloat(params.get("lat") ?? "0");
  const lon = parseFloat(params.get("lon") ?? "0");
  const name = params.get("name") ?? `${lat.toFixed(3)}, ${lon.toFixed(3)}`;

  const [data, setData] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);

  const load = async () => {
    if (!lat || !lon) return;
    setLoading(true);
    setError(null);
    try {
      const result = await forecast({ latitude: lat, longitude: lon, days: 7 });
      setData(result);
      if (result.days.length > 0) setSelectedDate(result.days[0].date);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load forecast.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [lat, lon]);

  const handleDayClick = (day: DayForecast) => setSelectedDate(day.date);

  return (
    <main className="min-h-screen px-4 py-8 max-w-2xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Link
          href={lat && lon ? `/?lat=${lat}&lon=${lon}&name=${encodeURIComponent(name)}` : "/"}
          className="w-9 h-9 rounded-xl bg-slate-800/60 border border-slate-700/40 flex items-center justify-center text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeft size={16} />
        </Link>
        <div>
          <h1 className="text-xl font-bold text-white">7-Day Forecast</h1>
          <p className="text-slate-500 text-sm">{decodeURIComponent(name)}</p>
        </div>
      </div>

      {error && (
        <div className="mb-6">
          <ErrorAlert message={error} onRetry={load} />
        </div>
      )}

      {loading && <LoadingState message="Loading 7-day forecast…" />}

      {!loading && data && (
        <div className="space-y-6 animate-fade-in">
          {/* Chart overview */}
          <section className="bg-slate-900/60 rounded-2xl border border-slate-700/40 p-5">
            <h2 className="text-slate-400 text-xs uppercase tracking-wider mb-4">Score Overview</h2>
            <ForecastChart
              days={data.days}
              onDayClick={handleDayClick}
              selectedDate={selectedDate ?? undefined}
            />
          </section>

          {/* Day cards */}
          <section className="space-y-3">
            {data.days.map((day) => (
              <SunsetCard
                key={day.date}
                day={day}
                defaultExpanded={day.date === selectedDate && day.date === data.days[0]?.date}
              />
            ))}
          </section>

          <p className="text-slate-600 text-xs text-center">
            Algorithm v{data.algorithm_version} · Generated {new Date(data.generated_at).toLocaleTimeString()}
          </p>
        </div>
      )}

      {!loading && !data && !error && (
        <div className="text-center py-20 text-slate-600">
          <p>No forecast data. Check your location.</p>
        </div>
      )}
    </main>
  );
}

export default function ForecastPage() {
  return (
    <Suspense fallback={<LoadingState message="Loading forecast…" />}>
      <ForecastContent />
    </Suspense>
  );
}
