"use client";

import { useCallback, useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Info } from "lucide-react";
import { forecast, isServiceBusy } from "@/lib/api";
import type { DayForecast, ForecastResponse, LocationState } from "@/lib/types";
import { loadCachedForecast, loadLocation, saveCachedForecast } from "@/lib/storage";
import { freshnessLabel, localToday } from "@/lib/utils";

import AppNav from "@/components/AppNav";
import { useSky } from "@/components/sky/SkyProvider";
import { usePrefersReducedMotion } from "@/lib/motion";
import SupportFooter from "@/components/SupportFooter";
import PageTransition from "@/components/PageTransition";
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
  /** The expanded card: undefined until the user chooses (today opens), null when all are shut. */
  const [openDate, setOpenDate] = useState<string | null | undefined>(undefined);
  const { setMood } = useSky();
  const reduce = usePrefersReducedMotion();

  const load = useCallback(async (loc: LocationState) => {
    // Paint the last forecast for this place first (as the Tonight tab does),
    // so a busy weather service leaves the week on screen, not an error.
    const cached = loadCachedForecast(loc, localToday());
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
      // The server's first day is the UTC date, which just after local
      // midnight is an evening that has already ended here.
      const upcoming = result.days.filter((d) => d.date >= localToday());
      setData(upcoming.length ? { ...result, days: upcoming } : result);
      setCachedAt(new Date().toISOString());
      saveCachedForecast(loc, result);
      const first = upcoming[0] ?? result.days[0];
      if (first) setSelectedDate(first.date);
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

  // The sky follows the day being looked at.
  const selectedDay = data?.days.find((d) => d.date === selectedDate);
  useEffect(() => {
    if (selectedDay) setMood(selectedDay.category);
  }, [selectedDay, setMood]);

  /** Choosing a day on the chart selects it, opens its card and brings it into view. */
  const handleDayClick = (day: DayForecast) => {
    setSelectedDate(day.date);
    setOpenDate(day.date);
    // Wait for the other card to fold so the scroll lands where it settles.
    setTimeout(() => {
      const el = document.getElementById(`day-${day.date}`);
      if (!el) return;
      const header = document.querySelector<HTMLElement>("[data-scrolled]")?.offsetHeight ?? 0;
      window.scrollTo({
        top: el.getBoundingClientRect().top + window.scrollY - header - 8,
        behavior: reduce ? "auto" : "smooth",
      });
    }, 460);
  };

  /** A card header toggles that card and selects its day. */
  const toggleCard = (day: DayForecast, isOpen: boolean) => {
    setSelectedDate(day.date);
    setOpenDate(isOpen ? null : day.date);
  };

  const expandedDate = openDate === undefined ? data?.days[0]?.date : openDate;

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

          <section className="flex flex-col gap-3 m-stagger">
            {data.days.map((day) => (
              <SunsetCard
                key={day.date}
                day={day}
                expanded={day.date === expandedDate}
                onToggle={() => toggleCard(day, day.date === expandedDate)}
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
    <PageTransition>
      <main className="min-h-screen text-gray-900 dark:text-white px-4 py-6 max-w-2xl mx-auto">
        <Suspense fallback={<LoadingState message="Loading forecast…" />}>
          <ForecastContent />
        </Suspense>
        <SupportFooter />
      </main>
    </PageTransition>
  );
}
