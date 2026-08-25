"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Camera } from "lucide-react";
import { predict } from "@/lib/api";
import type { LocationState, PredictResponse } from "@/lib/types";
import {
  loadCachedPrediction,
  loadLocation,
  loadPlaces,
  rememberPlace,
  saveCachedPrediction,
  saveLocation,
} from "@/lib/storage";
import { freshnessLabel } from "@/lib/utils";

import AppNav from "@/components/AppNav";
import DatePicker from "@/components/DatePicker";
import ErrorAlert from "@/components/ErrorAlert";
import EvidenceDrawer from "@/components/EvidenceDrawer";
import FirstRun from "@/components/FirstRun";
import LoadingState from "@/components/LoadingState";
import LocationSheet from "@/components/LocationSheet";
import SubmitPhotoModal from "@/components/SubmitPhotoModal";
import VerdictCard from "@/components/VerdictCard";
import RateSunset from "@/components/RateSunset";
import ViewingCurve from "@/components/ViewingCurve";

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function HomeContent() {
  const params = useSearchParams();

  const [hydrated, setHydrated] = useState(false);
  const [location, setLocation] = useState<LocationState | null>(null);
  const [places, setPlaces] = useState<LocationState[]>([]);
  const [selectedDate, setSelectedDate] = useState(todayIso());
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [cachedAt, setCachedAt] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [photoOpen, setPhotoOpen] = useState(false);

  /** Guards against a slow response for a place the user has already left. */
  const requestRef = useRef(0);

  const fetchPrediction = useCallback(
    async (loc: LocationState, date: string) => {
      const ticket = ++requestRef.current;
      setRefreshing(true);
      setError(null);

      try {
        const result = await predict({
          latitude: loc.latitude,
          longitude: loc.longitude,
          target_date: date,
        });
        if (ticket !== requestRef.current) return;

        setPrediction(result);
        setCachedAt(new Date().toISOString());
        saveCachedPrediction(loc, date, result);
      } catch (err) {
        if (ticket !== requestRef.current) return;
        setError(err instanceof Error ? err.message : "Couldn't reach the forecast.");
      } finally {
        if (ticket === requestRef.current) setRefreshing(false);
      }
    },
    []
  );

  /** Show whatever we already know about this place and date, then refresh. */
  const showThen = useCallback(
    (loc: LocationState, date: string) => {
      const cached = loadCachedPrediction(loc, date);
      setPrediction(cached?.prediction ?? null);
      setCachedAt(cached?.cachedAt ?? null);
      void fetchPrediction(loc, date);
    },
    [fetchPrediction]
  );

  // Boot: prefer a place passed in the URL (from the other tabs), else the one
  // we remembered. Nothing was persisted before, so every visit used to start
  // from an empty screen and a fresh permission prompt.
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

    const initial = fromUrl ?? loadLocation();

    setPlaces(loadPlaces());
    setHydrated(true);

    if (initial) {
      setLocation(initial);
      if (fromUrl) saveLocation(fromUrl);
      showThen(initial, todayIso());
    }
    // Boot once; later changes flow through the handlers below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleLocationSelect = useCallback(
    (loc: LocationState) => {
      setLocation(loc);
      saveLocation(loc);
      setPlaces(rememberPlace(loc));
      showThen(loc, selectedDate);
    },
    [selectedDate, showThen]
  );

  const handleDateChange = useCallback(
    (date: string) => {
      setSelectedDate(date);
      if (location) showThen(location, date);
    },
    [location, showThen]
  );

  // Nothing is known until localStorage has been read; rendering the first-run
  // screen before then would flash it at returning visitors on every load.
  if (!hydrated) {
    return <div className="min-h-screen" aria-hidden="true" />;
  }

  if (!location) {
    return (
      <>
        <AppNav location={null} active="tonight" />
        <FirstRun onLocationSelect={handleLocationSelect} />
      </>
    );
  }

  const showSkeleton = !prediction && refreshing;

  return (
    <>
      <AppNav
        location={location}
        active="tonight"
        onChangeLocation={() => setSheetOpen(true)}
      />

      <div className="flex items-center gap-2 mb-4">
        <div className="flex-1 min-w-0">
          {cachedAt && (
            <span className="inline-flex items-center gap-1.5 text-xs text-gray-600 dark:text-slate-400">
              {refreshing && (
                <span className="w-1.5 h-1.5 rounded-full bg-orange-500 animate-pulse" />
              )}
              {refreshing ? "Updating…" : `Updated ${freshnessLabel(cachedAt)}`}
            </span>
          )}
        </div>
        <DatePicker value={selectedDate} onChange={handleDateChange} />
      </div>

      {error && (
        <div className="mb-5">
          <ErrorAlert
            message={
              prediction
                ? `Showing the last reading — ${error}`
                : error
            }
            onRetry={() => fetchPrediction(location, selectedDate)}
          />
        </div>
      )}

      {showSkeleton && <LoadingState message="Reading the sky…" />}

      {prediction && (
        <div className="flex flex-col gap-4 animate-fade-in">
          <VerdictCard prediction={prediction} targetDate={selectedDate} />

          {/* Ratings are ground truth for the scoring engine, so they're only
              offered once the evening has actually happened. */}
          {selectedDate <= new Date().toISOString().slice(0, 10) && (
            <RateSunset
              location={location}
              targetDate={selectedDate}
              predictedScore={prediction.beauty_score_0_100}
            />
          )}

          <ViewingCurve
            windowScores={prediction.window_scores}
            bestPoint={prediction.best_window_point}
            sunsetTime={prediction.sunset_time}
            cloudHighPct={prediction.weather_summary.cloud_high_pct}
            twilightGradient={prediction.physics_component_breakdown.twilight_gradient_score}
          />

          <EvidenceDrawer prediction={prediction} />

          <button
            onClick={() => setPhotoOpen(true)}
            className="flex items-center justify-center gap-2 w-full py-3 rounded-xl bg-white dark:bg-slate-800/60 border border-gray-200 dark:border-slate-700/40 text-gray-700 dark:text-slate-300 hover:text-orange-700 dark:hover:text-orange-400 hover:border-orange-500/40 transition-colors text-sm font-medium"
          >
            <Camera size={16} />
            Share your sunset photo
          </button>
        </div>
      )}

      <LocationSheet
        open={sheetOpen}
        onClose={() => setSheetOpen(false)}
        current={location}
        places={places}
        onSelect={handleLocationSelect}
      />

      {photoOpen && (
        <SubmitPhotoModal
          latitude={location.latitude}
          longitude={location.longitude}
          locationName={location.name}
          defaultDate={selectedDate}
          onClose={() => setPhotoOpen(false)}
        />
      )}
    </>
  );
}

export default function HomePage() {
  return (
    <main className="min-h-screen bg-gray-50 dark:bg-slate-950 text-gray-900 dark:text-white px-4 py-6 max-w-2xl mx-auto">
      <Suspense fallback={<div className="min-h-screen" aria-hidden="true" />}>
        <HomeContent />
      </Suspense>
    </main>
  );
}
