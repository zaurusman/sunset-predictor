"use client";

import { useState } from "react";
import { Navigation } from "lucide-react";
import type { LocationState, SunsetCategory } from "@/lib/types";
import LocationSearch from "./LocationSearch";

interface FirstRunProps {
  onLocationSelect: (location: LocationState) => void;
}

/** Band widths mirror the thresholds in `scoreCategory`. */
const SCALE: { label: SunsetCategory; grow: number; className: string }[] = [
  { label: "Poor", grow: 30, className: "bg-red-600 dark:bg-red-400" },
  { label: "Decent", grow: 20, className: "bg-orange-600 dark:bg-orange-400" },
  { label: "Good", grow: 15, className: "bg-yellow-600 dark:bg-yellow-400" },
  { label: "Great", grow: 15, className: "bg-emerald-600 dark:bg-emerald-400" },
  { label: "Epic", grow: 20, className: "bg-violet-600 dark:bg-violet-400" },
];

/**
 * Shown once, on a first visit with nothing remembered.
 *
 * It also teaches the 0–100 scale before anyone is shown a number on it,
 * which is the cheapest fix for a score that otherwise means nothing.
 */
export default function FirstRun({ onLocationSelect }: FirstRunProps) {
  const [locating, setLocating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const useMyLocation = () => {
    if (!navigator.geolocation) {
      setError("Your browser can't share a location — search for a place instead.");
      return;
    }
    setLocating(true);
    setError(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocating(false);
        onLocationSelect({
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
          name: `${pos.coords.latitude.toFixed(3)}, ${pos.coords.longitude.toFixed(3)}`,
        });
      },
      () => {
        setLocating(false);
        setError("Location access was declined — search for a place instead.");
      },
      { timeout: 10_000 }
    );
  };

  return (
    <div className="flex flex-col gap-6 animate-fade-in">
      <div className="flex flex-col gap-2 pt-6">
        <h1 className="text-[28px] leading-tight font-bold tracking-tight text-gray-900 dark:text-white text-pretty">
          Where are you watching from?
        </h1>
        <p className="text-gray-700 dark:text-slate-300 text-sm leading-relaxed text-pretty">
          Afterglow reads tonight&rsquo;s cloud, haze and moisture, and tells you whether the
          sky is worth stepping outside for.
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <button
          onClick={useMyLocation}
          disabled={locating}
          className="flex items-center justify-center gap-2 h-12 rounded-full bg-orange-600 hover:bg-orange-700 disabled:opacity-60 text-white text-[15px] font-semibold transition-colors"
        >
          {locating ? (
            <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
          ) : (
            <Navigation size={17} />
          )}
          {locating ? "Finding you…" : "Use my location"}
        </button>

        <div className="flex items-center gap-3">
          <span className="h-px flex-1 bg-gray-200 dark:bg-slate-700" />
          <span className="text-xs text-gray-600 dark:text-slate-400">or search</span>
          <span className="h-px flex-1 bg-gray-200 dark:bg-slate-700" />
        </div>

        <LocationSearch onLocationSelect={onLocationSelect} showGeolocate={false} />

        {error && <p className="text-sm text-red-700 dark:text-red-400">{error}</p>}
      </div>

      <div className="bg-white dark:bg-slate-900/60 rounded-2xl border border-gray-200 dark:border-slate-700/40 p-5 flex flex-col gap-3">
        <h2 className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold">
          What the score means
        </h2>
        <div className="flex h-2 rounded-full overflow-hidden">
          {SCALE.map(({ label, grow, className }) => (
            <span key={label} className={className} style={{ flexGrow: grow }} />
          ))}
        </div>
        <div className="flex justify-between">
          {SCALE.map(({ label }) => (
            <span
              key={label}
              className="text-[10.5px] font-medium text-gray-700 dark:text-slate-300"
            >
              {label}
            </span>
          ))}
        </div>
        <p className="text-gray-700 dark:text-slate-300 text-xs leading-relaxed text-pretty">
          Most evenings land in the 40s and 50s. Past 70, Afterglow tells you it&rsquo;s
          worth changing your plans for.
        </p>
      </div>
    </div>
  );
}
