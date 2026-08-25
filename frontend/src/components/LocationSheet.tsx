"use client";

import { useEffect } from "react";
import { Check, X } from "lucide-react";
import type { LocationState } from "@/lib/types";
import { sameLocation } from "@/lib/storage";
import LocationSearch from "./LocationSearch";

interface LocationSheetProps {
  open: boolean;
  onClose: () => void;
  current: LocationState | null;
  places: LocationState[];
  onSelect: (location: LocationState) => void;
}

/**
 * Changing location, off the critical path.
 *
 * Picking a place used to be a gate every visit had to pass through; it now
 * lives behind the header chip, so the default experience is the reading.
 */
export default function LocationSheet({
  open,
  onClose,
  current,
  places,
  onSelect,
}: LocationSheetProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const handleSelect = (location: LocationState) => {
    onSelect(location);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center">
      <button
        aria-label="Close location picker"
        onClick={onClose}
        className="absolute inset-0 bg-slate-900/30 dark:bg-slate-950/60"
      />

      <div
        role="dialog"
        aria-modal="true"
        aria-label="Choose a location"
        className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto bg-white dark:bg-slate-900 rounded-t-3xl border-t border-x border-gray-200 dark:border-slate-700/50 px-4 pt-3 flex flex-col gap-4 shadow-2xl animate-slide-up"
        style={{ paddingBottom: "max(2rem, env(safe-area-inset-bottom))" }}
      >
        <div className="flex items-center gap-3">
          <h2 className="flex-1 text-lg font-bold tracking-tight text-gray-900 dark:text-white">
            Location
          </h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="w-11 h-11 flex-shrink-0 rounded-full flex items-center justify-center text-gray-600 dark:text-slate-400 hover:bg-gray-100 dark:hover:bg-slate-800 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        <LocationSearch onLocationSelect={handleSelect} currentLocation={current} />

        {places.length > 0 && (
          <div className="flex flex-col gap-1.5">
            <h3 className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold">
              Recent
            </h3>
            {places.map((place) => {
              const isCurrent = sameLocation(place, current);
              return (
                <button
                  key={`${place.latitude},${place.longitude}`}
                  onClick={() => handleSelect(place)}
                  className={`flex items-center gap-3 px-3.5 py-3 rounded-xl border text-left transition-colors ${
                    isCurrent
                      ? "bg-orange-50 dark:bg-orange-500/10 border-orange-500/50"
                      : "bg-white dark:bg-slate-800/40 border-gray-200 dark:border-slate-700/50 hover:border-orange-500/40"
                  }`}
                >
                  <span className="flex-1 min-w-0 truncate text-sm font-medium text-gray-900 dark:text-white">
                    {place.name}
                  </span>
                  {isCurrent && (
                    <Check size={15} className="flex-shrink-0 text-orange-600 dark:text-orange-400" />
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
