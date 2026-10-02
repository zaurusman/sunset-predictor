"use client";

import { useEffect, useState } from "react";
import { Bell, BellOff, Check, X } from "lucide-react";
import type { LocationState } from "@/lib/types";
import { sameLocation } from "@/lib/storage";
import { isStandalone } from "@/lib/install";
import { alertsAvailable, enableAlerts, isAlertOn, permission, setAlert } from "@/lib/push";
import LocationSearch from "./LocationSearch";

interface LocationSheetProps {
  open: boolean;
  onClose: () => void;
  current: LocationState | null;
  places: LocationState[];
  onSelect: (location: LocationState) => void;
  /** Changes when alerts are switched on elsewhere, so the bells re-check. */
  alertsVersion?: number;
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
  alertsVersion,
}: LocationSheetProps) {
  const [bellsVisible, setBellsVisible] = useState(false);
  const [blocked, setBlocked] = useState(false);
  const [, rerender] = useState(0);
  const [pending, setPending] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!open) return;
    const perm = permission();
    setBlocked(perm === "denied");
    // Bells only make sense where a push can actually arrive: the installed
    // app, or a browser that has already granted permission.
    if (perm === "unsupported" || (!isStandalone() && perm !== "granted")) {
      setBellsVisible(false);
      return;
    }
    let cancelled = false;
    alertsAvailable().then((ok) => {
      if (!cancelled) setBellsVisible(ok);
    });
    return () => {
      cancelled = true;
    };
  }, [open, alertsVersion]);

  if (!open) return null;

  const toggleBell = async (place: LocationState) => {
    const key = `${place.latitude},${place.longitude}`;
    setPending(key);
    try {
      if (permission() === "default") {
        await enableAlerts(place);
      } else {
        await setAlert(place, !isAlertOn(place));
      }
    } catch {
      // network failure — the bell simply stays as it was
    } finally {
      setBlocked(permission() === "denied");
      setPending(null);
      rerender((n) => n + 1);
    }
  };

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
        className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto bg-white dark:bg-slate-900 rounded-t-3xl border-t border-x border-gray-200 dark:border-slate-700/50 px-4 pt-3 flex flex-col gap-4 shadow-2xl m-sheet"
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
            {bellsVisible && (
              <p className="text-xs text-gray-600 dark:text-slate-400">
                {blocked
                  ? "Notifications are blocked — enable them for Afterglow in Settings."
                  : "Tap a bell to get pinged ~4 h before an Epic sunset there."}
              </p>
            )}
            {places.map((place) => {
              const isCurrent = sameLocation(place, current);
              const key = `${place.latitude},${place.longitude}`;
              const on = isAlertOn(place);
              return (
                <div key={key} className="flex items-center gap-2">
                  <button
                    onClick={() => handleSelect(place)}
                    className={`flex-1 min-w-0 flex items-center gap-3 px-3.5 py-3 rounded-xl border text-left transition-colors ${
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
                  {bellsVisible && (
                    <button
                      onClick={() => toggleBell(place)}
                      disabled={blocked || pending === key}
                      aria-pressed={on}
                      aria-label={on ? `Turn off Epic alerts for ${place.name}` : `Turn on Epic alerts for ${place.name}`}
                      className={`w-11 h-11 flex-shrink-0 rounded-xl border flex items-center justify-center transition-colors disabled:opacity-50 ${
                        on
                          ? "bg-orange-600 border-orange-600 text-white"
                          : "bg-white dark:bg-slate-800/40 border-gray-200 dark:border-slate-700/50 text-gray-500 dark:text-slate-400 hover:text-orange-600 dark:hover:text-orange-400"
                      }`}
                    >
                      {on ? <Bell size={16} /> : <BellOff size={16} />}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
