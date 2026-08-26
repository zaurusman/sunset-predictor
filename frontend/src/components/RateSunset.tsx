"use client";

import { useEffect, useState } from "react";
import { rateSunset } from "@/lib/api";
import type { LocationState } from "@/lib/types";

interface RateSunsetProps {
  location: LocationState;
  /** The evening being rated, "YYYY-MM-DD". */
  targetDate: string;
  /** What the model predicted, so we can show the gap after rating. */
  predictedScore: number;
}

/**
 * One-tap rating of how the sunset actually looked.
 *
 * WHY THIS IS HERE
 * ----------------
 * The engine has never been measured against reality. These ratings are the
 * only ground truth it will have, and they accrue at one per evening — every
 * night without this is a row that can't be recovered later.
 *
 * The wording deliberately invites low ratings. The previous ML attempt failed
 * partly because its labels came from posted photos, so the dataset contained
 * no bad evenings and the model could never learn to say "not tonight".
 */

const OPTIONS: { value: number; label: string; hint: string }[] = [
  { value: 1, label: "Nothing", hint: "Grey, no colour at all" },
  { value: 2, label: "Dull", hint: "A bit of colour, forgettable" },
  { value: 3, label: "Pleasant", hint: "Nice enough" },
  { value: 4, label: "Very good", hint: "Worth having stopped for" },
  { value: 5, label: "Exceptional", hint: "One of the year's best" },
];

/** Local-storage key so a rated evening stays rated across reloads. */
function storageKey(date: string, loc: LocationState): string {
  return `afterglow:rated:${date}:${loc.latitude.toFixed(2)},${loc.longitude.toFixed(2)}`;
}

export default function RateSunset({
  location,
  targetDate,
  predictedScore,
}: RateSunsetProps) {
  const [submitted, setSubmitted] = useState<number | null>(null);
  const [pending, setPending] = useState<number | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Restore any rating already given for this evening at this location.
  useEffect(() => {
    setSubmitted(null);
    setMessage(null);
    setError(null);
    try {
      const stored = window.localStorage.getItem(storageKey(targetDate, location));
      if (stored) setSubmitted(Number(stored));
    } catch {
      // localStorage unavailable (private mode) — rating simply won't persist.
    }
  }, [targetDate, location]);

  async function submit(value: number) {
    setPending(value);
    setError(null);
    try {
      const res = await rateSunset({
        latitude: location.latitude,
        longitude: location.longitude,
        rating: value,
        target_date: targetDate,
        location_name: location.name,
      });
      setSubmitted(value);
      setMessage(res.message);
      try {
        window.localStorage.setItem(storageKey(targetDate, location), String(value));
      } catch {
        // Non-fatal: the rating is stored server-side regardless.
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save that rating.");
    } finally {
      setPending(null);
    }
  }

  // ---------------------------------------------------------------------
  // Already rated — show the verdict against what the model said.
  // ---------------------------------------------------------------------
  if (submitted !== null) {
    const chosen = OPTIONS.find((o) => o.value === submitted);
    // Map 1–5 onto 0–100 to compare like with like.
    const implied = ((submitted - 1) / 4) * 100;
    const gap = Math.round(predictedScore - implied);

    return (
      <section className="bg-white dark:bg-slate-900/60 rounded-2xl border border-gray-200 dark:border-slate-700/40 p-4 flex flex-col gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold">
            You rated this
          </span>
          <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-gray-100 dark:bg-slate-800 text-gray-900 dark:text-white border border-gray-200 dark:border-slate-700">
            {chosen?.label ?? submitted}
          </span>
        </div>
        <p className="text-sm text-gray-700 dark:text-slate-300 leading-snug text-pretty">
          {message ??
            (Math.abs(gap) > 30
              ? `The model said ${Math.round(predictedScore)} — off by a lot. Logged.`
              : "Logged.")}
        </p>
        <button
          type="button"
          onClick={() => setSubmitted(null)}
          className="self-start text-xs text-gray-500 dark:text-slate-400 underline underline-offset-2 hover:text-gray-800 dark:hover:text-slate-200"
        >
          Change
        </button>
      </section>
    );
  }

  // ---------------------------------------------------------------------
  // Not yet rated
  // ---------------------------------------------------------------------
  return (
    <section className="bg-white dark:bg-slate-900/60 rounded-2xl border border-gray-200 dark:border-slate-700/40 p-4 flex flex-col gap-3">
      <div className="flex flex-col gap-0.5">
        <span className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold">
          How did it actually look?
        </span>
        <p className="text-xs text-gray-500 dark:text-slate-400 text-pretty">
          Rate the dull ones too — that&apos;s the half the model can&apos;t learn without.
        </p>
      </div>

      <div className="flex gap-1.5 flex-wrap">
        {OPTIONS.map((o) => (
          <button
            key={o.value}
            type="button"
            title={o.hint}
            disabled={pending !== null}
            onClick={() => submit(o.value)}
            className="flex-1 min-w-[64px] px-2 py-2 rounded-xl border border-gray-200 dark:border-slate-700 bg-gray-50 dark:bg-slate-800/60 text-xs font-semibold text-gray-800 dark:text-slate-200 hover:border-gray-400 dark:hover:border-slate-500 hover:bg-gray-100 dark:hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {pending === o.value ? "…" : o.label}
          </button>
        ))}
      </div>

      {error && (
        <p className="text-xs text-red-600 dark:text-red-400" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
