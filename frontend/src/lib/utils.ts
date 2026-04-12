/**
 * Shared utility functions for the Sunset Predictor frontend.
 */

import type { SunsetCategory } from "./types";

// ---------------------------------------------------------------------------
// Category styling
// ---------------------------------------------------------------------------

/** Tailwind text colour class for a sunset category. */
export function getCategoryColor(category: SunsetCategory): string {
  switch (category) {
    case "Epic":
      return "text-purple-400";
    case "Great":
      return "text-emerald-400";
    case "Good":
      return "text-yellow-400";
    case "Decent":
      return "text-amber-400";
    case "Poor":
      return "text-red-400";
    default:
      return "text-slate-400";
  }
}

/** Tailwind background colour class for a sunset category badge. */
export function getCategoryBgColor(category: SunsetCategory): string {
  switch (category) {
    case "Epic":
      return "bg-purple-500/20 text-purple-300 border-purple-500/30";
    case "Great":
      return "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
    case "Good":
      return "bg-yellow-500/20 text-yellow-300 border-yellow-500/30";
    case "Decent":
      return "bg-amber-500/20 text-amber-300 border-amber-500/30";
    case "Poor":
      return "bg-red-500/20 text-red-300 border-red-500/30";
    default:
      return "bg-slate-500/20 text-slate-300 border-slate-500/30";
  }
}

/**
 * Return a hex colour string for a numeric score (0–100).
 * Blends from red → amber → yellow → emerald → purple.
 */
export function getScoreHexColor(score: number): string {
  if (score >= 80) return "#a855f7"; // purple
  if (score >= 65) return "#34d399"; // emerald
  if (score >= 50) return "#fbbf24"; // amber
  if (score >= 30) return "#fb923c"; // orange
  return "#f87171"; // red
}

// ---------------------------------------------------------------------------
// Date / time formatting
// ---------------------------------------------------------------------------

/** Format an ISO datetime string as "HH:MM" in local browser time. */
export function formatTime(isoString: string): string {
  try {
    const dt = new Date(isoString);
    return dt.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch {
    return "--:--";
  }
}

/** Format an ISO date string as short weekday + date, e.g. "Fri 21 Jun". */
export function formatDateShort(isoString: string): string {
  try {
    const dt = new Date(isoString);
    return dt.toLocaleDateString(undefined, {
      weekday: "short",
      day: "numeric",
      month: "short",
    });
  } catch {
    return isoString;
  }
}

/** Format ISO date as "Monday, 21 June". */
export function formatDateLong(isoString: string): string {
  try {
    const dt = new Date(isoString);
    return dt.toLocaleDateString(undefined, {
      weekday: "long",
      day: "numeric",
      month: "long",
    });
  } catch {
    return isoString;
  }
}

/** True if the ISO date string refers to today. */
export function isToday(isoDateString: string): boolean {
  const today = new Date().toISOString().slice(0, 10);
  return isoDateString === today;
}

// ---------------------------------------------------------------------------
// Score helpers
// ---------------------------------------------------------------------------

/** Unicode emoji representing the score. */
export function scoreToEmoji(score: number): string {
  if (score >= 80) return "🌅";
  if (score >= 65) return "🧡";
  if (score >= 50) return "☁️";
  if (score >= 30) return "🌥️";
  return "🌧️";
}

/** Round to one decimal place. */
export function round1(n: number): number {
  return Math.round(n * 10) / 10;
}

// ---------------------------------------------------------------------------
// CSS helpers
// ---------------------------------------------------------------------------

/** Join class names, filtering falsy values. Minimal clsx-like helper. */
export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(" ");
}
