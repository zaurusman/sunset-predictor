/**
 * Shared utility functions for the Sunset Predictor frontend.
 */

import type { SunsetCategory } from "./types";

// ---------------------------------------------------------------------------
// Score thresholds
// ---------------------------------------------------------------------------

/**
 * The single source of truth for score banding. Every colour, label and bar in
 * the app derives from this — previously the dial banded at 80/65/50/30 while
 * the breakdown bars banded at 75/50/30, so one score could render in two
 * different colours on the same screen.
 */
export function scoreCategory(score: number): SunsetCategory {
  if (score >= 80) return "Epic";
  if (score >= 65) return "Great";
  if (score >= 50) return "Good";
  if (score >= 30) return "Decent";
  return "Poor";
}

// ---------------------------------------------------------------------------
// Category styling
//
// Light mode is the default theme, so every pair below is checked against a
// white/near-white card at WCAG AA (4.5:1 for normal text). The previous
// palette was dark-mode-only — `text-emerald-300` on `bg-emerald-500/20`
// measured 1.22:1 in light mode.
// ---------------------------------------------------------------------------

/** Tailwind text colour classes for a sunset category (light + dark). */
export function getCategoryColor(category: SunsetCategory): string {
  switch (category) {
    case "Epic":
      return "text-violet-700 dark:text-violet-300";
    case "Great":
      return "text-emerald-700 dark:text-emerald-300";
    case "Good":
      return "text-yellow-700 dark:text-yellow-300";
    case "Decent":
      return "text-orange-700 dark:text-orange-300";
    case "Poor":
      return "text-red-700 dark:text-red-300";
    default:
      return "text-slate-700 dark:text-slate-300";
  }
}

/** Tailwind classes for a category badge — background, text and border. */
export function getCategoryBgColor(category: SunsetCategory): string {
  switch (category) {
    case "Epic":
      return "bg-violet-50 text-violet-700 border-violet-200 dark:bg-violet-500/15 dark:text-violet-300 dark:border-violet-500/30";
    case "Great":
      return "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/15 dark:text-emerald-300 dark:border-emerald-500/30";
    case "Good":
      return "bg-yellow-50 text-yellow-700 border-yellow-200 dark:bg-yellow-500/15 dark:text-yellow-300 dark:border-yellow-500/30";
    case "Decent":
      return "bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-500/15 dark:text-orange-300 dark:border-orange-500/30";
    case "Poor":
      return "bg-red-50 text-red-700 border-red-200 dark:bg-red-500/15 dark:text-red-300 dark:border-red-500/30";
    default:
      return "bg-slate-100 text-slate-700 border-slate-200 dark:bg-slate-500/15 dark:text-slate-300 dark:border-slate-500/30";
  }
}

// ---------------------------------------------------------------------------
// Score colours
// ---------------------------------------------------------------------------

/** Light-mode score hexes — all pass AA for normal text on white. */
const SCORE_HEX_LIGHT: Record<SunsetCategory, string> = {
  Epic: "#6d28d9",
  Great: "#047857",
  Good: "#a16207",
  Decent: "#c2410c",
  Poor: "#b91c1c",
};

/** Dark-mode score hexes — all pass AA on the slate-950/900 surfaces. */
const SCORE_HEX_DARK: Record<SunsetCategory, string> = {
  Epic: "#a855f7",
  Great: "#34d399",
  Good: "#fbbf24",
  Decent: "#fb923c",
  Poor: "#f87171",
};

/**
 * Hex colour for a numeric score, for SVG and inline styles that cannot use
 * Tailwind's `dark:` variant. Pass the resolved theme so the value stays
 * legible on whichever surface it lands.
 */
export function getScoreHexColor(score: number, isDark = false): string {
  const table = isDark ? SCORE_HEX_DARK : SCORE_HEX_LIGHT;
  return table[scoreCategory(score)];
}

/** Tailwind text classes for a numeric score (light + dark). */
export function getScoreTextColor(score: number): string {
  return getCategoryColor(scoreCategory(score));
}

/**
 * Colour for a component sub-score (cloud quality, atmosphere, …).
 *
 * Same bands as the overall score — that consistency is the whole point — but
 * the top band renders emerald rather than violet. Violet reads as "an Epic
 * sunset", and a 95 for Atmosphere is a healthy input, not a verdict.
 */
export function getComponentHexColor(score: number, isDark = false): string {
  const category = scoreCategory(score);
  const band = category === "Epic" ? "Great" : category;
  return (isDark ? SCORE_HEX_DARK : SCORE_HEX_LIGHT)[band];
}

// ---------------------------------------------------------------------------
// Weather phrasing
// ---------------------------------------------------------------------------

/**
 * Aerosol optical depth is a research quantity, not something to show a person
 * shopping for a sunset. Collapse it to a word.
 */
export function hazeLabel(aod: number | null): string | null {
  if (aod === null || Number.isNaN(aod)) return null;
  if (aod < 0.1) return "Very low";
  if (aod < 0.2) return "Low";
  if (aod < 0.35) return "Moderate";
  if (aod < 0.5) return "High";
  return "Very high";
}

/**
 * Whether an explanation reads as helping or holding the score back.
 *
 * The backend sends reasons as plain strings with no polarity, so this is a
 * word-list heuristic. It only drives colour and grouping — a miss is cosmetic.
 */
const NEGATIVE_REASON_WORDS = [
  "block", "reduce", "mute", "poor", "rain", "haze", "heavy", "overcast",
  "obstruction", "dampen", "wash", "diffuse", "milky", "clip", "limited",
  "bad", "not ideal", "worst", "less", "few", "little",
];

export function isPositiveReason(reason: string): boolean {
  const lower = reason.toLowerCase();
  return !NEGATIVE_REASON_WORDS.some((w) => lower.includes(w));
}

// ---------------------------------------------------------------------------
// Date / time formatting
// ---------------------------------------------------------------------------

/** Format an ISO datetime string as "HH:MM" in local browser time. */
export function formatTime(isoString: string): string {
  try {
    const dt = new Date(isoString);
    return dt.toLocaleTimeString("en-GB", {
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
    return dt.toLocaleDateString("en-US", {
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
    return dt.toLocaleDateString("en-US", {
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

/**
 * Human phrasing for how long until a moment — "in 2h 14m", "in 6 min",
 * "just now", or null once it is more than a day away or already past.
 */
export function countdownTo(isoString: string, now: Date = new Date()): string | null {
  const target = new Date(isoString).getTime();
  if (Number.isNaN(target)) return null;

  const diffMs = target - now.getTime();
  if (diffMs < 0) return null;

  const mins = Math.round(diffMs / 60_000);
  if (mins < 1) return "any moment";
  if (mins < 60) return `in ${mins} min`;

  const hours = Math.floor(mins / 60);
  if (hours >= 24) return null;

  const rem = mins % 60;
  return rem === 0 ? `in ${hours}h` : `in ${hours}h ${rem}m`;
}

/** Relative phrasing for when a cached reading was taken. */
export function freshnessLabel(isoString: string, now: Date = new Date()): string {
  const then = new Date(isoString).getTime();
  if (Number.isNaN(then)) return "";

  const mins = Math.round((now.getTime() - then) / 60_000);
  if (mins < 2) return "just now";
  if (mins < 60) return `${mins} min ago`;

  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return formatDateShort(isoString);
}

// ---------------------------------------------------------------------------
// Score helpers
// ---------------------------------------------------------------------------

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
