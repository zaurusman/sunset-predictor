"use client";

import type { PredictResponse } from "@/lib/types";
import {
  countdownTo,
  formatTime,
  getCategoryBgColor,
  getScoreHexColor,
  isToday,
} from "@/lib/utils";
import { useIsDark } from "@/lib/useIsDark";

interface VerdictCardProps {
  prediction: PredictResponse;
  /** The date this reading is for, "YYYY-MM-DD". */
  targetDate: string;
}

/**
 * The answer in words, which is what people open the app for.
 * `go_outside_recommendation` has always been in the response and was never read.
 */
function headlineFor(prediction: PredictResponse, targetDate: string): string {
  const go = prediction.go_outside_recommendation;
  const today = new Date().toISOString().slice(0, 10);

  if (targetDate > today) return go ? "Looking promising" : "Nothing special yet";
  if (targetDate < today) return `A ${prediction.category.toLowerCase()} one`;
  if (go) return "Worth heading out";

  // The go-outside bar (70) sits above the Great band (65), so a 65–69 evening
  // is genuinely nice without being worth changing plans for. A flat "Not
  // tonight" here would contradict the green Great badge beside it.
  if (prediction.beauty_score_0_100 >= 50) return "Worth a glance";
  return "Not tonight";
}

/** Phrase the percentile as a comparison, which is what the number actually means. */
function rankPhrase(percentile: number): string {
  const pct = Math.round(percentile * 100);
  if (pct >= 97) return "top 3% of evenings here";
  if (pct >= 88) return `better than ${pct}% of evenings here`;
  if (pct <= 10) return `bottom ${Math.max(pct, 1)}% of evenings here`;
  return `better than ${pct}% of evenings here`;
}

const RADIUS = 27;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function VerdictCard({ prediction, targetDate }: VerdictCardProps) {
  const isDark = useIsDark();

  const score = Math.round(prediction.beauty_score_0_100);
  // The score is a rank against this location's own history, so say so — "62"
  // means little on its own, "better than 78% of evenings here" is the actual claim.
  const rank =
    prediction.climatology_is_local && prediction.climatology_percentile !== null
      ? prediction.climatology_percentile
      : null;
  const colour = getScoreHexColor(score, isDark);
  const headline = headlineFor(prediction, targetDate);
  const countdown = isToday(targetDate) ? countdownTo(prediction.sunset_time) : null;
  const why = prediction.reasons[0];

  return (
    <section className="bg-white dark:bg-slate-900/60 rounded-2xl border border-gray-200 dark:border-slate-700/40 p-5 flex flex-col gap-4">
      <div className="flex items-start gap-4">
        <div className="flex-1 flex flex-col gap-1.5 min-w-0">
          <span className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold">
            {isToday(targetDate) ? "Tonight" : targetDate}
          </span>
          <h1 className="text-[27px] leading-tight font-bold tracking-tight text-gray-900 dark:text-white text-pretty">
            {headline}
          </h1>
        </div>

        <div className="relative w-[62px] h-[62px] flex-shrink-0">
          <svg width="62" height="62" viewBox="0 0 62 62" aria-hidden="true">
            <circle
              cx="31"
              cy="31"
              r={RADIUS}
              fill="none"
              strokeWidth="6"
              className="stroke-gray-200 dark:stroke-slate-700"
            />
            <circle
              cx="31"
              cy="31"
              r={RADIUS}
              fill="none"
              stroke={colour}
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={`${(CIRCUMFERENCE * score) / 100} ${CIRCUMFERENCE}`}
              transform="rotate(-90 31 31)"
              style={{ transition: "stroke-dasharray 0.6s ease" }}
            />
          </svg>
          <div
            className="absolute inset-0 flex items-center justify-center text-[21px] font-bold tabular-nums tracking-tight"
            style={{ color: colour }}
          >
            {score}
          </div>
          <span className="sr-only">{score} out of 100</span>
        </div>
      </div>

      {/* Category and rank share a row; the reason gets its own full-width line.
          Keeping all three in one flex row starves the reason of horizontal
          space and wraps it to one word per line at 375px. */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className={`px-2.5 py-0.5 rounded-full border text-xs font-semibold ${getCategoryBgColor(prediction.category)}`}
          >
            {prediction.category}
          </span>
          {rank !== null && (
            <span className="text-xs text-gray-600 dark:text-slate-400">
              {rankPhrase(rank)}
            </span>
          )}
        </div>
        {why && (
          <p className="text-sm text-gray-700 dark:text-slate-300 leading-snug text-pretty">
            {why}
          </p>
        )}
      </div>

      {!prediction.climatology_is_local && (
        <p className="text-xs text-gray-500 dark:text-slate-400 leading-snug text-pretty">
          Still learning this location&apos;s normal — the score is ranked against a
          general baseline for now and will sharpen shortly.
        </p>
      )}

      <div className="h-px bg-gray-200 dark:bg-slate-700/60" />

      <div className="flex items-baseline gap-2">
        <span className="text-lg font-semibold tabular-nums tracking-tight text-gray-900 dark:text-white">
          Sunset {formatTime(prediction.sunset_time)}
        </span>
        {countdown && (
          <span className="text-sm text-gray-600 dark:text-slate-400">{countdown}</span>
        )}
      </div>
    </section>
  );
}
