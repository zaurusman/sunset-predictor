"use client";

import { useEffect, useId, useRef, useState } from "react";
import type { PredictResponse } from "@/lib/types";
import {
  countdownTo,
  formatTime,
  getCategoryBgColor,
  getScoreHexColor,
  isToday,
} from "@/lib/utils";
import { useIsDark } from "@/lib/useIsDark";
import {
  markSunsetPlayed,
  shouldPlaySunset,
  useCountUp,
  usePrefersReducedMotion,
} from "@/lib/motion";
import { SCORE_START_MS, useSky } from "./sky/SkyProvider";

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

  // The go-outside bar (75) sits above the Great band (72), so a 72–74 evening
  // is genuinely nice without being worth changing plans for. A flat "Not
  // tonight" here would contradict the green Great badge beside it.
  if (prediction.beauty_score_0_100 >= 50) return "Worth a glance";
  return "Not tonight";
}

/** Phrase the percentile as a comparison, which is what the number actually means. */
/**
 * Context line under the score.
 *
 * The score itself is absolute — "how good will the sky look". This says how
 * unusual that is HERE, and specifically here at this time of year: the rank
 * is taken against a seasonal window, so a pleasant August evening reads as
 * good for August rather than being buried under the winter's frontal skies.
 */
function rankPhrase(percentile: number, month: string): string {
  const pct = Math.round(percentile * 100);
  if (pct >= 97) return `among the best ${month} evenings here`;
  if (pct <= 10) return `quiet for ${month} here`;
  return `better than ${pct}% of ${month} evenings here`;
}

const RADIUS = 27;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function VerdictCard({ prediction, targetDate }: VerdictCardProps) {
  const isDark = useIsDark();
  const reduce = usePrefersReducedMotion();
  const { setMood, playSunset } = useSky();
  const ref = useRef<HTMLElement>(null);
  const ringId = useId();

  // Decided once, at mount: the sun sets on the first open of each day only.
  // This card renders only after the page has read localStorage, so reading it
  // here never differs from a server render.
  const [sunset] = useState(
    () => !reduce && isToday(targetDate) && shouldPlaySunset(targetDate),
  );
  const sunsetStarted = useRef(false);

  useEffect(() => {
    if (sunset && !sunsetStarted.current) {
      sunsetStarted.current = true;
      markSunsetPlayed(targetDate);
      playSunset(ref.current!, prediction.category);
    } else {
      setMood(prediction.category);
    }
  }, [sunset, prediction.category, targetDate, playSunset, setMood]);

  const monthName = new Date(prediction.sunset_time).toLocaleDateString(undefined, {
    month: "long",
  });

  const score = Math.round(prediction.beauty_score_0_100);
  // The score is absolute; the rank is context. It is deliberately secondary —
  // a rank cannot improve when the model improves, which is why it is no
  // longer the headline number.
  const rank =
    prediction.climatology_is_local && prediction.climatology_percentile !== null
      ? prediction.climatology_percentile
      : null;
  const colour = getScoreHexColor(score, isDark);
  const headline = headlineFor(prediction, targetDate);
  const countdown = isToday(targetDate) ? countdownTo(prediction.sunset_time) : null;
  const why = prediction.reasons[0];

  // The number counts up as the ring fills; after the sunset it waits for the
  // sun to touch the horizon. Later changes glide from the value on screen.
  const introDelay = sunset ? SCORE_START_MS : 0;
  const shown = useCountUp(score, {
    durationMs: sunset ? 1500 : 1000,
    delayMs: introDelay,
    enabled: !reduce,
  });
  const tipAngle = ((-90 + 3.6 * shown) * Math.PI) / 180;

  return (
    <section
      ref={ref}
      className="m-rise bg-white dark:bg-slate-900/60 rounded-2xl border border-gray-200 dark:border-slate-700/40 p-5 flex flex-col gap-4">
      <div className="flex items-start gap-4">
        <div className="flex-1 flex flex-col gap-1.5 min-w-0">
          <span className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold">
            {isToday(targetDate) ? "Tonight" : targetDate}
          </span>
          {/* Keyed so a new answer rises in word by word. */}
          <h1
            key={headline}
            className="text-[27px] leading-tight font-bold tracking-tight text-gray-900 dark:text-white text-pretty"
          >
            {headline.split(" ").map((word, i) => (
              <span key={i}>
                {i > 0 && " "}
                <span className="m-word">
                  <span style={{ animationDelay: `${i * 70}ms` }}>{word}</span>
                </span>
              </span>
            ))}
          </h1>
        </div>

        <div className="relative w-[62px] h-[62px] flex-shrink-0">
          <svg width="62" height="62" viewBox="0 0 62 62" aria-hidden="true" className="overflow-visible">
            <defs>
              {/* From the band's deeper shade to its brighter one, along the arc. */}
              <linearGradient id={ringId} x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor={getScoreHexColor(score, !isDark)} />
                <stop offset="100%" stopColor={colour} />
              </linearGradient>
            </defs>
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
              stroke={`url(#${ringId})`}
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={`${(CIRCUMFERENCE * shown) / 100} ${CIRCUMFERENCE}`}
              transform="rotate(-90 31 31)"
              // A zero-length dash still draws its round cap as a dot.
              opacity={shown < 1 ? 0 : 1}
            />
            {/* The travelling tip that leads the fill. */}
            <circle
              cx={31 + RADIUS * Math.cos(tipAngle)}
              cy={31 + RADIUS * Math.sin(tipAngle)}
              r="4.5"
              fill="#fff"
              stroke={colour}
              strokeWidth="2.5"
              style={{ filter: `drop-shadow(0 0 4px ${colour})`, opacity: shown < 1 ? 0 : 1 }}
            />
          </svg>
          <div
            className="absolute inset-0 flex items-center justify-center text-[21px] font-bold tabular-nums tracking-tight"
            style={{ color: colour }}
            aria-hidden="true"
          >
            {Math.round(shown)}
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
              {rankPhrase(rank, monthName)}
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
          Still learning what is normal here — the comparison below the score uses a
          general baseline for now. The score itself is unaffected.
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
