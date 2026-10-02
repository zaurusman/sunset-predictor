"use client";

import { ChevronDown } from "lucide-react";
import type { DayForecast } from "@/lib/types";
import {
  formatDateShort,
  formatTime,
  getCategoryBgColor,
  getScoreHexColor,
  isToday,
} from "@/lib/utils";
import { useIsDark } from "@/lib/useIsDark";
import { usePresence } from "@/lib/motion";
import ComponentBreakdown from "./ComponentBreakdown";
import ReasonsList from "./ReasonsList";
import ViewingCurve from "./ViewingCurve";

interface SunsetCardProps {
  day: DayForecast;
  /** Controlled by the page, so the chart can open a day and close the rest. */
  expanded: boolean;
  onToggle: () => void;
}

const MINI_R = 19;
const MINI_C = 2 * Math.PI * MINI_R;

export default function SunsetCard({ day, expanded, onToggle }: SunsetCardProps) {
  const isDark = useIsDark();
  // Stays mounted while the card folds shut, so closing animates too.
  const panel = usePresence(expanded, 450);

  const score = Math.round(day.beauty_score_0_100);
  const scoreColor = getScoreHexColor(score, isDark);
  const today = isToday(day.date);

  return (
    <div
      id={`day-${day.date}`}
      className={`rounded-2xl border transition-shadow duration-300 overflow-hidden ${
        today
          ? "border-orange-500/50 bg-white dark:bg-slate-900/90"
          : "border-gray-200 dark:border-slate-700/50 bg-white dark:bg-slate-900/60"
      } ${expanded ? "shadow-[0_8px_24px_-14px_rgba(15,23,42,.25)]" : ""}`}
    >
      <button
        className="w-full flex items-center gap-4 px-5 py-4 text-left"
        onClick={onToggle}
        aria-expanded={expanded}
      >
        {/* A mini version of Tonight's ring. */}
        <div className="relative w-[46px] h-[46px] flex-shrink-0">
          <svg width="46" height="46" viewBox="0 0 46 46" aria-hidden="true">
            <circle
              cx="23"
              cy="23"
              r={MINI_R}
              fill="none"
              strokeWidth="4"
              className="stroke-gray-200 dark:stroke-slate-700"
            />
            <circle
              cx="23"
              cy="23"
              r={MINI_R}
              fill="none"
              stroke={scoreColor}
              strokeWidth="4"
              strokeLinecap="round"
              strokeDasharray={`${(MINI_C * score) / 100} ${MINI_C}`}
              transform="rotate(-90 23 23)"
              opacity={score < 1 ? 0 : 1}
            />
          </svg>
          <span
            className="absolute inset-0 flex items-center justify-center text-sm font-bold tabular-nums"
            style={{ color: scoreColor }}
          >
            {score}
          </span>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-gray-900 dark:text-white font-semibold">
              {today ? "Today" : formatDateShort(day.date)}
            </span>
            <span
              className={`text-xs px-2 py-0.5 rounded-full border font-semibold ${getCategoryBgColor(day.category)}`}
            >
              {day.category}
            </span>
          </div>
          <div className="text-gray-600 dark:text-slate-400 text-sm mt-0.5 tabular-nums">
            Sunset {formatTime(day.sunset_time)}
          </div>
        </div>

        <ChevronDown
          size={16}
          className={`text-gray-500 dark:text-slate-400 transition-transform duration-[350ms] ${expanded ? "rotate-180" : ""}`}
        />
      </button>

      <div className="m-collapse" data-open={expanded}>
        <div>
          {panel.mounted && (
            <div className="px-5 pb-5 flex flex-col gap-4 border-t border-gray-200 dark:border-slate-700/40 pt-4">
              <ViewingCurve
                windowScores={day.window_scores}
                bestPoint={day.best_window_point}
                sunsetTime={day.sunset_time}
                dominantPathway={day.physics_component_breakdown.dominant_pathway}
              />

              <div>
                <h4 className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold mb-2">
                  Why
                </h4>
                <ReasonsList reasons={day.reasons} />
              </div>

              <div>
                <h4 className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold mb-3">
                  Breakdown
                </h4>
                <ComponentBreakdown breakdown={day.physics_component_breakdown} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
