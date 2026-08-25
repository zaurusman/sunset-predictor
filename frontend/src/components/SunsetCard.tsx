"use client";

import { useState } from "react";
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
import ComponentBreakdown from "./ComponentBreakdown";
import ReasonsList from "./ReasonsList";
import ViewingCurve from "./ViewingCurve";

interface SunsetCardProps {
  day: DayForecast;
  defaultExpanded?: boolean;
}

export default function SunsetCard({ day, defaultExpanded = false }: SunsetCardProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const isDark = useIsDark();

  const score = Math.round(day.beauty_score_0_100);
  const scoreColor = getScoreHexColor(score, isDark);
  const today = isToday(day.date);

  return (
    <div
      className={`rounded-2xl border transition-all duration-200 overflow-hidden ${
        today
          ? "border-orange-500/50 bg-white dark:bg-slate-900/90"
          : "border-gray-200 dark:border-slate-700/50 bg-white dark:bg-slate-900/60"
      }`}
    >
      <button
        className="w-full flex items-center gap-4 px-5 py-4 text-left"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        <div
          className="w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg tabular-nums flex-shrink-0"
          style={{
            background: `${scoreColor}1a`,
            border: `2px solid ${scoreColor}59`,
            color: scoreColor,
          }}
        >
          {score}
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
          className={`text-gray-500 dark:text-slate-400 transition-transform ${expanded ? "rotate-180" : ""}`}
        />
      </button>

      {expanded && (
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
  );
}
