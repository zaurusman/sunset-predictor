"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { DayForecast } from "@/lib/types";
import {
  formatDateShort,
  formatTime,
  getCategoryBgColor,
  getScoreHexColor,
  isToday,
} from "@/lib/utils";
import ComponentBreakdown from "./ComponentBreakdown";
import ReasonsList from "./ReasonsList";

interface SunsetCardProps {
  day: DayForecast;
  defaultExpanded?: boolean;
}

export default function SunsetCard({ day, defaultExpanded = false }: SunsetCardProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const scoreColor = getScoreHexColor(day.beauty_score_0_100);
  const today = isToday(day.date);

  return (
    <div
      className={`rounded-2xl border transition-all duration-200 overflow-hidden ${
        today
          ? "border-orange-500/40 bg-slate-900/90"
          : "border-slate-700/50 bg-slate-900/60"
      }`}
    >
      {/* Card header */}
      <button
        className="w-full flex items-center gap-4 px-5 py-4 text-left"
        onClick={() => setExpanded((e) => !e)}
      >
        {/* Score circle */}
        <div
          className="w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg flex-shrink-0"
          style={{
            background: `${scoreColor}22`,
            border: `2px solid ${scoreColor}55`,
            color: scoreColor,
          }}
        >
          {Math.round(day.beauty_score_0_100)}
        </div>

        {/* Date + category */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-white font-semibold">
              {today ? "Today" : formatDateShort(day.date)}
            </span>
            <span
              className={`text-xs px-2 py-0.5 rounded-full border font-medium ${getCategoryBgColor(day.category)}`}
            >
              {day.category}
            </span>
          </div>
          <div className="text-slate-500 text-sm mt-0.5">
            Sunset {formatTime(day.sunset_time)} · {Math.round(day.confidence_0_100)}% confidence
          </div>
        </div>

        {/* Expand toggle */}
        <div className="text-slate-600">
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-5 pb-5 space-y-4 border-t border-slate-700/40 pt-4">
          {/* Reasons */}
          <div>
            <h4 className="text-slate-400 text-xs uppercase tracking-wider mb-2">Why</h4>
            <ReasonsList reasons={day.reasons} />
          </div>

          {/* Component breakdown */}
          <div>
            <h4 className="text-slate-400 text-xs uppercase tracking-wider mb-2">Score Breakdown</h4>
            <ComponentBreakdown breakdown={day.physics_component_breakdown} />
          </div>
        </div>
      )}
    </div>
  );
}
