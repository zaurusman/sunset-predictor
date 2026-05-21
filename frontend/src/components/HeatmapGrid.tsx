"use client";

import { useState } from "react";
import type { HeatmapDay } from "@/lib/types";
import { getCategoryColor } from "@/lib/utils";

interface DayCell extends HeatmapDay {
  level: number;
}

interface Week {
  key: string;
  cells: (DayCell | null)[];
  monthLabel?: string;
}

// Full class strings so Tailwind JIT can detect them.
// Colors match getScoreHexColor / getCategoryBgColor in lib/utils.ts exactly.
const LEVEL_CLASSES: string[] = [
  "bg-gray-200 dark:bg-slate-700",  // 0: empty cell
  "bg-red-400",                      // 1: poor    (0–29)
  "bg-orange-400",                   // 2: decent  (30–49)
  "bg-amber-400",                    // 3: good    (50–64)
  "bg-emerald-400",                  // 4: great   (65–79)
  "bg-purple-500",                   // 5: epic    (80+)
];

const LEVEL_LABELS = ["No data", "Poor (0–29)", "Decent (30–49)", "Good (50–64)", "Great (65–79)", "Epic (80+)"];
const DAY_LABELS = ["Mon", "", "Wed", "", "Fri", "", "Sun"];

function scoreToLevel(score: number): number {
  if (score >= 80) return 5;
  if (score >= 65) return 4;
  if (score >= 50) return 3;
  if (score >= 30) return 2;
  return 1;
}

function buildWeeks(days: HeatmapDay[]): Week[] {
  if (days.length === 0) return [];

  const sorted = [...days].sort((a, b) => a.date.localeCompare(b.date));
  const byDate: Record<string, DayCell> = {};
  for (const d of sorted) {
    byDate[d.date] = { ...d, level: scoreToLevel(d.score) };
  }

  // Find the Monday of the week containing the first day
  const firstDate = new Date(sorted[0].date + "T12:00:00");
  const dow = (firstDate.getDay() + 6) % 7; // Mon=0 … Sun=6
  const cursor = new Date(firstDate);
  cursor.setDate(cursor.getDate() - dow);

  const lastDate = new Date(sorted[sorted.length - 1].date + "T12:00:00");

  const weeks: Week[] = [];
  let prevMonth = -1;

  while (cursor <= lastDate) {
    // Record the Monday of this week before the inner loop advances cursor
    const weekMonday = new Date(cursor);

    const cells: (DayCell | null)[] = [];
    for (let i = 0; i < 7; i++) {
      const dateStr = cursor.toISOString().slice(0, 10);
      cells.push(byDate[dateStr] ?? null);
      cursor.setDate(cursor.getDate() + 1);
    }

    const thisMonth = weekMonday.getMonth();
    let monthLabel: string | undefined;
    if (thisMonth !== prevMonth) {
      monthLabel = weekMonday.toLocaleDateString("en-US", { month: "short" });
      prevMonth = thisMonth;
    }

    weeks.push({ key: `w${weeks.length}`, cells, monthLabel });
  }

  return weeks;
}

export default function HeatmapGrid({ days }: { days: HeatmapDay[] }) {
  const [hovered, setHovered] = useState<DayCell | null>(null);
  const weeks = buildWeeks(days);

  return (
    <div>
      {/* Hover info bar */}
      <div className="h-6 mb-3 text-sm">
        {hovered ? (
          <span className="text-gray-700 dark:text-slate-300">
            {new Date(hovered.date + "T12:00:00").toLocaleDateString("en-US", {
              weekday: "short",
              year: "numeric",
              month: "short",
              day: "numeric",
            })}
            {" — "}
            <strong>{Math.round(hovered.score)}</strong>
            <span className="text-gray-500 dark:text-slate-400">/100 · </span>
            <span className={getCategoryColor(hovered.category)}>{hovered.category}</span>
          </span>
        ) : (
          <span className="text-gray-400 dark:text-slate-500 text-xs">
            Hover a cell to see score
          </span>
        )}
      </div>

      <div className="overflow-x-auto pb-2">
        <div className="flex items-start gap-[3px] min-w-max">
          {/* Day-of-week labels */}
          <div className="flex flex-col gap-[3px] mt-[20px] mr-1">
            {DAY_LABELS.map((label, i) => (
              <div
                key={i}
                className="w-3 h-3 flex items-center justify-end text-[9px] text-gray-400 dark:text-slate-500 leading-none"
              >
                {label}
              </div>
            ))}
          </div>

          {/* Week columns */}
          {weeks.map((week) => (
            <div key={week.key} className="flex flex-col gap-[3px]">
              {/* Month label row */}
              <div className="h-[16px] text-[10px] text-gray-500 dark:text-slate-400 leading-none whitespace-nowrap">
                {week.monthLabel ?? ""}
              </div>

              {/* Day cells */}
              {week.cells.map((cell, i) => (
                <div
                  key={i}
                  className={`w-3 h-3 rounded-sm transition-opacity ${
                    cell
                      ? `${LEVEL_CLASSES[cell.level]} cursor-pointer hover:opacity-70`
                      : "bg-transparent"
                  }`}
                  onMouseEnter={() => cell && setHovered(cell)}
                  onMouseLeave={() => setHovered(null)}
                />
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-1.5 mt-3 flex-wrap">
        <span className="text-xs text-gray-500 dark:text-slate-400">Less</span>
        {LEVEL_CLASSES.map((cls, i) => (
          <div
            key={i}
            title={LEVEL_LABELS[i]}
            className={`w-3 h-3 rounded-sm ${cls} border border-gray-200/50 dark:border-slate-600/30`}
          />
        ))}
        <span className="text-xs text-gray-500 dark:text-slate-400">More</span>
      </div>
    </div>
  );
}
