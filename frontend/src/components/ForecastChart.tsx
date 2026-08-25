"use client";

import { useEffect, useRef, useState } from "react";
import type { DayForecast } from "@/lib/types";
import { formatDateShort, formatTime, getScoreHexColor } from "@/lib/utils";
import { useIsDark } from "@/lib/useIsDark";

interface ForecastChartProps {
  days: DayForecast[];
  onDayClick?: (day: DayForecast) => void;
  selectedDate?: string;
}

// The SVG user-space is 1:1 with CSS pixels, so type stays the same size at
// every container width. Width is measured; height is fixed.
const DEFAULT_W = 320; // used for SSR + the first client render
const VB_H = 200;
const PAD_TOP = 18; // room for the value label above each bar
const PAD_BOTTOM = 20; // room for the two-line date label
const PAD_LEFT = 26; // room for the y-axis labels
const PAD_RIGHT = 6;

const PLOT_X = PAD_LEFT;
const PLOT_Y = PAD_TOP;
const PLOT_H = VB_H - PAD_TOP - PAD_BOTTOM;
const BASELINE = PLOT_Y + PLOT_H;

const Y_TICKS = [0, 25, 50, 75, 100];
const CORNER = 4;

/** Bar with rounded top corners only (matches the old recharts radius=[4,4,0,0]). */
function barPath(x: number, y: number, w: number, h: number): string {
  const r = Math.min(CORNER, w / 2, h);
  return [
    `M${x},${y + h}`,
    `V${y + r}`,
    `A${r},${r} 0 0 1 ${x + r},${y}`,
    `H${x + w - r}`,
    `A${r},${r} 0 0 1 ${x + w},${y + r}`,
    `V${y + h}`,
    "Z",
  ].join(" ");
}

/** "Tue" / "25" — split over two lines so 7 labels fit at mobile widths. */
function axisLabel(isoDate: string): { weekday: string; day: string } {
  try {
    const dt = new Date(isoDate);
    return {
      weekday: dt.toLocaleDateString("en-US", { weekday: "short" }),
      day: dt.toLocaleDateString("en-US", { day: "numeric" }),
    };
  } catch {
    return { weekday: isoDate, day: "" };
  }
}

export default function ForecastChart({
  days,
  onDayClick,
  selectedDate,
}: ForecastChartProps) {
  const isDark = useIsDark();
  const [hoveredDate, setHoveredDate] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(DEFAULT_W);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      const w = entry.contentRect.width;
      if (w > 0) setWidth(w);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const plotW = Math.max(1, width - PAD_LEFT - PAD_RIGHT);

  const gridColor = isDark ? "#1e293b" : "#e2e8f0";
  // Both axes carry 9-10px text, so both need 4.5:1. The previous pair leaned
  // one step too light on each side — #94a3b8 is 2.65:1 on white, and #64748b
  // is 4.0:1 on slate-950. This pair clears the bar in both themes.
  const xAxisColor = isDark ? "#94a3b8" : "#64748b";
  const yAxisColor = xAxisColor;
  const hoverColor = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)";

  const slotW = plotW / days.length;
  const barW = Math.min(40, slotW * 0.55);
  const interactive = Boolean(onDayClick);

  return (
    <div ref={containerRef} className="w-full">
      {days.length > 0 && (
        <svg
          viewBox={`0 0 ${width} ${VB_H}`}
          width="100%"
          role="img"
          aria-label="Sunset beauty score for each of the next 7 days"
          className="block h-auto select-none overflow-visible"
        >
          {/* Horizontal grid + y-axis labels */}
          {Y_TICKS.map((tick) => {
            const y = BASELINE - (tick / 100) * PLOT_H;
            return (
              <g key={tick}>
                <line
                  x1={PLOT_X}
                  x2={PLOT_X + plotW}
                  y1={y}
                  y2={y}
                  stroke={gridColor}
                  strokeWidth={1}
                  strokeDasharray="3 3"
                />
                <text
                  x={PLOT_X - 6}
                  y={y}
                  fill={yAxisColor}
                  fontSize={10}
                  textAnchor="end"
                  dominantBaseline="middle"
                >
                  {tick}
                </text>
              </g>
            );
          })}

          {days.map((day, i) => {
            const score = day.beauty_score_0_100;
            const rounded = Math.round(score);
            const slotX = PLOT_X + i * slotW;
            const barX = slotX + (slotW - barW) / 2;
            const barH = Math.max(
              2,
              (Math.max(0, Math.min(100, score)) / 100) * PLOT_H,
            );
            const barY = BASELINE - barH;
            const dimmed = Boolean(selectedDate) && day.date !== selectedDate;
            const { weekday, day: dayNum } = axisLabel(day.date);

            return (
              <g
                key={day.date}
                role={interactive ? "button" : undefined}
                tabIndex={interactive ? 0 : undefined}
                aria-label={`${formatDateShort(day.date)}: score ${rounded} out of 100, ${day.category}, sunset ${formatTime(day.sunset_time)}`}
                aria-pressed={
                  interactive ? day.date === selectedDate : undefined
                }
                onClick={() => onDayClick?.(day)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onDayClick?.(day);
                  }
                }}
                onMouseEnter={() => setHoveredDate(day.date)}
                onMouseLeave={() => setHoveredDate(null)}
                onFocus={() => setHoveredDate(day.date)}
                onBlur={() => setHoveredDate(null)}
                style={{ cursor: interactive ? "pointer" : "default" }}
                className="focus:outline-none [&:focus-visible>rect]:stroke-orange-500"
              >
                <title>
                  {`${formatDateShort(day.date)} — ${rounded}/100 · ${day.category} · sunset ${formatTime(day.sunset_time)}`}
                </title>

                {/* Hit area + hover/focus highlight */}
                <rect
                  x={slotX}
                  y={PLOT_Y}
                  width={slotW}
                  height={PLOT_H + PAD_BOTTOM}
                  rx={4}
                  fill={hoveredDate === day.date ? hoverColor : "transparent"}
                  stroke="transparent"
                  strokeWidth={1}
                />

                <path
                  d={barPath(barX, barY, barW, barH)}
                  fill={getScoreHexColor(score, isDark)}
                  opacity={dimmed ? 0.45 : 1}
                />

                {/* Score above the bar */}
                <text
                  x={slotX + slotW / 2}
                  y={barY - 5}
                  fill={xAxisColor}
                  fontSize={10}
                  fontWeight={600}
                  textAnchor="middle"
                  opacity={dimmed ? 0.7 : 1}
                >
                  {rounded}
                </text>

                {/* Two-line date label below the baseline */}
                <text
                  x={slotX + slotW / 2}
                  y={BASELINE + 9}
                  fill={xAxisColor}
                  fontSize={9}
                  textAnchor="middle"
                  opacity={dimmed ? 0.7 : 1}
                >
                  {weekday}
                </text>
                <text
                  x={slotX + slotW / 2}
                  y={BASELINE + 18}
                  fill={yAxisColor}
                  fontSize={9}
                  textAnchor="middle"
                  opacity={dimmed ? 0.7 : 1}
                >
                  {dayNum}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
