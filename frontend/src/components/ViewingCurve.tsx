"use client";

import { useState } from "react";
import { WINDOW_POINTS } from "@/lib/types";
import { formatTime, getScoreHexColor } from "@/lib/utils";
import { useIsDark } from "@/lib/useIsDark";

interface ViewingCurveProps {
  /** Physics score at each sampled moment, keyed by window point. */
  windowScores: Record<string, number>;
  /** The window point that scored highest. */
  bestPoint: string;
  /** ISO datetime of sunset — the anchor the offsets are measured from. */
  sunsetTime: string;
}

/** Minutes each window point sits from sunset. */
const OFFSET_MINUTES: Record<string, number> = {
  "-15m": -15,
  sunset: 0,
  "+15m": 15,
  "+30m": 30,
};

const VIEW_W = 300;
const VIEW_H = 100;

/** Cardinal-spline path through the points, so the curve reads as a trend. */
function smoothPath(pts: { x: number; y: number }[]): string {
  if (pts.length < 2) return "";

  let d = `M${pts[0].x},${pts[0].y}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] ?? pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] ?? p2;
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C${cp1x},${cp1y} ${cp2x},${cp2y} ${p2.x},${p2.y}`;
  }
  return d;
}

/**
 * The four moments the backend actually scores, drawn as a curve.
 *
 * The API has always returned a score per window point and named the best one;
 * the old timeline ignored both and drew a marker fixed at the midpoint.
 */
export default function ViewingCurve({
  windowScores,
  bestPoint,
  sunsetTime,
}: ViewingCurveProps) {
  const isDark = useIsDark();

  const available = WINDOW_POINTS.filter((p) => typeof windowScores[p] === "number");
  const peak = available.includes(bestPoint as (typeof WINDOW_POINTS)[number])
    ? bestPoint
    : available[0];

  const [selected, setSelected] = useState<string | null>(null);
  const active = selected ?? peak;

  // Older responses (or a weather override) can arrive without window scores;
  // there is nothing honest to draw from one point.
  if (available.length < 2) return null;

  const sunsetMs = new Date(sunsetTime).getTime();
  const clockFor = (point: string) =>
    Number.isFinite(sunsetMs)
      ? formatTime(new Date(sunsetMs + (OFFSET_MINUTES[point] ?? 0) * 60_000).toISOString())
      : "--:--";

  const step = VIEW_W / (available.length - 1);
  const pts = available.map((point, i) => ({
    point,
    score: windowScores[point],
    x: i * step,
    y: VIEW_H - 12 - (windowScores[point] / 100) * (VIEW_H - 26),
  }));

  const line = smoothPath(pts);
  const area = `${line} L${VIEW_W},${VIEW_H} L0,${VIEW_H} Z`;
  // orange-700 rather than -600: the peak label is 12px, so it needs 4.5:1.
  const accent = isDark ? "#fb923c" : "#c2410c";
  const activePt = pts.find((p) => p.point === active) ?? pts[0];

  return (
    <section className="bg-white dark:bg-slate-900/60 rounded-2xl border border-gray-200 dark:border-slate-700/40 p-5">
      <div className="flex items-baseline gap-2 mb-3">
        <h2 className="flex-1 text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold">
          When to look
        </h2>
        <span className="text-xs font-semibold tabular-nums" style={{ color: accent }}>
          peak {clockFor(peak)}
        </span>
      </div>

      <svg
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="none"
        className="w-full h-24 overflow-visible"
        role="img"
        aria-label={`Sunset quality peaks at ${clockFor(peak)}`}
      >
        <defs>
          <linearGradient id="viewingCurveFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={accent} stopOpacity="0.24" />
            <stop offset="100%" stopColor={accent} stopOpacity="0" />
          </linearGradient>
        </defs>

        <path d={area} fill="url(#viewingCurveFill)" />
        <path
          d={line}
          fill="none"
          stroke={accent}
          strokeWidth="2.5"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />

        {pts.map((p) => {
          const isActive = p.point === active;
          return (
            <circle
              key={p.point}
              cx={p.x}
              cy={p.y}
              r={isActive ? 5.5 : 3.5}
              fill={isActive ? accent : isDark ? "#475569" : "#cbd5e1"}
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>

      <div className="grid grid-cols-4 gap-1 mt-2">
        {pts.map((p) => {
          const isActive = p.point === active;
          return (
            <button
              key={p.point}
              onClick={() => setSelected(p.point)}
              aria-pressed={isActive}
              className={
                isActive
                  ? "flex flex-col items-center justify-center gap-0.5 min-h-[44px] rounded-lg bg-orange-50 dark:bg-orange-500/10 text-orange-700 dark:text-orange-300"
                  : "flex flex-col items-center justify-center gap-0.5 min-h-[44px] rounded-lg text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              }
            >
              <span className="text-xs font-semibold tabular-nums">{clockFor(p.point)}</span>
              <span className="text-[10px] opacity-80">
                {p.point === "sunset" ? "sunset" : p.point}
              </span>
            </button>
          );
        })}
      </div>

      <p className="text-gray-600 dark:text-slate-400 text-xs mt-3 leading-relaxed">
        {active === peak ? (
          <>
            Best colour around{" "}
            <span className="font-semibold text-gray-900 dark:text-white">{clockFor(peak)}</span>
            {peak !== "sunset" && peak.startsWith("+")
              ? " — after the sun is down, when the afterglow lights the high cloud."
              : "."}
          </>
        ) : (
          <>
            At {clockFor(active)} the sky scores{" "}
            <span
              className="font-semibold tabular-nums"
              style={{ color: getScoreHexColor(activePt.score, isDark) }}
            >
              {Math.round(activePt.score)}
            </span>
            . Peak is {clockFor(peak)}.
          </>
        )}
      </p>
    </section>
  );
}
