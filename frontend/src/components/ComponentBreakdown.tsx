"use client";

import type { PhysicsBreakdown } from "@/lib/types";
import { getComponentHexColor } from "@/lib/utils";
import { useIsDark } from "@/lib/useIsDark";

interface ComponentBreakdownProps {
  breakdown: PhysicsBreakdown;
}

interface ComponentRow {
  key: keyof Pick<
    PhysicsBreakdown,
    "cloud_quality_score" | "atmosphere_score" | "moisture_score" | "horizon_score"
  >;
  label: string;
}

const COMPONENTS: ComponentRow[] = [
  { key: "cloud_quality_score", label: "Cloud quality" },
  { key: "atmosphere_score", label: "Atmosphere" },
  { key: "moisture_score", label: "Moisture" },
  { key: "horizon_score", label: "Horizon" },
];

/**
 * The four scored components.
 *
 * Each row is a single baseline-aligned line — label, weight, value — rather
 * than two competing columns, which used to collide at 375px once "(42%
 * weight)" wrapped underneath the label and ran into the description text.
 */
export default function ComponentBreakdown({ breakdown }: ComponentBreakdownProps) {
  const isDark = useIsDark();

  return (
    <div className="flex flex-col gap-3.5">
      {COMPONENTS.map(({ key, label }) => {
        const score = breakdown[key];
        const weight = breakdown.component_weights[key.replace("_score", "")] ?? 0;
        const colour = getComponentHexColor(score, isDark);

        return (
          <div key={key} className="flex flex-col gap-1.5">
            <div className="flex items-baseline gap-2">
              <span className="flex-1 min-w-0 text-sm font-medium text-gray-900 dark:text-slate-200">
                {label}
              </span>
              <span className="text-xs text-gray-600 dark:text-slate-400 tabular-nums">
                {Math.round(weight * 100)}%
              </span>
              <span
                className="text-sm font-semibold tabular-nums w-7 text-right"
                style={{ color: colour }}
              >
                {Math.round(score)}
              </span>
            </div>
            <div className="h-1.5 bg-gray-200 dark:bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${score}%`, backgroundColor: colour }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
