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
    "cloud_quality_score" | "atmosphere_score" | "moisture_score"
  >;
  label: string;
}

/**
 * The three SCORED components. Horizon is no longer among them — it is a gate
 * (below), because it is a property of where you stand rather than of tonight,
 * and as a weighted component it only ever added a constant to every score.
 */
const COMPONENTS: ComponentRow[] = [
  { key: "cloud_quality_score", label: "Cloud quality" },
  { key: "atmosphere_score", label: "Atmosphere" },
  { key: "moisture_score", label: "Moisture" },
];

/** A gate reduces the whole score rather than contributing a share of it. */
interface Gate {
  label: string;
  value: number;
  /** Explains what a reduced value means, shown when the gate is biting. */
  reason: string;
}

/** Below this a gate is worth surfacing; above it, it isn't doing anything. */
const GATE_VISIBLE_BELOW = 0.97;

/**
 * Human-readable name per pathway. Mirrors PATHWAY_LABELS in the scoring
 * engine — a sunset can be beautiful in several unrelated ways, and the score
 * alone does not tell you which one tonight is.
 */
const PATHWAY_LABELS: Record<string, string> = {
  lit_cloud: "Lit clouds",
  twilight_gradient: "Clear-sky gradient",
  crepuscular: "Sun rays",
  breaking_storm: "Breaking storm",
  horizon_band: "Band under the cloud",
};

/** Below this a pathway isn't really happening; listing it would be noise. */
const PATHWAY_VISIBLE_ABOVE = 12;

function gatesFrom(breakdown: PhysicsBreakdown): Gate[] {
  const gates: Gate[] = [];

  if (breakdown.light_corridor_factor !== null) {
    gates.push({
      label: "Light path",
      value: breakdown.light_corridor_factor,
      reason: "Cloud upstream is shading the sky here, whatever it looks like overhead",
    });
  }
  gates.push({
    label: "Rain",
    value: breakdown.precipitation_gate,
    reason: "Rain at sunset replaces the colour rather than dimming it",
  });
  gates.push({
    label: "Horizon",
    value: breakdown.horizon_gate,
    reason: "Your horizon hides the lowest, brightest part of the sky",
  });

  return gates.filter((g) => g.value < GATE_VISIBLE_BELOW);
}

export default function ComponentBreakdown({ breakdown }: ComponentBreakdownProps) {
  const isDark = useIsDark();
  const gates = gatesFrom(breakdown);

  const pathways = Object.entries(breakdown.pathway_scores ?? {})
    .filter(([, v]) => v >= PATHWAY_VISIBLE_ABOVE)
    .sort((a, b) => b[1] - a[1]);

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

      {pathways.length > 0 && (
        <div className="flex flex-col gap-2 pt-1 border-t border-gray-200 dark:border-slate-700/60">
          <span className="text-gray-600 dark:text-slate-400 text-[11px] uppercase tracking-wider font-semibold">
            Ways tonight could be beautiful
          </span>
          {pathways.map(([key, value]) => {
            const isDominant = key === breakdown.dominant_pathway;
            return (
              <div key={key} className="flex items-baseline gap-2">
                <span
                  className={
                    isDominant
                      ? "text-sm font-semibold text-gray-900 dark:text-slate-100"
                      : "text-sm text-gray-600 dark:text-slate-400"
                  }
                >
                  {PATHWAY_LABELS[key] ?? key}
                </span>
                <span className="flex-1 text-xs text-gray-600 dark:text-slate-400 tabular-nums text-right">
                  {Math.round(value)}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {gates.length > 0 && (
        <div className="flex flex-col gap-2 pt-1 border-t border-gray-200 dark:border-slate-700/60">
          <span className="text-gray-600 dark:text-slate-400 text-[11px] uppercase tracking-wider font-semibold">
            Holding it back
          </span>
          {gates.map((g) => (
            <div key={g.label} className="flex flex-col gap-0.5">
              <div className="flex items-baseline gap-2">
                <span className="text-sm font-medium text-gray-900 dark:text-slate-200">
                  {g.label}
                </span>
                <span className="flex-1 text-xs text-gray-600 dark:text-slate-400 tabular-nums text-right">
                  −{Math.round((1 - g.value) * 100)}%
                </span>
              </div>
              <span className="text-xs text-gray-500 dark:text-slate-400 leading-snug text-pretty">
                {g.reason}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
