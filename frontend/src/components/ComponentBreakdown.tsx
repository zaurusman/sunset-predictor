"use client";

import type { PhysicsBreakdown } from "@/lib/types";

interface ComponentBreakdownProps {
  breakdown: PhysicsBreakdown;
}

interface ComponentRow {
  key: keyof Omit<PhysicsBreakdown, "weighted_physics_score" | "component_weights">;
  label: string;
  description: string;
}

const COMPONENTS: ComponentRow[] = [
  { key: "cloud_quality_score", label: "Cloud Quality", description: "High/mid cloud distribution for colour" },
  { key: "atmosphere_score", label: "Atmosphere", description: "Visibility, aerosol, clarity" },
  { key: "moisture_score", label: "Moisture", description: "Rain and humidity conditions" },
  { key: "horizon_score", label: "Horizon", description: "Obstruction by terrain / buildings" },
];

function ScoreBar({ score }: { score: number }) {
  const colour =
    score >= 75 ? "#34d399" :
    score >= 50 ? "#fbbf24" :
    score >= 30 ? "#fb923c" :
    "#f87171";

  return (
    <div className="flex items-center gap-3 w-full">
      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${score}%`, backgroundColor: colour }}
        />
      </div>
      <span className="text-sm font-semibold tabular-nums w-8 text-right" style={{ color: colour }}>
        {Math.round(score)}
      </span>
    </div>
  );
}

export default function ComponentBreakdown({ breakdown }: ComponentBreakdownProps) {
  return (
    <div className="space-y-4">
      {COMPONENTS.map(({ key, label, description }) => {
        const score = breakdown[key] as number;
        const weight = breakdown.component_weights[key.replace("_score", "")] ?? 0;
        return (
          <div key={key} className="space-y-1.5">
            <div className="flex items-center justify-between text-sm">
              <div>
                <span className="text-slate-200 font-medium">{label}</span>
                <span className="text-slate-500 ml-2 text-xs">({Math.round(weight * 100)}% weight)</span>
              </div>
              <span className="text-slate-500 text-xs">{description}</span>
            </div>
            <ScoreBar score={score} />
          </div>
        );
      })}

      {/* Overall physics score */}
      <div className="pt-3 border-t border-slate-700/50 space-y-1.5">
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-300 font-semibold">Physics Score</span>
          <span className="text-slate-500 text-xs">weighted average</span>
        </div>
        <ScoreBar score={breakdown.weighted_physics_score} />
      </div>
    </div>
  );
}
