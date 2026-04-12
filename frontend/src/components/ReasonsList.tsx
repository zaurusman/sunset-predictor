"use client";

import { AlertCircle, CheckCircle2, Cloud, Droplets, Eye, Mountain, Sun } from "lucide-react";

interface ReasonsListProps {
  reasons: string[];
}

function getReasonIcon(reason: string) {
  const lower = reason.toLowerCase();
  if (lower.includes("cloud")) return <Cloud size={14} className="flex-shrink-0 mt-0.5" />;
  if (lower.includes("rain") || lower.includes("precipitation")) return <Droplets size={14} className="flex-shrink-0 mt-0.5" />;
  if (lower.includes("visibility") || lower.includes("haze") || lower.includes("air") || lower.includes("aerosol")) return <Eye size={14} className="flex-shrink-0 mt-0.5" />;
  if (lower.includes("horizon") || lower.includes("obstruction")) return <Mountain size={14} className="flex-shrink-0 mt-0.5" />;
  if (lower.includes("sun") || lower.includes("light") || lower.includes("colour") || lower.includes("color")) return <Sun size={14} className="flex-shrink-0 mt-0.5" />;
  return <CheckCircle2 size={14} className="flex-shrink-0 mt-0.5" />;
}

function isPositive(reason: string): boolean {
  const lower = reason.toLowerCase();
  const negativeWords = ["block", "reduce", "mute", "poor", "rain", "haze", "heavy", "overcast", "obstruction", "dampen", "wash", "diffuse", "milky", "clip", "limited", "bad", "not ideal", "worst"];
  return !negativeWords.some((w) => lower.includes(w));
}

export default function ReasonsList({ reasons }: ReasonsListProps) {
  if (!reasons.length) return null;

  return (
    <div className="space-y-2">
      {reasons.map((reason, i) => {
        const positive = isPositive(reason);
        return (
          <div
            key={i}
            className="flex items-start gap-3 px-4 py-3 rounded-xl bg-slate-800/50 border border-slate-700/40"
          >
            <span className={positive ? "text-emerald-400" : "text-amber-400"}>
              {getReasonIcon(reason)}
            </span>
            <span className="text-slate-300 text-sm leading-relaxed">{reason}</span>
          </div>
        );
      })}
    </div>
  );
}
