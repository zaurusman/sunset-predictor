"use client";

import { CheckCircle2, Cloud, Droplets, Eye, Mountain, Sun } from "lucide-react";
import { isPositiveReason } from "@/lib/utils";

interface ReasonsListProps {
  reasons: string[];
}

function getReasonIcon(reason: string) {
  const lower = reason.toLowerCase();
  const props = { size: 14, className: "flex-shrink-0 mt-0.5" };

  if (lower.includes("cloud")) return <Cloud {...props} />;
  if (lower.includes("rain") || lower.includes("precipitation")) return <Droplets {...props} />;
  if (
    lower.includes("visibility") ||
    lower.includes("haze") ||
    lower.includes("air") ||
    lower.includes("aerosol")
  )
    return <Eye {...props} />;
  if (lower.includes("horizon") || lower.includes("obstruction")) return <Mountain {...props} />;
  if (
    lower.includes("sun") ||
    lower.includes("light") ||
    lower.includes("colour") ||
    lower.includes("color")
  )
    return <Sun {...props} />;
  return <CheckCircle2 {...props} />;
}

export default function ReasonsList({ reasons }: ReasonsListProps) {
  if (!reasons.length) return null;

  return (
    <div className="flex flex-col gap-2">
      {reasons.map((reason, i) => {
        const positive = isPositiveReason(reason);
        return (
          <div
            key={i}
            className="flex items-start gap-3 px-4 py-3 rounded-xl bg-gray-50 dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700/40"
          >
            <span
              className={
                positive
                  ? "text-emerald-700 dark:text-emerald-400"
                  : "text-orange-700 dark:text-orange-400"
              }
            >
              {getReasonIcon(reason)}
            </span>
            <span className="text-gray-800 dark:text-slate-200 text-sm leading-relaxed text-pretty">
              {reason}
            </span>
          </div>
        );
      })}
    </div>
  );
}
