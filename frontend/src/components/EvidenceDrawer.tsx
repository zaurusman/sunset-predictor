"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type { PredictResponse } from "@/lib/types";
import { hazeLabel, isPositiveReason } from "@/lib/utils";
import ComponentBreakdown from "./ComponentBreakdown";

interface EvidenceDrawerProps {
  prediction: PredictResponse;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-50 dark:bg-slate-800/50 rounded-lg p-2.5">
      <div className="text-gray-600 dark:text-slate-400 text-xs mb-0.5">{label}</div>
      <div className="text-gray-900 dark:text-white font-semibold text-sm tabular-nums">
        {value}
      </div>
    </div>
  );
}

/**
 * Why the score is what it is — one line each way when closed, the full
 * breakdown when opened. Replaces the old always-on stack of four bars and
 * nine raw readings, which put every fact at the same weight.
 */
export default function EvidenceDrawer({ prediction }: EvidenceDrawerProps) {
  const [open, setOpen] = useState(false);

  const helping = prediction.reasons.filter(isPositiveReason);
  const hurting = prediction.reasons.filter((r) => !isPositiveReason(r));
  // One line each way, but the side that matches the verdict leads. Always
  // putting the positive first meant a "Not tonight" evening opened with
  // "clear air will help colours pop", which reads as a contradiction.
  const positiveFirst = prediction.beauty_score_0_100 >= 50;
  const helpingLine = helping.slice(0, 1).map((text) => ({ text, positive: true }));
  const hurtingLine = hurting.slice(0, 1).map((text) => ({ text, positive: false }));
  const summary = positiveFirst
    ? [...helpingLine, ...hurtingLine]
    : [...hurtingLine, ...helpingLine];

  const w = prediction.weather_summary;
  const haze = hazeLabel(w.aerosol_optical_depth);

  return (
    <section className="bg-white dark:bg-slate-900/60 rounded-2xl border border-gray-200 dark:border-slate-700/40 overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="w-full flex items-center gap-2.5 px-5 py-4 text-left"
      >
        <h2 className="flex-1 text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold">
          Why this score
        </h2>
        <span className="text-xs text-gray-600 dark:text-slate-400">
          {open ? "Less" : "Details"}
        </span>
        <ChevronDown
          size={15}
          className={`text-gray-500 dark:text-slate-400 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      <div className="px-5 pb-4 flex flex-col gap-2.5">
        {summary.map(({ text, positive }) => (
          <div key={text} className="flex items-start gap-2.5">
            <span
              className={`w-1.5 h-1.5 rounded-full flex-shrink-0 mt-[7px] ${
                positive ? "bg-emerald-600 dark:bg-emerald-400" : "bg-red-600 dark:bg-red-400"
              }`}
            />
            <span className="text-sm text-gray-800 dark:text-slate-200 leading-snug text-pretty">
              {text}
            </span>
          </div>
        ))}
      </div>

      {open && (
        <div className="px-5 py-5 border-t border-gray-200 dark:border-slate-700/40 bg-gray-50 dark:bg-slate-900/40 flex flex-col gap-5">
          <ComponentBreakdown breakdown={prediction.physics_component_breakdown} />

          <div>
            <h3 className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold mb-3">
              Weather at sunset
            </h3>
            <div className="grid grid-cols-3 gap-2">
              <Stat label="Cloud" value={`${Math.round(w.cloud_total_pct)}%`} />
              <Stat label="High cloud" value={`${Math.round(w.cloud_high_pct)}%`} />
              <Stat label="Low cloud" value={`${Math.round(w.cloud_low_pct)}%`} />
              <Stat label="Visibility" value={`${Math.round(w.visibility_km)} km`} />
              <Stat label="Humidity" value={`${Math.round(w.humidity_pct)}%`} />
              <Stat label="Rain" value={`${w.precipitation_mm} mm`} />
              {haze && (
                <Stat
                  label={w.aerosol_is_estimated ? "Haze (est.)" : "Haze"}
                  value={haze}
                />
              )}
              <Stat label="Temp" value={`${Math.round(w.temperature_c)}°C`} />
              <Stat label="Wind" value={`${Math.round(w.wind_speed_kmh)} km/h`} />
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
