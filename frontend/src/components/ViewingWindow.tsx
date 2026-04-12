"use client";

import { Clock } from "lucide-react";
import { formatTime } from "@/lib/utils";

interface ViewingWindowProps {
  sunsetTime: string;
  windowStart: string;
  windowEnd: string;
}

export default function ViewingWindow({
  sunsetTime,
  windowStart,
  windowEnd,
}: ViewingWindowProps) {
  return (
    <div className="bg-slate-800/50 border border-slate-700/40 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3 text-slate-400 text-xs uppercase tracking-wider">
        <Clock size={12} />
        <span>Best Viewing Window</span>
      </div>

      <div className="flex items-center justify-between gap-2">
        {/* Start */}
        <div className="text-center">
          <div className="text-slate-500 text-xs mb-1">From</div>
          <div className="text-white font-semibold tabular-nums text-lg">
            {formatTime(windowStart)}
          </div>
        </div>

        {/* Timeline bar */}
        <div className="flex-1 relative h-6 flex items-center">
          <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                background: "linear-gradient(90deg, #fb923c, #f59e0b, #a855f7)",
                width: "100%",
              }}
            />
          </div>

          {/* Sunset marker */}
          <div
            className="absolute flex flex-col items-center -translate-x-1/2"
            style={{ left: "50%" }}
          >
            <div className="w-3 h-3 rounded-full bg-orange-400 border-2 border-slate-900 shadow-lg shadow-orange-500/40" />
            <div className="text-orange-400 text-xs mt-1 whitespace-nowrap font-medium">
              {formatTime(sunsetTime)}
            </div>
          </div>
        </div>

        {/* End */}
        <div className="text-center">
          <div className="text-slate-500 text-xs mb-1">To</div>
          <div className="text-white font-semibold tabular-nums text-lg">
            {formatTime(windowEnd)}
          </div>
        </div>
      </div>

      <p className="text-slate-500 text-xs mt-3 text-center">
        Best 10 min before sunset through 25 min after — this is informational and does not affect the score.
      </p>
    </div>
  );
}
