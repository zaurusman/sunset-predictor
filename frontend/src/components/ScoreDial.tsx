"use client";

import { useEffect, useState } from "react";
import type { SunsetCategory } from "@/lib/types";
import { getCategoryBgColor, getScoreHexColor } from "@/lib/utils";

interface ScoreDialProps {
  score: number;
  category: SunsetCategory;
  confidence: number;
  size?: number;
}

export default function ScoreDial({
  score,
  category,
  confidence,
  size = 200,
}: ScoreDialProps) {
  const [animatedScore, setAnimatedScore] = useState(0);

  // Animate score fill on mount
  useEffect(() => {
    const timer = setTimeout(() => setAnimatedScore(score), 100);
    return () => clearTimeout(timer);
  }, [score]);

  const radius = (size - 20) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * radius;

  // We draw a 270-degree arc (from 135° to 405°, i.e. bottom-left around to bottom-right)
  // using stroke-dasharray trick
  const arcFraction = 0.75; // 270 degrees = 75% of circle
  const arcLength = circumference * arcFraction;
  const offset = circumference * arcFraction * (1 - animatedScore / 100);

  const colour = getScoreHexColor(score);

  // SVG rotation: start arc at 135 degrees (7-o'clock position)
  const rotation = 135;

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="rotate-0"
        >
          <defs>
            <linearGradient id="scoreGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#f87171" />
              <stop offset="35%" stopColor="#fb923c" />
              <stop offset="65%" stopColor="#fbbf24" />
              <stop offset="85%" stopColor="#34d399" />
              <stop offset="100%" stopColor="#a855f7" />
            </linearGradient>
          </defs>

          {/* Background track */}
          <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill="none"
            stroke="#1e293b"
            strokeWidth="12"
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeDashoffset={0}
            strokeLinecap="round"
            transform={`rotate(${rotation} ${cx} ${cy})`}
          />

          {/* Score arc */}
          <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill="none"
            stroke="url(#scoreGrad)"
            strokeWidth="12"
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform={`rotate(${rotation} ${cx} ${cy})`}
            style={{
              transition: "stroke-dashoffset 1s cubic-bezier(0.4, 0, 0.2, 1)",
              filter: `drop-shadow(0 0 8px ${colour}66)`,
            }}
          />
        </svg>

        {/* Center content */}
        <div
          className="absolute inset-0 flex flex-col items-center justify-center gap-1"
          style={{ top: "10px" }}
        >
          <span
            className="font-bold tabular-nums leading-none"
            style={{
              fontSize: size * 0.22,
              color: colour,
              textShadow: `0 0 20px ${colour}88`,
            }}
          >
            {Math.round(animatedScore)}
          </span>
          <span
            className="text-slate-400 font-medium uppercase tracking-widest"
            style={{ fontSize: size * 0.065 }}
          >
            / 100
          </span>
        </div>
      </div>

      {/* Category badge */}
      <div
        className={`px-4 py-1.5 rounded-full border text-sm font-semibold tracking-wide ${getCategoryBgColor(category)}`}
      >
        {category}
      </div>

      {/* Confidence */}
      <div className="text-slate-500 text-xs">
        {Math.round(confidence)}% confidence
      </div>
    </div>
  );
}
