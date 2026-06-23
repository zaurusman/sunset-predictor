import {
  getConfidenceFillColor,
  getConfidenceLevel,
  getConfidenceTier,
} from "@/lib/utils";

interface ConfidenceMeterProps {
  /** Raw confidence value (0–100); rendered as a 4-segment tier meter. */
  confidence: number;
  /** Show the "Confidence" caption below the bars. */
  label?: boolean;
  className?: string;
}

/**
 * Signal-bar style confidence indicator: four rising segments, filled by tier
 * and tier-tinted, with a "Confidence" caption below. The tier and the raw
 * percentage are intentionally not shown — the filled-bar count conveys level.
 */
export default function ConfidenceMeter({
  confidence,
  label = true,
  className = "",
}: ConfidenceMeterProps) {
  const tier = getConfidenceTier(confidence);
  const level = getConfidenceLevel(confidence);
  const fill = getConfidenceFillColor(confidence);

  return (
    <div
      className={`inline-flex flex-col items-center gap-1 ${className}`}
      role="img"
      aria-label={`Confidence: ${tier}`}
    >
      <span className="flex items-end gap-[3px]" aria-hidden="true">
        {[0, 1, 2, 3].map((i) => (
          <span
            key={i}
            className={`w-1.5 rounded-[2px] ${i < level ? fill : "bg-gray-200 dark:bg-slate-700"}`}
            style={{ height: 5 + i * 3 }}
          />
        ))}
      </span>
      {label && (
        <span className="text-[10px] font-medium uppercase tracking-wider text-gray-400 dark:text-slate-500">
          Confidence
        </span>
      )}
    </div>
  );
}
