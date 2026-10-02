/**
 * Colours for the rating moment. Each rating warms to its own soft tone with
 * dark, readable ink — a saturated fill hid which option had been tapped.
 */
export interface RatingTone {
  /** Button background once chosen. */
  fill: string;
  /** Confirmation chip background. */
  chip: string;
  /** Text on fill and chip. */
  ink: string;
  /** Check ring, check mark and the soft shadow under the chosen button. */
  ring: string;
  rays: string[];
}

type Value = 1 | 2 | 3 | 4 | 5;

const LIGHT: Record<Value, RatingTone> = {
  1: { fill: "#e2e8f0", chip: "#e2e8f0", ink: "#1e293b", ring: "#64748b", rays: ["#94a3b8", "#cbd5e1"] },
  2: { fill: "#e7e5e4", chip: "#e7e5e4", ink: "#292524", ring: "#78716c", rays: ["#a8a29e", "#d6d3d1"] },
  3: { fill: "linear-gradient(135deg,#fef3c7,#fed7aa)", chip: "#fde8c8", ink: "#7c2d12", ring: "#ea580c", rays: ["#fbbf24", "#fb923c", "#fdba74"] },
  4: { fill: "linear-gradient(135deg,#fed7aa,#fbcfe8)", chip: "#fcdcd0", ink: "#831843", ring: "#db2777", rays: ["#fb923c", "#f472b6", "#fdba74"] },
  5: { fill: "linear-gradient(120deg,#e9d5ff,#fbcfe8 50%,#fed7aa)", chip: "#f1dcf5", ink: "#5b21b6", ring: "#9333ea", rays: ["#c084fc", "#f472b6", "#fb923c", "#fbbf24"] },
};

const DARK: Record<Value, RatingTone> = {
  1: { fill: "#334155", chip: "#334155", ink: "#f1f5f9", ring: "#94a3b8", rays: ["#64748b", "#94a3b8"] },
  2: { fill: "#44403c", chip: "#44403c", ink: "#fafaf9", ring: "#a8a29e", rays: ["#78716c", "#a8a29e"] },
  3: { fill: "linear-gradient(135deg,#78350f,#7c2d12)", chip: "#7c2d12", ink: "#ffedd5", ring: "#fb923c", rays: ["#fbbf24", "#fb923c", "#fdba74"] },
  4: { fill: "linear-gradient(135deg,#7c2d12,#831843)", chip: "#831843", ink: "#fce7f3", ring: "#f472b6", rays: ["#fb923c", "#f472b6", "#fdba74"] },
  5: { fill: "linear-gradient(120deg,#4c1d95,#831843 50%,#7c2d12)", chip: "#4c1d95", ink: "#f3e8ff", ring: "#c084fc", rays: ["#c084fc", "#f472b6", "#fb923c", "#fbbf24"] },
};

export function ratingTone(value: Value, dark: boolean): RatingTone {
  return (dark ? DARK : LIGHT)[value];
}
