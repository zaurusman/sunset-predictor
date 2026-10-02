import type { SunsetCategory } from "./types";

/** What the sky band shows. "Neutral" = History and first run; "Day" = before the sun sets. */
export type SkyMood = SunsetCategory | "Neutral" | "Day";

const PAGE_LIGHT = "#f8fafc";
const PAGE_DARK = "#020617";

// Pastel on purpose: the band sets a mood; the data keeps the focus.
const LIGHT: Record<SkyMood, string> = {
  Day: `linear-gradient(180deg,#bfdbfe 0%,#dbeafe 24%,${PAGE_LIGHT} 50%)`,
  Poor: `linear-gradient(180deg,#cbd5e1 0%,#e2e8f0 22%,${PAGE_LIGHT} 44%)`,
  Decent: `linear-gradient(180deg,#e8d9cd 0%,#f1e8e0 22%,${PAGE_LIGHT} 44%)`,
  Good: `linear-gradient(180deg,#f9d9b0 0%,#fce9d2 22%,${PAGE_LIGHT} 44%)`,
  Great: `linear-gradient(180deg,#f7c8bd 0%,#fbd8c3 18%,#fdeedd 30%,${PAGE_LIGHT} 46%)`,
  Epic: `linear-gradient(180deg,#d8c7ef 0%,#f3c9d9 15%,#fbdcc0 28%,${PAGE_LIGHT} 46%)`,
  Neutral: `linear-gradient(180deg,#ece4dc 0%,#f3eee9 22%,${PAGE_LIGHT} 44%)`,
};

// Same hue families, deep and dim, over slate-950.
const DARK: Record<SkyMood, string> = {
  Day: `linear-gradient(180deg,#1e3a5f 0%,#172a46 24%,${PAGE_DARK} 50%)`,
  Poor: `linear-gradient(180deg,#1e293b 0%,#111827 24%,${PAGE_DARK} 46%)`,
  Decent: `linear-gradient(180deg,#2c241f 0%,#1c1715 24%,${PAGE_DARK} 46%)`,
  Good: `linear-gradient(180deg,#3a2a17 0%,#241a10 24%,${PAGE_DARK} 46%)`,
  Great: `linear-gradient(180deg,#3d1f22 0%,#2c1a17 20%,#1d140f 32%,${PAGE_DARK} 48%)`,
  Epic: `linear-gradient(180deg,#2e1f4d 0%,#3b1d3a 16%,#3a2418 30%,${PAGE_DARK} 48%)`,
  Neutral: `linear-gradient(180deg,#24201d 0%,#17141a 24%,${PAGE_DARK} 46%)`,
};

export function skyGradient(mood: SkyMood, dark: boolean): string {
  return (dark ? DARK : LIGHT)[mood];
}
