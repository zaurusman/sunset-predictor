# Motion & Sky Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the approved motion & sky design (spec: `docs/superpowers/specs/2026-10-02-motion-and-sky-design.md`, mockups Option 2 v3.2 and 7-day v1) on top of Next 16.

**Architecture:** A layout-level `SkyProvider` owns a fixed sky layer (palettes, the sunset sequence, cross-fades) and exposes `setMood` / `playSunset`. Small, pure helpers (`lib/motion.ts`, `lib/sky.ts`, `lib/ratingTones.ts`) carry every rule that can be unit-tested. Declarative motion lives in `globals.css` as named keyframes + utility classes; interruptible or data-driven motion (sun, count-up, rays) uses the Web Animations API. Tab changes use React `<ViewTransition>` with `transitionTypes` on `<Link>`. Overlays get exit animations through a `usePresence` hook.

**Tech Stack:** Next 16.3.8, React 19.3 (`ViewTransition`), Tailwind 3, Web Animations API, Vitest (dev only, new).

## Global Constraints

- No new **runtime** dependencies. Vitest is a devDependency for the pure helpers.
- Every existing element and behaviour stays (spec: "Every existing element and behaviour stays").
- Content is readable from the first frame; no animation gates information.
- Arrival easing `cubic-bezier(.16,1,.3,1)`; exits faster than entrances.
- Reduced motion = OS `prefers-reduced-motion`: no spatial movement, sun, rays or drifting glow; colour/opacity changes and confirmations stay as short fades. No in-app toggle.
- Sunset sequence: sun sinks ~1950 ms → horizon flare → score starts → dusk cross-fade ~2200 ms (slow-in). First open of the day only (per date in `localStorage`).
- The Poor…Epic label never animates through categories; only the number counts.
- Text ≥ 11 px for functional text; contrast ≥ 4.5:1 (Impeccable hook findings are triaged, not ignored).
- Shell exports `NODE_ENV=production`: run npm/next via `$CLEAN` (see the Next 16 plan's wrapper).
- Branch `feature/motion-sky` on top of `origin/main` (b8c85fb). Finish with a PR; never push to `main`.

## File map

| File | Responsibility |
|---|---|
| `frontend/vitest.config.ts` (new) | Vitest for `src/**/*.test.ts`, node env, `@` alias |
| `frontend/src/lib/motion.ts` (new) | Easing/duration tokens, `staggerDelay`, sunset-played storage, `usePrefersReducedMotion`, `useCountUp`, `usePresence`, `useScrolled` |
| `frontend/src/lib/sky.ts` (new) | Mood type, light/dark palettes, `skyGradient()` |
| `frontend/src/lib/ratingTones.ts` (new) | Per-rating fill/ink/ring/ray colours (light + dark) |
| `frontend/src/components/sky/SkyProvider.tsx` (new) | Context + backdrop: layers, sun, horizon flare, glow; `setMood`, `playSunset` |
| `frontend/src/app/globals.css` | Keyframes, motion utility classes, view-transition CSS, reduced-motion overrides, body background |
| `frontend/src/app/layout.tsx` | Mount `SkyProvider` (replaces the ambient radial overlay) |
| `frontend/src/components/AppNav.tsx` | Sticky frosted header, directional tab links, anchored header, morphing tab highlight |
| `frontend/src/components/PageTransition.tsx` (new) | `<ViewTransition>` wrapper with the tab enter/exit map |
| `frontend/src/components/VerdictCard.tsx` | Gradient ring + travelling tip + count-up, headline rise, triggers sunset/mood |
| `frontend/src/components/ViewingCurve.tsx` | Reveal, scrubber, peak pop (used on Tonight and inside 7-day cards) |
| `frontend/src/components/RateSunset.tsx` | The rating moment wired to the real submit flow |
| `frontend/src/components/{EvidenceDrawer,DatePicker,LocationSheet,IosInstallSheet,SubmitPhotoModal,ErrorAlert,InstallPrompt}.tsx` | Shared drawer/overlay/alert motion |
| `frontend/src/components/{ForecastChart,SunsetCard,ComponentBreakdown,ReasonsList}.tsx` | 7-day motion + chart↔card linking |
| `frontend/src/components/HeatmapGrid.tsx`, `frontend/src/app/heatmap/page.tsx` | History wave + month bars + neutral mood |
| `frontend/src/app/{page,forecast/page,heatmap/page}.tsx` | Transparent `main`, page transition wrapper, moods |

---

### Task 1: Test harness and pure helpers

**Files:**
- Create: `frontend/vitest.config.ts`, `frontend/src/lib/motion.ts`, `frontend/src/lib/sky.ts`, `frontend/src/lib/ratingTones.ts`
- Test: `frontend/src/lib/motion.test.ts`, `frontend/src/lib/sky.test.ts`, `frontend/src/lib/ratingTones.test.ts`
- Modify: `frontend/package.json` (`"test": "vitest run"`, devDependency `vitest`)

**Interfaces — Produces:**
- `motion.ts`: `EASE_OUT: string`, `staggerDelay(index: number, step: number, cap: number): number`, `SUNSET_PLAYED_KEY = "afterglow:sunsetPlayed"`, `shouldPlaySunset(dateIso: string, store?: Pick<Storage,"getItem">): boolean`, `markSunsetPlayed(dateIso: string, store?: Pick<Storage,"setItem">): void`, hooks `usePrefersReducedMotion(): boolean`, `useCountUp(target: number, opts: {durationMs: number; delayMs?: number; enabled?: boolean}): number`, `usePresence(open: boolean, exitMs: number): {mounted: boolean; closing: boolean}`, `useScrolled(threshold?: number): boolean`
- `sky.ts`: `type SkyMood = SunsetCategory | "Neutral" | "Day"`, `skyGradient(mood: SkyMood, dark: boolean): string`
- `ratingTones.ts`: `interface RatingTone { fill: string; chip: string; ink: string; ring: string; rays: string[] }`, `ratingTone(value: 1|2|3|4|5, dark: boolean): RatingTone`

- [ ] **Step 1: Install Vitest and add the script**

```bash
cd frontend && $CLEAN npm install -D vitest@^3
```
`package.json` scripts: `"test": "vitest run"`.

`frontend/vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
```

- [ ] **Step 2: Write failing tests**

`frontend/src/lib/motion.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { markSunsetPlayed, shouldPlaySunset, staggerDelay, SUNSET_PLAYED_KEY } from "./motion";

function memory(): Storage {
  const m = new Map<string, string>();
  return {
    getItem: (k) => m.get(k) ?? null,
    setItem: (k, v) => void m.set(k, String(v)),
    removeItem: (k) => void m.delete(k),
    clear: () => m.clear(),
    key: (i) => [...m.keys()][i] ?? null,
    get length() { return m.size; },
  };
}

describe("staggerDelay", () => {
  it("steps per index and never exceeds the cap", () => {
    expect(staggerDelay(0, 70, 280)).toBe(0);
    expect(staggerDelay(2, 70, 280)).toBe(140);
    expect(staggerDelay(9, 70, 280)).toBe(280);
  });
});

describe("sunset played", () => {
  it("plays once per date", () => {
    const s = memory();
    expect(shouldPlaySunset("2026-10-03", s)).toBe(true);
    markSunsetPlayed("2026-10-03", s);
    expect(s.getItem(SUNSET_PLAYED_KEY)).toBe("2026-10-03");
    expect(shouldPlaySunset("2026-10-03", s)).toBe(false);
    expect(shouldPlaySunset("2026-10-04", s)).toBe(true);
  });
  it("plays when storage throws (private mode) but never crashes", () => {
    const broken = { getItem: () => { throw new Error("denied"); } };
    expect(shouldPlaySunset("2026-10-03", broken)).toBe(true);
    expect(() => markSunsetPlayed("2026-10-03", { setItem: () => { throw new Error("denied"); } })).not.toThrow();
  });
});
```

`frontend/src/lib/sky.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { skyGradient } from "./sky";

describe("skyGradient", () => {
  it("returns a distinct gradient per mood and theme", () => {
    const moods = ["Poor", "Decent", "Good", "Great", "Epic", "Neutral", "Day"] as const;
    const light = moods.map((m) => skyGradient(m, false));
    const dark = moods.map((m) => skyGradient(m, true));
    expect(new Set(light).size).toBe(moods.length);
    expect(new Set(dark).size).toBe(moods.length);
    light.concat(dark).forEach((g) => expect(g.startsWith("linear-gradient(180deg")).toBe(true));
  });
  it("fades into the page colour", () => {
    expect(skyGradient("Epic", false)).toContain("#f8fafc");
    expect(skyGradient("Epic", true)).toContain("#020617");
  });
});
```

`frontend/src/lib/ratingTones.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { ratingTone } from "./ratingTones";

describe("ratingTone", () => {
  it("gives every rating a readable ink and at least two ray colours", () => {
    for (const v of [1, 2, 3, 4, 5] as const) {
      for (const dark of [false, true]) {
        const t = ratingTone(v, dark);
        expect(t.ink).toMatch(/^#[0-9a-f]{6}$/i);
        expect(t.rays.length).toBeGreaterThanOrEqual(2);
      }
    }
  });
  it("matches the approved light tones", () => {
    expect(ratingTone(3, false).fill).toBe("linear-gradient(135deg,#fef3c7,#fed7aa)");
    expect(ratingTone(5, false).ink).toBe("#5b21b6");
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd frontend && $CLEAN npm test`
Expected: FAIL — modules `./motion`, `./sky`, `./ratingTones` not found.

- [ ] **Step 4: Implement the helpers**

`frontend/src/lib/sky.ts`:
```ts
import type { SunsetCategory } from "./types";

/** What the sky band shows. "Neutral" = History/first run; "Day" = before the sun sets. */
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
```

`frontend/src/lib/ratingTones.ts`:
```ts
export interface RatingTone {
  fill: string; // button background once chosen
  chip: string; // confirmation chip background
  ink: string;  // text on fill/chip — dark in light mode, light in dark mode
  ring: string; // check ring + glow
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
```

`frontend/src/lib/motion.ts`:
```ts
"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";

/** Confident arrival; matches the approved demos. */
export const EASE_OUT = "cubic-bezier(.16,1,.3,1)";

export function staggerDelay(index: number, step: number, cap: number): number {
  return Math.min(index * step, cap);
}

export const SUNSET_PLAYED_KEY = "afterglow:sunsetPlayed";

/** The full sunset plays on the first open of each day; storage failures default to playing. */
export function shouldPlaySunset(dateIso: string, store: Pick<Storage, "getItem"> = localStorage): boolean {
  try {
    return store.getItem(SUNSET_PLAYED_KEY) !== dateIso;
  } catch {
    return true;
  }
}

export function markSunsetPlayed(dateIso: string, store: Pick<Storage, "setItem"> = localStorage): void {
  try {
    store.setItem(SUNSET_PLAYED_KEY, dateIso);
  } catch {
    // Private mode: it simply plays again next time.
  }
}

const REDUCE = "(prefers-reduced-motion: reduce)";

export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    (cb) => {
      const mq = window.matchMedia(REDUCE);
      mq.addEventListener("change", cb);
      return () => mq.removeEventListener("change", cb);
    },
    () => window.matchMedia(REDUCE).matches,
    () => false,
  );
}

/** Counts from the currently shown value to `target`; jumps when disabled. */
export function useCountUp(target: number, { durationMs, delayMs = 0, enabled = true }: { durationMs: number; delayMs?: number; enabled?: boolean }): number {
  const [value, setValue] = useState(enabled ? 0 : target);
  const shown = useRef(value);
  useEffect(() => {
    if (!enabled) {
      shown.current = target;
      setValue(target);
      return;
    }
    const from = shown.current;
    let raf = 0;
    const start = performance.now() + delayMs;
    const tick = (now: number) => {
      const p = Math.min(1, Math.max(0, (now - start) / durationMs));
      const eased = 1 - Math.pow(1 - p, 4);
      shown.current = from + (target - from) * eased;
      setValue(shown.current);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs, delayMs, enabled]);
  return value;
}

/** Keeps an element mounted for `exitMs` after `open` turns false, so it can animate out. */
export function usePresence(open: boolean, exitMs: number): { mounted: boolean; closing: boolean } {
  const [mounted, setMounted] = useState(open);
  if (open && !mounted) setMounted(true); // adjust during render, not in an effect
  useEffect(() => {
    if (open || !mounted) return;
    const t = setTimeout(() => setMounted(false), exitMs);
    return () => clearTimeout(t);
  }, [open, mounted, exitMs]);
  return { mounted, closing: mounted && !open };
}

/** True once the page has scrolled past `threshold` px (for the frosted header). */
export function useScrolled(threshold = 6): boolean {
  return useSyncExternalStore(
    (cb) => {
      window.addEventListener("scroll", cb, { passive: true });
      return () => window.removeEventListener("scroll", cb);
    },
    () => window.scrollY > threshold,
    () => false,
  );
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd frontend && $CLEAN npm test` → all pass. Then `$CLEAN npm run type-check` → exit 0.

- [ ] **Step 6: Commit** — `git add frontend/package.json frontend/package-lock.json frontend/vitest.config.ts frontend/src/lib/motion.ts frontend/src/lib/sky.ts frontend/src/lib/ratingTones.ts frontend/src/lib/*.test.ts` → `feat(motion): pure helpers for sky moods, rating tones and motion timing`

---

### Task 2: Global motion CSS

**Files:** Modify `frontend/src/app/globals.css`, `frontend/tailwind.config.ts`

- [ ] **Step 1: Add the motion system to `globals.css`** (append; keep existing rules)

```css
/* ── Page background: the sky layer paints above this ─────────────── */
body { background: #f8fafc; }
.dark body { background: #020617; }

/* ── Motion system ──────────────────────────────────────────────────── */
:root { --ease-out: cubic-bezier(.16, 1, .3, 1); }

@keyframes rise { from { opacity: 0; transform: translateY(14px); } }
@keyframes rise-sm { from { opacity: 0; transform: translateY(8px); } }
@keyframes fade { from { opacity: 0; } }
@keyframes word-rise { from { transform: translateY(105%); } }
@keyframes grow-y { from { transform: scaleY(0); } }
@keyframes grow-x { from { transform: scaleX(0); } }
@keyframes pop { 0% { transform: scale(0); } 60% { transform: scale(1.35); } 100% { transform: scale(1); } }
@keyframes halo { from { transform: scale(1); opacity: .7; } to { transform: scale(3.2); opacity: 0; } }
@keyframes reveal-x { from { clip-path: inset(0 100% 0 0); } to { clip-path: inset(0 0 0 0); } }
@keyframes sweep { from { left: 0%; opacity: .8; } 99% { opacity: .8; } to { left: var(--sweep-to, 70%); opacity: 0; } }
@keyframes sheet-in { from { transform: translateY(100%); } }
@keyframes sheet-out { to { transform: translateY(100%); } }
@keyframes backdrop-in { from { opacity: 0; } }
@keyframes backdrop-out { to { opacity: 0; } }
@keyframes modal-in { from { opacity: 0; transform: scale(.96); } }
@keyframes modal-out { to { opacity: 0; transform: scale(.96); } }
@keyframes drop-in { from { opacity: 0; transform: translateY(-6px) scale(.98); } }
@keyframes alert-in { from { opacity: 0; transform: translateY(-8px); } }

.m-rise { animation: rise .56s var(--ease-out) backwards; }
.m-rise-sm { animation: rise-sm .42s var(--ease-out) backwards; }
.m-fade { animation: fade .3s ease-out backwards; }
.m-word { display: inline-block; overflow: hidden; vertical-align: bottom; padding-bottom: 2px; }
.m-word > span { display: inline-block; animation: word-rise .6s var(--ease-out) backwards; }
.m-grow-y { transform-box: fill-box; transform-origin: bottom; animation: grow-y .7s var(--ease-out) backwards; }
.m-grow-x { transform-origin: left; animation: grow-x .7s var(--ease-out) backwards; }
.m-pop { transform-box: fill-box; transform-origin: center; animation: pop .42s var(--ease-out) backwards; }
.m-halo { transform-box: fill-box; transform-origin: center; animation: halo .9s ease-out both; }
.m-reveal { animation: reveal-x 1.1s var(--ease-out) backwards; }
.m-sweep { animation: sweep 1.1s var(--ease-out) both; }
.m-sheet { animation: sheet-in .38s var(--ease-out) backwards; }
.m-sheet[data-closing="true"] { animation: sheet-out .22s ease-in forwards; }
.m-backdrop { animation: backdrop-in .2s ease-out backwards; }
.m-backdrop[data-closing="true"] { animation: backdrop-out .22s ease-in forwards; }
.m-modal { animation: modal-in .3s var(--ease-out) backwards; }
.m-modal[data-closing="true"] { animation: modal-out .18s ease-in forwards; }
.m-drop { transform-origin: top center; animation: drop-in .22s var(--ease-out) backwards; }
.m-alert { animation: alert-in .35s var(--ease-out) backwards; }
.m-press { transition: transform .12s ease-out; }
.m-press:active { transform: scale(.95); }

/* Height animation for drawers and day cards. */
.m-collapse { display: grid; grid-template-rows: 0fr; transition: grid-template-rows .45s var(--ease-out); }
.m-collapse[data-open="true"] { grid-template-rows: 1fr; }
.m-collapse > * { overflow: hidden; min-height: 0; }

/* ── Tab view transitions (React <ViewTransition> + Link transitionTypes) ── */
::view-transition { pointer-events: none; }
::view-transition-group(app-header) { animation: none; z-index: 100; }
::view-transition-old(app-header) { display: none; }
::view-transition-new(app-header) { animation: none; }
::view-transition-group(tab-pill) { animation-duration: .38s; animation-timing-function: var(--ease-out); z-index: 101; }
::view-transition-new(tab-pill), ::view-transition-old(tab-pill) { animation: pill-stretch .38s var(--ease-out) both; height: 100%; }
@keyframes pill-stretch { 50% { transform: scaleX(1.3); } }
::view-transition-old(.tab-forward) { animation: .17s ease-in both vt-out; --dx: -20px; }
::view-transition-new(.tab-forward) { animation: .36s var(--ease-out) .05s both vt-in; --dx: 26px; }
::view-transition-old(.tab-back) { animation: .17s ease-in both vt-out; --dx: 20px; }
::view-transition-new(.tab-back) { animation: .36s var(--ease-out) .05s both vt-in; --dx: -26px; }
@keyframes vt-out { to { opacity: 0; translate: var(--dx); filter: blur(4px); } }
@keyframes vt-in { from { opacity: 0; translate: var(--dx); filter: blur(6px); } }

/* ── Reduced motion: keep colour/opacity, drop movement ─────────────── */
@media (prefers-reduced-motion: reduce) {
  .m-rise, .m-rise-sm, .m-word > span, .m-grow-y, .m-grow-x, .m-pop, .m-reveal, .m-drop, .m-alert, .m-modal { animation: fade .2s ease-out backwards; }
  .m-sheet, .m-sheet[data-closing="true"] { animation: none; }
  .m-halo, .m-sweep { animation: none; opacity: 0; }
  .m-press:active { transform: none; }
  .m-collapse { transition: none; }
  ::view-transition-old(*), ::view-transition-new(*), ::view-transition-group(*) { animation-duration: 0s !important; animation-delay: 0s !important; }
}
```

- [ ] **Step 2: Remove the now-unused Tailwind animations** (`fade-in`, `slide-up`, `score-fill` and their keyframes) from `tailwind.config.ts`, and replace their three usages (`animate-fade-in` in `page.tsx`, `forecast/page.tsx`, `heatmap/page.tsx`, `FirstRun.tsx`; `animate-slide-up` in `IosInstallSheet.tsx`) with `m-fade` / `m-sheet`. Grep must return nothing: `grep -rn "animate-fade-in\|animate-slide-up\|score-fill" frontend/src frontend/tailwind.config.ts`.

- [ ] **Step 3: Verify** — `$CLEAN npm run type-check`, `$CLEAN npx next build` pass.

- [ ] **Step 4: Commit** — `feat(motion): global motion keyframes, view-transition and reduced-motion CSS`

---

### Task 3: The sky (provider, backdrop, sunset sequence)

**Files:** Create `frontend/src/components/sky/SkyProvider.tsx`; modify `frontend/src/app/layout.tsx`; make `<main>` transparent in `app/page.tsx`, `app/forecast/page.tsx`, `app/heatmap/page.tsx` (drop `bg-gray-50 dark:bg-slate-950`).

**Interfaces — Consumes:** `skyGradient`, `SkyMood` (Task 1), `usePrefersReducedMotion`, `EASE_OUT`. **Produces:** `useSky(): { setMood(mood: SkyMood): void; playSunset(horizon: HTMLElement, mood: SkyMood): number }` — `playSunset` returns ms until the score should start filling (0 when reduced motion).

- [ ] **Step 1: Write `SkyProvider.tsx`**

```tsx
"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef } from "react";
import { useTheme } from "next-themes";
import { skyGradient, type SkyMood } from "@/lib/sky";
import { EASE_OUT, usePrefersReducedMotion } from "@/lib/motion";

interface Sky {
  setMood: (mood: SkyMood) => void;
  /** Runs the sunset over `horizon` (the score card). Returns ms until the score should start. */
  playSunset: (horizon: HTMLElement, mood: SkyMood) => number;
}

const SkyContext = createContext<Sky>({ setMood: () => {}, playSunset: () => 0 });
export const useSky = () => useContext(SkyContext);

const SUN_MS = 1950;
const DUSK_MS = 2200;
const FADE_MS = 900;

export default function SkyProvider({ children }: { children: React.ReactNode }) {
  const { resolvedTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  const reduce = usePrefersReducedMotion();
  const layers = [useRef<HTMLDivElement>(null), useRef<HTMLDivElement>(null)];
  const front = useRef(0);
  const mood = useRef<SkyMood>("Neutral");
  const sun = useRef<HTMLDivElement>(null);
  const flare = useRef<HTMLDivElement>(null);
  const glow = useRef<HTMLDivElement>(null);

  const stopAll = () =>
    [...layers.map((l) => l.current), sun.current, flare.current, glow.current].forEach((el) =>
      el?.getAnimations().forEach((a) => a.cancel()),
    );

  const settle = (m: SkyMood) => {
    const f = layers[front.current].current!, b = layers[1 - front.current].current!;
    f.style.background = skyGradient(m, dark);
    f.style.opacity = "1";
    b.style.opacity = "0";
    glow.current!.style.opacity = m === "Epic" ? "0.7" : "0";
    glow.current!.dataset.breathe = m === "Epic" && !reduce ? "true" : "false";
  };

  const crossTo = useCallback((m: SkyMood, ms: number, delay = 0, easing = "ease-in-out") => {
    const f = layers[front.current].current!, b = layers[1 - front.current].current!;
    b.style.background = skyGradient(m, dark);
    b.animate([{ opacity: 0 }, { opacity: 1 }], { duration: ms, delay, easing, fill: "forwards" });
    const out = f.animate([{ opacity: 1 }, { opacity: 0 }], { duration: ms, delay, easing, fill: "forwards" });
    front.current = 1 - front.current;
    out.onfinish = () => { stopAll(); settle(m); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dark]);

  const setMood = useCallback((m: SkyMood) => {
    if (m === mood.current) return;
    mood.current = m;
    if (reduce) { stopAll(); settle(m); return; }
    crossTo(m, FADE_MS);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduce, crossTo]);

  const playSunset = useCallback((horizon: HTMLElement, m: SkyMood) => {
    mood.current = m;
    stopAll();
    if (reduce) { settle(m); return 0; }
    const y = horizon.getBoundingClientRect().top - 8;
    settle("Day");
    const s = sun.current!, h = flare.current!;
    s.animate(
      [
        { transform: "translateY(30px) scale(.85)", opacity: 0 },
        { transform: "translateY(46px) scale(1)", opacity: 1, offset: 0.15 },
        { transform: `translateY(${y - 30}px) scale(1)`, opacity: 1, offset: 0.8 },
        { transform: `translateY(${y + 40}px) scale(.92)`, opacity: 0.85 },
      ],
      { duration: SUN_MS, easing: "cubic-bezier(.4,0,.5,1)" },
    );
    h.style.top = `${y}px`;
    h.animate(
      [{ opacity: 0, transform: "scaleX(.3)" }, { opacity: 1, transform: "scaleX(1)", offset: 0.35 }, { opacity: 0, transform: "scaleX(1.05)" }],
      { duration: 1350, delay: SUN_MS - 525, easing: "ease-out" },
    );
    // Dusk: a slow cross-fade from day into tonight's colour — never a reveal from a point.
    const f = layers[front.current].current!, b = layers[1 - front.current].current!;
    b.style.background = skyGradient(m, dark);
    b.animate([{ opacity: 0 }, { opacity: 0.35, offset: 0.4 }, { opacity: 1 }], { duration: DUSK_MS, delay: SUN_MS - 450, easing: "ease-in-out", fill: "forwards" });
    const out = f.animate([{ opacity: 1 }, { opacity: 0.8, offset: 0.4 }, { opacity: 0 }], { duration: DUSK_MS, delay: SUN_MS - 450, easing: "ease-in-out", fill: "forwards" });
    front.current = 1 - front.current;
    out.onfinish = () => { stopAll(); settle(m); };
    return SUN_MS - 450;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduce, dark]);

  // Theme switch: repaint the settled mood in the other palette.
  useEffect(() => { stopAll(); settle(mood.current); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [dark]);

  const value = useMemo(() => ({ setMood, playSunset }), [setMood, playSunset]);

  return (
    <SkyContext.Provider value={value}>
      <div className="fixed inset-0 pointer-events-none z-0" aria-hidden="true">
        <div ref={layers[0]} className="absolute inset-0" style={{ background: skyGradient("Neutral", false) }} />
        <div ref={layers[1]} className="absolute inset-0" style={{ opacity: 0 }} />
        <div ref={glow} className="sky-glow absolute -left-[20%] -right-[20%] -top-16 h-64 opacity-0" />
        <div ref={sun} className="sky-sun absolute left-1/2 top-0 opacity-0" />
        <div ref={flare} className="sky-flare absolute left-[6%] right-[6%] opacity-0" />
      </div>
      {children}
    </SkyContext.Provider>
  );
}
```

Append to `globals.css`:
```css
.sky-sun { width: 190px; height: 190px; margin: -95px 0 0 -95px; border-radius: 50%;
  background: radial-gradient(circle, #fffbeb 0%, #fff1d6 14%, #fdba74 30%, rgba(251,146,60,.55) 44%, rgba(251,146,60,.18) 60%, rgba(251,146,60,0) 72%); }
.dark .sky-sun { opacity: .85; }
.sky-flare { height: 3px; margin-top: -1.5px; border-radius: 9999px;
  background: radial-gradient(ellipse 50% 100% at 50% 50%, #fb923c 0%, rgba(251,146,60,.5) 45%, rgba(251,146,60,0) 100%);
  box-shadow: 0 0 18px 6px rgba(251,146,60,.35); }
.sky-glow { background: radial-gradient(ellipse 50% 45% at 50% 40%, rgba(251,146,60,.28), transparent 70%); }
.sky-glow[data-breathe="true"] { animation: breathe 6s ease-in-out infinite alternate; }
@keyframes breathe { from { opacity: .55; transform: translateX(-4%); } to { opacity: 1; transform: translateX(4%); } }
@media (prefers-reduced-motion: reduce) { .sky-glow[data-breathe="true"] { animation: none; } }
```
(The `.sky-flare` glow is the literal horizon light — triage the Impeccable `dark-glow` finding as domain-appropriate.)

- [ ] **Step 2: Mount it in `layout.tsx`** — inside `ThemeProvider`, replace the ambient radial `<div className="fixed inset-0 …">` and the `relative z-10` wrapper with:
```tsx
<SkyProvider>
  <div className="relative z-10">{children}</div>
</SkyProvider>
```

- [ ] **Step 3: Moods on the simple pages** — `heatmap/page.tsx` and the first-run branch of `page.tsx`: `const { setMood } = useSky(); useEffect(() => setMood("Neutral"), [setMood]);` (Tonight's prediction mood comes from Task 5; 7 days from Task 9.)

- [ ] **Step 4: Verify in the browser** (`preview_start` backend-8016 + a `next dev` config for this worktree): History shows the neutral warm band in light and dark; `document.querySelector('[aria-hidden] > div').style.background` changes when the theme toggles. No console errors.

- [ ] **Step 5: Commit** — `feat(sky): layout-level sky with moods, cross-fades and the sunset sequence`

---

### Task 4: Header, tabs and page transitions

**Files:** Modify `AppNav.tsx`; create `components/PageTransition.tsx`; wrap the content of `app/page.tsx`, `app/forecast/page.tsx`, `app/heatmap/page.tsx`.

**Interfaces — Consumes:** `useScrolled`. **Produces:** `<PageTransition>{children}</PageTransition>`.

- [ ] **Step 1: `PageTransition.tsx`**

```tsx
"use client";

import { ViewTransition } from "react";

/** Directional slide between tabs; browser back/forward (no type) swaps instantly. */
export default function PageTransition({ children }: { children: React.ReactNode }) {
  const map = { "tab-forward": "tab-forward", "tab-back": "tab-back", default: "none" } as const;
  return (
    <ViewTransition enter={map} exit={map} default="none">
      <div>{children}</div>
    </ViewTransition>
  );
}
```

- [ ] **Step 2: AppNav** — root becomes the sticky, anchored, frosted header; tab links carry direction; the active highlight is a shared element:

```tsx
const order: AppTab[] = ["tonight", "forecast", "heatmap"];
const scrolled = useScrolled();
// root
<div style={{ viewTransitionName: "app-header" }} data-scrolled={scrolled}
  className="sticky top-0 z-30 -mx-4 px-4 pt-2 pb-3 mb-3 flex flex-col gap-3 transition-[background-color,box-shadow] duration-200 data-[scrolled=true]:bg-slate-50/75 dark:data-[scrolled=true]:bg-slate-950/70 data-[scrolled=true]:backdrop-blur-lg data-[scrolled=true]:shadow-[0_1px_0_rgba(15,23,42,.06)]">
// location pill, theme toggle, nav container: replace bg-white / bg-gray-100 with
//   "bg-white/70 dark:bg-slate-900/60 backdrop-blur-md border-white/80 dark:border-slate-700/50" and add "m-press"
// each tab Link:
<Link transitionTypes={[order.indexOf(tab.id) > order.indexOf(active) ? "tab-forward" : "tab-back"]} …>
  {isActive && <span aria-hidden="true" style={{ viewTransitionName: "tab-pill" }} className="absolute inset-0 rounded-lg bg-white dark:bg-slate-800 shadow-sm" />}
  <span className="relative">{tab.label}</span>
</Link>
```
(Active link keeps its text classes but drops its own `bg-white … shadow-sm`; links become `relative`.) `ThemeToggle.tsx` gets the same frosted classes.

- [ ] **Step 3: Wrap each page's content** — in all three pages, wrap the client content component in `<PageTransition>` inside `<main>` (keep `SupportFooter` inside it too).

- [ ] **Step 4: Verify** — the logo (`/logo.png`, orange wordmark) stays legible on every mood in light and dark (spec: brand text on the sky uses the darker orange); if it washes out on Epic/Great, give the logo link a `drop-shadow(0 1px 0 rgba(255,255,255,.6))` in light mode. At 375 px: Tonight → 7 days slides content left with blur while the header stays still and the highlight slides+stretches; 7 days → Tonight slides right; browser back swaps instantly; scrolling 10 px frosts the header; with emulated reduced motion the tab change is instant. No console errors.

- [ ] **Step 5: Commit** — `feat(nav): frosted sticky header and directional tab transitions`

---

### Task 5: Verdict card — ring, count-up, headline, sunset trigger

**Files:** Modify `VerdictCard.tsx`.

**Interfaces — Consumes:** `useSky`, `useCountUp`, `usePrefersReducedMotion`, `shouldPlaySunset`, `markSunsetPlayed`, `isToday`.

- [ ] **Step 1: Implement**
  - `const ref = useRef<HTMLElement>(null)` on the `<section>`; add `m-rise`.
  - Effect keyed on `[prediction.category, targetDate]`: if `isToday(targetDate) && shouldPlaySunset(today)` → `markSunsetPlayed(today); setIntroDelay(playSunset(ref.current!, prediction.category))`; else `setMood(prediction.category)` and `setIntroDelay(0)`. Use a ref guard so the sunset runs once per mount.
  - `const shown = useCountUp(score, { durationMs: introDelay ? 1500 : 1000, delayMs: introDelay, enabled: !reduce })`; ring dasharray and the number use `shown` (number rounded). Remove the old inline `transition: stroke-dasharray`.
  - Ring stroke: `<linearGradient id="verdict-ring">` from the dark-mode hex to the light-mode hex of `colour`'s band (use `getScoreHexColor(score, true)` → `getScoreHexColor(score, false)` in light, reversed in dark); stroke `url(#verdict-ring)`.
  - Tip: `<circle r="4.5" fill="#fff" stroke={colour} strokeWidth="2.5" style={{ filter: `drop-shadow(0 0 4px ${colour})`, opacity: shown < 1 ? 0 : 1 }}>` at angle `(-90 + 3.6 * shown)°` on radius 27 (centre 31,31).
  - Headline: `key={headline}` on `<h1>`; render words as `<span className="m-word"><span style={{ animationDelay: `${i * 70}ms` }}>{word}</span></span>` joined by spaces; keep `text-pretty`.
  - Category pill and colours are computed from final values (unchanged logic) — no ticking.
- [ ] **Step 2: Verify** — clear `afterglow:sunsetPlayed`, reload `/`: sun sinks, flare at the card's top edge, number counts from 0 once the sun touches, sky fades to tonight's palette. Reload again: no sun, quick count-up. Change date in the picker: number glides, headline re-rises, sky cross-fades. Reduced motion: no sun, number shows final value immediately.
- [ ] **Step 3: Commit** — `feat(tonight): sunset intro, counting ring with travelling tip, rising headline`

---

### Task 6: "When to look" reveal

**Files:** Modify `ViewingCurve.tsx`.

- [ ] **Step 1: Implement** (keeps the time buttons, captions and selection logic unchanged)
  - Wrap the `<svg>` in `<div className="relative">`; give the svg `className="… m-reveal"` with `style={{ animationDelay: "250ms" }}`.
  - Scrubber: `<span aria-hidden className="m-sweep absolute top-1 bottom-0 w-px border-l border-dashed" style={{ borderColor: accent, ["--sweep-to" as string]: `${(peakPt.x / VIEW_W) * 100}%`, animationDelay: "250ms" }} />`.
  - Peak circle: add `className="m-pop"` and `style={{ animationDelay: "1.02s" }}` to the active circle when it is the peak; add a sibling halo circle `className="m-halo"` (`fill="none" stroke={accent}`, delay `1.1s`).
  - Raise the `text-[10px]` offset labels to `text-[11px]` (functional text floor).
- [ ] **Step 2: Verify** — on Tonight and inside an opened 7-day card: curve reveals left→right, dashed marker sweeps and fades at the peak, peak dot pops once with a halo; tapping a time still updates the caption. Reduced motion: shows immediately.
- [ ] **Step 3: Commit** — `feat(curve): reveal, sweep and peak pulse for When to look`

---

### Task 7: The rating moment

**Files:** Modify `RateSunset.tsx`.

**Interfaces — Consumes:** `ratingTone`, `useIsDark`, `usePrefersReducedMotion`, `rateSunset` (unchanged API), `SUPPORT_URL`.

- [ ] **Step 1: Implement** — one `<section ref={cardRef} className="relative … m-rise">` that renders either the picker or the confirmation (stacked in one grid cell; the inactive one `invisible`), keeping all existing state and copy:
  - `phase: "pick" | "sending" | "done"`; `chosen: number | null`; restored-from-storage → `phase="done"` with `animate=false`.
  - Tap `o.value`: set `chosen`, `phase="sending"`; button style `{ background: tone.fill, color: tone.ink, borderColor: "transparent", boxShadow: `0 6px 18px -6px ${tone.ring}66` }`; unless reduced motion: bounce (`el.animate([{transform:"scale(1)"},{transform:"scale(.93)",offset:.25},{transform:"scale(1.06)",offset:.6},{transform:"scale(1)"}],{duration:520})`), 12 rays (absolute `span`s, 4×10 px, rounded, `tone.rays[k % n]`, rotated around the button centre and animated outward/fading over 700 ms, removed on finish), other buttons → `opacity .25, scale(.94)` (staggered 30 ms). Then call `submit(value)`.
  - On success: wait until ≥ 780 ms after the tap, then measure card height `h0`, switch to `phase="done"`, measure `h1`, `cardRef.current.animate([{height:`${h0}px`},{height:`${h1}px`}],{duration:420,easing:EASE_OUT})`; the confirmation renders: check ring (`stroke-dasharray 120`, animate offset 120→0 over 520 ms, delay 200) and mark (26→0, 300 ms, delay 620) in `tone.ring`; "Saved — you saw <chip>" (chip `tone.chip` / `tone.ink`, pop-in delay 420); then the existing message logic (`res.message` or the "The model said N — off by a lot. Logged." / "Logged." text); the coffee link when `submitted === 5`; "Change my rating" (existing behaviour: back to picker).
  - On error: revert the fill and options (cancel animations), `phase="pick"`, show the existing error line.
  - `aria-live="polite"` on the confirmation; buttons `disabled` while sending; keep `title={o.hint}`.
- [ ] **Step 2: Verify against the local backend** — success: tap each rating once (use "Change my rating" between) → readable fill, rays, confirmation with the server message; reload → confirmation shows without animation. Failure: stop the backend preview, tap → fill reverts, error shown. Dark mode tones readable. Reduced motion: no rays/bounce, fades only.
- [ ] **Step 3: Commit** — `feat(rating): sunset-toned tap, rays and an animated saved confirmation`

---

### Task 8: Tonight page, drawers and overlays

**Files:** Modify `app/page.tsx`, `EvidenceDrawer.tsx`, `DatePicker.tsx`, `LocationSheet.tsx`, `IosInstallSheet.tsx`, `SubmitPhotoModal.tsx`, `ErrorAlert.tsx`, `InstallPrompt.tsx`.

- [ ] **Step 1: Tonight page** — card stack: give each direct child `m-rise` with `animationDelay = staggerDelay(i, 70, 280)` (VerdictCard already rises at 0). Photo button gets `m-press`. Change `{photoOpen && <SubmitPhotoModal …/>}` to `<SubmitPhotoModal open={photoOpen} …/>`.
- [ ] **Step 2: EvidenceDrawer** — always render the panel inside `<div className="m-collapse" data-open={open}><div>{presence.mounted && …}</div></div>` with `const presence = usePresence(open, 450)`; inner content `m-rise-sm`.
- [ ] **Step 3: DatePicker** — dropdown: `usePresence(open, 180)`; `className="… m-drop"`, while closing add `opacity-0 transition-opacity duration-150`.
- [ ] **Step 4: Sheets** — `LocationSheet`, `IosInstallSheet`: replace `if (!open) return null` with `const p = usePresence(open, 220); if (!p.mounted) return null;`; backdrop `m-backdrop data-closing={p.closing}`, panel `m-sheet data-closing={p.closing}`. Keep Escape/close handlers.
- [ ] **Step 5: Modal** — `SubmitPhotoModal` takes `open: boolean`; `usePresence(open, 180)`; overlay `m-backdrop`, dialog `m-modal`, both with `data-closing`.
- [ ] **Step 6: Alerts** — `ErrorAlert` root and `InstallPrompt` card get `m-alert`.
- [ ] **Step 7: Verify** — each overlay opens and closes with motion (close via button, backdrop and Escape); form state in the photo modal survives while open; reduced motion: sheets appear/disappear without sliding.
- [ ] **Step 8: Commit** — `feat(motion): drawers, sheets, modal and alerts animate in and out`

---

### Task 9: 7 days — linked chart and animated cards

**Files:** Modify `app/forecast/page.tsx`, `ForecastChart.tsx`, `SunsetCard.tsx`, `ComponentBreakdown.tsx`, `ReasonsList.tsx`.

- [ ] **Step 1: Page state** — `selectedDate` (existing) drives everything: `openDate` state (initially today's date); `select(date, fromChart)` sets both, calls `setMood(day.category)`, and when `fromChart` scrolls the card into view after 460 ms: `const el = document.getElementById(`day-${date}`); window.scrollTo({ top: el.getBoundingClientRect().top + scrollY - headerHeight - 8, behavior: reduce ? "auto" : "smooth" })` where `headerHeight` is the sticky header's `offsetHeight`. Initial mood = today's category.
- [ ] **Step 2: ForecastChart** — bar `<path>` gets `className="m-grow-y"` + `style={{ animationDelay: `${200 + i * 60}ms` }}`; value label `m-fade` delayed `600 + i*60` ms; dim non-selected to `opacity 0.35` with `transition: opacity .35s`; selected bar plays a 380 ms `scaleY(1)→1.06→1` nudge (WAAPI on click). Raise 9–10 px chart text to 11 px.
- [ ] **Step 3: SunsetCard** — props become `{ day, expanded, onToggle }` (controlled); root `id={`day-${day.date}`}` and `m-rise`; score circle → 46 px mini ring (`r=19`, stroke 4, `getScoreHexColor`), number centred; panel in `m-collapse` with `usePresence(expanded, 450)`; open card gets `shadow-[0_8px_24px_-14px_rgba(15,23,42,.25)]`; chevron rotation uses `transition-transform duration-[350ms]`.
- [ ] **Step 4: Inside a card** — `ReasonsList` items `m-rise-sm` with `staggerDelay(i, 55, 330) + 120` ms; `ComponentBreakdown` bar fill `m-grow-x` delayed `450 + i*60` ms (CSS transform, keep the `width` style); remove its `transition-all duration-700`.
- [ ] **Step 5: Verify** — arrive on 7 days: bars grow in sequence, today's card opens; tap Monday's bar → bar emphasised, sky cross-fades to Monday's palette, Monday's card opens (others close) and scrolls under the sticky header; tap a card header → toggles and selects; keyboard Enter on a bar works; reduced motion: no growth/slides, selection still works.
- [ ] **Step 6: Commit** — `feat(forecast): chart selects and opens the day; animated bars and cards`

---

### Task 10: History

**Files:** Modify `HeatmapGrid.tsx`, `app/heatmap/page.tsx`.

- [ ] **Step 1:** Each filled cell gets `m-fade` with `animationDelay = Math.min(weekIndex * 12 + dayIndex * 6, 600)` ms. "Best months" bars get `m-grow-x` delayed `staggerDelay(i, 80, 240)`. Raise `text-[9px]`/`text-[10px]` labels to `text-[11px]` only if layout still fits at 375 px; otherwise leave and note it.
- [ ] **Step 2: Verify** — 6m ↔ 12m replays the wave; neutral sky in light and dark.
- [ ] **Step 3: Commit** — `feat(history): heatmap wave and growing month bars`

---

### Task 11: Full verification and PR

- [ ] `npm test`, `npm run type-check`, `npm run lint` (0 errors), `next build` — all pass.
- [ ] Browser matrix at 375 px and desktop, light and dark, with and without emulated reduced motion: Tonight (first open of day + repeat), 7 days, History, every overlay, first-run screen; no console errors; Impeccable hook findings triaged.
- [ ] Measure gzip JS per page as in the Next 16 PR; report the delta.
- [ ] Push `feature/motion-sky`, open the PR with screenshots/notes, and hand back for the user's local check before merge.
