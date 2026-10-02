"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";

/** Confident arrival — the easing every approved demo used. */
export const EASE_OUT = "cubic-bezier(.16,1,.3,1)";

/** Delay for the i-th item of a staggered list, capped so long lists never wait. */
export function staggerDelay(index: number, step: number, cap: number): number {
  return Math.min(index * step, cap);
}

export const SUNSET_PLAYED_KEY = "afterglow:sunsetPlayed";

/**
 * The full sunset plays on the first open of each day only — Afterglow is a
 * daily glance, and a two-second moment every visit would become friction.
 * A storage failure defaults to playing.
 */
export function shouldPlaySunset(
  dateIso: string,
  store: Pick<Storage, "getItem"> = localStorage,
): boolean {
  try {
    return store.getItem(SUNSET_PLAYED_KEY) !== dateIso;
  } catch {
    return true;
  }
}

export function markSunsetPlayed(
  dateIso: string,
  store: Pick<Storage, "setItem"> = localStorage,
): void {
  try {
    store.setItem(SUNSET_PLAYED_KEY, dateIso);
  } catch {
    // Private mode: it simply plays again next time.
  }
}

const REDUCE = "(prefers-reduced-motion: reduce)";

export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const mq = window.matchMedia(REDUCE);
      mq.addEventListener("change", onChange);
      return () => mq.removeEventListener("change", onChange);
    },
    () => window.matchMedia(REDUCE).matches,
    () => false,
  );
}

/**
 * Counts from the value currently on screen to `target` (so a refresh glides
 * rather than restarting at 0). Jumps straight to `target` when disabled.
 * Changing `restart` counts up from 0 again.
 */
export function useCountUp(
  target: number,
  {
    durationMs,
    delayMs = 0,
    enabled = true,
    restart = 0,
  }: { durationMs: number; delayMs?: number; enabled?: boolean; restart?: number },
): number {
  const [value, setValue] = useState(enabled ? 0 : target);
  const shown = useRef(value);
  const lastRestart = useRef(restart);

  useEffect(() => {
    if (restart !== lastRestart.current) {
      lastRestart.current = restart;
      if (enabled) shown.current = 0;
    }
    const from = shown.current;
    let raf = 0;
    const start = performance.now() + (enabled ? delayMs : 0);
    const duration = enabled ? durationMs : 0;
    const tick = (now: number) => {
      const p = duration === 0 ? 1 : Math.min(1, Math.max(0, (now - start) / duration));
      const eased = 1 - Math.pow(1 - p, 4);
      shown.current = from + (target - from) * eased;
      setValue(shown.current);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs, delayMs, enabled, restart]);

  return value;
}

/** Keeps something mounted for `exitMs` after `open` turns false, so it can animate out. */
export function usePresence(open: boolean, exitMs: number): { mounted: boolean; closing: boolean } {
  const [mounted, setMounted] = useState(open);
  // Adjusting state during render (not in an effect) is React's recommended
  // pattern for state derived from a prop change.
  if (open && !mounted) setMounted(true);

  useEffect(() => {
    if (open || !mounted) return;
    const timer = setTimeout(() => setMounted(false), exitMs);
    return () => clearTimeout(timer);
  }, [open, mounted, exitMs]);

  return { mounted, closing: mounted && !open };
}

/** True once the page has scrolled past `threshold` px — the cue to frost the header. */
export function useScrolled(threshold = 6): boolean {
  return useSyncExternalStore(
    (onChange) => {
      window.addEventListener("scroll", onChange, { passive: true });
      return () => window.removeEventListener("scroll", onChange);
    },
    () => window.scrollY > threshold,
    () => false,
  );
}
