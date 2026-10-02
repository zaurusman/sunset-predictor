"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef } from "react";
import { useTheme } from "next-themes";
import { skyGradient, type SkyMood } from "@/lib/sky";
import { usePrefersReducedMotion } from "@/lib/motion";

interface Sky {
  /** Cross-fade the band to a mood (no-op if it is already showing). */
  setMood: (mood: SkyMood) => void;
  /**
   * The Tonight opening: the sun sinks to the top edge of `horizon` (the score
   * card), a line of light flares there, then the sky fades from day into
   * `mood` like dusk. Returns the ms until the score should start filling.
   */
  playSunset: (horizon: HTMLElement, mood: SkyMood) => number;
}

const SkyContext = createContext<Sky>({ setMood: () => {}, playSunset: () => 0 });

export function useSky(): Sky {
  return useContext(SkyContext);
}

const SUN_MS = 1950;
const DUSK_MS = 2200;
const FADE_MS = 900;
/** The score starts as the sun touches the horizon, slightly before it is gone. */
export const SCORE_START_MS = SUN_MS - 450;

/**
 * The sky band behind every page. It lives in the root layout so it persists
 * across tab changes and can cross-fade between them.
 */
export default function SkyProvider({ children }: { children: React.ReactNode }) {
  const { resolvedTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  const reduce = usePrefersReducedMotion();

  const layerA = useRef<HTMLDivElement>(null);
  const layerB = useRef<HTMLDivElement>(null);
  const sun = useRef<HTMLDivElement>(null);
  const flare = useRef<HTMLDivElement>(null);
  const glow = useRef<HTMLDivElement>(null);

  /** Which layer is currently on top. */
  const frontIsA = useRef(true);
  const mood = useRef<SkyMood>("Neutral");
  const darkRef = useRef(dark);
  const reduceRef = useRef(reduce);

  const layers = () => {
    const a = layerA.current!, b = layerB.current!;
    return frontIsA.current ? { front: a, back: b } : { front: b, back: a };
  };

  const stopAll = () => {
    for (const el of [layerA.current, layerB.current, sun.current, flare.current, glow.current]) {
      el?.getAnimations().forEach((a) => a.cancel());
    }
  };

  /** Paint `m` with no animation, leaving everything in its resting state. */
  const settle = useCallback((m: SkyMood) => {
    const { front, back } = layers();
    front.style.background = skyGradient(m, darkRef.current);
    front.style.opacity = "1";
    back.style.opacity = "0";
    sun.current!.style.opacity = "0";
    flare.current!.style.opacity = "0";
    const epic = m === "Epic";
    glow.current!.style.opacity = epic ? "0.7" : "0";
    glow.current!.dataset.breathe = epic && !reduceRef.current ? "true" : "false";
  }, []);

  /** Fade the hidden layer in with `m` and the visible one out, then swap roles. */
  const fadeTo = useCallback(
    (m: SkyMood, keyframesIn: Keyframe[], keyframesOut: Keyframe[], timing: KeyframeAnimationOptions) => {
      const { front, back } = layers();
      back.style.background = skyGradient(m, darkRef.current);
      back.animate(keyframesIn, { ...timing, fill: "forwards" });
      const out = front.animate(keyframesOut, { ...timing, fill: "forwards" });
      frontIsA.current = !frontIsA.current;
      out.onfinish = () => {
        stopAll();
        settle(m);
      };
      if (m === "Epic") {
        glow.current!.animate([{ opacity: 0 }, { opacity: 0.7 }], {
          duration: 1200,
          delay: (timing.delay as number) + (timing.duration as number) * 0.6,
          fill: "forwards",
        });
      } else {
        glow.current!.dataset.breathe = "false";
        glow.current!.animate([{ opacity: Number(glow.current!.style.opacity || 0) }, { opacity: 0 }], {
          duration: 400,
          fill: "forwards",
        });
      }
    },
    [settle],
  );

  const setMood = useCallback(
    (m: SkyMood) => {
      if (m === mood.current) return;
      mood.current = m;
      if (reduceRef.current) {
        stopAll();
        settle(m);
        return;
      }
      fadeTo(m, [{ opacity: 0 }, { opacity: 1 }], [{ opacity: 1 }, { opacity: 0 }], {
        duration: FADE_MS,
        delay: 0,
        easing: "ease-in-out",
      });
    },
    [fadeTo, settle],
  );

  const playSunset = useCallback(
    (horizon: HTMLElement, m: SkyMood) => {
      mood.current = m;
      stopAll();
      if (reduceRef.current) {
        settle(m);
        return 0;
      }
      const y = horizon.getBoundingClientRect().top - 8;
      settle("Day");

      sun.current!.animate(
        [
          { transform: "translateY(30px) scale(.85)", opacity: 0 },
          { transform: "translateY(46px) scale(1)", opacity: 1, offset: 0.15 },
          { transform: `translateY(${y - 30}px) scale(1)`, opacity: 1, offset: 0.8 },
          { transform: `translateY(${y + 40}px) scale(.92)`, opacity: 0.85 },
        ],
        { duration: SUN_MS, easing: "cubic-bezier(.4,0,.5,1)" },
      );

      flare.current!.style.top = `${y}px`;
      flare.current!.animate(
        [
          { opacity: 0, transform: "scaleX(.3)" },
          { opacity: 1, transform: "scaleX(1)", offset: 0.35 },
          { opacity: 0, transform: "scaleX(1.05)" },
        ],
        { duration: 1350, delay: SUN_MS - 525, easing: "ease-out" },
      );

      // Dusk: a slow cross-fade from day into tonight's colour — never a
      // reveal spreading from a point.
      fadeTo(
        m,
        [{ opacity: 0 }, { opacity: 0.35, offset: 0.4 }, { opacity: 1 }],
        [{ opacity: 1 }, { opacity: 0.8, offset: 0.4 }, { opacity: 0 }],
        { duration: DUSK_MS, delay: SUN_MS - 450, easing: "ease-in-out" },
      );
      return SCORE_START_MS;
    },
    [fadeTo, settle],
  );

  // Keep the latest theme and motion preference for the callbacks above, and
  // repaint the resting sky in the other palette when the theme flips.
  useEffect(() => {
    darkRef.current = dark;
    reduceRef.current = reduce;
    stopAll();
    settle(mood.current);
  }, [dark, reduce, settle]);

  const value = useMemo(() => ({ setMood, playSunset }), [setMood, playSunset]);

  return (
    <SkyContext.Provider value={value}>
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden" aria-hidden="true">
        <div ref={layerA} className="sky-initial absolute inset-0" />
        <div ref={layerB} className="absolute inset-0 opacity-0" />
        <div ref={glow} className="sky-glow absolute -left-[20%] -right-[20%] -top-16 h-64 opacity-0" />
        <div ref={sun} className="sky-sun absolute left-1/2 top-0 opacity-0" />
        <div ref={flare} className="sky-flare absolute left-[6%] right-[6%] top-0 opacity-0" />
      </div>
      {children}
    </SkyContext.Provider>
  );
}
