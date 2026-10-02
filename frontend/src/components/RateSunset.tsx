"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Coffee } from "lucide-react";
import { rateSunset } from "@/lib/api";
import type { LocationState } from "@/lib/types";
import { ratingTone } from "@/lib/ratingTones";
import { EASE_OUT, usePrefersReducedMotion } from "@/lib/motion";
import { useIsDark } from "@/lib/useIsDark";
import { SUPPORT_URL } from "./SupportFooter";

interface RateSunsetProps {
  location: LocationState;
  /** The evening being rated, "YYYY-MM-DD". */
  targetDate: string;
  /** What the model predicted, so we can show the gap after rating. */
  predictedScore: number;
}

/**
 * One-tap rating of how the sunset actually looked.
 *
 * WHY THIS IS HERE
 * ----------------
 * The engine has never been measured against reality. These ratings are the
 * only ground truth it will have, and they accrue at one per evening — every
 * night without this is a row that can't be recovered later.
 *
 * The wording deliberately invites low ratings. The previous ML attempt failed
 * partly because its labels came from posted photos, so the dataset contained
 * no bad evenings and the model could never learn to say "not tonight".
 */

const OPTIONS: { value: 1 | 2 | 3 | 4 | 5; label: string; hint: string }[] = [
  { value: 1, label: "Nothing", hint: "Grey, no colour at all" },
  { value: 2, label: "Dull", hint: "A bit of colour, forgettable" },
  { value: 3, label: "Pleasant", hint: "Nice enough" },
  { value: 4, label: "Very good", hint: "Worth having stopped for" },
  { value: 5, label: "Exceptional", hint: "One of the year's best" },
];

/** Local-storage key so a rated evening stays rated across reloads. */
function storageKey(date: string, loc: LocationState): string {
  return `afterglow:rated:${date}:${loc.latitude.toFixed(2)},${loc.longitude.toFixed(2)}`;
}

/** How long the tap moment plays before the card turns into the confirmation. */
const TAP_MOMENT_MS = 780;

type Phase = "pick" | "sending" | "done";

export default function RateSunset({
  location,
  targetDate,
  predictedScore,
}: RateSunsetProps) {
  const isDark = useIsDark();
  const reduce = usePrefersReducedMotion();

  const [phase, setPhase] = useState<Phase>("pick");
  /** The rating tapped (while sending) or saved (once done). */
  const [chosen, setChosen] = useState<OPTIONS_VALUE | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** False when the confirmation was restored from storage: it just appears. */
  const [animateDone, setAnimateDone] = useState(false);
  /** The picker has finished fading out and leaves the layout. */
  const [pickGone, setPickGone] = useState(false);

  const cardRef = useRef<HTMLElement>(null);
  const pickRef = useRef<HTMLDivElement>(null);
  const doneRef = useRef<HTMLDivElement>(null);
  const raysRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<SVGCircleElement>(null);
  const markRef = useRef<SVGPathElement>(null);
  const chipRef = useRef<HTMLSpanElement>(null);
  const buttons = useRef(new Map<number, HTMLButtonElement>());
  /** Card height just before the confirmation replaces the picker. */
  const heightBefore = useRef(0);

  // Restore any rating already given for this evening at this location.
  useEffect(() => {
    let stored: string | null = null;
    try {
      stored = window.localStorage.getItem(storageKey(targetDate, location));
    } catch {
      // localStorage unavailable (private mode) — rating simply won't persist.
    }
    const value = Number(stored);
    const restored = stored && value >= 1 && value <= 5 ? (value as OPTIONS_VALUE) : null;
    setChosen(restored);
    setPhase(restored ? "done" : "pick");
    setPickGone(restored !== null);
    setAnimateDone(false);
    setMessage(null);
    setError(null);
  }, [targetDate, location]);

  /** The other choices step back while the tapped one celebrates. */
  function stepBack(except: number) {
    let i = 0;
    for (const [value, el] of buttons.current) {
      if (value === except) continue;
      el.animate([{ opacity: 1, transform: "none" }, { opacity: 0.25, transform: "scale(.94)" }], {
        duration: 260,
        delay: 180 + i++ * 30,
        easing: EASE_OUT,
        fill: "forwards",
      });
    }
  }

  /** Little rays of light burst out of the tapped button. */
  function burst(btn: HTMLElement, colours: string[]) {
    const layer = raysRef.current;
    if (!layer) return;
    const lr = layer.getBoundingClientRect();
    const br = btn.getBoundingClientRect();
    const cx = br.left + br.width / 2 - lr.left;
    const cy = br.top + br.height / 2 - lr.top;
    const n = 12;
    for (let k = 0; k < n; k++) {
      const ray = document.createElement("span");
      ray.className = "absolute w-1 h-2.5 rounded-full";
      ray.style.background = colours[k % colours.length];
      ray.style.left = `${cx - 2}px`;
      ray.style.top = `${cy - 5}px`;
      layer.appendChild(ray);
      const deg = (k / n) * 360 + 90;
      const r0 = Math.max(br.width, br.height) / 2 - 4;
      const r1 = r0 + 22 + (k % 3) * 6;
      ray.animate(
        [
          { transform: `rotate(${deg}deg) translateY(${-r0}px) scaleY(.3)`, opacity: 0 },
          { transform: `rotate(${deg}deg) translateY(${-(r0 + 8)}px) scaleY(1)`, opacity: 1, offset: 0.3 },
          { transform: `rotate(${deg}deg) translateY(${-r1}px) scaleY(.4)`, opacity: 0 },
        ],
        { duration: 700, easing: "cubic-bezier(.2,.7,.3,1)" },
      ).onfinish = () => ray.remove();
    }
  }

  /** Undo the tap moment, e.g. when the save failed. */
  function resetButtons() {
    for (const el of buttons.current.values()) el.getAnimations().forEach((a) => a.cancel());
    raysRef.current?.replaceChildren();
  }

  async function submit(value: OPTIONS_VALUE) {
    // The tap moment plays for at least this long, however fast the save is.
    const tapMoment = new Promise((r) => setTimeout(r, reduce ? 0 : TAP_MOMENT_MS));
    const btn = buttons.current.get(value);
    setChosen(value);
    setPhase("sending");
    setError(null);
    if (!reduce && btn) {
      btn.animate(
        [
          { transform: "scale(1)" },
          { transform: "scale(.93)", offset: 0.25 },
          { transform: "scale(1.06)", offset: 0.6 },
          { transform: "scale(1)" },
        ],
        { duration: 520, easing: "ease-out" },
      );
      burst(btn, ratingTone(value, isDark).rays);
      stepBack(value);
    }

    try {
      const res = await rateSunset({
        latitude: location.latitude,
        longitude: location.longitude,
        rating: value,
        target_date: targetDate,
        location_name: location.name,
      });
      try {
        window.localStorage.setItem(storageKey(targetDate, location), String(value));
      } catch {
        // Non-fatal: the rating is stored server-side regardless.
      }
      // Let the tap moment finish before the card turns into the confirmation.
      await tapMoment;
      heightBefore.current = cardRef.current?.offsetHeight ?? 0;
      setMessage(res.message);
      setAnimateDone(true);
      setPickGone(false);
      setPhase("done");
    } catch (e) {
      resetButtons();
      setChosen(null);
      setPhase("pick");
      setError(e instanceof Error ? e.message : "Could not save that rating.");
    }
  }

  // The card becomes a confirmation: the picker lifts away, the check draws
  // itself and the chip pops in.
  useLayoutEffect(() => {
    if (phase !== "done" || !animateDone) return;
    const pick = pickRef.current;
    const done = doneRef.current;
    if (!pick || !done) return;
    pick.animate([{ opacity: 1, transform: "none" }, { opacity: 0, transform: "translateY(-6px)" }], {
      duration: reduce ? 120 : 220,
      easing: "ease-in",
      fill: "forwards",
    }).onfinish = () => setPickGone(true);
    done.animate([{ opacity: 0, transform: "translateY(8px)" }, { opacity: 1, transform: "none" }], {
      duration: reduce ? 150 : 420,
      delay: reduce ? 0 : 120,
      easing: EASE_OUT,
      fill: "backwards",
    });
    ringRef.current?.animate([{ strokeDashoffset: 120 }, { strokeDashoffset: 0 }], {
      duration: reduce ? 1 : 520,
      delay: reduce ? 0 : 200,
      easing: "cubic-bezier(.65,0,.35,1)",
      fill: "both",
    });
    markRef.current?.animate([{ strokeDashoffset: 26 }, { strokeDashoffset: 0 }], {
      duration: reduce ? 1 : 300,
      delay: reduce ? 0 : 620,
      easing: "ease-out",
      fill: "both",
    });
    if (!reduce) {
      chipRef.current?.animate(
        [
          { transform: "scale(.6)", opacity: 0 },
          { transform: "scale(1.12)", opacity: 1, offset: 0.6 },
          { transform: "scale(1)" },
        ],
        { duration: 440, delay: 420, easing: "ease-out", fill: "backwards" },
      );
    }
  }, [phase, animateDone, reduce]);

  // Once the picker leaves the layout, settle the card to the confirmation's
  // height instead of jumping or leaving a gap.
  useLayoutEffect(() => {
    if (!pickGone || !animateDone || reduce) return;
    const card = cardRef.current;
    if (!card) return;
    resetButtons();
    card.animate([{ height: `${heightBefore.current}px` }, { height: `${card.offsetHeight}px` }], {
      duration: 420,
      easing: EASE_OUT,
    });
  }, [pickGone, animateDone, reduce]);

  function changeRating() {
    const reopen = () => {
      resetButtons();
      setChosen(null);
      setPhase("pick");
      setPickGone(false);
      setAnimateDone(false);
      setMessage(null);
      if (!reduce) {
        requestAnimationFrame(() =>
          pickRef.current?.animate([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], {
            duration: 320,
            easing: EASE_OUT,
          }),
        );
      }
    };
    const done = doneRef.current;
    if (!done || reduce) return reopen();
    done.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 160, fill: "forwards" }).onfinish = reopen;
  }

  const option = OPTIONS.find((o) => o.value === chosen);
  const tone = chosen ? ratingTone(chosen, isDark) : null;
  const showDone = phase === "done" && chosen !== null;
  // Map 1–5 onto 0–100 to compare like with like.
  const gap = chosen ? Math.round(predictedScore - ((chosen - 1) / 4) * 100) : 0;

  return (
    <section
      ref={cardRef}
      className="m-rise relative bg-white dark:bg-slate-900/60 rounded-2xl border border-gray-200 dark:border-slate-700/40 p-4"
    >
      <div ref={raysRef} className="absolute inset-0 pointer-events-none z-10" aria-hidden="true" />

      {/* Picker and confirmation share one grid cell, so the swap never jumps. */}
      <div className="grid">
        {!pickGone && (
          <div
            ref={pickRef}
            className={`[grid-area:1/1] flex flex-col gap-3 ${showDone ? "pointer-events-none" : ""}`}
            aria-hidden={showDone || undefined}
          >
            <div className="flex flex-col gap-0.5">
              <span className="text-gray-600 dark:text-slate-400 text-xs uppercase tracking-wider font-semibold">
                How did it actually look?
              </span>
              <p className="text-xs text-gray-500 dark:text-slate-400 text-pretty">
                Rate the dull ones too — that&apos;s the half the model can&apos;t learn without.
              </p>
            </div>

            <div className="grid grid-cols-4 gap-1.5">
              {OPTIONS.map((o) => {
                const picked = chosen === o.value && phase !== "pick";
                const t = picked ? ratingTone(o.value, isDark) : null;
                return (
                  <button
                    key={o.value}
                    ref={(el) => {
                      if (el) buttons.current.set(o.value, el);
                      else buttons.current.delete(o.value);
                    }}
                    type="button"
                    title={o.hint}
                    disabled={phase !== "pick"}
                    onClick={() => submit(o.value)}
                    style={
                      t
                        ? {
                            background: t.fill,
                            color: t.ink,
                            borderColor: "transparent",
                            boxShadow: `0 6px 18px -6px ${t.ring}66`,
                          }
                        : undefined
                    }
                    className={`${o.value === 5 ? "col-span-4" : ""} relative min-h-[44px] px-0.5 py-2 rounded-xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-800/60 text-xs font-semibold text-gray-800 dark:text-slate-200 enabled:hover:border-gray-400 dark:enabled:hover:border-slate-500 disabled:cursor-default transition-[background,border-color,box-shadow] duration-300`}
                  >
                    {o.label}
                  </button>
                );
              })}
            </div>

            {error && (
              <p className="text-xs text-red-600 dark:text-red-400" role="alert">
                {error}
              </p>
            )}
          </div>
        )}

        {showDone && tone && (
          <div ref={doneRef} className="[grid-area:1/1] self-start flex items-start gap-3" aria-live="polite">
            <svg width="44" height="44" viewBox="0 0 44 44" className="flex-none" aria-hidden="true">
              <circle
                ref={ringRef}
                cx="22"
                cy="22"
                r="19"
                fill="none"
                stroke={tone.ring}
                strokeWidth="3"
                strokeLinecap="round"
                transform="rotate(-90 22 22)"
                strokeDasharray="120"
              />
              <path
                ref={markRef}
                d="M14 22.5 L19.5 28 L30 16.5"
                fill="none"
                stroke={tone.ring}
                strokeWidth="3.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeDasharray="26"
              />
            </svg>
            <div className="flex flex-col gap-1 min-w-0">
              <p className="flex items-center gap-1.5 flex-wrap text-sm font-semibold text-gray-900 dark:text-white">
                {animateDone ? "Saved — you saw" : "You saw"}
                <span
                  ref={chipRef}
                  className="inline-flex items-center h-6 px-2.5 rounded-full text-[13px] font-bold"
                  style={{ background: tone.chip, color: tone.ink }}
                >
                  {option?.label ?? chosen}
                </span>
              </p>
              <p className="text-sm text-gray-700 dark:text-slate-300 leading-snug text-pretty">
                {message ??
                  (Math.abs(gap) > 30
                    ? `The model said ${Math.round(predictedScore)} — off by a lot. Logged.`
                    : "Logged.")}
              </p>
              {chosen === 5 && (
                // Someone who just saw one of the year's best is the most likely to chip in.
                <a
                  href={SUPPORT_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="m-press self-start inline-flex items-center gap-1.5 min-h-[44px] px-4 rounded-full text-sm font-medium text-orange-900 dark:text-orange-200 bg-orange-50 dark:bg-orange-500/10 border border-orange-200 dark:border-orange-500/30 hover:border-orange-500/60 transition-colors"
                >
                  <Coffee size={14} className="text-orange-500" />
                  Glad you caught it. Buy me a coffee?
                </a>
              )}
              <button
                type="button"
                onClick={changeRating}
                className="self-start min-h-[32px] text-xs text-orange-800 dark:text-orange-300 underline underline-offset-2 hover:text-orange-950 dark:hover:text-orange-200"
              >
                Change my rating
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

type OPTIONS_VALUE = (typeof OPTIONS)[number]["value"];
