/**
 * Home-Screen install state.
 *
 * iOS only delivers web push to Home-Screen apps, so on iPhone the install
 * pitch IS the alerts pitch. Everything here is best-effort: storage may be
 * unavailable and every browser exposes a different subset of these APIs.
 */

export type Platform = "ios" | "android" | "desktop" | "unsupported";
export type PromptKind = "install" | "alerts";

const VISITS_KEY = "afterglow:visits";
const SESSION_KEY = "afterglow:visitCounted";
const promptKey = (kind: PromptKind) => `afterglow:prompt:${kind}`;

const SNOOZE_MS = 14 * 24 * 60 * 60 * 1000;
const MAX_DISMISSALS = 3;

interface PromptState {
  dismissals: number;
  snoozedUntil: number;
}

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // storage unavailable — prompts will simply reappear next visit
  }
}

/** In-app browsers (Instagram, Facebook, WhatsApp, …) can't install. */
function isInAppBrowser(ua: string): boolean {
  return /FBAN|FBAV|Instagram|WhatsApp|Line\/|Twitter|LinkedInApp|Snapchat|TikTok/i.test(ua);
}

function iosVersion(ua: string): number | null {
  const m = ua.match(/OS (\d+)_(\d+)/);
  return m ? Number(m[1]) + Number(m[2]) / 100 : null;
}

export function detectPlatform(): Platform {
  if (typeof navigator === "undefined") return "unsupported";
  const ua = navigator.userAgent;
  if (isInAppBrowser(ua)) return "unsupported";
  const isIos =
    /iPhone|iPad|iPod/.test(ua) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1); // iPadOS
  if (isIos) {
    const v = iosVersion(ua);
    return v !== null && v < 16.04 ? "unsupported" : "ios"; // web push needs 16.4
  }
  if (/Android/i.test(ua)) return "android";
  return "desktop";
}

export function isStandalone(): boolean {
  if (typeof window === "undefined") return false;
  return (
    window.matchMedia?.("(display-mode: standalone)").matches ||
    (navigator as Navigator & { standalone?: boolean }).standalone === true
  );
}

export function pushSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    "serviceWorker" in navigator &&
    "PushManager" in window &&
    "Notification" in window
  );
}

/** Counts at most once per browser session; returns the running total. */
export function recordVisit(): number {
  let visits = read<number>(VISITS_KEY, 0);
  try {
    if (!sessionStorage.getItem(SESSION_KEY)) {
      sessionStorage.setItem(SESSION_KEY, "1");
      visits += 1;
      write(VISITS_KEY, visits);
    }
  } catch {
    // no sessionStorage — leave the count alone
  }
  return visits;
}

export function canShowPrompt(kind: PromptKind): boolean {
  const s = read<PromptState>(promptKey(kind), { dismissals: 0, snoozedUntil: 0 });
  return s.dismissals < MAX_DISMISSALS && Date.now() >= s.snoozedUntil;
}

export function dismissPrompt(kind: PromptKind): void {
  const s = read<PromptState>(promptKey(kind), { dismissals: 0, snoozedUntil: 0 });
  write(promptKey(kind), { dismissals: s.dismissals + 1, snoozedUntil: Date.now() + SNOOZE_MS });
}

// ── Android / Chromium native install prompt ─────────────────────────────────

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

let deferred: BeforeInstallPromptEvent | null = null;
const listeners = new Set<() => void>();
let captured = false;

/** Call once on mount; the event fires early and only once per page load. */
export function captureInstallEvent(): void {
  if (captured || typeof window === "undefined") return;
  captured = true;
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferred = e as BeforeInstallPromptEvent;
    listeners.forEach((cb) => cb());
  });
}

export function hasInstallEvent(): boolean {
  return deferred !== null;
}

export function onInstallEvent(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export async function triggerInstall(): Promise<boolean> {
  if (!deferred) return false;
  await deferred.prompt();
  const { outcome } = await deferred.userChoice;
  deferred = null;
  return outcome === "accepted";
}
