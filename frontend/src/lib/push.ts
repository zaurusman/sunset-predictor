/**
 * Web Push opt-in and per-place alert bells.
 *
 * The belled places are mirrored in localStorage so the bells render instantly;
 * the backend copy (keyed by the push endpoint) is what the hourly alert run reads.
 */

import { getVapidKey, subscribePush, unsubscribePush } from "./api";
import { MAX_SAVED_PLACES, sameLocation } from "./storage";
import { pushSupported } from "./install";
import type { LocationState } from "./types";

const ALERT_PLACES_KEY = "afterglow:alertPlaces";

export function permission(): NotificationPermission | "unsupported" {
  return pushSupported() ? Notification.permission : "unsupported";
}

export function loadAlertPlaces(): LocationState[] {
  try {
    const raw = localStorage.getItem(ALERT_PLACES_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveAlertPlaces(places: LocationState[]): void {
  try {
    localStorage.setItem(ALERT_PLACES_KEY, JSON.stringify(places));
  } catch {
    // ignore
  }
}

export function isAlertOn(place: LocationState): boolean {
  return loadAlertPlaces().some((p) => sameLocation(p, place));
}

let vapidCheck: Promise<boolean> | null = null;

/** Whether the server has alerts configured (cached for the page's lifetime). */
export function alertsAvailable(): Promise<boolean> {
  if (!pushSupported()) return Promise.resolve(false);
  vapidCheck ??= getVapidKey().then(() => true, () => false);
  return vapidCheck;
}

function urlBase64ToUint8Array(base64: string): Uint8Array<ArrayBuffer> {
  const padded = (base64 + "=".repeat((4 - (base64.length % 4)) % 4)).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(padded);
  return Uint8Array.from(raw, (c) => c.charCodeAt(0));
}

async function currentSubscription(create: boolean): Promise<PushSubscription | null> {
  const reg = await navigator.serviceWorker.ready;
  const existing = await reg.pushManager.getSubscription();
  if (existing || !create) return existing;
  return reg.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: urlBase64ToUint8Array(await getVapidKey()),
  });
}

async function sync(places: LocationState[]): Promise<void> {
  if (places.length === 0) {
    const sub = await currentSubscription(false);
    if (sub) {
      await unsubscribePush(sub.endpoint);
      await sub.unsubscribe();
    }
    return;
  }
  const sub = await currentSubscription(true);
  if (!sub) return;
  await subscribePush({
    subscription: sub.toJSON(),
    places,
    tz: Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC",
  });
}

/**
 * Ask for permission (MUST be called from a tap handler — iOS requires a user
 * gesture) and turn the bell on for *place*. Returns false if not granted.
 */
export async function enableAlerts(place: LocationState): Promise<boolean> {
  if (!pushSupported()) return false;
  const result = await Notification.requestPermission();
  if (result !== "granted") return false;
  await setAlert(place, true);
  return true;
}

/** Toggle one place's bell and push the full list to the backend. */
export async function setAlert(place: LocationState, on: boolean): Promise<LocationState[]> {
  const others = loadAlertPlaces().filter((p) => !sameLocation(p, place));
  const next = (on ? [place, ...others] : others).slice(0, MAX_SAVED_PLACES);
  await sync(next);
  saveAlertPlaces(next);
  return next;
}
