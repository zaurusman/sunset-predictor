/**
 * Browser-side Web Push plumbing: register the worker, ask for permission,
 * and keep the server's idea of this device in sync with the browser's.
 *
 * Everything here is defensive about support. Push is unavailable in a
 * surprising number of real situations — iOS Safari outside an installed
 * PWA, private windows, Firefox with push disabled — and the UI needs to say
 * so rather than offer a toggle that silently does nothing.
 */

import type { LocationState } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

const SW_PATH = "/sw.js";

export interface NotificationConfig {
  enabled: boolean;
  vapid_public_key: string;
  default_threshold: number;
  default_lead_minutes: number;
}

export type PushSupport =
  | { supported: true }
  | { supported: false; reason: string };

/**
 * Why this browser can't do push, in words worth showing a user.
 *
 * The iOS case is the one that matters: Safari supports push ONLY for a site
 * added to the home screen, so telling an iPhone user "not supported" would
 * be wrong — there is something they can do about it.
 */
export function checkSupport(): PushSupport {
  if (typeof window === "undefined") return { supported: false, reason: "" };

  if (!("serviceWorker" in navigator)) {
    return { supported: false, reason: "This browser doesn't support background alerts." };
  }
  if (!("PushManager" in window)) {
    const isIOS =
      /iPad|iPhone|iPod/.test(navigator.userAgent) ||
      // iPadOS 13+ reports itself as a Mac; the touch points give it away.
      (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    if (isIOS) {
      return {
        supported: false,
        reason: "On iPhone, add Afterglow to your Home Screen first — Safari only allows alerts for installed apps.",
      };
    }
    return { supported: false, reason: "This browser doesn't support push notifications." };
  }
  if (!("Notification" in window)) {
    return { supported: false, reason: "This browser doesn't support notifications." };
  }
  return { supported: true };
}

/** Ask the server whether alerts are configured, and for its VAPID key. */
export async function fetchConfig(): Promise<NotificationConfig | null> {
  try {
    const res = await fetch(`${API_BASE}/notifications/config`);
    if (!res.ok) return null;
    return (await res.json()) as NotificationConfig;
  } catch {
    return null;
  }
}

/**
 * The VAPID key travels as base64url text but PushManager wants raw bytes.
 * Getting this wrong fails at subscribe() with an opaque DOMException, so it
 * is worth doing explicitly rather than inline.
 */
function urlBase64ToUint8Array(base64String: string): Uint8Array<ArrayBuffer> {
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = window.atob(base64);
  // Allocate the ArrayBuffer explicitly: `new Uint8Array(length)` widens to
  // ArrayBufferLike, which applicationServerKey (a BufferSource) rejects.
  const output = new Uint8Array(new ArrayBuffer(raw.length));
  for (let i = 0; i < raw.length; i += 1) output[i] = raw.charCodeAt(i);
  return output;
}

async function getRegistration(): Promise<ServiceWorkerRegistration> {
  const existing = await navigator.serviceWorker.getRegistration(SW_PATH);
  if (existing) return existing;
  return navigator.serviceWorker.register(SW_PATH);
}

/** The browser's current subscription, if it already has one. */
export async function getExistingSubscription(): Promise<PushSubscription | null> {
  const support = checkSupport();
  if (!support.supported) return null;
  try {
    const reg = await navigator.serviceWorker.getRegistration(SW_PATH);
    if (!reg) return null;
    return await reg.pushManager.getSubscription();
  } catch {
    return null;
  }
}

export class PermissionDeniedError extends Error {
  constructor() {
    // Once denied, the browser will not re-prompt — only the user can undo it
    // in site settings, so the message has to say where to go.
    super(
      "Notifications are blocked for this site. Re-enable them in your browser's site settings, then try again.",
    );
    this.name = "PermissionDeniedError";
  }
}

/**
 * Turn alerts on for this browser at this place.
 *
 * Subscribes with the push service, then registers the result with our
 * backend. If the backend rejects it, the browser subscription is rolled back
 * so the two never disagree about whether alerts are on.
 */
export async function subscribe(params: {
  location: LocationState;
  vapidPublicKey: string;
  threshold: number;
  leadMinutes: number;
}): Promise<void> {
  const permission = await Notification.requestPermission();
  if (permission !== "granted") throw new PermissionDeniedError();

  const reg = await getRegistration();
  // Wait for activation: subscribing against an installing worker throws.
  await navigator.serviceWorker.ready;

  const existing = await reg.pushManager.getSubscription();
  const subscription =
    existing ??
    (await reg.pushManager.subscribe({
      // Required by every browser that implements push — an unencrypted
      // subscription is not an option.
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(params.vapidPublicKey),
    }));

  const json = subscription.toJSON();
  const res = await fetch(`${API_BASE}/notifications/subscribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      endpoint: subscription.endpoint,
      keys: { p256dh: json.keys?.p256dh, auth: json.keys?.auth },
      latitude: params.location.latitude,
      longitude: params.location.longitude,
      location_name: params.location.name,
      threshold: params.threshold,
      lead_minutes: params.leadMinutes,
    }),
  });

  if (!res.ok) {
    // Only unsubscribe what we just created: an existing subscription may
    // belong to a working registration we should not tear down.
    if (!existing) await subscription.unsubscribe().catch(() => {});
    let detail = res.statusText;
    try {
      detail = (await res.json())?.detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
}

/** Turn alerts off — on the server first, then in the browser. */
export async function unsubscribe(): Promise<void> {
  const subscription = await getExistingSubscription();
  if (!subscription) return;

  // Server first: if this fails we keep the browser subscription, so the UI
  // still shows "on" and the state stays truthful. The reverse order would
  // leave the server pushing to an endpoint that no longer exists.
  await fetch(`${API_BASE}/notifications/unsubscribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ endpoint: subscription.endpoint }),
  });

  await subscription.unsubscribe().catch(() => {});
}

/**
 * Does the server still know about this browser?
 *
 * Render's free tier wipes the subscription store on every redeploy, which
 * leaves a browser holding a live subscription for a server that has
 * forgotten it — alerts silently stop with the UI still saying "on". This is
 * how the UI catches that and offers to re-register.
 */
export async function serverKnowsUs(subscription: PushSubscription): Promise<boolean> {
  try {
    const res = await fetch(
      `${API_BASE}/notifications/status?endpoint=${encodeURIComponent(subscription.endpoint)}`,
    );
    return res.ok;
  } catch {
    // A network failure is not evidence the server forgot us; don't cry wolf.
    return true;
  }
}
