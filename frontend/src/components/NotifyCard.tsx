"use client";

import { useCallback, useEffect, useState } from "react";
import { Bell, BellOff, Loader2 } from "lucide-react";
import type { LocationState } from "@/lib/types";
import {
  checkSupport,
  fetchConfig,
  getExistingSubscription,
  PermissionDeniedError,
  serverKnowsUs,
  subscribe,
  unsubscribe,
  type NotificationConfig,
} from "@/lib/push";

interface NotifyCardProps {
  location: LocationState;
}

/** What the alert is worth interrupting someone for. */
const THRESHOLDS = [
  { value: 80, label: "Only exceptional", hint: "Epic skies (80+)" },
  { value: 70, label: "Worth heading out", hint: "The app's own bar (70+)" },
  { value: 50, label: "Anything decent", hint: "Most evenings (50+)" },
];

const LEADS = [
  { value: 60, label: "1h before" },
  { value: 120, label: "2h before" },
  { value: 180, label: "3h before" },
];

type Status = "loading" | "unavailable" | "off" | "on" | "stale";

export default function NotifyCard({ location }: NotifyCardProps) {
  const [status, setStatus] = useState<Status>("loading");
  const [config, setConfig] = useState<NotificationConfig | null>(null);
  const [reason, setReason] = useState<string>("");
  const [threshold, setThreshold] = useState(70);
  const [leadMinutes, setLeadMinutes] = useState(120);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const support = checkSupport();
      const cfg = await fetchConfig();
      if (cancelled) return;

      // Server off, or browser can't: either way there is no working toggle
      // to offer. The iOS reason is worth showing; the rest is not.
      if (!cfg?.enabled) {
        setStatus("unavailable");
        setReason("");
        return;
      }
      if (!support.supported) {
        setStatus("unavailable");
        setReason(support.reason);
        return;
      }

      setConfig(cfg);
      setThreshold(cfg.default_threshold);
      setLeadMinutes(cfg.default_lead_minutes);

      const existing = await getExistingSubscription();
      if (cancelled) return;
      if (!existing) {
        setStatus("off");
        return;
      }

      // A live browser subscription the server has forgotten looks identical
      // to a working one from here — ask before claiming alerts are on.
      const known = await serverKnowsUs(existing);
      if (cancelled) return;
      setStatus(known ? "on" : "stale");
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const turnOn = useCallback(async () => {
    if (!config) return;
    setBusy(true);
    setError(null);
    try {
      await subscribe({
        location,
        vapidPublicKey: config.vapid_public_key,
        threshold,
        leadMinutes,
      });
      setStatus("on");
    } catch (err) {
      setError(
        err instanceof PermissionDeniedError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Couldn't turn on alerts.",
      );
    } finally {
      setBusy(false);
    }
  }, [config, location, threshold, leadMinutes]);

  const turnOff = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      await unsubscribe();
      setStatus("off");
    } catch {
      setError("Couldn't turn off alerts. Try again.");
    } finally {
      setBusy(false);
    }
  }, []);

  // Loading and unavailable-with-nothing-to-say both render nothing: a card
  // that flashes in and out on every load is worse than no card.
  if (status === "loading") return null;
  if (status === "unavailable" && !reason) return null;

  if (status === "unavailable") {
    return (
      <div className="rounded-2xl bg-white dark:bg-slate-800/60 border border-gray-200 dark:border-slate-700/40 p-4">
        <p className="text-sm text-gray-600 dark:text-slate-400">{reason}</p>
      </div>
    );
  }

  const isOn = status === "on";

  return (
    <div className="rounded-2xl bg-white dark:bg-slate-800/60 border border-gray-200 dark:border-slate-700/40 p-4 flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            {isOn ? <Bell size={15} /> : <BellOff size={15} />}
            Sunset alerts
          </h3>
          {/* Only the place name can run long enough to need truncating; the
              fixed copy must wrap, or a 375px screen cuts it mid-sentence. */}
          {isOn ? (
            <p className="text-xs text-gray-600 dark:text-slate-400 mt-0.5 truncate">
              On for {location.name}
            </p>
          ) : (
            <p className="text-xs text-gray-600 dark:text-slate-400 mt-0.5">
              {status === "stale"
                ? "Alerts stopped — tap to switch them back on"
                : "Get a heads-up when tonight looks good"}
            </p>
          )}
        </div>

        {/* 14px label, so the fill needs 4.5:1 behind it. White on orange-600
            is 3.56:1 and white on orange-500 is worse — hence orange-700 in
            light, and dark text on bright orange in dark. */}
        <button
          onClick={isOn ? turnOff : turnOn}
          disabled={busy}
          aria-pressed={isOn}
          className={`flex items-center justify-center gap-2 h-11 px-4 flex-shrink-0 rounded-xl text-sm font-semibold transition-colors disabled:opacity-60 ${
            isOn
              ? "bg-gray-100 dark:bg-slate-700/60 text-gray-700 dark:text-slate-300 hover:text-gray-900 dark:hover:text-white"
              : "bg-orange-700 text-white hover:bg-orange-800 dark:bg-orange-400 dark:text-slate-950 dark:hover:bg-orange-300"
          }`}
        >
          {busy && <Loader2 size={14} className="animate-spin" />}
          {isOn ? "Turn off" : status === "stale" ? "Reconnect" : "Turn on"}
        </button>
      </div>

      {/* Settings are only meaningful before subscribing — changing them after
          would need a re-subscribe, so we ask up front instead. */}
      {!isOn && (
        <div className="flex flex-col gap-2 pt-1">
          <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-slate-400">
            <span className="w-20 flex-shrink-0">Tell me about</span>
            <select
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="flex-1 min-w-0 h-11 px-3 rounded-xl bg-gray-100/80 dark:bg-slate-800/80 border border-gray-200 dark:border-slate-700 text-base sm:text-sm text-gray-900 dark:text-white focus:outline-none focus:border-orange-500/60"
            >
              {THRESHOLDS.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label} — {t.hint}
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-slate-400">
            <span className="w-20 flex-shrink-0">Warn me</span>
            <select
              value={leadMinutes}
              onChange={(e) => setLeadMinutes(Number(e.target.value))}
              className="flex-1 min-w-0 h-11 px-3 rounded-xl bg-gray-100/80 dark:bg-slate-800/80 border border-gray-200 dark:border-slate-700 text-base sm:text-sm text-gray-900 dark:text-white focus:outline-none focus:border-orange-500/60"
            >
              {LEADS.map((l) => (
                <option key={l.value} value={l.value}>
                  {l.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}

      {error && (
        <p className="text-xs text-red-700 dark:text-red-400" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
