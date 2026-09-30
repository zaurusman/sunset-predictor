"use client";

import { useEffect, useState } from "react";
import { Bell, Smartphone, X } from "lucide-react";
import type { LocationState } from "@/lib/types";
import {
  canShowPrompt,
  captureInstallEvent,
  detectPlatform,
  dismissPrompt,
  hasInstallEvent,
  isStandalone,
  onInstallEvent,
  recordVisit,
  triggerInstall,
  type PromptKind,
} from "@/lib/install";
import { alertsAvailable, enableAlerts, loadAlertPlaces, permission } from "@/lib/push";
import IosInstallSheet from "./IosInstallSheet";

interface Props {
  location: LocationState;
  onAlertsChanged?: () => void;
}

/**
 * One quiet card under the verdict. Browser tab → "add to Home Screen";
 * installed app → "turn on Epic alerts". Never on a first visit, never above
 * the answer, snoozed for two weeks when dismissed.
 */
export default function InstallPrompt({ location, onAlertsChanged }: Props) {
  const [kind, setKind] = useState<PromptKind | null>(null);
  const [iosSheet, setIosSheet] = useState(false);
  const [busy, setBusy] = useState(false);
  const [denied, setDenied] = useState(false);

  useEffect(() => {
    captureInstallEvent();
    const visits = recordVisit();
    const platform = detectPlatform();
    let cancelled = false;

    const decide = async () => {
      if (platform === "unsupported") return setKind(null);

      if (isStandalone()) {
        const wantsAlerts =
          permission() === "default" &&
          loadAlertPlaces().length === 0 &&
          canShowPrompt("alerts") &&
          (await alertsAvailable());
        if (!cancelled) setKind(wantsAlerts ? "alerts" : null);
        return;
      }

      const installable = platform === "ios" || hasInstallEvent();
      if (!cancelled) setKind(visits >= 2 && installable && canShowPrompt("install") ? "install" : null);
    };

    void decide();
    // Chromium may fire beforeinstallprompt after mount.
    const off = onInstallEvent(() => void decide());
    return () => {
      cancelled = true;
      off();
    };
  }, []);

  if (!kind) return null;

  const dismiss = () => {
    dismissPrompt(kind);
    setKind(null);
  };

  const act = async () => {
    if (kind === "install") {
      if (detectPlatform() === "ios") return setIosSheet(true);
      const accepted = await triggerInstall();
      if (accepted) setKind(null);
      return;
    }
    setBusy(true);
    try {
      const ok = await enableAlerts(location);
      if (ok) {
        setKind(null);
        onAlertsChanged?.();
      } else {
        setDenied(permission() === "denied");
      }
    } catch {
      setDenied(false);
    } finally {
      setBusy(false);
    }
  };

  const isInstall = kind === "install";

  return (
    <>
      <div className="relative flex items-start gap-3 p-4 rounded-2xl bg-gradient-to-br from-orange-50 to-rose-50 dark:from-orange-500/10 dark:to-rose-500/10 border border-orange-200/70 dark:border-orange-500/20">
        <span className="w-10 h-10 flex-shrink-0 rounded-xl flex items-center justify-center bg-white/80 dark:bg-slate-900/60 text-orange-600 dark:text-orange-400">
          {isInstall ? <Smartphone size={18} /> : <Bell size={18} />}
        </span>
        <div className="flex-1 min-w-0 pr-6">
          <p className="text-sm font-semibold text-gray-900 dark:text-white">
            {isInstall ? "Never miss an epic sunset" : `Epic sunset alerts for ${location.name}`}
          </p>
          <p className="mt-0.5 text-xs text-gray-700 dark:text-slate-300">
            {isInstall
              ? "Add Afterglow to your Home Screen and we'll ping you when one's coming."
              : denied
                ? "Notifications are blocked — enable them for Afterglow in Settings."
                : "One ping, about 4 hours before sunset, only when it's going to be Epic."}
          </p>
          {!denied && (
            <button
              onClick={act}
              disabled={busy}
              className="mt-3 inline-flex items-center min-h-[40px] px-4 rounded-full text-sm font-semibold text-white bg-orange-600 hover:bg-orange-700 disabled:opacity-60 transition-colors"
            >
              {isInstall ? "Add to Home Screen" : busy ? "Turning on…" : "Turn on alerts"}
            </button>
          )}
        </div>
        <button
          onClick={dismiss}
          aria-label="Not now"
          className="absolute top-2 right-2 w-9 h-9 rounded-full flex items-center justify-center text-gray-500 dark:text-slate-400 hover:bg-white/60 dark:hover:bg-slate-800/60"
        >
          <X size={15} />
        </button>
      </div>
      <IosInstallSheet open={iosSheet} onClose={() => setIosSheet(false)} />
    </>
  );
}
