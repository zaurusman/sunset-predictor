"use client";

import { useEffect } from "react";
import { PlusSquare, Share, X } from "lucide-react";

interface Props {
  open: boolean;
  onClose: () => void;
}

/** Safari has no install prompt, so we show the two taps it takes. */
export default function IosInstallSheet({ open, onClose }: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const steps = [
    { icon: <Share size={18} />, text: <>Tap <strong>Share</strong> in Safari&apos;s toolbar</> },
    { icon: <PlusSquare size={18} />, text: <>Choose <strong>Add to Home Screen</strong></> },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center">
      <button
        aria-label="Close"
        onClick={onClose}
        className="absolute inset-0 bg-slate-900/30 dark:bg-slate-950/60"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Add Afterglow to your Home Screen"
        className="relative w-full max-w-2xl bg-white dark:bg-slate-900 rounded-t-3xl border-t border-x border-gray-200 dark:border-slate-700/50 px-5 pt-4 flex flex-col gap-4 shadow-2xl animate-slide-up"
        style={{ paddingBottom: "max(2rem, env(safe-area-inset-bottom))" }}
      >
        <div className="flex items-center gap-3">
          <h2 className="flex-1 text-lg font-bold tracking-tight text-gray-900 dark:text-white">
            Add to Home Screen
          </h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="w-11 h-11 rounded-full flex items-center justify-center text-gray-600 dark:text-slate-400 hover:bg-gray-100 dark:hover:bg-slate-800"
          >
            <X size={16} />
          </button>
        </div>
        <ol className="flex flex-col gap-3">
          {steps.map((s, i) => (
            <li key={i} className="flex items-center gap-3 text-sm text-gray-800 dark:text-slate-200">
              <span className="w-9 h-9 flex-shrink-0 rounded-xl flex items-center justify-center bg-orange-50 dark:bg-orange-500/10 text-orange-600 dark:text-orange-400">
                {s.icon}
              </span>
              <span>
                <span className="text-gray-500 dark:text-slate-500 mr-1">{i + 1}.</span>
                {s.text}
              </span>
            </li>
          ))}
        </ol>
        <p className="text-xs text-gray-600 dark:text-slate-400">
          Then open Afterglow from your Home Screen and turn on Epic sunset alerts.
        </p>
      </div>
    </div>
  );
}
