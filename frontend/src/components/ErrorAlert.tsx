"use client";

import { AlertTriangle, CloudOff, RefreshCw } from "lucide-react";
import clsx from "clsx";

interface ErrorAlertProps {
  message: string;
  onRetry?: () => void;
  /**
   * "busy": the weather service is temporarily out of capacity (see
   * isServiceBusy). Nothing is broken and it passes on its own, so it reads
   * as a calm notice rather than a red error.
   */
  variant?: "error" | "busy";
}

export default function ErrorAlert({ message, onRetry, variant = "error" }: ErrorAlertProps) {
  const busy = variant === "busy";
  const Icon = busy ? CloudOff : AlertTriangle;

  return (
    <div
      role={busy ? "status" : "alert"}
      className={clsx(
        "flex items-start gap-3 px-4 py-3 rounded-xl border text-sm",
        busy
          ? "bg-white dark:bg-slate-800/60 border-gray-200 dark:border-slate-700/40"
          : "bg-red-500/10 border-red-500/30"
      )}
    >
      <Icon
        size={16}
        className={clsx(
          "flex-shrink-0 mt-0.5",
          busy ? "text-gray-500 dark:text-slate-400" : "text-red-400"
        )}
      />
      <div className="flex-1 min-w-0">
        <span className={clsx("text-pretty", busy ? "text-gray-700 dark:text-slate-300" : "text-red-300")}>
          {message}
        </span>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className={clsx(
            "flex items-center gap-1 text-xs font-medium flex-shrink-0",
            busy
              ? "text-orange-700 dark:text-orange-400 hover:text-orange-800 dark:hover:text-orange-300"
              : "text-red-400 hover:text-red-300"
          )}
        >
          <RefreshCw size={12} />
          {busy ? "Try again" : "Retry"}
        </button>
      )}
    </div>
  );
}
