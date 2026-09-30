import { Coffee } from "lucide-react";

export const SUPPORT_URL = "https://buymeacoffee.com/afterglowsunset";

/**
 * Quiet support link at the bottom of every tab. Deliberately low-key: people
 * open Afterglow for a daily glance, so this must never compete with the forecast.
 */
export default function SupportFooter() {
  return (
    <footer className="mt-10 pt-5 pb-2 border-t border-gray-200 dark:border-slate-800 flex flex-col items-center gap-2 text-center">
      <p className="text-xs text-gray-500 dark:text-slate-500">
        Afterglow is free and made by one person. If it helped you catch a good one, you can help keep it running.
      </p>
      <a
        href={SUPPORT_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1.5 min-h-[44px] px-4 rounded-full text-sm font-medium text-gray-700 dark:text-slate-300 bg-white dark:bg-slate-800/60 border border-gray-200 dark:border-slate-700/50 hover:border-orange-500/40 transition-colors"
      >
        <Coffee size={14} className="text-orange-500" />
        Buy me a coffee
      </a>
    </footer>
  );
}
