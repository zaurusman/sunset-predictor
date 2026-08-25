"use client";

import Image from "next/image";
import Link from "next/link";
import { ChevronDown, MapPin } from "lucide-react";
import type { LocationState } from "@/lib/types";
import ThemeToggle from "./ThemeToggle";

export type AppTab = "tonight" | "forecast" | "heatmap";

interface AppNavProps {
  location: LocationState | null;
  active: AppTab;
  /** Opens the location sheet. Omit to render the place as a static label. */
  onChangeLocation?: () => void;
}

const TABS: { id: AppTab; label: string; href: string }[] = [
  { id: "tonight", label: "Tonight", href: "/" },
  { id: "forecast", label: "7 days", href: "/forecast" },
  { id: "heatmap", label: "History", href: "/heatmap" },
];

/**
 * Builds a tab href that carries the current place, so moving between views
 * never loses it. The home page previously received these params and ignored
 * them, which is why the back arrow dropped you onto the empty state.
 */
function hrefFor(base: string, location: LocationState | null): string {
  if (!location) return base;
  const params = new URLSearchParams({
    lat: String(location.latitude),
    lon: String(location.longitude),
    name: location.name,
  });
  return `${base}?${params.toString()}`;
}

export default function AppNav({ location, active, onChangeLocation }: AppNavProps) {
  return (
    <div className="flex flex-col gap-3 mb-5">
      <div className="flex items-center gap-2">
        <Link
          href={hrefFor("/", location)}
          className="flex-1 min-w-0 flex items-center min-h-[44px]"
          aria-label="Afterglow home"
        >
          <Image
            src="/logo.png"
            alt="Afterglow"
            width={168}
            height={28}
            className="h-7 w-auto"
            priority
          />
        </Link>

        {location &&
          (onChangeLocation ? (
            <button
              onClick={onChangeLocation}
              className="flex items-center gap-1.5 h-11 max-w-[45%] px-3 rounded-full bg-white dark:bg-slate-800/60 border border-gray-200 dark:border-slate-700/50 text-gray-700 dark:text-slate-300 text-sm font-medium hover:border-orange-500/40 transition-colors"
            >
              <MapPin size={12} className="flex-shrink-0" />
              <span className="truncate">{location.name}</span>
              <ChevronDown size={12} className="flex-shrink-0 text-gray-400 dark:text-slate-500" />
            </button>
          ) : (
            <span className="flex items-center gap-1.5 h-11 max-w-[45%] px-3 rounded-full bg-white dark:bg-slate-800/60 border border-gray-200 dark:border-slate-700/50 text-gray-700 dark:text-slate-300 text-sm font-medium">
              <MapPin size={13} className="flex-shrink-0" />
              <span className="truncate">{location.name}</span>
            </span>
          ))}

        <ThemeToggle />
      </div>

      <nav className="flex gap-1 p-1 rounded-xl bg-gray-100 dark:bg-slate-900/70 border border-gray-200 dark:border-slate-700/40">
        {TABS.map((tab) => {
          const isActive = tab.id === active;
          return (
            <Link
              key={tab.id}
              href={hrefFor(tab.href, location)}
              aria-current={isActive ? "page" : undefined}
              className={
                isActive
                  ? "flex-1 flex items-center justify-center min-h-[44px] rounded-lg text-sm font-semibold bg-white dark:bg-slate-800 text-gray-900 dark:text-white shadow-sm"
                  : "flex-1 flex items-center justify-center min-h-[44px] rounded-lg text-sm font-medium text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              }
            >
              {tab.label}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
