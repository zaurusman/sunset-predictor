"use client";

import Image from "next/image";
import Link from "next/link";
import { ChevronDown, MapPin } from "lucide-react";
import type { LocationState } from "@/lib/types";
import { useScrolled } from "@/lib/motion";
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

/** Tab order, so a link knows whether it moves the page forward or back. */
const ORDER: AppTab[] = ["tonight", "forecast", "heatmap"];

/** Frosted surfaces that float over the sky. */
const FROST = "bg-white/70 dark:bg-slate-900/60 backdrop-blur-md border border-white/80 dark:border-slate-700/50";

export default function AppNav({ location, active, onChangeLocation }: AppNavProps) {
  const scrolled = useScrolled();
  return (
    // Named so the view transition keeps the header still while the page
    // underneath it slides; it frosts once content scrolls beneath it.
    <div
      style={{ viewTransitionName: "app-header" }}
      data-scrolled={scrolled}
      className="sticky top-0 z-30 -mx-4 px-4 pt-2 pb-3 mb-3 flex flex-col gap-3 transition-[background-color,box-shadow] duration-200 data-[scrolled=true]:bg-slate-50/75 dark:data-[scrolled=true]:bg-slate-950/70 data-[scrolled=true]:backdrop-blur-lg data-[scrolled=true]:shadow-[0_1px_0_rgba(15,23,42,.06)]"
    >
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
              className={`m-press flex items-center gap-1.5 h-11 max-w-[45%] px-3 rounded-full ${FROST} text-gray-700 dark:text-slate-300 text-sm font-medium hover:border-orange-500/40 transition-colors`}
            >
              <MapPin size={12} className="flex-shrink-0" />
              <span className="truncate">{location.name}</span>
              <ChevronDown size={12} className="flex-shrink-0 text-gray-400 dark:text-slate-500" />
            </button>
          ) : (
            <span className={`flex items-center gap-1.5 h-11 max-w-[45%] px-3 rounded-full ${FROST} text-gray-700 dark:text-slate-300 text-sm font-medium`}>
              <MapPin size={13} className="flex-shrink-0" />
              <span className="truncate">{location.name}</span>
            </span>
          ))}

        <ThemeToggle />
      </div>

      <nav className={`flex gap-1 p-1 rounded-xl ${FROST}`}>
        {TABS.map((tab) => {
          const isActive = tab.id === active;
          return (
            <Link
              key={tab.id}
              href={hrefFor(tab.href, location)}
              aria-current={isActive ? "page" : undefined}
              transitionTypes={[ORDER.indexOf(tab.id) > ORDER.indexOf(active) ? "tab-forward" : "tab-back"]}
              className={
                isActive
                  ? "relative flex-1 flex items-center justify-center min-h-[44px] rounded-lg text-sm font-semibold text-gray-900 dark:text-white"
                  : "m-press relative flex-1 flex items-center justify-center min-h-[44px] rounded-lg text-sm font-medium text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              }
            >
              {/* One shared highlight: the transition slides it to the new tab. */}
              {isActive && (
                <span
                  aria-hidden="true"
                  style={{ viewTransitionName: "tab-pill" }}
                  className="absolute inset-0 rounded-lg bg-white dark:bg-slate-800 shadow-sm"
                />
              )}
              <span className="relative">{tab.label}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
