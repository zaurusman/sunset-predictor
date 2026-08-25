"use client";

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";

/**
 * True when the dark palette is active.
 *
 * Reports light until after mount, which is what the server renders
 * (`defaultTheme="light"`), so colour-carrying SVG never trips a hydration
 * mismatch the way an unguarded `resolvedTheme` read does.
 */
export function useIsDark(): boolean {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  return mounted && resolvedTheme === "dark";
}
