"use client";

import { ViewTransition } from "react";

/**
 * Directional slide between tabs; browser back/forward (no type) swaps instantly.
 * Must wrap the page's outermost element: a DOM node above the boundary
 * suppresses its enter/exit.
 */
export default function PageTransition({ children }: { children: React.ReactNode }) {
  const map = { "tab-forward": "tab-forward", "tab-back": "tab-back", default: "none" } as const;
  return (
    <ViewTransition enter={map} exit={map} default="none">
      {children}
    </ViewTransition>
  );
}
