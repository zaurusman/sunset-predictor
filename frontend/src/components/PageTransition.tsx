"use client";

import { ViewTransition } from "react";

/**
 * Directional slide between tabs; browser back/forward (no type) swaps instantly.
 * Must wrap the page's outermost element: a DOM node above the boundary
 * suppresses its enter/exit.
 *
 * The leaving and arriving pages get different classes because Safari also
 * draws the leaving page as a "new" image; styled like an arriving page, it
 * faded back in under the new page until the transition ended.
 */
const ENTER = { "tab-forward": "page-in-forward", "tab-back": "page-in-back", default: "none" } as const;
const EXIT = { "tab-forward": "page-out-forward", "tab-back": "page-out-back", default: "none" } as const;

export default function PageTransition({ children }: { children: React.ReactNode }) {
  return (
    <ViewTransition enter={ENTER} exit={EXIT} default="none">
      {children}
    </ViewTransition>
  );
}
