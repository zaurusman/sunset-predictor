"use client";

import { useEffect } from "react";

/** Registers /sw.js once per page load. Renders nothing. */
export default function ServiceWorkerRegistrar() {
  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;
    navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => {
      // Push is a nice-to-have; never surface SW failures to the user.
    });
  }, []);
  return null;
}
