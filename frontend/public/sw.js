/**
 * Afterglow service worker — receives sunset alerts and opens the app.
 *
 * Deliberately minimal: no offline caching, no fetch handler. A cache layer
 * here would shadow Next.js's own asset hashing and serve stale bundles after
 * a deploy, and the app already renders from a localStorage cache on boot.
 * The only job is push.
 */

self.addEventListener("install", () => {
  // Take over immediately rather than waiting for every tab to close —
  // otherwise a first-time subscriber's worker stays "waiting" and the first
  // alert has nothing active to wake.
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("push", (event) => {
  // A push with no payload still has to show something: browsers revoke push
  // permission from workers that receive a push and display nothing.
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch {
    data = {};
  }

  const title = data.title || "Tonight's sunset";
  const options = {
    body: data.body || "Conditions look worth a look.",
    icon: "/icon-192.png",
    badge: "/badge-72.png",
    // Same tag every time, so a re-send replaces the old alert instead of
    // stacking two notifications about the same evening.
    tag: "afterglow-sunset",
    renotify: true,
    data: { url: data.url || "/" },
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const target = event.notification.data?.url || "/";

  event.waitUntil(
    self.clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then((clients) => {
        // Reuse an open Afterglow tab if there is one — focusing beats piling
        // up a new tab every evening.
        for (const client of clients) {
          if (client.url.includes(self.location.origin) && "focus" in client) {
            client.navigate(target);
            return client.focus();
          }
        }
        return self.clients.openWindow(target);
      }),
  );
});
