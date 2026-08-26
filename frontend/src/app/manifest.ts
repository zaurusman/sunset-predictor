import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Afterglow",
    short_name: "Afterglow",
    description: "How beautiful will tonight's sunset be? Get a score, reasons, and the best time to watch.",
    start_url: "/",
    display: "standalone",
    background_color: "#04050A",
    theme_color: "#04050A",
    icons: [
      {
        src: "/icon-192.png",
        sizes: "192x192",
        type: "image/png",
      },
      {
        src: "/icon-512.png",
        sizes: "512x512",
        type: "image/png",
      },
    ],
  };
}
