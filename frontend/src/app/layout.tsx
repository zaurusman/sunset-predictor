import type { Metadata, Viewport } from "next";
import "./globals.css";
import ThemeProvider from "@/components/ThemeProvider";
import ServiceWorkerRegistrar from "@/components/ServiceWorkerRegistrar";
import SkyProvider from "@/components/sky/SkyProvider";

export const metadata: Metadata = {
  title: "Afterglow",
  description: "How beautiful will tonight's sunset be? Get a score, reasons, and the best time to watch.",
  keywords: ["sunset", "weather", "forecast", "beauty score"],
  appleWebApp: {
    title: "Afterglow",
    capable: true,
    statusBarStyle: "black-translucent",
  },
};

export const viewport: Viewport = {
  themeColor: "#04050A",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen antialiased">
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false}>
          <ServiceWorkerRegistrar />
          <SkyProvider>
            <div className="relative z-10">{children}</div>
          </SkyProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
