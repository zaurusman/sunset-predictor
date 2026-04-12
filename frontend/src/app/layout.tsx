import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sunset Predictor",
  description: "How beautiful will tonight's sunset be? Get a score, reasons, and the best time to watch.",
  keywords: ["sunset", "weather", "forecast", "beauty score"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 antialiased">
        {/* Subtle ambient gradient overlay */}
        <div
          className="fixed inset-0 pointer-events-none z-0"
          style={{
            background:
              "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(249,115,22,0.08) 0%, transparent 60%)",
          }}
        />
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  );
}
