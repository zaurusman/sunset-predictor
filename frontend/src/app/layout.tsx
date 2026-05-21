import type { Metadata } from "next";
import "./globals.css";
import ThemeProvider from "@/components/ThemeProvider";

export const metadata: Metadata = {
  title: "Afterglow",
  description: "How beautiful will tonight's sunset be? Get a score, reasons, and the best time to watch.",
  keywords: ["sunset", "weather", "forecast", "beauty score"],
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
          {/* Subtle ambient gradient overlay */}
          <div
            className="fixed inset-0 pointer-events-none z-0"
            style={{
              background:
                "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(249,115,22,0.06) 0%, transparent 60%)",
            }}
          />
          <div className="relative z-10">{children}</div>
        </ThemeProvider>
      </body>
    </html>
  );
}
