"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch — only render after mount
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="w-11 h-11 flex-shrink-0" />;

  const isDark = theme === "dark";

  return (
    <button
      onClick={() => setTheme(isDark ? "light" : "dark")}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      className="m-press w-11 h-11 flex-shrink-0 rounded-xl flex items-center justify-center text-gray-600 hover:text-orange-600 bg-white/70 hover:bg-white/90 border border-white/80 backdrop-blur-md dark:text-slate-400 dark:hover:text-orange-400 dark:bg-slate-900/60 dark:hover:bg-slate-800/70 dark:border-slate-700/50 transition-colors"
    >
      {isDark ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}
