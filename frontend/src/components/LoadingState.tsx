"use client";

interface LoadingStateProps {
  message?: string;
}

export default function LoadingState({ message = "Fetching forecast…" }: LoadingStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16">
      {/* Animated sunset rings */}
      <div className="relative w-16 h-16">
        <div className="absolute inset-0 rounded-full border-2 border-orange-500/20 animate-ping" />
        <div className="absolute inset-2 rounded-full border-2 border-orange-400/40 animate-pulse" />
        <div className="absolute inset-4 rounded-full bg-orange-500/30 animate-pulse" />
      </div>
      <p className="text-slate-400 text-sm">{message}</p>
    </div>
  );
}
