"use client";

import { useEffect, useState } from "react";
import { Brain, CheckCircle, XCircle } from "lucide-react";
import { getModelInfo } from "@/lib/api";

export default function ModelInfoPanel() {
  const [info, setInfo] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="rounded-xl bg-slate-800/50 border border-slate-700/40 p-4 animate-pulse">
        <div className="h-4 bg-slate-700 rounded w-1/3" />
      </div>
    );
  }

  if (!info) return null;

  const loaded = info.loaded as boolean;

  return (
    <div className="rounded-xl bg-slate-800/50 border border-slate-700/40 p-4 text-sm">
      <div className="flex items-center gap-2 mb-3">
        <Brain size={14} className="text-slate-400" />
        <span className="text-slate-400 text-xs uppercase tracking-wider">ML Model</span>
        {loaded ? (
          <CheckCircle size={12} className="text-emerald-400 ml-auto" />
        ) : (
          <XCircle size={12} className="text-amber-400 ml-auto" />
        )}
      </div>

      {loaded ? (
        <div className="space-y-1.5 text-xs">
          <Row label="Trained" value={String(info.trained_at ?? "—").slice(0, 10)} />
          <Row label="Train / Val" value={`${info.n_train} / ${info.n_val}`} />
          <Row label="RMSE" value={String(info.rmse ?? "—")} />
          <Row label="Spearman r" value={String(info.spearman_r ?? "—")} />
          <Row label="Blend α" value={String(info.blend_alpha ?? "—")} />
        </div>
      ) : (
        <p className="text-slate-500 text-xs">
          No trained model — running in physics-only mode.
          <br />
          Run <code className="text-orange-400">python scripts/train_model.py</code> to train.
        </p>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-300 font-mono">{value}</span>
    </div>
  );
}
