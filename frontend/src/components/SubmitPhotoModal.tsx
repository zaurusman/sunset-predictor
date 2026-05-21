"use client";

import { useCallback, useRef, useState } from "react";
import { Camera, CheckCircle, Upload, X } from "lucide-react";
import { submitPhoto } from "@/lib/api";

interface Props {
  latitude: number;
  longitude: number;
  locationName: string;
  defaultDate: string; // "YYYY-MM-DD"
  onClose: () => void;
}

type Status = "idle" | "submitting" | "success" | "error";

export default function SubmitPhotoModal({
  latitude,
  longitude,
  locationName,
  defaultDate,
  onClose,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [photoDate, setPhotoDate] = useState(defaultDate);
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f && f.type.startsWith("image/")) handleFile(f);
    },
    [handleFile]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleSubmit = async () => {
    if (!file) return;
    setStatus("submitting");
    setErrorMsg("");
    try {
      await submitPhoto({
        photo: file,
        latitude,
        longitude,
        photoDate,
        locationName,
        userMessage: message,
      });
      setStatus("success");
    } catch (err) {
      setStatus("error");
      setErrorMsg(err instanceof Error ? err.message : "Something went wrong.");
    }
  };

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose]
  );

  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm px-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
      onKeyDown={handleKeyDown}
      role="dialog"
      aria-modal="true"
      aria-label="Submit sunset photo"
      tabIndex={-1}
    >
      <div className="relative w-full max-w-md bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-700/60 rounded-2xl shadow-2xl p-6 flex flex-col gap-5">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 dark:text-slate-500 hover:text-gray-600 dark:hover:text-slate-300 transition-colors"
          aria-label="Close"
        >
          <X size={18} />
        </button>

        {/* Header */}
        <div className="flex items-center gap-2.5">
          <Camera size={20} className="text-orange-400" />
          <h2 className="text-gray-900 dark:text-white font-semibold text-lg">Share your sunset</h2>
        </div>

        {status === "success" ? (
          <div className="flex flex-col items-center gap-4 py-6 text-center">
            <CheckCircle size={48} className="text-orange-400" />
            <p className="text-gray-900 dark:text-white font-medium">Photo submitted!</p>
            <p className="text-gray-500 dark:text-slate-400 text-sm">
              Thanks for sharing. Your photo and the stats from that day have been sent.
            </p>
            <button
              onClick={onClose}
              className="mt-2 px-6 py-2 rounded-xl bg-orange-500 hover:bg-orange-400 text-white text-sm font-medium transition-colors"
            >
              Done
            </button>
          </div>
        ) : (
          <>
            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              className={`relative cursor-pointer rounded-xl border-2 border-dashed transition-colors flex flex-col items-center justify-center overflow-hidden
                ${dragging
                  ? "border-orange-400 bg-orange-500/10"
                  : "border-gray-300 dark:border-slate-600 hover:border-gray-400 dark:hover:border-slate-500 bg-gray-100/40 dark:bg-slate-800/40"
                }
                ${preview ? "h-48" : "h-36"}
              `}
            >
              {preview ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={preview}
                  alt="Preview"
                  className="h-full w-full object-cover"
                />
              ) : (
                <div className="flex flex-col items-center gap-2 text-gray-400 dark:text-slate-500 pointer-events-none select-none">
                  <Upload size={24} />
                  <span className="text-sm">Click or drag a photo here</span>
                  <span className="text-xs text-gray-300 dark:text-slate-600">JPEG · PNG · WebP · HEIC — max 10 MB</span>
                </div>
              )}
              {preview && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/30 opacity-0 hover:opacity-100 transition-opacity">
                  <span className="text-white text-xs font-medium">Change photo</span>
                </div>
              )}
            </div>
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/heic,image/heif"
              className="hidden"
              onChange={handleInputChange}
            />

            {/* Date */}
            <div className="flex flex-col gap-1.5">
              <label className="text-gray-400 dark:text-slate-400 text-xs uppercase tracking-wider">
                Date taken
              </label>
              <input
                type="date"
                value={photoDate}
                onChange={(e) => setPhotoDate(e.target.value)}
                max={new Date().toISOString().slice(0, 10)}
                className="w-full bg-gray-100/60 dark:bg-slate-800/60 border border-gray-200 dark:border-slate-700/50 rounded-lg px-3 py-2 text-gray-900 dark:text-white text-sm focus:outline-none focus:border-orange-500/50 transition-colors"
              />
            </div>

            {/* Location (read-only) */}
            <div className="flex flex-col gap-1.5">
              <label className="text-gray-400 dark:text-slate-400 text-xs uppercase tracking-wider">
                Location
              </label>
              <div className="w-full bg-gray-100/40 dark:bg-slate-800/40 border border-gray-200/30 dark:border-slate-700/30 rounded-lg px-3 py-2 text-gray-600 dark:text-slate-300 text-sm truncate">
                {locationName || `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`}
              </div>
            </div>

            {/* Optional message */}
            <div className="flex flex-col gap-1.5">
              <label className="text-gray-400 dark:text-slate-400 text-xs uppercase tracking-wider">
                Message (optional)
              </label>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Tell me about this sunset…"
                maxLength={1000}
                rows={3}
                className="w-full bg-gray-100/60 dark:bg-slate-800/60 border border-gray-200 dark:border-slate-700/50 rounded-lg px-3 py-2 text-gray-900 dark:text-white text-sm placeholder-gray-400 dark:placeholder-slate-600 focus:outline-none focus:border-orange-500/50 transition-colors resize-none"
              />
            </div>

            {/* Error */}
            {status === "error" && (
              <p className="text-red-500 dark:text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                {errorMsg}
              </p>
            )}

            {/* Submit */}
            <button
              onClick={handleSubmit}
              disabled={!file || status === "submitting"}
              className="w-full py-2.5 rounded-xl bg-orange-500 hover:bg-orange-400 disabled:bg-gray-200 dark:disabled:bg-slate-700 disabled:text-gray-400 dark:disabled:text-slate-500 disabled:cursor-not-allowed text-white font-medium text-sm transition-colors flex items-center justify-center gap-2"
            >
              {status === "submitting" ? (
                <>
                  <span className="inline-block h-3.5 w-3.5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                  Sending…
                </>
              ) : (
                "Send photo"
              )}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
