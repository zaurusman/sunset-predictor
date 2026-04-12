"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MapPin, Navigation, Search, X } from "lucide-react";
import { geocode } from "@/lib/api";
import type { GeocodingResult, LocationState } from "@/lib/types";

interface LocationSearchProps {
  onLocationSelect: (location: LocationState) => void;
  currentLocation?: LocationState | null;
  disabled?: boolean;
}

export default function LocationSearch({
  onLocationSelect,
  currentLocation,
  disabled = false,
}: LocationSearchProps) {
  const [query, setQuery] = useState(currentLocation?.name ?? "");
  const [results, setResults] = useState<GeocodingResult[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [geoLoading, setGeoLoading] = useState(false);
  const [geoError, setGeoError] = useState<string | null>(null);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    function handle(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, []);

  const search = useCallback(async (q: string) => {
    if (q.trim().length < 2) {
      setResults([]);
      return;
    }
    const res = await geocode(q);
    setResults(res);
    setShowDropdown(res.length > 0);
  }, []);

  const handleInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => search(val), 300);
  };

  const handleSelect = (result: GeocodingResult) => {
    const name = [result.name, result.admin1, result.country]
      .filter(Boolean)
      .join(", ");
    setQuery(name);
    setShowDropdown(false);
    setResults([]);
    onLocationSelect({
      latitude: result.latitude,
      longitude: result.longitude,
      name,
    });
  };

  const handleGeolocate = () => {
    if (!navigator.geolocation) {
      setGeoError("Geolocation not supported by your browser.");
      return;
    }
    setGeoLoading(true);
    setGeoError(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        setGeoLoading(false);
        const name = `${latitude.toFixed(3)}, ${longitude.toFixed(3)}`;
        setQuery(name);
        onLocationSelect({ latitude, longitude, name });
      },
      (err) => {
        setGeoLoading(false);
        setGeoError("Location access denied. Please search manually.");
      },
      { timeout: 10_000 }
    );
  };

  const clearInput = () => {
    setQuery("");
    setResults([]);
    setShowDropdown(false);
  };

  return (
    <div ref={containerRef} className="relative w-full max-w-lg">
      <div className="flex gap-2">
        {/* Search input */}
        <div className="relative flex-1">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none"
          />
          <input
            type="text"
            value={query}
            onChange={handleInput}
            onFocus={() => results.length > 0 && setShowDropdown(true)}
            placeholder="Search location…"
            disabled={disabled}
            className="w-full bg-slate-800/80 border border-slate-700 rounded-xl pl-9 pr-9 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-orange-500/60 focus:ring-1 focus:ring-orange-500/20 transition-colors disabled:opacity-50"
          />
          {query && (
            <button
              onClick={clearInput}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* Geolocation button */}
        <button
          onClick={handleGeolocate}
          disabled={geoLoading || disabled}
          title="Use my location"
          className="flex items-center justify-center w-12 h-12 rounded-xl bg-slate-800/80 border border-slate-700 text-slate-400 hover:text-orange-400 hover:border-orange-500/40 transition-colors disabled:opacity-50"
        >
          {geoLoading ? (
            <div className="w-4 h-4 border-2 border-slate-500 border-t-orange-400 rounded-full animate-spin" />
          ) : (
            <Navigation size={16} />
          )}
        </button>
      </div>

      {/* Error message */}
      {geoError && (
        <p className="mt-1 text-xs text-red-400">{geoError}</p>
      )}

      {/* Dropdown */}
      {showDropdown && results.length > 0 && (
        <div className="absolute z-50 w-full mt-1 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl overflow-hidden">
          {results.map((result) => {
            const label = [result.name, result.admin1, result.country]
              .filter(Boolean)
              .join(", ");
            return (
              <button
                key={result.id}
                onClick={() => handleSelect(result)}
                className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-slate-700/60 transition-colors text-sm"
              >
                <MapPin size={14} className="text-slate-500 flex-shrink-0" />
                <span className="text-white truncate">{label}</span>
                <span className="ml-auto text-slate-500 text-xs flex-shrink-0">
                  {result.country_code}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
