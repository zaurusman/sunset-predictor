"use client";

import { useEffect, useRef, useState } from "react";
import { Calendar, ChevronDown, ChevronLeft, ChevronRight } from "lucide-react";

interface DatePickerProps {
  value: string; // "YYYY-MM-DD"
  onChange: (date: string) => void;
  disabled?: boolean;
}

// ── helpers ────────────────────────────────────────────────────────────────

/** Parse an ISO date string as a local midnight Date (avoids UTC offset issues). */
function parseIso(iso: string): Date {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d);
}

/** Serialize a local Date to "YYYY-MM-DD". */
function toIso(d: Date): string {
  return [
    d.getFullYear(),
    String(d.getMonth() + 1).padStart(2, "0"),
    String(d.getDate()).padStart(2, "0"),
  ].join("-");
}

/** Today's local date as "YYYY-MM-DD". */
function todayIso(): string {
  return toIso(new Date());
}

function formatLabel(iso: string): string {
  if (iso === todayIso()) return "Today";
  const d = parseIso(iso);
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

const WEEKDAYS = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];
const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

// ── component ──────────────────────────────────────────────────────────────

export default function DatePicker({ value, onChange, disabled = false }: DatePickerProps) {
  const todayDate = new Date();
  todayDate.setHours(0, 0, 0, 0);

  // Minimum selectable date: 1 year back
  const minDate = new Date(todayDate);
  minDate.setFullYear(todayDate.getFullYear() - 1);

  const [open, setOpen] = useState(false);
  const [viewYear, setViewYear] = useState(() => parseIso(value).getFullYear());
  const [viewMonth, setViewMonth] = useState(() => parseIso(value).getMonth());
  const containerRef = useRef<HTMLDivElement>(null);

  // Close when clicking outside
  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, []);

  // Keep view in sync when value is changed externally
  useEffect(() => {
    const d = parseIso(value);
    setViewYear(d.getFullYear());
    setViewMonth(d.getMonth());
  }, [value]);

  function prevMonth() {
    if (viewMonth === 0) { setViewMonth(11); setViewYear((y) => y - 1); }
    else setViewMonth((m) => m - 1);
  }
  function nextMonth() {
    if (viewMonth === 11) { setViewMonth(0); setViewYear((y) => y + 1); }
    else setViewMonth((m) => m + 1);
  }

  const viewFirst = new Date(viewYear, viewMonth, 1);
  const canPrev = viewFirst > new Date(minDate.getFullYear(), minDate.getMonth(), 1);
  const canNext = viewFirst < new Date(todayDate.getFullYear(), todayDate.getMonth(), 1);

  // Build the day grid: leading nulls + day numbers + trailing nulls
  const firstWeekday = viewFirst.getDay();
  const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();
  const cells: (number | null)[] = [
    ...Array(firstWeekday).fill(null),
    ...Array.from({ length: daysInMonth }, (_, i) => i + 1),
  ];
  while (cells.length % 7 !== 0) cells.push(null);

  function selectDay(day: number) {
    const iso = toIso(new Date(viewYear, viewMonth, day));
    onChange(iso);
    setOpen(false);
  }

  function goToday() {
    onChange(todayIso());
    setOpen(false);
  }

  return (
    <div ref={containerRef} className="relative">
      {/* Trigger button */}
      <button
        onClick={() => !disabled && setOpen((o) => !o)}
        disabled={disabled}
        className="flex items-center gap-2 px-4 py-2.5 bg-slate-800/80 border border-slate-700 rounded-xl text-sm text-slate-300 hover:text-orange-400 hover:border-orange-500/40 transition-colors disabled:opacity-50 whitespace-nowrap"
      >
        <Calendar size={14} className="flex-shrink-0" />
        <span>{formatLabel(value)}</span>
        <ChevronDown size={12} className="text-slate-500 flex-shrink-0" />
      </button>

      {/* Calendar popover */}
      {open && (
        <div className="absolute z-50 top-full mt-2 left-1/2 -translate-x-1/2 bg-slate-800 border border-slate-700 rounded-2xl shadow-2xl p-4 w-72">
          {/* Month/year navigation */}
          <div className="flex items-center justify-between mb-3">
            <button
              onClick={prevMonth}
              disabled={!canPrev}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 disabled:opacity-25 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft size={14} />
            </button>
            <span className="text-sm font-semibold text-white">
              {MONTH_NAMES[viewMonth]} {viewYear}
            </span>
            <button
              onClick={nextMonth}
              disabled={!canNext}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 disabled:opacity-25 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight size={14} />
            </button>
          </div>

          {/* Weekday headers */}
          <div className="grid grid-cols-7 mb-1">
            {WEEKDAYS.map((d) => (
              <div key={d} className="text-center text-slate-500 text-xs py-1 font-medium">
                {d}
              </div>
            ))}
          </div>

          {/* Day grid */}
          <div className="grid grid-cols-7 gap-y-0.5">
            {cells.map((day, idx) => {
              if (day === null) return <div key={`e${idx}`} />;

              const cellDate = new Date(viewYear, viewMonth, day);
              const cellIso = toIso(cellDate);
              const isSelected = cellIso === value;
              const isToday = cellDate.getTime() === todayDate.getTime();
              const isDisabled = cellDate > todayDate || cellDate < minDate;

              let cls =
                "text-xs h-8 w-full rounded-lg font-medium transition-colors ";
              if (isDisabled) {
                cls += "text-slate-600 cursor-not-allowed";
              } else if (isSelected) {
                cls += "bg-orange-500 text-white";
              } else if (isToday) {
                cls +=
                  "border border-orange-500/50 text-orange-400 hover:bg-orange-500/10";
              } else {
                cls += "text-slate-300 hover:bg-slate-700";
              }

              return (
                <button
                  key={day}
                  onClick={() => !isDisabled && selectDay(day)}
                  disabled={isDisabled}
                  className={cls}
                >
                  {day}
                </button>
              );
            })}
          </div>

          {/* Footer — jump to today */}
          {value !== todayIso() && (
            <div className="mt-3 pt-3 border-t border-slate-700/60">
              <button
                onClick={goToday}
                className="w-full text-xs text-orange-400 hover:text-orange-300 transition-colors py-0.5"
              >
                Back to today
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
