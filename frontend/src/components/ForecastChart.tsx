"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DayForecast } from "@/lib/types";
import { formatDateShort, formatTime, getScoreHexColor } from "@/lib/utils";

interface ForecastChartProps {
  days: DayForecast[];
  onDayClick?: (day: DayForecast) => void;
  selectedDate?: string;
}

interface TooltipPayload {
  payload?: DayForecast & { displayDate: string };
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: DayForecast & { displayDate: string } }> }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-3 shadow-2xl text-sm">
      <div className="font-semibold text-white mb-1">{d.displayDate}</div>
      <div className="text-orange-400 font-bold text-lg">{d.beauty_score_0_100}</div>
      <div className="text-slate-400">{d.category}</div>
      <div className="text-slate-500 text-xs mt-1">Sunset {formatTime(d.sunset_time)}</div>
    </div>
  );
}

export default function ForecastChart({
  days,
  onDayClick,
  selectedDate,
}: ForecastChartProps) {
  const chartData = days.map((d) => ({
    ...d,
    displayDate: formatDateShort(d.date),
  }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart
        data={chartData}
        margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
        onClick={(data) => {
          if (data?.activePayload?.[0] && onDayClick) {
            onDayClick(data.activePayload[0].payload as DayForecast);
          }
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
        <XAxis
          dataKey="displayDate"
          tick={{ fill: "#94a3b8", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fill: "#64748b", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={30}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
        <Bar dataKey="beauty_score_0_100" radius={[4, 4, 0, 0]} maxBarSize={40}>
          {chartData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={getScoreHexColor(entry.beauty_score_0_100)}
              opacity={selectedDate && entry.date !== selectedDate ? 0.45 : 1}
              style={{ cursor: onDayClick ? "pointer" : "default" }}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
