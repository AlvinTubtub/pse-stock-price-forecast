"use client";

import { useState } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  ReferenceArea,
} from "recharts";
import type { OhlcvPoint } from "@/lib/types";
import { useInteractiveChart } from "./useInteractiveChart";
import InteractiveChartToolbar from "./InteractiveChartToolbar";

export default function HistoryChart({ data }: { data: OhlcvPoint[] }) {
  const chart = useInteractiveChart({ data, xKey: "date" });

  const [visibleSeries, setVisibleSeries] = useState({
    open: true,
    high: true,
    low: true,
    close: true,
    volume: true,
  });

  const toggleSeries = (key: keyof typeof visibleSeries) => {
    setVisibleSeries((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const startDate = chart.visibleData[0]?.date;
  const endDate = chart.visibleData[chart.visibleData.length - 1]?.date;

  return (
    <div className="w-full select-none space-y-3">
      {/* Top Header: Series Toggles & Date Info */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-slate-400 font-medium">Series:</span>
          <button
            type="button"
            onClick={() => toggleSeries("open")}
            className={`px-2 py-0.5 rounded border transition-colors cursor-pointer ${
              visibleSeries.open
                ? "bg-amber-500/10 border-amber-500/50 text-amber-400 font-medium"
                : "bg-slate-800/50 border-slate-700 text-slate-500 line-through"
            }`}
          >
            Open
          </button>
          <button
            type="button"
            onClick={() => toggleSeries("high")}
            className={`px-2 py-0.5 rounded border transition-colors cursor-pointer ${
              visibleSeries.high
                ? "bg-green-500/10 border-green-500/50 text-green-400 font-medium"
                : "bg-slate-800/50 border-slate-700 text-slate-500 line-through"
            }`}
          >
            High
          </button>
          <button
            type="button"
            onClick={() => toggleSeries("low")}
            className={`px-2 py-0.5 rounded border transition-colors cursor-pointer ${
              visibleSeries.low
                ? "bg-red-500/10 border-red-500/50 text-red-400 font-medium"
                : "bg-slate-800/50 border-slate-700 text-slate-500 line-through"
            }`}
          >
            Low
          </button>
          <button
            type="button"
            onClick={() => toggleSeries("close")}
            className={`px-2 py-0.5 rounded border transition-colors cursor-pointer ${
              visibleSeries.close
                ? "bg-blue-500/10 border-blue-500/50 text-blue-400 font-medium"
                : "bg-slate-800/50 border-slate-700 text-slate-500 line-through"
            }`}
          >
            Close
          </button>
          <button
            type="button"
            onClick={() => toggleSeries("volume")}
            className={`px-2 py-0.5 rounded border transition-colors cursor-pointer ${
              visibleSeries.volume
                ? "bg-sky-500/10 border-sky-500/50 text-sky-400 font-medium"
                : "bg-slate-800/50 border-slate-700 text-slate-500 line-through"
            }`}
          >
            Volume
          </button>
        </div>

        <div className="text-slate-400 text-[11px] font-mono whitespace-nowrap">
          {startDate && endDate ? `${startDate} – ${endDate}` : null}{" "}
          <span className="text-slate-500">
            ({chart.visibleCount.toLocaleString()}/{chart.totalCount.toLocaleString()} pts)
          </span>
        </div>
      </div>

      {/* Relative Chart Container with Overlaid Floating Controls */}
      <div
        ref={chart.containerRef}
        className={`w-full relative ${
          chart.isBoxZoomActive
            ? "cursor-crosshair"
            : chart.isPanModeActive
            ? "cursor-grab active:cursor-grabbing"
            : "cursor-grab"
        }`}
      >
        {/* Floating Control Toolbar overlaid inside top-right of chart */}
        <InteractiveChartToolbar
          onZoomIn={chart.zoomIn}
          onZoomOut={chart.zoomOut}
          onResetView={chart.resetView}
          isBoxZoomActive={chart.isBoxZoomActive}
          onToggleBoxZoom={chart.toggleBoxZoom}
          isPanModeActive={chart.isPanModeActive}
          onTogglePanMode={chart.togglePanMode}
          className="absolute top-2 right-4 z-20"
        />

        <ResponsiveContainer width="100%" height={430}>
          <ComposedChart
            data={chart.visibleData}
            margin={{ top: 15, right: 10, left: 0, bottom: 0 }}
            onMouseDown={(e) => e && e.activeLabel && chart.handleMouseDown(e.activeLabel)}
            onMouseMove={(e) => e && e.activeLabel && chart.handleMouseMove(e.activeLabel)}
            onMouseUp={chart.handleMouseUp}
            onMouseLeave={chart.handleMouseUp}
          >
            <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={35} />
            <YAxis
              yAxisId="price"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              domain={["auto", "auto"]}
              tickFormatter={(val) => `₱${val}`}
            />
            <YAxis yAxisId="volume" orientation="right" tick={{ fill: "#94a3b8", fontSize: 11 }} hide />
            <Tooltip
              contentStyle={{
                background: "#1e293b",
                border: "1px solid #334155",
                borderRadius: 8,
                fontSize: 12,
                boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.5)",
              }}
              labelStyle={{ color: "#f8fafc", fontWeight: 600, marginBottom: 4 }}
              formatter={(value: any, name: any) => {
                const num = Number(value);
                if (isNaN(num)) return [value, name];
                if (name === "Volume") {
                  return [num.toLocaleString(), "Volume"];
                }
                return [`₱${num.toFixed(2)}`, name];
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} />

            {/* Volume Area */}
            {visibleSeries.volume && (
              <Area
                yAxisId="volume"
                type="monotone"
                dataKey="volume"
                name="Volume"
                fill="#3b82f6"
                stroke="none"
                fillOpacity={0.15}
              />
            )}

            {/* OHLC Lines */}
            {visibleSeries.open && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="open"
                name="Open"
                stroke="#f59e0b"
                dot={false}
                strokeWidth={1.2}
              />
            )}
            {visibleSeries.high && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="high"
                name="High"
                stroke="#22c55e"
                dot={false}
                strokeWidth={1.2}
              />
            )}
            {visibleSeries.low && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="low"
                name="Low"
                stroke="#ef4444"
                dot={false}
                strokeWidth={1.2}
              />
            )}
            {visibleSeries.close && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="close"
                name="Close"
                stroke="#60a5fa"
                dot={false}
                strokeWidth={2}
              />
            )}

            {/* Box Zoom Highlight Box */}
            {chart.refAreaLeft && chart.refAreaRight && (
              <ReferenceArea
                yAxisId="price"
                x1={chart.refAreaLeft}
                x2={chart.refAreaRight}
                stroke="#60a5fa"
                strokeOpacity={0.8}
                fill="#3b82f6"
                fillOpacity={0.25}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
