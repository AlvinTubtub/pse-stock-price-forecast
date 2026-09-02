"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ReferenceArea,
} from "recharts";
import { useInteractiveChart } from "./useInteractiveChart";
import InteractiveChartToolbar from "./InteractiveChartToolbar";

const MODEL_COLORS: Record<string, string> = {
  Actual: "#22c55e",
  ARIMA: "#f59e0b",
  "Lag-Informed Regression": "#3b82f6",
  LSTM: "#a855f7",
  "Naive baseline": "#64748b",
  "Naive Baseline": "#64748b",
};

function formatShortDate(dateStr: string): string {
  if (!dateStr) return "";
  try {
    if (/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
      const [y, m, d] = dateStr.split("-").map(Number);
      const dt = new Date(y, m - 1, d);
      return dt.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    }
    const dt = new Date(dateStr);
    return dt.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  } catch {
    return dateStr;
  }
}

function formatFullDate(dateStr: string): string {
  if (!dateStr) return "";
  try {
    if (/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
      const [y, m, d] = dateStr.split("-").map(Number);
      const dt = new Date(y, m - 1, d);
      return dt.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
    }
    const dt = new Date(dateStr);
    return dt.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  } catch {
    return dateStr;
  }
}

export interface PredictionChartProps {
  dates?: string[];
  actual: number[];
  byModel: Record<string, number[]>;
  selectedModel?: string;
  liveStartDate?: string;
}

export default function PredictionChart({
  dates,
  actual,
  byModel,
  selectedModel,
  liveStartDate,
}: PredictionChartProps) {
  const data = actual.map((value, i) => {
    const rawDate = dates?.[i] || `Day ${i + 1}`;
    const row: Record<string, any> = {
      step: rawDate,
      displayDate: dates?.[i] ? formatShortDate(rawDate) : `Day ${i + 1}`,
      fullDate: dates?.[i] ? formatFullDate(rawDate) : `Day ${i + 1}`,
      Actual: value,
    };
    for (const [model, series] of Object.entries(byModel)) {
      if (series[i] !== undefined) row[model] = series[i];
    }
    return row;
  });

  const seriesNames = ["Actual", ...Object.keys(byModel)];
  const chart = useInteractiveChart({ data, xKey: "step" });

  if (data.length === 0) {
    return (
      <div className="flex h-[360px] items-center justify-center rounded-lg border border-dashed border-dark-border bg-dark-bg/40 px-6 text-center">
        <div className="max-w-md space-y-2">
          <p className="text-sm font-medium text-slate-200">Waiting for the first realized production forecast</p>
          <p className="text-xs leading-relaxed text-slate-400">
            ForecastPH will add a point here after a forecast is issued and the corresponding trading
            session&apos;s closing price becomes available. Historical research backtests are kept separate.
          </p>
        </div>
      </div>
    );
  }

  const renderTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    const point = payload[0]?.payload;
    const headerDate = point?.fullDate || point?.displayDate || point?.step;

    return (
      <div className="bg-dark-card border border-dark-border rounded-xl p-3 shadow-xl text-xs space-y-1.5 min-w-[210px]">
        <p className="font-semibold text-white border-b border-dark-border pb-1.5 mb-1.5">
          {headerDate}
        </p>
        {payload.map((entry: any) => {
          const isActual = entry.dataKey === "Actual";
          const isSelected = selectedModel && entry.dataKey === selectedModel;
          const isNaive =
            entry.dataKey === "Naive baseline" || entry.dataKey === "Naive Baseline";
          const label = isNaive ? "Naive Baseline" : entry.dataKey;

          return (
            <div key={entry.dataKey} className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-full inline-block shrink-0"
                  style={{ backgroundColor: entry.color }}
                />
                <span
                  className={`font-medium ${
                    isActual
                      ? "text-green-400 font-semibold"
                      : isSelected
                      ? "text-brand-400 font-semibold"
                      : "text-slate-300"
                  }`}
                >
                  {label}
                  {isSelected && (
                    <span className="ml-1 text-[10px] text-brand-400 uppercase font-semibold">
                      (Selected)
                    </span>
                  )}
                </span>
              </div>
              <span className="font-mono text-white font-semibold">
                ₱
                {Number(entry.value).toLocaleString("en-PH", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="w-full select-none space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-400">
        <span className="text-slate-400">
          Showing {chart.visibleCount} of {chart.totalCount} trading sessions &middot; Hover points
          to inspect prices
        </span>
        <div className="flex items-center gap-2">
          <span className="hidden sm:inline text-slate-500">
            Wheel = Zoom &middot; Drag = Pan &middot; Double-click = Reset
          </span>
          <InteractiveChartToolbar
            onZoomIn={() => chart.zoomIn(0.2)}
            onZoomOut={() => chart.zoomOut(0.2)}
            onResetView={chart.resetView}
            isBoxZoomActive={chart.isBoxZoomActive}
            onToggleBoxZoom={chart.toggleBoxZoom}
            isPanModeActive={chart.isPanModeActive}
            onTogglePanMode={chart.togglePanMode}
          />
        </div>
      </div>

      <div
        ref={chart.containerRef}
        onDoubleClick={chart.resetView}
        className={`w-full relative ${
          chart.isBoxZoomActive
            ? "cursor-crosshair"
            : chart.isPanModeActive || chart.isDragging
            ? "cursor-grabbing"
            : "cursor-grab"
        }`}
        title="Scroll mouse wheel to zoom · Drag horizontally to pan · Double-click to reset view"
      >
        <ResponsiveContainer width="100%" height={360}>
          <LineChart
            data={chart.visibleData}
            margin={{ top: 10, right: 15, left: 10, bottom: 5 }}
            onMouseDown={(e) => e && e.activeLabel && chart.handleMouseDown(e.activeLabel)}
            onMouseMove={(e) => e && e.activeLabel && chart.handleMouseMove(e.activeLabel)}
            onMouseUp={chart.handleMouseUp}
            onMouseLeave={chart.handleMouseUp}
          >
            <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
            {liveStartDate && <ReferenceLine x={liveStartDate} stroke="#34d399" strokeDasharray="4 4" label={{ value: "Live forecast begins", fill: "#6ee7b7", fontSize: 10, position: "top" }} />}
            <XAxis
              dataKey="step"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickFormatter={(val) => {
                const item = data.find((d) => d.step === val);
                return item ? item.displayDate : val;
              }}
              minTickGap={35}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              domain={["auto", "auto"]}
              tickFormatter={(val) => `₱${Number(val).toLocaleString()}`}
              label={{
                value: "Price (₱)",
                angle: -90,
                position: "insideLeft",
                fill: "#64748b",
                fontSize: 11,
              }}
            />
            <Tooltip content={renderTooltip} />
            <Legend
              wrapperStyle={{ fontSize: 12, paddingTop: 10 }}
              formatter={(value) => {
                const isSelected = selectedModel && value === selectedModel;
                const isNaive = value === "Naive baseline" || value === "Naive Baseline";
                const displayLabel = isNaive ? "Naive Baseline" : value;
                return (
                  <span className={isSelected ? "font-bold text-white" : "text-slate-300"}>
                    {displayLabel}
                    {isSelected && " (Selected)"}
                  </span>
                );
              }}
            />

            {seriesNames.map((name) => {
              const isActual = name === "Actual";
              const isSelected = selectedModel && name === selectedModel;
              const isNaive = name === "Naive baseline" || name === "Naive Baseline";

              return (
                <Line
                  key={name}
                  type="monotone"
                  dataKey={name}
                  stroke={MODEL_COLORS[name] ?? "#94a3b8"}
                  strokeWidth={isActual ? 2.5 : isSelected ? 2.5 : 1.5}
                  strokeDasharray={isNaive ? "4 4" : undefined}
                  dot={data.length === 1 ? { r: 4 } : false}
                  activeDot={{ r: 4 }}
                />
              );
            })}

            {/* Box zoom reference selection area */}
            {chart.isBoxZoomActive && chart.refAreaLeft && chart.refAreaRight && (
              <ReferenceArea
                x1={chart.refAreaLeft}
                x2={chart.refAreaRight}
                strokeOpacity={0.3}
                fill="#38bdf8"
                fillOpacity={0.2}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
