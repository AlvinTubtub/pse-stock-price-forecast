"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";
import { useInteractiveChart } from "./useInteractiveChart";

const COLORS: Record<string, string> = {
  "Lag-Informed Regression": "#60a5fa",
  ARIMA: "#f59e0b",
  LSTM: "#a855f7",
  "Naive baseline": "#64748b",
  Actual: "#22c55e",
};

export default function PredictionChart({
  actual,
  byModel,
}: {
  actual: number[];
  byModel: Record<string, number[]>;
}) {
  const data = actual.map((value, i) => {
    const row: Record<string, any> = { step: `Day ${i + 1}`, stepNum: i + 1, Actual: value };
    for (const [model, series] of Object.entries(byModel)) {
      if (series[i] !== undefined) row[model] = series[i];
    }
    return row;
  });

  const seriesNames = ["Actual", ...Object.keys(byModel)];

  const chart = useInteractiveChart({ data, xKey: "step" });

  return (
    <div className="w-full select-none">
      <div className="flex justify-end text-[11px] text-slate-500 mb-1">
        <span>Mouse wheel = Zoom &middot; Drag = Pan</span>
      </div>

      <div
        ref={chart.containerRef}
        onDoubleClick={chart.resetView}
        className={`w-full relative cursor-grab ${chart.isDragging ? "cursor-grabbing" : ""}`}
        title="Scroll mouse wheel to zoom in/out · Drag horizontally to pan · Double-click to reset view"
      >
        <ResponsiveContainer width="100%" height={350}>
          <LineChart
            data={chart.visibleData}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
            onMouseDown={(e) => e && e.activeLabel && chart.handleMouseDown(e.activeLabel)}
            onMouseMove={(e) => e && e.activeLabel && chart.handleMouseMove(e.activeLabel)}
            onMouseUp={chart.handleMouseUp}
            onMouseLeave={chart.handleMouseUp}
          >
            <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="step" tick={{ fill: "#94a3b8", fontSize: 11 }} />
            <YAxis
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              domain={["auto", "auto"]}
              tickFormatter={(val) => `₱${val}`}
            />
            <Tooltip
              contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: "#e2e8f0" }}
              formatter={(val: any, name: any) => [`₱${Number(val).toFixed(2)}`, name]}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {seriesNames.map((name) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                stroke={COLORS[name] ?? "#94a3b8"}
                strokeWidth={name === "Actual" ? 2.5 : 1.5}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
