"use client";

import React from "react";

export interface InteractiveChartToolbarProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetView: () => void;
  isBoxZoomActive: boolean;
  onToggleBoxZoom: () => void;
  isPanModeActive: boolean;
  onTogglePanMode: () => void;
  className?: string;
}

export default function InteractiveChartToolbar({
  onZoomIn,
  onZoomOut,
  onResetView,
  isBoxZoomActive,
  onToggleBoxZoom,
  isPanModeActive,
  onTogglePanMode,
  className = "",
}: InteractiveChartToolbarProps) {
  return (
    <div
      className={`inline-flex items-center gap-0.5 bg-[#182030]/90 backdrop-blur-md border border-slate-700/60 rounded-xl p-1 shadow-lg ${className}`}
    >
      {/* 1. Zoom In (+) */}
      <button
        type="button"
        onClick={onZoomIn}
        title="Zoom In (+)"
        className="w-7 h-7 flex items-center justify-center text-slate-300 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors cursor-pointer"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2.2} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m8-7.5H4" />
        </svg>
      </button>

      {/* 2. Zoom Out (−) */}
      <button
        type="button"
        onClick={onZoomOut}
        title="Zoom Out (−)"
        className="w-7 h-7 flex items-center justify-center text-slate-300 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors cursor-pointer"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2.2} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 12h-15" />
        </svg>
      </button>

      {/* 3. Box Zoom (🔍) */}
      <button
        type="button"
        onClick={onToggleBoxZoom}
        title="Box Zoom — Select a range to zoom"
        className={`w-7 h-7 flex items-center justify-center rounded-lg transition-colors cursor-pointer ${
          isBoxZoomActive
            ? "bg-[#1c304a] text-[#38bdf8] font-bold"
            : "text-slate-300 hover:text-white hover:bg-slate-700/50"
        }`}
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM10.5 7.5v6m3-3h-6"
          />
        </svg>
      </button>

      {/* 4. Pan Mode (↔) */}
      <button
        type="button"
        onClick={onTogglePanMode}
        title="Pan — Drag chart left/right"
        className={`w-7 h-7 flex items-center justify-center rounded-lg transition-colors cursor-pointer ${
          isPanModeActive
            ? "bg-[#1c304a] text-[#38bdf8] font-bold"
            : "text-slate-300 hover:text-white hover:bg-slate-700/50"
        }`}
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 3v18m-9-9h18M5 9l-3 3 3 3m14-6l3 3-3 3M9 5l3-3 3 3m-6 14l3 3 3-3"
          />
        </svg>
      </button>

      {/* 5. Reset View (↺) */}
      <button
        type="button"
        onClick={onResetView}
        title="Reset View (↺)"
        className="w-7 h-7 flex items-center justify-center text-slate-300 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors cursor-pointer"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M16.023 9.348h4.992v-.001M21 12a9 9 0 11-18 0 9 9 0 0115.357-6.357l3.228 3.357"
          />
        </svg>
      </button>
    </div>
  );
}
