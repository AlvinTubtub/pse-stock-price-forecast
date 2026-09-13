import type { ChartInteractionResult } from "@/hooks/useChartTouchGestures";

type ChartGestureControlsProps = Pick<
  ChartInteractionResult,
  "zoomIn" | "zoomOut" | "canZoomIn" | "canZoomOut"
>;

export default function ChartGestureControls({
  zoomIn,
  zoomOut,
  canZoomIn,
  canZoomOut,
}: ChartGestureControlsProps) {
  const buttonClass = "rounded-md border border-dark-border bg-dark-bg px-2.5 py-1 text-[11px] font-semibold text-slate-300 outline-none transition-colors hover:border-brand-500 hover:text-white focus-visible:ring-2 focus-visible:ring-brand-400 disabled:cursor-not-allowed disabled:opacity-40";

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-400">
      <span>
        Pinch or wheel to zoom <span aria-hidden="true">•</span> drag horizontally to pan
      </span>
      <div className="flex items-center gap-1.5" aria-label="Chart viewport controls">
        <button type="button" onClick={zoomIn} disabled={!canZoomIn} className={buttonClass}>
          + Zoom In
        </button>
        <button type="button" onClick={zoomOut} disabled={!canZoomOut} className={buttonClass}>
          − Zoom Out
        </button>
      </div>
    </div>
  );
}

export function ChartResetButton({
  visible,
  onReset,
}: {
  visible: boolean;
  onReset: () => void;
}) {
  if (!visible) return null;

  return (
    <button
      type="button"
      onClick={onReset}
      onPointerDown={(event) => event.stopPropagation()}
      aria-label="Reset chart zoom"
      className="absolute right-3 top-3 z-20 rounded-lg border border-dark-border bg-dark-card/90 px-2.5 py-1.5 text-[11px] font-semibold text-white shadow-md backdrop-blur-sm outline-none transition-colors hover:border-brand-500 hover:text-brand-300 focus-visible:ring-2 focus-visible:ring-brand-400"
    >
      Reset
    </button>
  );
}
