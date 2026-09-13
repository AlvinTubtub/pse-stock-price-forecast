export interface ChartViewport {
  startIndex: number;
  endIndex: number;
}

export interface ViewportBounds {
  minIndex: number;
  maxIndex: number;
  minWindow: number;
}

export interface PointPosition {
  x: number;
  y: number;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function viewportSize(viewport: ChartViewport): number {
  return Math.max(0, viewport.endIndex - viewport.startIndex + 1);
}

export function fullViewport(totalPoints: number): ChartViewport {
  return totalPoints > 0
    ? { startIndex: 0, endIndex: totalPoints - 1 }
    : { startIndex: 0, endIndex: -1 };
}

export function normalizeViewport(
  viewport: ChartViewport,
  bounds: ViewportBounds,
): ChartViewport {
  const minIndex = Number.isFinite(bounds.minIndex) ? Math.ceil(bounds.minIndex) : 0;
  const maxIndex = Number.isFinite(bounds.maxIndex) ? Math.floor(bounds.maxIndex) : minIndex - 1;
  const available = maxIndex - minIndex + 1;

  if (available <= 0) return { startIndex: minIndex, endIndex: minIndex - 1 };

  const minimumWindow = clamp(Math.round(bounds.minWindow) || 1, 1, available);
  const requestedWindow = Number.isFinite(viewport.startIndex) && Number.isFinite(viewport.endIndex)
    ? Math.round(viewport.endIndex - viewport.startIndex + 1)
    : available;
  const windowSize = clamp(requestedWindow, minimumWindow, available);
  const requestedStart = Number.isFinite(viewport.startIndex)
    ? Math.round(viewport.startIndex)
    : minIndex;
  const startIndex = clamp(requestedStart, minIndex, maxIndex - windowSize + 1);

  return { startIndex, endIndex: startIndex + windowSize - 1 };
}

export function panViewport(
  viewport: ChartViewport,
  offset: number,
  bounds: ViewportBounds,
): ChartViewport {
  const normalized = normalizeViewport(viewport, bounds);
  const size = viewportSize(normalized);
  if (size === 0) return normalized;

  return normalizeViewport(
    {
      startIndex: normalized.startIndex + Math.round(offset),
      endIndex: normalized.endIndex + Math.round(offset),
    },
    bounds,
  );
}

export function zoomViewportAtIndex(
  viewport: ChartViewport,
  windowScale: number,
  anchorIndex: number,
  targetRatio: number,
  bounds: ViewportBounds,
): ChartViewport {
  const normalized = normalizeViewport(viewport, bounds);
  const size = viewportSize(normalized);
  if (size <= 1 || !Number.isFinite(windowScale) || windowScale <= 0) return normalized;

  const available = bounds.maxIndex - bounds.minIndex + 1;
  const minimumWindow = clamp(Math.round(bounds.minWindow) || 1, 1, available);
  const nextSize = clamp(Math.round(size * windowScale), minimumWindow, available);
  const ratio = clamp(Number.isFinite(targetRatio) ? targetRatio : 0.5, 0, 1);
  const safeAnchor = clamp(
    Number.isFinite(anchorIndex) ? anchorIndex : normalized.startIndex + (size - 1) / 2,
    bounds.minIndex,
    bounds.maxIndex,
  );
  const nextStart = Math.round(safeAnchor - ratio * (nextSize - 1));

  return normalizeViewport(
    { startIndex: nextStart, endIndex: nextStart + nextSize - 1 },
    { ...bounds, minWindow: minimumWindow },
  );
}

export function pointDistance(left: PointPosition, right: PointPosition): number {
  return Math.hypot(right.x - left.x, right.y - left.y);
}

export function pointMidpoint(left: PointPosition, right: PointPosition): PointPosition {
  return { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 };
}

export function viewportsEqual(left: ChartViewport, right: ChartViewport): boolean {
  return left.startIndex === right.startIndex && left.endIndex === right.endIndex;
}

export function isViewportModified(
  viewport: ChartViewport,
  defaultViewport: ChartViewport,
  bounds: ViewportBounds,
): boolean {
  return !viewportsEqual(
    normalizeViewport(viewport, bounds),
    normalizeViewport(defaultViewport, bounds),
  );
}

export function wheelDeltaToScale(deltaY: number): number {
  if (!Number.isFinite(deltaY) || deltaY === 0) return 1;
  const exponent = clamp(deltaY * 0.01, -0.12, 0.12);
  return Math.exp(exponent);
}

export function pixelDragToIndexShift(
  deltaPixels: number,
  chartWidth: number,
  visiblePoints: number,
): number {
  if (
    !Number.isFinite(deltaPixels) ||
    !Number.isFinite(chartWidth) ||
    !Number.isFinite(visiblePoints) ||
    chartWidth <= 0 ||
    visiblePoints <= 0
  ) {
    return 0;
  }
  return -deltaPixels / (chartWidth / visiblePoints);
}

export function removeActivePointer(
  pointers: Map<number, PointPosition>,
  pointerId: number,
): number {
  pointers.delete(pointerId);
  return pointers.size;
}
