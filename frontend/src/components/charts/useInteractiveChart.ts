"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";

export interface UseInteractiveChartOptions<T> {
  data: T[];
  xKey: keyof T;
}

interface ViewportRange {
  startIndex: number;
  endIndex: number;
}

const MIN_POINTS = 10;

export function useInteractiveChart<T extends Record<string, any>>({
  data,
  xKey,
}: UseInteractiveChartOptions<T>) {
  const totalCount = data.length;

  // Single consolidated state object for viewport range (prevents desynchronization)
  const [range, setRange] = useState<ViewportRange>(() => ({
    startIndex: 0,
    endIndex: Math.max(0, totalCount - 1),
  }));

  const [isBoxZoomActive, setIsBoxZoomActive] = useState<boolean>(false);
  const [isPanModeActive, setIsPanModeActive] = useState<boolean>(false);
  const [refAreaLeft, setRefAreaLeft] = useState<string | number | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<string | number | null>(null);
  const [isMouseDown, setIsMouseDown] = useState<boolean>(false);
  const [dragStartLabel, setDragStartLabel] = useState<string | number | null>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);

  // Sync index bounds whenever the base data changes
  useEffect(() => {
    setRange({
      startIndex: 0,
      endIndex: Math.max(0, data.length - 1),
    });
    setRefAreaLeft(null);
    setRefAreaRight(null);
  }, [data]);

  const findIndexByLabel = useCallback(
    (label: string | number) => {
      return data.findIndex((item) => item[xKey] === label);
    },
    [data, xKey]
  );

  const resetView = useCallback(() => {
    if (totalCount === 0) return;
    setRange({
      startIndex: 0,
      endIndex: Math.max(0, totalCount - 1),
    });
    setRefAreaLeft(null);
    setRefAreaRight(null);
    setIsBoxZoomActive(false);
    setIsPanModeActive(false);
  }, [totalCount]);

  const zoomIn = useCallback(
    (factor = 0.2) => {
      if (totalCount <= MIN_POINTS) return;

      setRange((prev) => {
        const currentSpan = prev.endIndex - prev.startIndex + 1;
        if (currentSpan <= MIN_POINTS) return prev;

        const delta = Math.max(1, Math.floor(currentSpan * factor));
        const targetSpan = Math.max(MIN_POINTS, currentSpan - delta);
        const center = (prev.startIndex + prev.endIndex) / 2;
        const half = (targetSpan - 1) / 2;

        let newStart = Math.round(center - half);
        let newEnd = newStart + targetSpan - 1;

        if (newStart < 0) {
          newStart = 0;
          newEnd = Math.min(totalCount - 1, newStart + targetSpan - 1);
        }
        if (newEnd >= totalCount) {
          newEnd = totalCount - 1;
          newStart = Math.max(0, newEnd - targetSpan + 1);
        }

        if (newStart < 0) newStart = 0;
        if (newEnd >= totalCount) newEnd = totalCount - 1;
        if (newStart > newEnd) newStart = newEnd;

        return { startIndex: newStart, endIndex: newEnd };
      });
    },
    [totalCount]
  );

  const zoomOut = useCallback(
    (factor = 0.2) => {
      if (totalCount === 0) return;

      setRange((prev) => {
        const currentSpan = prev.endIndex - prev.startIndex + 1;
        if (prev.startIndex === 0 && prev.endIndex === totalCount - 1) {
          return prev;
        }

        const delta = Math.max(1, Math.floor(currentSpan * factor));
        const targetSpan = Math.min(totalCount, currentSpan + delta);
        const center = (prev.startIndex + prev.endIndex) / 2;
        const half = (targetSpan - 1) / 2;

        let newStart = Math.round(center - half);
        let newEnd = newStart + targetSpan - 1;

        if (newStart < 0) {
          newStart = 0;
          newEnd = Math.min(totalCount - 1, newStart + targetSpan - 1);
        }
        if (newEnd >= totalCount) {
          newEnd = totalCount - 1;
          newStart = Math.max(0, newEnd - targetSpan + 1);
        }

        if (newStart < 0) newStart = 0;
        if (newEnd >= totalCount) newEnd = totalCount - 1;
        if (newStart > newEnd) newStart = newEnd;

        return { startIndex: newStart, endIndex: newEnd };
      });
    },
    [totalCount]
  );

  const toggleBoxZoom = useCallback(() => {
    setIsBoxZoomActive((prev) => {
      const next = !prev;
      if (next) setIsPanModeActive(false);
      return next;
    });
    setRefAreaLeft(null);
    setRefAreaRight(null);
  }, []);

  const togglePanMode = useCallback(() => {
    setIsPanModeActive((prev) => {
      const next = !prev;
      if (next) setIsBoxZoomActive(false);
      return next;
    });
  }, []);

  // Native non-passive wheel listener for scroll wheel zooming
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.deltaY < 0) {
        // Wheel UP = Zoom In
        zoomIn(0.15);
      } else if (e.deltaY > 0) {
        // Wheel DOWN = Zoom Out
        zoomOut(0.15);
      }
    };

    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      el.removeEventListener("wheel", handleWheel);
    };
  }, [zoomIn, zoomOut]);

  // Drag handlers for mouse drag panning & box zoom selection
  const handleMouseDown = useCallback(
    (label: string | number) => {
      if (!label) return;
      setIsMouseDown(true);
      setDragStartLabel(label);

      if (isBoxZoomActive) {
        setRefAreaLeft(label);
        setRefAreaRight(label);
      }
    },
    [isBoxZoomActive]
  );

  const handleMouseMove = useCallback(
    (label: string | number) => {
      if (!isMouseDown || !label) return;

      if (isBoxZoomActive) {
        setRefAreaRight(label);
      } else if (dragStartLabel !== null) {
        // Drag to pan
        const startIdx = findIndexByLabel(dragStartLabel);
        const currIdx = findIndexByLabel(label);

        if (startIdx >= 0 && currIdx >= 0 && startIdx !== currIdx) {
          const delta = currIdx - startIdx;
          setRange((prev) => {
            const span = prev.endIndex - prev.startIndex;
            let newStart = prev.startIndex - delta;
            let newEnd = newStart + span;

            if (newStart < 0) {
              newStart = 0;
              newEnd = Math.min(totalCount - 1, newStart + span);
            } else if (newEnd >= totalCount) {
              newEnd = totalCount - 1;
              newStart = Math.max(0, newEnd - span);
            }

            return { startIndex: newStart, endIndex: newEnd };
          });

          setDragStartLabel(label);
        }
      }
    },
    [isMouseDown, isBoxZoomActive, dragStartLabel, findIndexByLabel, totalCount]
  );

  const handleMouseUp = useCallback(() => {
    setIsMouseDown(false);
    setDragStartLabel(null);

    if (isBoxZoomActive && refAreaLeft !== null && refAreaRight !== null) {
      if (refAreaLeft !== refAreaRight) {
        const idx1 = findIndexByLabel(refAreaLeft);
        const idx2 = findIndexByLabel(refAreaRight);
        if (idx1 >= 0 && idx2 >= 0) {
          const s = Math.min(idx1, idx2);
          const e = Math.max(idx1, idx2);
          if (e - s >= 2) {
            setRange({
              startIndex: Math.max(0, s),
              endIndex: Math.min(totalCount - 1, e),
            });
          }
        }
      }
      setRefAreaLeft(null);
      setRefAreaRight(null);
    }
  }, [isBoxZoomActive, refAreaLeft, refAreaRight, findIndexByLabel, totalCount]);

  const visibleData = useMemo(() => {
    if (data.length === 0) return [];
    const s = Math.max(0, Math.min(range.startIndex, data.length - 1));
    const e = Math.max(s, Math.min(range.endIndex, data.length - 1));
    return data.slice(s, e + 1);
  }, [data, range.startIndex, range.endIndex]);

  return {
    visibleData,
    startIndex: range.startIndex,
    endIndex: range.endIndex,
    totalCount,
    visibleCount: visibleData.length,
    containerRef,
    zoomIn,
    zoomOut,
    resetView,
    isBoxZoomActive,
    toggleBoxZoom,
    isPanModeActive,
    togglePanMode,
    refAreaLeft,
    refAreaRight,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    isMouseDown,
    isDragging: isMouseDown,
  };
}
