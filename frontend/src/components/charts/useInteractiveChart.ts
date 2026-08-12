"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";

export interface UseInteractiveChartOptions<T> {
  data: T[];
  xKey: keyof T;
}

export function useInteractiveChart<T extends Record<string, any>>({
  data,
  xKey,
}: UseInteractiveChartOptions<T>) {
  const totalCount = data.length;

  const [startIndex, setStartIndex] = useState<number>(0);
  const [endIndex, setEndIndex] = useState<number>(Math.max(0, totalCount - 1));
  const [isBoxZoomActive, setIsBoxZoomActive] = useState<boolean>(false);
  const [isPanModeActive, setIsPanModeActive] = useState<boolean>(false);
  const [refAreaLeft, setRefAreaLeft] = useState<string | number | null>(null);
  const [refAreaRight, setRefAreaRight] = useState<string | number | null>(null);
  const [isMouseDown, setIsMouseDown] = useState<boolean>(false);
  const [dragStartLabel, setDragStartLabel] = useState<string | number | null>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);

  // Sync index bounds if data changes
  useEffect(() => {
    if (data.length > 0) {
      setStartIndex(0);
      setEndIndex(data.length - 1);
    }
  }, [data]);

  const findIndexByLabel = useCallback(
    (label: string | number) => {
      return data.findIndex((item) => item[xKey] === label);
    },
    [data, xKey]
  );

  const resetView = useCallback(() => {
    if (totalCount === 0) return;
    setStartIndex(0);
    setEndIndex(totalCount - 1);
    setRefAreaLeft(null);
    setRefAreaRight(null);
    setIsBoxZoomActive(false);
    setIsPanModeActive(false);
  }, [totalCount]);

  const zoomIn = useCallback(
    (factor = 0.2) => {
      if (totalCount === 0) return;
      setStartIndex((prevStart) => {
        let currentEnd = totalCount - 1;
        setEndIndex((prevEnd) => {
          currentEnd = prevEnd;
          return prevEnd;
        });

        const len = currentEnd - prevStart;
        if (len <= 5) return prevStart;
        const change = Math.max(1, Math.floor(len * factor));
        const newStart = Math.min(currentEnd - 5, prevStart + Math.floor(change / 2));
        const newEnd = Math.max(newStart + 5, currentEnd - Math.ceil(change / 2));
        setEndIndex(newEnd);
        return newStart;
      });
    },
    [totalCount]
  );

  const zoomOut = useCallback(
    (factor = 0.2) => {
      if (totalCount === 0) return;
      setStartIndex((prevStart) => {
        let currentEnd = totalCount - 1;
        setEndIndex((prevEnd) => {
          currentEnd = prevEnd;
          return prevEnd;
        });

        const len = currentEnd - prevStart;
        if (len >= totalCount - 1) {
          setEndIndex(totalCount - 1);
          return 0;
        }
        const change = Math.max(1, Math.floor(len * factor));
        let newStart = Math.max(0, prevStart - Math.floor(change / 2));
        let newEnd = Math.min(totalCount - 1, currentEnd + Math.ceil(change / 2));

        if (newStart === 0) {
          newEnd = Math.min(totalCount - 1, newStart + len + change);
        } else if (newEnd === totalCount - 1) {
          newStart = Math.max(0, newEnd - (len + change));
        }

        setEndIndex(newEnd);
        return newStart;
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
          setStartIndex((prevStart) => {
            let currentEnd = totalCount - 1;
            setEndIndex((prevEnd) => {
              currentEnd = prevEnd;
              return prevEnd;
            });

            const len = currentEnd - prevStart;
            let newStart = prevStart - delta;
            let newEnd = newStart + len;

            if (newStart < 0) {
              newStart = 0;
              newEnd = len;
            } else if (newEnd >= totalCount) {
              newEnd = totalCount - 1;
              newStart = Math.max(0, newEnd - len);
            }

            setEndIndex(newEnd);
            return newStart;
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
          const newStart = Math.min(idx1, idx2);
          const newEnd = Math.max(idx1, idx2);
          if (newEnd - newStart >= 2) {
            setStartIndex(newStart);
            setEndIndex(newEnd);
          }
        }
      }
      setRefAreaLeft(null);
      setRefAreaRight(null);
    }
  }, [isBoxZoomActive, refAreaLeft, refAreaRight, findIndexByLabel]);

  const visibleData = useMemo(() => {
    if (data.length === 0) return [];
    const s = Math.max(0, Math.min(startIndex, data.length - 1));
    const e = Math.max(s, Math.min(endIndex, data.length - 1));
    return data.slice(s, e + 1);
  }, [data, startIndex, endIndex]);

  return {
    visibleData,
    startIndex,
    endIndex,
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
