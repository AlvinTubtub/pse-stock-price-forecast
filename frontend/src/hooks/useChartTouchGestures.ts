"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEventHandler, RefObject } from "react";
import {
  fullViewport,
  isViewportModified,
  normalizeViewport,
  panViewport,
  pixelDragToIndexShift,
  pointDistance,
  pointMidpoint,
  removeActivePointer,
  viewportSize,
  viewportsEqual,
  wheelDeltaToScale,
  zoomViewportAtIndex,
  type ChartViewport,
  type PointPosition,
  type ViewportBounds,
} from "@/lib/chartViewport";

interface UseChartTouchGesturesOptions {
  totalPoints: number;
  minWindow: number;
  resetKey?: string | number;
}

interface PanStart {
  point: PointPosition;
  viewport: ChartViewport;
  intent: "pending" | "horizontal" | "vertical";
}

interface PinchStart {
  distance: number;
  anchorIndex: number;
  viewport: ChartViewport;
}

export interface ChartInteractionResult {
  viewport: ChartViewport;
  zoomIn: () => void;
  zoomOut: () => void;
  reset: () => void;
  canZoomIn: boolean;
  canZoomOut: boolean;
  canPan: boolean;
  isDragging: boolean;
  isViewportModified: boolean;
  surfaceRef: RefObject<HTMLDivElement>;
  handlers: {
    onPointerDown: PointerEventHandler<HTMLDivElement>;
    onPointerMove: PointerEventHandler<HTMLDivElement>;
    onPointerUp: PointerEventHandler<HTMLDivElement>;
    onPointerCancel: PointerEventHandler<HTMLDivElement>;
    onLostPointerCapture: PointerEventHandler<HTMLDivElement>;
  };
}

const INTENT_THRESHOLD_PX = 7;
const HORIZONTAL_INTENT_RATIO = 1.15;

export function useChartInteractions({
  totalPoints,
  minWindow,
  resetKey,
}: UseChartTouchGesturesOptions): ChartInteractionResult {
  const maxIndex = totalPoints - 1;
  const bounds: ViewportBounds = { minIndex: 0, maxIndex, minWindow };
  const [storedViewport, setStoredViewport] = useState<ChartViewport>(() => fullViewport(totalPoints));
  const viewport = normalizeViewport(storedViewport, bounds);
  const viewportRef = useRef(viewport);
  const pointersRef = useRef(new Map<number, PointPosition>());
  const mousePointerRef = useRef<number | null>(null);
  const panRef = useRef<PanStart | null>(null);
  const pinchRef = useRef<PinchStart | null>(null);
  const frameRef = useRef<number | null>(null);
  const pendingViewportRef = useRef<ChartViewport | null>(null);
  const surfaceRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  viewportRef.current = viewport;

  const applyPendingViewport = useCallback(() => {
    frameRef.current = null;
    const pending = pendingViewportRef.current;
    pendingViewportRef.current = null;
    if (pending) setStoredViewport(pending);
  }, []);

  const scheduleViewport = useCallback((nextViewport: ChartViewport) => {
    pendingViewportRef.current = normalizeViewport(nextViewport, bounds);
    if (frameRef.current === null) {
      frameRef.current = window.requestAnimationFrame(applyPendingViewport);
    }
  }, [applyPendingViewport, bounds.maxIndex, bounds.minIndex, bounds.minWindow]);

  const currentViewport = useCallback(
    () => pendingViewportRef.current ?? viewportRef.current,
    [],
  );

  const reset = useCallback(() => {
    if (frameRef.current !== null) window.cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    pendingViewportRef.current = null;
    setStoredViewport(fullViewport(totalPoints));
  }, [totalPoints]);

  useEffect(() => {
    reset();
    pointersRef.current.clear();
    mousePointerRef.current = null;
    panRef.current = null;
    pinchRef.current = null;
    setIsDragging(false);
  }, [reset, resetKey]);

  useEffect(() => () => {
    if (frameRef.current !== null) window.cancelAnimationFrame(frameRef.current);
  }, []);

  const startPinch = useCallback((element: HTMLDivElement) => {
    const points = [...pointersRef.current.values()];
    if (points.length < 2) return;
    const [left, right] = points;
    const distance = pointDistance(left, right);
    const midpoint = pointMidpoint(left, right);
    const rect = element.getBoundingClientRect();
    const ratio = rect.width > 0 ? Math.min(1, Math.max(0, (midpoint.x - rect.left) / rect.width)) : 0.5;
    const baseViewport = currentViewport();
    pinchRef.current = {
      distance: Math.max(distance, 1),
      anchorIndex: baseViewport.startIndex + ratio * Math.max(0, viewportSize(baseViewport) - 1),
      viewport: baseViewport,
    };
    panRef.current = null;
  }, [currentViewport]);

  const onPointerDown = useCallback<PointerEventHandler<HTMLDivElement>>((event) => {
    if (event.pointerType === "mouse") {
      if (event.button !== 0) return;
      mousePointerRef.current = event.pointerId;
      panRef.current = {
        point: { x: event.clientX, y: event.clientY },
        viewport: currentViewport(),
        intent: "pending",
      };
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch {
        // Pointer capture can be unavailable if the pointer has already ended.
      }
      return;
    }

    if (event.pointerType !== "touch") return;
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Pointer capture may be unavailable after a browser-driven cancellation.
    }

    if (pointersRef.current.size === 1) {
      panRef.current = {
        point: { x: event.clientX, y: event.clientY },
        viewport: currentViewport(),
        intent: "pending",
      };
      pinchRef.current = null;
    } else if (pointersRef.current.size === 2) {
      startPinch(event.currentTarget);
    }
  }, [currentViewport, startPinch]);

  const onPointerMove = useCallback<PointerEventHandler<HTMLDivElement>>((event) => {
    if (event.pointerType === "mouse") {
      const pan = panRef.current;
      if (mousePointerRef.current !== event.pointerId || !pan) return;
      const deltaX = event.clientX - pan.point.x;
      if (pan.intent === "pending" && Math.abs(deltaX) > INTENT_THRESHOLD_PX) {
        pan.intent = "horizontal";
        setIsDragging(true);
      }
      if (pan.intent !== "horizontal") return;

      const rect = event.currentTarget.getBoundingClientRect();
      scheduleViewport(panViewport(
        pan.viewport,
        pixelDragToIndexShift(deltaX, rect.width, viewportSize(pan.viewport)),
        bounds,
      ));
      event.preventDefault();
      return;
    }

    if (event.pointerType !== "touch" || !pointersRef.current.has(event.pointerId)) return;
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const rect = event.currentTarget.getBoundingClientRect();

    if (pointersRef.current.size >= 2 && pinchRef.current) {
      const [left, right] = [...pointersRef.current.values()];
      const distance = pointDistance(left, right);
      const midpoint = pointMidpoint(left, right);
      const targetRatio = rect.width > 0
        ? Math.min(1, Math.max(0, (midpoint.x - rect.left) / rect.width))
        : 0.5;
      scheduleViewport(zoomViewportAtIndex(
        pinchRef.current.viewport,
        pinchRef.current.distance / Math.max(distance, 1),
        pinchRef.current.anchorIndex,
        targetRatio,
        bounds,
      ));
      event.preventDefault();
      return;
    }

    const pan = panRef.current;
    if (!pan || pointersRef.current.size !== 1) return;
    const deltaX = event.clientX - pan.point.x;
    const deltaY = event.clientY - pan.point.y;

    if (pan.intent === "pending") {
      if (Math.abs(deltaX) > INTENT_THRESHOLD_PX && Math.abs(deltaX) > Math.abs(deltaY) * HORIZONTAL_INTENT_RATIO) {
        pan.intent = "horizontal";
      } else if (Math.abs(deltaY) > INTENT_THRESHOLD_PX && Math.abs(deltaY) >= Math.abs(deltaX) * HORIZONTAL_INTENT_RATIO) {
        pan.intent = "vertical";
      }
    }

    if (pan.intent !== "horizontal" || rect.width <= 0) return;
    const pointOffset = pixelDragToIndexShift(deltaX, rect.width, viewportSize(pan.viewport));
    scheduleViewport(panViewport(pan.viewport, pointOffset, bounds));
    event.preventDefault();
  }, [bounds.maxIndex, bounds.minIndex, bounds.minWindow, scheduleViewport]);

  const finishPointer = useCallback<PointerEventHandler<HTMLDivElement>>((event) => {
    if (event.pointerType === "mouse") {
      if (mousePointerRef.current !== event.pointerId) return;
      mousePointerRef.current = null;
      panRef.current = null;
      pinchRef.current = null;
      setIsDragging(false);
      try {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
      } catch {
        // Lost capture is already a complete cleanup for this pointer.
      }
      return;
    }

    if (event.pointerType !== "touch") return;
    removeActivePointer(pointersRef.current, event.pointerId);
    try {
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    } catch {
      // A cancelled pointer can lose capture before this handler runs.
    }

    if (pointersRef.current.size >= 2) {
      startPinch(event.currentTarget);
    } else if (pointersRef.current.size === 1) {
      const point = [...pointersRef.current.values()][0];
      panRef.current = { point, viewport: currentViewport(), intent: "pending" };
      pinchRef.current = null;
    } else {
      panRef.current = null;
      pinchRef.current = null;
    }
  }, [currentViewport, startPinch]);

  const zoomBy = useCallback((scale: number) => {
    const base = currentViewport();
    const anchor = base.startIndex + Math.max(0, viewportSize(base) - 1) / 2;
    scheduleViewport(zoomViewportAtIndex(base, scale, anchor, 0.5, bounds));
  }, [bounds.maxIndex, bounds.minIndex, bounds.minWindow, currentViewport, scheduleViewport]);

  const onWheel = useCallback((event: WheelEvent) => {
    if (event.deltaY === 0 || totalPoints <= 1) return;
    if (event.target instanceof Element && event.target.closest("button")) return;
    const surface = surfaceRef.current;
    if (!surface) return;
    const rect = surface.getBoundingClientRect();
    if (rect.width <= 0) return;

    const base = currentViewport();
    const anchorRatio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const anchorIndex = base.startIndex + anchorRatio * Math.max(0, viewportSize(base) - 1);
    const next = zoomViewportAtIndex(
      base,
      wheelDeltaToScale(event.deltaY),
      anchorIndex,
      anchorRatio,
      bounds,
    );
    if (viewportsEqual(next, base)) return;

    event.preventDefault();
    scheduleViewport(next);
  }, [bounds.maxIndex, bounds.minIndex, bounds.minWindow, currentViewport, scheduleViewport, totalPoints]);

  useEffect(() => {
    const surface = surfaceRef.current;
    if (!surface) return;
    surface.addEventListener("wheel", onWheel, { passive: false });
    return () => surface.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  const size = viewportSize(viewport);
  const minimum = Math.min(Math.max(1, minWindow), totalPoints);

  return {
    viewport,
    zoomIn: () => zoomBy(0.75),
    zoomOut: () => zoomBy(1.35),
    reset,
    canZoomIn: size > minimum,
    canZoomOut: size < totalPoints,
    canPan: size > 0 && size < totalPoints,
    isDragging,
    isViewportModified: isViewportModified(viewport, fullViewport(totalPoints), bounds),
    surfaceRef,
    handlers: {
      onPointerDown,
      onPointerMove,
      onPointerUp: finishPointer,
      onPointerCancel: finishPointer,
      onLostPointerCapture: finishPointer,
    },
  };
}
