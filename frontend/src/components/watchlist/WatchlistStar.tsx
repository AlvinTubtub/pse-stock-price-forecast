"use client";

import React from "react";
import { useWatchlist } from "@/context/WatchlistContext";

export interface WatchlistStarProps {
  symbol: string;
  showLabel?: boolean;
  className?: string;
  size?: "sm" | "md";
}

export default function WatchlistStar({
  symbol,
  showLabel = false,
  className = "",
  size = "md",
}: WatchlistStarProps) {
  const { isWatching, toggleWatchlist } = useWatchlist();
  const watching = isWatching(symbol);

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    toggleWatchlist(symbol);
  };

  const iconSizes = {
    sm: "w-4 h-4",
    md: "w-4 h-4",
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      title={watching ? `Remove ${symbol} from Watchlist` : `Add ${symbol} to Watchlist`}
      aria-label={watching ? `Watching ${symbol}` : `Add ${symbol} to Watchlist`}
      className={`inline-flex items-center gap-1 transition-all cursor-pointer select-none rounded-lg font-medium ${
        showLabel
          ? watching
            ? "px-2 py-1 text-[10px] bg-amber-500/15 border border-amber-500/40 text-amber-300 hover:bg-amber-500/25"
            : "px-1.5 py-1 text-[9px] bg-brand-500/10 border border-brand-500/40 text-brand-300 hover:bg-brand-500/20 hover:text-brand-200 rounded-md"
          : watching
          ? "p-1.5 text-amber-400 hover:text-amber-300 hover:bg-amber-400/10 rounded-md"
          : "p-1 text-amber-400 hover:text-amber-300"
      } ${className}`}
    >
      {/* Star Icon */}
      {watching ? (
        <svg
          className={`${showLabel ? "w-3 h-3" : iconSizes[size]} fill-amber-400 text-amber-400 shrink-0`}
          viewBox="0 0 24 24"
        >
          <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" />
        </svg>
      ) : (
        <svg
          className={`${showLabel ? "w-3 h-3" : iconSizes[size]} fill-none stroke-currentColor stroke-2 shrink-0`}
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
          />
        </svg>
      )}

      {showLabel && (
        <span className="leading-none whitespace-nowrap">
          {watching ? "Watching" : "Add to Watchlist"}
        </span>
      )}
    </button>
  );
}
