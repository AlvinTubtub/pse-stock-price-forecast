"use client";

import { useState } from "react";

export interface CompanyLogoProps {
  symbol: string;
  name?: string;
  size?: "xs" | "sm" | "md" | "lg" | "xl";
  className?: string;
}

const SIZE_CONFIGS = {
  xs: {
    container: "w-5 h-5 rounded-md p-0.5",
    text: "text-[8px]",
  },
  sm: {
    container: "w-7 h-7 rounded-lg p-0.5",
    text: "text-[10px]",
  },
  md: {
    container: "w-10 h-10 rounded-xl p-1",
    text: "text-xs",
  },
  lg: {
    container: "w-12 h-12 rounded-xl p-1.5",
    text: "text-sm",
  },
  xl: {
    container: "w-16 h-16 rounded-2xl p-2",
    text: "text-base",
  },
};

export default function CompanyLogo({
  symbol,
  name,
  size = "md",
  className = "",
}: CompanyLogoProps) {
  const [hasError, setHasError] = useState(false);
  const config = SIZE_CONFIGS[size] || SIZE_CONFIGS.md;
  const sym = symbol?.toUpperCase() || "";

  if (hasError || !sym) {
    return (
      <div
        className={`${config.container} bg-brand-600/10 border border-brand-500/30 text-brand-400 font-bold flex items-center justify-center shrink-0 uppercase select-none ${className}`}
        title={name || sym}
      >
        <span className={config.text}>{sym.slice(0, 3)}</span>
      </div>
    );
  }

  return (
    <div
      className={`${config.container} bg-white border border-slate-200/90 dark:border-slate-700/60 shadow-xs flex items-center justify-center shrink-0 overflow-hidden ${className}`}
      title={name ? `${name} (${sym})` : sym}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`/logos/${sym}.png`}
        alt={`${sym} official logo`}
        className="w-full h-full object-contain"
        loading="lazy"
        onError={() => setHasError(true)}
      />
    </div>
  );
}
