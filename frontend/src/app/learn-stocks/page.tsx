"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  PSE_MARKET_SCHEDULE,
  PSE_MARKET_SCHEDULE_SOURCE,
  BROKER_DIRECTORY,
  EDUCATIONAL_VIDEOS,
  GLOSSARY_TERMS,
} from "@/lib/learnData";

export default function LearnStocksPage() {
  // Accordion open states (default: first open, or based on hash)
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    "trading-101": true,
  });

  // Glossary category filter state
  const [termCategory, setTermCategory] = useState<"all" | "market" | "forecastph">("all");

  // Track expanded terms in glossary
  const [expandedTerms, setExpandedTerms] = useState<Record<string, boolean>>({});

  // Auto-expand accordion if URL contains a hash
  useEffect(() => {
    const handleHash = () => {
      const hash = window.location.hash.replace("#", "");
      if (!hash) return;

      const aliasMap: Record<string, string> = {
        "forecast-prediction": "how-to-read",
        "how-to-read": "how-to-read",
        "trading-101": "trading-101",
        "pse-basics": "pse-basics",
        "trading-terms": "trading-terms",
        "forecast-accuracy": "forecast-accuracy",
        "forecast-charts": "forecast-charts",
        "watch-learn": "watch-learn",
        "pse-brokers": "pse-brokers",
        "research-methodology": "research-methodology",
      };

      const targetId = aliasMap[hash] || hash;
      setOpenSections({ [targetId]: true });

      // Smooth scroll to the target section
      setTimeout(() => {
        const el = document.getElementById(targetId);
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 100);
    };

    handleHash();
    window.addEventListener("hashchange", handleHash);
    return () => window.removeEventListener("hashchange", handleHash);
  }, []);

  const toggleSection = (id: string) => {
    setOpenSections((prev) => (prev[id] ? {} : { [id]: true }));
  };

  const toggleTerm = (term: string) => {
    setExpandedTerms((prev) => ({
      ...prev,
      [term]: !prev[term],
    }));
  };

  const filteredTerms = GLOSSARY_TERMS.filter((t) => {
    if (termCategory === "all") return true;
    return t.category === termCategory;
  });

  return (
    <div className="animate-[fadeIn_0.3s_ease-out] max-w-5xl mx-auto space-y-8 pb-12">
      {/* 1. Header */}
      <div className="bg-dark-card border border-dark-border rounded-2xl p-6 sm:p-8 text-center space-y-3 shadow-sm">
        <span className="inline-block px-3 py-1 text-xs font-semibold text-brand-400 bg-brand-500/10 border border-brand-500/25 rounded-full">
          Educational Center
        </span>
        <h1 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">
          Learn Stocks
        </h1>
        <p className="text-sm sm:text-base text-slate-300 max-w-2xl mx-auto leading-relaxed">
          Learn the basics of Philippine stock trading and understand how to interpret ForecastPH predictions.
        </p>
      </div>

      {/* Accordion List */}
      <div className="space-y-4">
        {/* ================================================================
            1. STOCK TRADING 101
        ================================================================ */}
        <section id="trading-101" className="bg-dark-card border border-dark-border rounded-2xl overflow-hidden shadow-sm">
          <button
            type="button"
            onClick={() => toggleSection("trading-101")}
            aria-expanded={Boolean(openSections["trading-101"])}
            className="w-full text-left px-6 py-5 flex items-center justify-between gap-4 hover:bg-white/5 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">📘</span>
              <div>
                <h2 className="text-lg font-bold text-white leading-tight">1. Stock Trading 101</h2>
                <p className="text-xs text-slate-400 mt-0.5">Foundational concepts of stocks, shares, dividends, and risk</p>
              </div>
            </div>
            <span className="text-xl font-mono text-brand-400 w-8 h-8 rounded-lg bg-dark-bg border border-dark-border flex items-center justify-center shrink-0">
              {openSections["trading-101"] ? "−" : "+"}
            </span>
          </button>

          {openSections["trading-101"] && (
            <div className="px-6 pb-6 pt-2 border-t border-dark-border/60 space-y-5 text-sm text-slate-300 leading-relaxed">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-1.5">
                  <h3 className="font-bold text-white text-base">What is a Stock?</h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    A stock (or equity) represents fractional ownership in a corporation. When you buy a stock, you become a <strong className="text-white">shareholder</strong> and own a tiny portion of the company&apos;s overall assets and future earnings.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-1.5">
                  <h3 className="font-bold text-white text-base">Shares & Public Companies</h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    A company&apos;s total ownership is divided into individual units called <strong className="text-white">shares</strong>. Public companies offer their shares on the Philippine Stock Exchange (PSE) so any verified retail or institutional investor can buy and sell them.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-1.5">
                  <h3 className="font-bold text-white text-base">Capital Gains vs. Losses</h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    A <strong className="text-emerald-400">capital gain</strong> occurs when you sell a stock for a higher price than what you paid for it. A <strong className="text-red-400">capital loss</strong> occurs if the market price drops and you sell below your purchase cost.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-1.5">
                  <h3 className="font-bold text-white text-base">Dividends</h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Some profitable companies return a portion of their earnings directly to shareholders as <strong className="text-amber-400">cash dividends</strong> or additional <strong className="text-amber-400">stock dividends</strong>, providing passive income in addition to share price growth.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-1.5">
                  <h3 className="font-bold text-white text-base">Investing vs. Trading</h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    <strong className="text-white">Investing</strong> focuses on long-term wealth accumulation by holding fundamentally sound businesses over years. <strong className="text-white">Trading</strong> focuses on short-term price movements over days, weeks, or months to capture price volatility.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-1.5">
                  <h3 className="font-bold text-white text-base">Risk & Diversification</h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Stock prices fluctuate continuously based on supply, demand, and news. Spreading capital across diverse industries (e.g. banking, utilities, retail, real estate) helps prevent a downturn in one company from erasing your total portfolio value.
                  </p>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* ================================================================
            2. PHILIPPINE STOCK EXCHANGE BASICS
        ================================================================ */}
        <section id="pse-basics" className="bg-dark-card border border-dark-border rounded-2xl overflow-hidden shadow-sm">
          <button
            type="button"
            onClick={() => toggleSection("pse-basics")}
            aria-expanded={Boolean(openSections["pse-basics"])}
            className="w-full text-left px-6 py-5 flex items-center justify-between gap-4 hover:bg-white/5 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">🏛️</span>
              <div>
                <h2 className="text-lg font-bold text-white leading-tight">2. Philippine Stock Exchange Basics</h2>
                <p className="text-xs text-slate-400 mt-0.5">Authoritative PSE trading hours, sessions, holidays, and mechanics</p>
              </div>
            </div>
            <span className="text-xl font-mono text-brand-400 w-8 h-8 rounded-lg bg-dark-bg border border-dark-border flex items-center justify-center shrink-0">
              {openSections["pse-basics"] ? "−" : "+"}
            </span>
          </button>

          {openSections["pse-basics"] && (
            <div className="px-6 pb-6 pt-2 border-t border-dark-border/60 space-y-5 text-sm text-slate-300 leading-relaxed">
              <p>
                The <strong className="text-white">Philippine Stock Exchange (PSE)</strong> is the national equity exchange of the Philippines. It operates an automated matching engine connecting licensed brokers and investors nationwide.
              </p>

              {/* Official Trading Schedule */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider text-slate-400">
                  Official PSE Trading Schedule (Monday – Friday, Non-Holidays)
                </h3>
                <a
                  href={PSE_MARKET_SCHEDULE_SOURCE.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-block text-xs text-brand-400 hover:text-brand-300"
                >
                  Source: Philippine Stock Exchange — checked {PSE_MARKET_SCHEDULE_SOURCE.checkedOn} ↗
                </a>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {PSE_MARKET_SCHEDULE.map((item) => (
                    <div key={item.phase} className="bg-dark-bg/80 border border-dark-border rounded-xl p-3.5 space-y-1">
                      <span className="text-[11px] font-bold text-brand-400 uppercase tracking-wide">{item.time}</span>
                      <h4 className="text-sm font-semibold text-white">{item.phase}</h4>
                      <p className="text-xs text-slate-400 leading-normal">{item.description}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Holiday & Trading Awareness */}
              <div className="p-4 bg-brand-950/30 border border-brand-500/30 rounded-xl space-y-2">
                <h4 className="text-xs font-bold text-brand-300 uppercase tracking-wide flex items-center gap-1.5">
                  <span>📅</span> Philippine Holidays & Calendar Guard
                </h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  The PSE is closed on all regular and special non-working Philippine public holidays (such as Ninoy Aquino Day, National Heroes Day, Holy Week, etc.). ForecastPH includes an automated calendar guard that skips holiday ingestion and generates forecasts only for valid upcoming PSE trading sessions.
                </p>
              </div>
            </div>
          )}
        </section>

        {/* ================================================================
            3. ESSENTIAL TRADING TERMS
        ================================================================ */}
        <section id="trading-terms" className="bg-dark-card border border-dark-border rounded-2xl overflow-hidden shadow-sm">
          <button
            type="button"
            onClick={() => toggleSection("trading-terms")}
            aria-expanded={Boolean(openSections["trading-terms"])}
            className="w-full text-left px-6 py-5 flex items-center justify-between gap-4 hover:bg-white/5 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">📖</span>
              <div>
                <h2 className="text-lg font-bold text-white leading-tight">3. Essential Trading Terms</h2>
                <p className="text-xs text-slate-400 mt-0.5">Quick reference glossary for market concepts and ForecastPH metrics</p>
              </div>
            </div>
            <span className="text-xl font-mono text-brand-400 w-8 h-8 rounded-lg bg-dark-bg border border-dark-border flex items-center justify-center shrink-0">
              {openSections["trading-terms"] ? "−" : "+"}
            </span>
          </button>

          {openSections["trading-terms"] && (
            <div className="px-6 pb-6 pt-2 border-t border-dark-border/60 space-y-4 text-sm text-slate-300 leading-relaxed">
              {/* Category Filter */}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setTermCategory("all")}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
                    termCategory === "all"
                      ? "bg-brand-600 text-white"
                      : "bg-dark-bg border border-dark-border text-slate-400 hover:text-white"
                  }`}
                >
                  All Terms ({GLOSSARY_TERMS.length})
                </button>
                <button
                  type="button"
                  onClick={() => setTermCategory("market")}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
                    termCategory === "market"
                      ? "bg-brand-600 text-white"
                      : "bg-dark-bg border border-dark-border text-slate-400 hover:text-white"
                  }`}
                >
                  Market Terms (12)
                </button>
                <button
                  type="button"
                  onClick={() => setTermCategory("forecastph")}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
                    termCategory === "forecastph"
                      ? "bg-brand-600 text-white"
                      : "bg-dark-bg border border-dark-border text-slate-400 hover:text-white"
                  }`}
                >
                  ForecastPH Terminology (8)
                </button>
              </div>

              {/* Grid of Compact Term Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-2">
                {filteredTerms.map((t) => (
                  <div key={t.term} className="bg-dark-bg/80 border border-dark-border hover:border-brand-500/40 rounded-xl transition-colors">
                    <button
                      type="button"
                      onClick={() => toggleTerm(t.term)}
                      aria-expanded={Boolean(expandedTerms[t.term])}
                      className="w-full p-4 text-left space-y-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-inset focus:ring-brand-500"
                    >
                    <div className="flex items-start justify-between gap-2">
                      <h4 className="font-bold text-white text-sm">{t.term}</h4>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-dark-card border border-dark-border text-brand-400 font-mono">
                        {t.category === "forecastph" ? "ForecastPH" : "Market"}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 leading-snug">{t.shortDef}</p>
                    {expandedTerms[t.term] && (
                      <p className="text-xs text-slate-400 pt-2 border-t border-dark-border/60 leading-relaxed">
                        {t.detailedDef}
                      </p>
                    )}
                    <span className="inline-block text-[11px] text-brand-400 font-medium">
                      {expandedTerms[t.term] ? "Show less ↑" : "Learn more ↓"}
                    </span>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* ================================================================
            4. HOW TO READ A FORECASTPH PREDICTION
        ================================================================ */}
        <section id="how-to-read" className="bg-dark-card border border-dark-border rounded-2xl overflow-hidden shadow-sm">
          <button
            type="button"
            onClick={() => toggleSection("how-to-read")}
            aria-expanded={Boolean(openSections["how-to-read"])}
            className="w-full text-left px-6 py-5 flex items-center justify-between gap-4 hover:bg-white/5 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">🎯</span>
              <div>
                <h2 className="text-lg font-bold text-white leading-tight">4. How to Read a ForecastPH Prediction</h2>
                <p className="text-xs text-slate-400 mt-0.5">Practical walkthrough of predicted values, models, and real-world factors</p>
              </div>
            </div>
            <span className="text-xl font-mono text-brand-400 w-8 h-8 rounded-lg bg-dark-bg border border-dark-border flex items-center justify-center shrink-0">
              {openSections["how-to-read"] ? "−" : "+"}
            </span>
          </button>

          {openSections["how-to-read"] && (
            <div className="px-6 pb-6 pt-2 border-t border-dark-border/60 space-y-5 text-sm text-slate-300 leading-relaxed">
              <p>
                Every trading day after the PSE closes at 3:15 PM PHT, ForecastPH ingests the latest official quotations report and produces a price forecast for the next upcoming session.
              </p>

              {/* Concrete Example Card */}
              <div className="bg-dark-bg/90 border border-brand-500/40 rounded-2xl p-5 sm:p-6 shadow-[0_0_20px_rgba(59,130,246,0.1)] space-y-4">
                <div className="flex items-center justify-between border-b border-dark-border/80 pb-3">
                  <span className="text-xs font-semibold text-brand-400 uppercase tracking-wide">Sample Prediction Walkthrough</span>
                  <span className="text-xs px-2 py-0.5 rounded bg-brand-500/20 text-brand-300 border border-brand-500/30">Next Trading Session</span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center sm:text-left">
                  <div className="p-3 bg-dark-card border border-dark-border rounded-xl">
                    <p className="text-xs text-slate-400">Current Close</p>
                    <p className="text-xl font-bold text-white font-mono mt-0.5">₱100.00</p>
                  </div>
                  <div className="p-3 bg-dark-card border border-dark-border rounded-xl">
                    <p className="text-xs text-slate-400">Forecasted Close</p>
                    <p className="text-xl font-bold text-brand-400 font-mono mt-0.5">₱102.50</p>
                  </div>
                  <div className="p-3 bg-dark-card border border-dark-border rounded-xl">
                    <p className="text-xs text-slate-400">Expected Movement</p>
                    <p className="text-xl font-bold text-emerald-400 font-mono mt-0.5">▲ +2.50%</p>
                    <p className="text-[11px] text-emerald-300">+₱2.50</p>
                  </div>
                  <div className="p-3 bg-dark-card border border-dark-border rounded-xl">
                    <p className="text-xs text-slate-400">Selected Model</p>
                    <p className="text-base font-bold text-white mt-0.5">LSTM</p>
                    <p className="text-[11px] text-brand-400">Best test-set RMSE</p>
                  </div>
                </div>

                <div className="p-4 bg-dark-card border border-dark-border rounded-xl space-y-2">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider">What this means:</h4>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    &quot;The model estimates that the next trading day&apos;s closing price could be approximately <strong className="text-white">₱102.50</strong>.&quot;
                  </p>
                  <p className="text-xs text-amber-300 font-medium">
                    ⚠️ This does <strong className="underline">not</strong> mean the stock is guaranteed to reach ₱102.50.
                  </p>
                </div>
              </div>

              {/* Factors influencing prices */}
              <div className="space-y-2">
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Why actual prices diverge from forecasts:</h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Machine learning models evaluate historical numerical price and volume patterns. However, live markets react to unexpected real-time events that no purely historical model can anticipate:
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                  <div className="p-3 bg-dark-bg border border-dark-border rounded-xl text-xs space-y-1">
                    <strong className="text-white">Company Disclosures:</strong> Earnings surprises, dividend declarations, executive changes, or mergers.
                  </div>
                  <div className="p-3 bg-dark-bg border border-dark-border rounded-xl text-xs space-y-1">
                    <strong className="text-white">Economic Indicators:</strong> Inflation reports, central bank BSP interest rate decisions, and GDP numbers.
                  </div>
                  <div className="p-3 bg-dark-bg border border-dark-border rounded-xl text-xs space-y-1">
                    <strong className="text-white">Geopolitical Developments:</strong> Global commodity price shifts (oil, metals), foreign exchange swings (USD/PHP).
                  </div>
                  <div className="p-3 bg-dark-bg border border-dark-border rounded-xl text-xs space-y-1">
                    <strong className="text-white">Market Liquidity & Shocks:</strong> Sudden institutional block sales or foreign fund rebalancing events.
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* ================================================================
            5. UNDERSTANDING FORECAST ACCURACY
        ================================================================ */}
        <section id="forecast-accuracy" className="bg-dark-card border border-dark-border rounded-2xl overflow-hidden shadow-sm">
          <button
            type="button"
            onClick={() => toggleSection("forecast-accuracy")}
            aria-expanded={Boolean(openSections["forecast-accuracy"])}
            className="w-full text-left px-6 py-5 flex items-center justify-between gap-4 hover:bg-white/5 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">📊</span>
              <div>
                <h2 className="text-lg font-bold text-white leading-tight">5. Understanding Forecast Accuracy</h2>
                <p className="text-xs text-slate-400 mt-0.5">Plain-language guides to RMSE, MAE, MASE, and R²</p>
              </div>
            </div>
            <span className="text-xl font-mono text-brand-400 w-8 h-8 rounded-lg bg-dark-bg border border-dark-border flex items-center justify-center shrink-0">
              {openSections["forecast-accuracy"] ? "−" : "+"}
            </span>
          </button>

          {openSections["forecast-accuracy"] && (
            <div className="px-6 pb-6 pt-2 border-t border-dark-border/60 space-y-4 text-sm text-slate-300 leading-relaxed">
              <p>
                ForecastPH does not cherry-pick high accuracy numbers. Every company detail page reports rigorous out-of-sample test metrics across 60 chronological trading sessions:
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-white text-base">RMSE (Root Mean Squared Error)</h3>
                    <span className="text-xs text-brand-400 font-mono">₱ Pesos</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    <strong className="text-white">Interpretation:</strong> Lower RMSE indicates smaller overall errors. Because RMSE squares each error, it penalizes occasional large forecast misses more heavily than small consistent misses.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-white text-base">MAE (Mean Absolute Error)</h3>
                    <span className="text-xs text-brand-400 font-mono">₱ Pesos</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    <strong className="text-white">Interpretation:</strong> Lower MAE means predictions are generally closer to actual prices. If a stock has an MAE of ₱0.80, it means the model&apos;s predictions were off by an average of 80 centavos.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-white text-base">MASE (Mean Absolute Scaled Error)</h3>
                    <span className="text-xs text-emerald-400 font-mono">&lt; 1.0 = Beats Naive</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    <strong className="text-white">Interpretation:</strong> Compares the model against a baseline that simply assumes tomorrow&apos;s close equals today&apos;s close. A MASE below 1.0 indicates the model added predictive value beyond a naive guess.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-white text-base">R² (Goodness-of-Fit)</h3>
                    <span className="text-xs text-amber-400 font-mono">0.0 to 1.0</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    <strong className="text-white">Interpretation:</strong> Represents the proportion of variance in test-set prices explained by the model. Higher R² indicates a closer statistical fit, but R² alone does not guarantee future trading profitability.
                  </p>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* ================================================================
            6. UNDERSTANDING FORECASTPH CHARTS
        ================================================================ */}
        <section id="forecast-charts" className="bg-dark-card border border-dark-border rounded-2xl overflow-hidden shadow-sm">
          <button
            type="button"
            onClick={() => toggleSection("forecast-charts")}
            aria-expanded={Boolean(openSections["forecast-charts"])}
            className="w-full text-left px-6 py-5 flex items-center justify-between gap-4 hover:bg-white/5 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">📈</span>
              <div>
                <h2 className="text-lg font-bold text-white leading-tight">6. Understanding ForecastPH Charts</h2>
                <p className="text-xs text-slate-400 mt-0.5">Visual guide to interpreting the 4 charts on every company page</p>
              </div>
            </div>
            <span className="text-xl font-mono text-brand-400 w-8 h-8 rounded-lg bg-dark-bg border border-dark-border flex items-center justify-center shrink-0">
              {openSections["forecast-charts"] ? "−" : "+"}
            </span>
          </button>

          {openSections["forecast-charts"] && (
            <div className="px-6 pb-6 pt-2 border-t border-dark-border/60 space-y-4 text-sm text-slate-300 leading-relaxed">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-2">
                  <h3 className="font-bold text-white text-sm flex items-center gap-2">
                    <span className="text-brand-400">1.</span> Historical OHLCV Line Chart
                  </h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Displays continuous historical price movements alongside trading volume. Includes PSE EDGE-style date-range quick filters (1M, 3M, 6M, 1Y) and interactive zoom and pan controls.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-2">
                  <h3 className="font-bold text-white text-sm flex items-center gap-2">
                    <span className="text-brand-400">2.</span> Next-Day Prediction Chart
                  </h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Shows the latest actual closing price connected by dashed projection lines to all three candidate model predictions (Lag Regression, ARIMA, LSTM) for the upcoming trading session.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-2">
                  <h3 className="font-bold text-white text-sm flex items-center gap-2">
                    <span className="text-brand-400">3.</span> Backtest: Predicted vs. Actual
                  </h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Visualizes the 60-session out-of-sample backtest followed by live forward-tested forecasts. Closer alignment between the predicted line and actual price line indicates smaller historical forecast errors.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-2">
                  <h3 className="font-bold text-white text-sm flex items-center gap-2">
                    <span className="text-brand-400">4.</span> Forecast Error Over Time
                  </h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Plots the exact daily error (Predicted Close minus Actual Close in ₱). Values closer to the center zero-line indicate accurate forecasts, while spikes show unexpected volatility days.
                  </p>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* ================================================================
            7. WATCH & LEARN (LAZY-LOADED VIDEOS)
        ================================================================ */}
        <section id="watch-learn" className="bg-dark-card border border-dark-border rounded-2xl overflow-hidden shadow-sm">
          <button
            type="button"
            onClick={() => toggleSection("watch-learn")}
            aria-expanded={Boolean(openSections["watch-learn"])}
            className="w-full text-left px-6 py-5 flex items-center justify-between gap-4 hover:bg-white/5 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">🎬</span>
              <div>
                <h2 className="text-lg font-bold text-white leading-tight">7. Watch & Learn</h2>
                <p className="text-xs text-slate-400 mt-0.5">Curated educational videos on Philippine stocks, charts, and risk management</p>
              </div>
            </div>
            <span className="text-xl font-mono text-brand-400 w-8 h-8 rounded-lg bg-dark-bg border border-dark-border flex items-center justify-center shrink-0">
              {openSections["watch-learn"] ? "−" : "+"}
            </span>
          </button>

          {openSections["watch-learn"] && (
            <div className="px-6 pb-6 pt-2 border-t border-dark-border/60 space-y-6 text-sm text-slate-300 leading-relaxed">
              <p className="text-xs text-slate-400">
                Videos are loaded on demand to conserve mobile data and maintain fast page performance.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {EDUCATIONAL_VIDEOS.map((v) => (
                  <div key={v.id} className="bg-dark-bg/80 border border-dark-border rounded-xl p-4 space-y-3">
                    <span className="inline-block px-2.5 py-0.5 rounded text-[10px] font-semibold bg-brand-500/15 text-brand-300 border border-brand-500/30">
                      {v.topic}
                    </span>
                    <h3 className="font-bold text-white text-sm leading-snug">{v.title}</h3>

                    {/* Responsive video container */}
                    <div className="relative w-full aspect-video rounded-lg overflow-hidden bg-slate-900 border border-dark-border">
                      <iframe
                        src={`https://www.youtube-nocookie.com/embed/${v.youtubeId}`}
                        title={v.title}
                        loading="lazy"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                        className="absolute inset-0 w-full h-full"
                      />
                    </div>

                    <p className="text-xs text-slate-400 leading-relaxed">{v.description}</p>
                    <div className="pt-2 border-t border-dark-border/50 flex items-center justify-between text-xs">
                      <span className="text-slate-500 font-medium">{v.channel}</span>
                      <a
                        href={v.directUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-brand-400 hover:text-brand-300 font-medium inline-flex items-center gap-1"
                      >
                        Open on YouTube ↗
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* ================================================================
            8. PSE TRADING PARTICIPANTS & ONLINE BROKERS
        ================================================================ */}
        <section id="pse-brokers" className="bg-dark-card border border-dark-border rounded-2xl overflow-hidden shadow-sm">
          <button
            type="button"
            onClick={() => toggleSection("pse-brokers")}
            aria-expanded={Boolean(openSections["pse-brokers"])}
            className="w-full text-left px-6 py-5 flex items-center justify-between gap-4 hover:bg-white/5 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">💼</span>
              <div>
                <h2 className="text-lg font-bold text-white leading-tight">8. PSE Trading Participants & Online Brokers</h2>
                <p className="text-xs text-slate-400 mt-0.5">Directory of Philippine stock brokerage platforms; verify current status before opening an account</p>
              </div>
            </div>
            <span className="text-xl font-mono text-brand-400 w-8 h-8 rounded-lg bg-dark-bg border border-dark-border flex items-center justify-center shrink-0">
              {openSections["pse-brokers"] ? "−" : "+"}
            </span>
          </button>

          {openSections["pse-brokers"] && (
            <div className="px-6 pb-6 pt-2 border-t border-dark-border/60 space-y-5 text-sm text-slate-300 leading-relaxed">
              {/* Mandatory Neutrality Disclaimer */}
              <div className="p-4 bg-dark-bg border border-dark-border/80 rounded-xl space-y-1.5 text-xs text-slate-400 leading-relaxed">
                <p className="font-semibold text-slate-300">Important Disclaimer:</p>
                <p>
                  ForecastPH is not affiliated with or endorsed by the brokers listed here. Broker information is provided for educational purposes. Verify current registration, requirements, fees, and services directly with the PSE, SEC, and the broker before opening an account.
                </p>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3">
                <a
                  href="https://www.pse.com.ph/directory/#tp1"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-brand-400 hover:text-brand-300 font-medium inline-flex items-center gap-1"
                >
                  View Complete PSE Trading Participant Directory ↗
                </a>
              </div>

              {/* Broker Cards Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 pt-1">
                {BROKER_DIRECTORY.map((b) => (
                  <div
                    key={b.id}
                    className="bg-dark-bg/80 border border-dark-border rounded-xl p-4 flex flex-col justify-between space-y-3"
                  >
                    <div className="space-y-1.5">
                      <div className="flex items-start justify-between gap-2">
                        <h3 className="font-bold text-white text-base leading-tight">{b.name}</h3>
                        <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 whitespace-nowrap">
                          PSE directory reference
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 font-medium">{b.parentEntity}</p>
                      <p className="text-[11px] text-slate-400">{b.pseStatus}</p>
                      <p className="text-xs text-slate-300 pt-1 leading-relaxed">{b.description}</p>
                    </div>

                    <div className="pt-3 border-t border-dark-border/60 flex items-center justify-between text-xs">
                      <a
                        href={b.websiteUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-brand-400 hover:text-brand-300 font-semibold"
                      >
                        Official Website ↗
                      </a>
                      <a
                        href={b.pseDirectoryUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-slate-400 hover:text-slate-200"
                      >
                        PSE Directory ↗
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* ================================================================
            9. ABOUT THE FORECASTPH RESEARCH
        ================================================================ */}
        <section id="research-methodology" className="bg-dark-card border border-dark-border rounded-2xl overflow-hidden shadow-sm">
          <button
            type="button"
            onClick={() => toggleSection("research-methodology")}
            aria-expanded={Boolean(openSections["research-methodology"])}
            className="w-full text-left px-6 py-5 flex items-center justify-between gap-4 hover:bg-white/5 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">🔬</span>
              <div>
                <h2 className="text-lg font-bold text-white leading-tight">9. About the ForecastPH Research</h2>
                <p className="text-xs text-slate-400 mt-0.5">Machine learning architectures, walk-forward evaluation, and research methodology</p>
              </div>
            </div>
            <span className="text-xl font-mono text-brand-400 w-8 h-8 rounded-lg bg-dark-bg border border-dark-border flex items-center justify-center shrink-0">
              {openSections["research-methodology"] ? "−" : "+"}
            </span>
          </button>

          {openSections["research-methodology"] && (
            <div className="px-6 pb-6 pt-2 border-t border-dark-border/60 space-y-6 text-sm text-slate-300 leading-relaxed">
              <p>
                ForecastPH is an academic capstone research project evaluating whether machine learning and deep learning models can outperform traditional time-series methods on the Philippine Stock Exchange across diverse market sectors.
              </p>

              {/* The 3 Core Models */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xl">🧮</span>
                    <h3 className="font-bold text-white text-base">Lag Regression</h3>
                  </div>
                  <p className="text-xs font-semibold text-brand-400">&quot;The Pattern Spotter&quot;</p>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    An interpretable machine learning model analyzing historical price and volume lags. Uses Partial Autocorrelation Function (PACF) to identify repeating lag cycles and LASSO regularization to eliminate non-informative features without overfitting.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xl">📈</span>
                    <h3 className="font-bold text-white text-base">ARIMA</h3>
                  </div>
                  <p className="text-xs font-semibold text-blue-400">&quot;The Trend Tracker&quot;</p>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    A classical statistical time-series standard (Autoregressive Integrated Moving Average). Uses statistical differencing to achieve stationarity, tracks underlying trends, and models error autocorrelation with strict convergence verification.
                  </p>
                </div>

                <div className="bg-dark-bg/80 border border-dark-border p-4.5 rounded-xl space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xl">🧠</span>
                    <h3 className="font-bold text-white text-base">LSTM</h3>
                  </div>
                  <p className="text-xs font-semibold text-purple-400">&quot;Deep Sequence Memory&quot;</p>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    A recurrent neural network with specialized input, forget, and output gates. Learns non-linear temporal dependencies across historical multi-day sequences using normalized training data with zero lookahead leakage.
                  </p>
                </div>
              </div>

              {/* Research Methodology Notes */}
              <div className="p-4.5 bg-dark-bg border border-dark-border rounded-xl space-y-2 text-xs text-slate-400">
                <h4 className="font-bold text-slate-200 uppercase tracking-wide">Research Governance & Dual-Track Separation:</h4>
                <ul className="list-disc list-inside space-y-1 text-slate-300">
                  <li><strong className="text-white">Formal Benchmark Track:</strong> Frozen cross-validation folds (5 expanding-window folds) evaluated with non-parametric hypothesis tests (Friedman test, Wilcoxon signed-rank test with Holm correction).</li>
                  <li><strong className="text-white">Production Refresh Track:</strong> Deployed models refreshed weekly with challenger gatekeeping to serve daily live predictions without mutating formal research baselines.</li>
                  <li><strong className="text-white">Baseline Naive Benchmark:</strong> Every model is evaluated against the random walk naive benchmark (tomorrow&apos;s price = today&apos;s price) to ensure real added predictive value (MASE &lt; 1.0).</li>
                </ul>
              </div>
            </div>
          )}
        </section>
      </div>

      {/* Bottom CTA */}
      <div className="p-6 bg-dark-card border border-dark-border rounded-2xl text-center space-y-3 shadow-sm">
        <h3 className="text-lg font-bold text-white">Ready to inspect live predictions?</h3>
        <p className="text-xs text-slate-400 max-w-lg mx-auto">
          Explore the 15 Philippine companies tracked across banking, property, retail, mining, and utilities.
        </p>
        <div className="pt-1 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/companies"
            className="px-5 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white text-sm font-semibold shadow-md shadow-brand-500/20 transition-all"
          >
            Explore Companies →
          </Link>
          <Link
            href="/watchlist"
            className="px-5 py-2.5 rounded-xl bg-dark-bg border border-dark-border hover:border-slate-500 text-slate-200 text-sm font-semibold transition-all"
          >
            My Watchlist ★
          </Link>
        </div>
      </div>
    </div>
  );
}
