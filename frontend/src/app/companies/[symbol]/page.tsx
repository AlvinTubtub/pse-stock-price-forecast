import { notFound } from "next/navigation";
import Link from "next/link";
import HistoryChart from "@/components/charts/HistoryChart";
import NextDayPredictionChart from "@/components/charts/NextDayPredictionChart";
import PredictionChart from "@/components/charts/PredictionChart";
import ErrorChart from "@/components/charts/ErrorChart";
import ChangeBadge from "@/components/ChangeBadge";
import StatCard from "@/components/StatCard";
import { getAllSymbols, getCompanyDetail } from "@/lib/data";
import { formatDate, formatNum, formatPeso, formatPct } from "@/lib/format";

export async function generateStaticParams() {
  const symbols = await getAllSymbols();
  return symbols.map((symbol) => ({ symbol }));
}

export default async function CompanyDetailPage({ params }: { params: { symbol: string } }) {
  const company = await getCompanyDetail(params.symbol);
  if (!company) notFound();

  const modelOrder = ["lag_reg", "arima", "lstm", "naive"];
  const modelLabels: Record<string, string> = {
    lag_reg: "Lag-Informed Regression",
    arima: "ARIMA",
    lstm: "LSTM",
    naive: "Naive baseline",
  };

  const selectedModelKey =
    Object.keys(modelLabels).find((key) => modelLabels[key] === company.model) || "arima";
  const selectedMetrics = company.metrics[selectedModelKey] || {
    rmse: "--",
    mae: "--",
    mase: "--",
    r2: "--",
  };

  const maseVal = parseFloat(String(selectedMetrics.mase));
  const beatsNaive = !isNaN(maseVal) && maseVal < 1.0;

  return (
    <div className="space-y-8">
      {/* 1. Header & Navigation */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <Link
            href="/companies"
            className="text-xs font-semibold text-brand-400 hover:text-brand-300 transition-colors uppercase tracking-wider mb-1 inline-block"
          >
            ← Back to Companies
          </Link>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold text-white tracking-tight">{company.symbol}</h1>
            <span className="text-xs px-2.5 py-1 rounded-md bg-dark-bg border border-dark-border text-slate-300 font-medium">
              {company.sector}
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-0.5">{company.name}</p>
        </div>

        {/* Date context */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400 bg-dark-card border border-dark-border px-3.5 py-2 rounded-xl">
          {company.dataAsOf && (
            <div>
              <span className="text-slate-500">Data as of: </span>
              <strong className="text-slate-300 font-medium">{formatDate(company.dataAsOf)}</strong>
            </div>
          )}
          {company.dataAsOf && company.forecastDate && <span>&middot;</span>}
          {company.forecastDate && (
            <div>
              <span className="text-slate-500">Forecast for: </span>
              <strong className="text-brand-300 font-semibold">
                {formatDate(company.forecastDate)}
              </strong>
            </div>
          )}
        </div>
      </div>

      {/* 2. Top Stats Grid */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Previous Close" value={formatPeso(company.previousClose)} />
        <StatCard
          label="Forecasted Close"
          value={formatPeso(company.predictedClose)}
          accent="text-white"
        />
        <div className="bg-dark-card border border-dark-border rounded-xl p-5 shadow-sm">
          <p className="text-xs uppercase tracking-wide text-slate-400 mb-2">Expected Change</p>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-white">{formatPeso(company.pesoChange)}</span>
            <ChangeBadge pctChange={company.pctChange} />
          </div>
        </div>
        <div className="bg-dark-card border border-dark-border rounded-xl p-5 shadow-sm">
          <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Selected Model</p>
          <p className="text-base font-bold text-white truncate" title={company.model}>
            {company.model}
          </p>
          <p className="text-[11px] text-brand-400 mt-0.5">Lowest test-set RMSE</p>
        </div>
      </section>

      {/* 3. Selected Model Summary Panel */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-4 pb-3 border-b border-dark-border/60">
          <div>
            <h2 className="text-lg font-semibold text-white">
              Selected Model Summary: {company.model}
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Metrics calculated strictly for this company&apos;s out-of-sample test split.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`text-xs px-2.5 py-1 rounded-full font-semibold border ${
                beatsNaive
                  ? "bg-green-500/10 text-green-400 border-green-500/30"
                  : "bg-amber-500/10 text-amber-400 border-amber-500/30"
              }`}
            >
              {beatsNaive ? "✓ Beats Naive Baseline (MASE < 1.0)" : "⚠ Worse Than Naive (MASE ≥ 1.0)"}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-dark-bg border border-dark-border rounded-lg p-3.5">
            <p className="text-xs text-slate-400 mb-0.5">Test RMSE</p>
            <p className="text-lg font-bold text-white font-mono">
              {formatNum(selectedMetrics.rmse, 4)}
            </p>
            <p className="text-[11px] text-slate-500 mt-0.5">Scale-dependent error</p>
          </div>
          <div className="bg-dark-bg border border-dark-border rounded-lg p-3.5">
            <p className="text-xs text-slate-400 mb-0.5">Test MAE</p>
            <p className="text-lg font-bold text-white font-mono">
              {formatNum(selectedMetrics.mae, 4)}
            </p>
            <p className="text-[11px] text-slate-500 mt-0.5">Mean absolute error</p>
          </div>
          <div className="bg-dark-bg border border-dark-border rounded-lg p-3.5">
            <p className="text-xs text-slate-400 mb-0.5">MASE</p>
            <p
              className={`text-lg font-bold font-mono ${beatsNaive ? "text-green-400" : "text-amber-400"}`}
            >
              {formatNum(selectedMetrics.mase, 4)}
            </p>
            <p className="text-[11px] text-slate-500 mt-0.5">&lt; 1.0 beats naive</p>
          </div>
          <div className="bg-dark-bg border border-dark-border rounded-lg p-3.5">
            <p className="text-xs text-slate-400 mb-0.5">Goodness-of-Fit (R²)</p>
            <p className="text-lg font-bold text-white font-mono">
              {formatNum(selectedMetrics.r2, 4)}
            </p>
            <p className="text-[11px] text-slate-500 mt-0.5">Test set variance explained</p>
          </div>
        </div>

        <p className="text-xs text-slate-400 mt-3.5 leading-relaxed">
          <strong className="text-slate-300">Note: </strong>
          MASE below 1.0 indicates lower forecast error than the naive baseline (predicting
          tomorrow&apos;s close equals today&apos;s close). R² is a supplementary goodness-of-fit
          metric and is not a forecast confidence probability.
        </p>
      </section>

      {/* 4. Historical OHLCV */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Historical OHLCV</h2>
        <HistoryChart data={company.ohlcv} />
      </section>

      {/* 5. Next-Day Prediction */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-1">Next-Day Prediction</h2>
        <p className="text-sm text-slate-400 mb-4">
          Latest actual close with model predictions for the next trading session.
        </p>
        <NextDayPredictionChart
          ohlcv={company.ohlcv}
          previousClose={company.previousClose}
          nextClose={company.nextClose}
          forecastDate={company.forecastDate}
          dataAsOf={company.dataAsOf}
        />
      </section>

      {/* 6. Backtest: Predicted vs. Actual */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-1">
          Backtest: Predicted vs. Actual (Last 60 Sessions)
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          Historical backtest comparing each model&apos;s predictions with the actual closing price.
        </p>
        <PredictionChart
          dates={company.backtestDates}
          actual={company.backtestActual}
          byModel={company.backtestByModel}
          selectedModel={company.model}
        />
      </section>

      {/* 7. Forecast Error Over Time */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-1">Forecast Error Over Time</h2>
        <p className="text-sm text-slate-400 mb-4">
          Forecast error over the 60-session backtest window, calculated as predicted close minus
          actual close (₱).
        </p>
        <ErrorChart
          dates={company.backtestDates}
          actual={company.backtestActual}
          byModel={company.backtestByModel}
          selectedModel={company.model}
        />
      </section>

      {/* 8. Model Performance Table */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6 overflow-x-auto">
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-white mb-1">
            Model Performance for {company.symbol}
          </h2>
          <p className="text-xs text-slate-400">
            Chronological backtest evaluation metrics across all forecasting models and the naive
            baseline.
          </p>
        </div>

        <table className="w-full text-sm">
          <thead className="text-xs text-slate-400 uppercase bg-dark-bg/80 border-b border-dark-border">
            <tr>
              <th className="text-left py-2 px-3">Model</th>
              <th className="text-right py-2 px-3">RMSE (₱)</th>
              <th className="text-right py-2 px-3">MAE (₱)</th>
              <th className="text-right py-2 px-3">MASE</th>
              <th className="text-right py-2 px-3">R²</th>
            </tr>
          </thead>
          <tbody>
            {modelOrder
              .filter((m) => company.metrics[m])
              .map((m) => (
                <tr
                  key={m}
                  className={`border-b border-dark-border/50 ${
                    modelLabels[m] === company.model ? "bg-brand-500/5" : ""
                  }`}
                >
                  <td className="py-2.5 px-3 font-medium text-white">
                    {modelLabels[m]}
                    {modelLabels[m] === company.model && (
                      <span className="ml-2 text-[10px] uppercase text-brand-400 border border-brand-500/40 rounded px-1.5 py-0.5">
                        Selected
                      </span>
                    )}
                    {m === "naive" && (
                      <span className="ml-2 text-[10px] uppercase text-slate-400 border border-slate-600 rounded px-1.5 py-0.5">
                        Benchmark
                      </span>
                    )}
                  </td>
                  <td className="text-right py-2.5 px-3 font-mono">
                    {formatNum(company.metrics[m].rmse)}
                  </td>
                  <td className="text-right py-2.5 px-3 font-mono">
                    {formatNum(company.metrics[m].mae)}
                  </td>
                  <td className="text-right py-2.5 px-3 font-mono">
                    {formatNum(company.metrics[m].mase)}
                  </td>
                  <td className="text-right py-2.5 px-3 font-mono">
                    {formatNum(company.metrics[m].r2)}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>

        <div className="mt-4 pt-3 border-t border-dark-border/60 text-xs text-slate-400 space-y-1">
          <p>
            &bull; <strong className="text-slate-300">Model Selection: </strong>
            Selected model is determined by the lowest test-set RMSE on the held-out test window.
          </p>
          <p>
            &bull; <strong className="text-slate-300">MASE Benchmark: </strong>
            MASE &lt; 1.0 indicates better performance than the naive baseline, MASE = 1.0 indicates
            approximately equal performance, and MASE &gt; 1.0 indicates worse performance.
          </p>
          <p>
            &bull; <strong className="text-slate-300">R² Interpretation: </strong>
            R² measures in-sample/test-set explained variance in price levels and is not a forecast
            confidence probability.
          </p>
        </div>
      </section>
    </div>
  );
}
