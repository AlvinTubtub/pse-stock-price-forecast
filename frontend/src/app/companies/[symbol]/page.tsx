import { notFound } from "next/navigation";
import Link from "next/link";
import HistoryChart from "@/components/charts/HistoryChart";
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

  const maseNum =
    typeof selectedMetrics.mase === "number"
      ? selectedMetrics.mase
      : parseFloat(String(selectedMetrics.mase));

  let baselineStatus = "Worse Than Naive";
  let baselineBadgeClass = "bg-amber-500/10 text-amber-400 border-amber-500/30";

  if (!Number.isNaN(maseNum)) {
    if (maseNum < 1.0) {
      baselineStatus = "Beats Naive";
      baselineBadgeClass = "bg-green-500/10 text-green-400 border-green-500/30";
    } else if (maseNum === 1.0) {
      baselineStatus = "Approximately Equal to Naive";
      baselineBadgeClass = "bg-slate-700/40 text-slate-300 border-slate-600/40";
    }
  }

  return (
    <div className="space-y-8">
      {/* 1. Header with Breadcrumb, Company Info, and Dates */}
      <div>
        <Link href="/companies" className="text-sm text-brand-400 hover:text-brand-300">
          ← Back to Company List
        </Link>
        <div className="flex flex-wrap items-start justify-between gap-4 mt-2">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold text-white">{company.symbol}</h1>
              <span className="text-xs px-2.5 py-1 rounded-md bg-dark-bg border border-dark-border text-slate-300 font-medium">
                {company.sector}
              </span>
            </div>
            <p className="text-slate-400 mt-1">{company.name}</p>
          </div>
          <div className="flex flex-col items-end gap-1.5">
            <ChangeBadge pctChange={company.pctChange} />
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
              {company.dataAsOf && <span>Data as of: {formatDate(company.dataAsOf)}</span>}
              {company.dataAsOf && company.forecastDate && <span>&middot;</span>}
              {company.forecastDate && (
                <span className="text-brand-300">
                  Forecast for: {formatDate(company.forecastDate)}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 2. Headline StatCards */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Previous Close" value={formatPeso(company.previousClose)} />
        <StatCard
          label="Forecasted Close"
          value={formatPeso(company.predictedClose)}
          accent="text-brand-400"
          sublabel={
            company.forecastDate ? `For ${formatDate(company.forecastDate)}` : undefined
          }
        />
        <StatCard
          label="Predicted Change"
          value={`${company.pesoChange >= 0 ? "+" : ""}${formatPeso(company.pesoChange)}`}
          accent={company.pesoChange >= 0 ? "text-green-400" : "text-red-400"}
          sublabel={formatPct(company.pctChange)}
        />
        <StatCard
          label="Selected Model"
          value={company.model}
          sublabel="Lowest test-set RMSE"
          accent="text-white"
        />
      </section>

      {/* 3. Selected Model Summary & Baseline Status */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-4 pb-3 border-b border-dark-border/60">
          <div>
            <h2 className="text-base font-semibold text-white">Selected Model Summary</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Selected model is the model with the lowest test-set RMSE for {company.symbol}.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Baseline Comparison:</span>
            <span
              className={`inline-flex items-center px-2.5 py-1 rounded text-xs font-semibold border ${baselineBadgeClass}`}
            >
              {baselineStatus}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
          <div className="bg-dark-bg border border-dark-border rounded-lg p-3.5">
            <p className="text-xs text-slate-400 mb-1">Test RMSE</p>
            <p className="text-lg font-semibold text-white font-mono">
              {formatNum(selectedMetrics.rmse)}
            </p>
            <p className="text-[11px] text-slate-500 mt-0.5">Primary selection metric (₱)</p>
          </div>

          <div className="bg-dark-bg border border-dark-border rounded-lg p-3.5">
            <p className="text-xs text-slate-400 mb-1">Test MAE</p>
            <p className="text-lg font-semibold text-white font-mono">
              {formatNum(selectedMetrics.mae)}
            </p>
            <p className="text-[11px] text-slate-500 mt-0.5">Mean absolute error (₱)</p>
          </div>

          <div className="bg-dark-bg border border-dark-border rounded-lg p-3.5">
            <p className="text-xs text-slate-400 mb-1">MASE</p>
            <p className="text-lg font-semibold text-white font-mono">
              {formatNum(selectedMetrics.mase)}
            </p>
            <p className="text-[11px] text-slate-500 mt-0.5">&lt; 1.0 beats naive baseline</p>
          </div>

          <div className="bg-dark-bg border border-dark-border rounded-lg p-3.5">
            <p className="text-xs text-slate-400 mb-1">Test R²</p>
            <p className="text-lg font-semibold text-white font-mono">
              {formatNum(selectedMetrics.r2)}
            </p>
            <p className="text-[11px] text-slate-500 mt-0.5">Explained price variance</p>
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

      {/* 5. Backtest: Predicted vs. Actual */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-1">
          Backtest: Predicted vs. Actual (last 60 sessions)
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          Historical backtest comparing each model&apos;s predictions with the actual closing price.
        </p>
        <PredictionChart actual={company.backtestActual} byModel={company.backtestByModel} />
      </section>

      {/* 6. Forecast Error Over Time */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-1">Forecast Error Over Time</h2>
        <p className="text-sm text-slate-400 mb-4">
          Forecast error over the backtest window, calculated as predicted close minus actual close
          (₱).
        </p>
        <ErrorChart actual={company.backtestActual} byModel={company.backtestByModel} />
      </section>

      {/* 7. Model Performance Table */}
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
