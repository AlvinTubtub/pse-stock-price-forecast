import { notFound } from "next/navigation";
import Link from "next/link";
import HistoryChart from "@/components/charts/HistoryChart";
import PredictionChart from "@/components/charts/PredictionChart";
import ErrorChart from "@/components/charts/ErrorChart";
import ChangeBadge from "@/components/ChangeBadge";
import StatCard from "@/components/StatCard";
import { getAllSymbols, getCompanyDetail } from "@/lib/data";
import { formatNum, formatPeso } from "@/lib/format";

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

  return (
    <div className="space-y-8">
      <div>
        <Link href="/companies" className="text-sm text-brand-400 hover:text-brand-300">
          ← Back to Company List
        </Link>
        <div className="flex flex-wrap items-start justify-between gap-4 mt-2">
          <div>
            <h1 className="text-3xl font-bold text-white">{company.symbol}</h1>
            <p className="text-slate-400">{company.name} &middot; {company.sector}</p>
          </div>
          <ChangeBadge pctChange={company.pctChange} />
        </div>
      </div>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Previous Close" value={formatPeso(company.previousClose)} />
        <StatCard label="Predicted Next Close" value={formatPeso(company.predictedClose)} accent="text-brand-400" />
        <StatCard label="Predicted Change" value={`${company.pesoChange >= 0 ? "+" : ""}${formatPeso(company.pesoChange)}`} accent={company.pesoChange >= 0 ? "text-green-400" : "text-red-400"} />
        <StatCard label="Model Confidence (R²-based)" value={`${company.confidence}%`} sublabel={company.model} />
      </section>

      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Historical OHLCV</h2>
        <HistoryChart data={company.ohlcv} />
      </section>

      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-1">Backtest: Predicted vs. Actual (last 60 sessions)</h2>
        <p className="text-sm text-slate-400 mb-4">How each model's rolling forecast compared to the actual closing price.</p>
        <PredictionChart actual={company.backtestActual} byModel={company.backtestByModel} />
      </section>

      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-1">Forecast Error Over Time</h2>
        <p className="text-sm text-slate-400 mb-4">Predicted minus actual close (₱), per model, across the backtest window.</p>
        <ErrorChart actual={company.backtestActual} byModel={company.backtestByModel} />
      </section>

      <section className="bg-dark-card border border-dark-border rounded-xl p-6 overflow-x-auto">
        <h2 className="text-lg font-semibold text-white mb-4">Model Performance for {company.symbol}</h2>
        <table className="w-full text-sm">
          <thead className="text-xs text-slate-400 uppercase bg-dark-bg/80 border-b border-dark-border">
            <tr>
              <th className="text-left py-2 px-3">Model</th>
              <th className="text-right py-2 px-3">RMSE</th>
              <th className="text-right py-2 px-3">MAE</th>
              <th className="text-right py-2 px-3">MASE</th>
              <th className="text-right py-2 px-3">R²</th>
            </tr>
          </thead>
          <tbody>
            {modelOrder
              .filter((m) => company.metrics[m])
              .map((m) => (
                <tr key={m} className={`border-b border-dark-border/50 ${modelLabels[m] === company.model ? "bg-brand-500/5" : ""}`}>
                  <td className="py-2 px-3 font-medium text-white">
                    {modelLabels[m]}
                    {modelLabels[m] === company.model && (
                      <span className="ml-2 text-[10px] uppercase text-brand-400 border border-brand-500/40 rounded px-1.5 py-0.5">
                        Selected
                      </span>
                    )}
                  </td>
                  <td className="text-right py-2 px-3">{formatNum(company.metrics[m].rmse)}</td>
                  <td className="text-right py-2 px-3">{formatNum(company.metrics[m].mae)}</td>
                  <td className="text-right py-2 px-3">{formatNum(company.metrics[m].mase)}</td>
                  <td className="text-right py-2 px-3">{formatNum(company.metrics[m].r2)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
