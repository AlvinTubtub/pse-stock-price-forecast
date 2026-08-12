import ModelBarChart from "@/components/charts/ModelBarChart";
import StatCard from "@/components/StatCard";
import { getMetrics } from "@/lib/data";
import { formatNum } from "@/lib/format";

export default async function ComparePage() {
  const metrics = await getMetrics();

  if (!metrics) {
    return <p className="text-slate-400">Model performance data isn&apos;t available yet.</p>;
  }

  const models = Object.keys(metrics.aggregate);
  const rmseData = models.map((m) => ({ model: m, value: metrics.aggregate[m].rmse }));
  const maeData = models.map((m) => ({ model: m, value: metrics.aggregate[m].mae }));
  const maseData = models.map((m) => ({ model: m, value: metrics.aggregate[m].mase }));
  const r2Data = models.map((m) => ({ model: m, value: metrics.aggregate[m].r2 }));

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">Model Performance</h1>
        <p className="text-slate-400 text-sm">
          Aggregate backtest metrics across all tracked companies, averaged per forecasting model.
        </p>
      </div>

      <section className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <StatCard label="Best Performing Model (lowest MASE)" value={metrics.bestModel} accent="text-green-400" />
        <StatCard label="Weakest Performing Model" value={metrics.worstModel} accent="text-amber-400" />
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <h2 className="text-sm font-semibold text-white mb-2">RMSE (lower is better)</h2>
          <ModelBarChart data={rmseData} label="RMSE (₱)" />
        </div>
        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <h2 className="text-sm font-semibold text-white mb-2">MAE (lower is better)</h2>
          <ModelBarChart data={maeData} label="MAE (₱)" />
        </div>
        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <h2 className="text-sm font-semibold text-white mb-2">MASE (lower is better, &lt;1 beats naive)</h2>
          <ModelBarChart data={maseData} label="MASE" />
        </div>
        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <h2 className="text-sm font-semibold text-white mb-2">R² (higher is better)</h2>
          <ModelBarChart data={r2Data} label="R²" />
        </div>
      </section>

      <section className="bg-dark-card border border-dark-border rounded-xl p-6 overflow-x-auto">
        <h2 className="text-lg font-semibold text-white mb-4">Aggregate Metrics Table</h2>
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
            {models.map((m) => (
              <tr key={m} className={`border-b border-dark-border/50 ${m === metrics.bestModel ? "bg-brand-500/5" : ""}`}>
                <td className="py-2 px-3 font-medium text-white">{m}</td>
                <td className="text-right py-2 px-3">{formatNum(metrics.aggregate[m].rmse)}</td>
                <td className="text-right py-2 px-3">{formatNum(metrics.aggregate[m].mae)}</td>
                <td className="text-right py-2 px-3">{formatNum(metrics.aggregate[m].mase)}</td>
                <td className="text-right py-2 px-3">{formatNum(metrics.aggregate[m].r2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-2">Per-Company Best Model</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 text-sm">
          {Object.entries(metrics.perCompany).map(([symbol, info]) => (
            <div key={symbol} className="bg-dark-bg border border-dark-border rounded-lg p-3">
              <p className="font-semibold text-white">{symbol}</p>
              <p className="text-slate-400 text-xs">{info.bestModel}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
