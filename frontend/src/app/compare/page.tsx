import Link from "next/link";
import ModelBarChart from "@/components/charts/ModelBarChart";
import CompanyLogo from "@/components/CompanyLogo";
import StatCard from "@/components/StatCard";
import { getMetrics } from "@/lib/data";
import { formatNum } from "@/lib/format";

interface StatTestsData {
  friedman?: {
    statistic?: number;
    p_value?: number;
    n_companies?: number;
  };
  best_model_consistency?: {
    dominant_model?: string;
    dominant_count?: number;
    total_companies?: number;
    min_required?: number;
    pass?: boolean;
    counts?: Record<string, number>;
  };
  wilcoxon_holm_posthoc?: Record<
    string,
    {
      statistic?: number;
      p_value?: number;
      holm_p_value?: number;
    }
  >;
}

function calculateMedian(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export default async function ComparePage() {
  const metrics = await getMetrics();

  if (!metrics || !metrics.perCompany) {
    return <p className="text-slate-400">Model performance data isn&apos;t available yet.</p>;
  }

  const symbols = Object.keys(metrics.perCompany).sort();
  const totalCompanies = symbols.length;

  const MODEL_CONFIGS = [
    { id: "arima", name: "ARIMA", isTrained: true },
    { id: "lag_reg", name: "Lag-Informed Regression", isTrained: true },
    { id: "lstm", name: "LSTM", isTrained: true },
    { id: "naive", name: "Naive baseline", isTrained: false },
  ];

  const modelStats = MODEL_CONFIGS.map((config) => {
    const isTrained = config.isTrained;
    const wins = isTrained
      ? symbols.filter((s) => metrics.perCompany[s]?.bestModel === config.name).length
      : 0;
    const winRate = isTrained && totalCompanies > 0 ? (wins / totalCompanies) * 100 : null;

    const rmses: number[] = [];
    const maes: number[] = [];
    const mases: number[] = [];
    const r2s: number[] = [];
    let beatNaiveCount = 0;

    for (const sym of symbols) {
      const m = metrics.perCompany[sym]?.metrics?.[config.id];
      if (m) {
        const rmse = typeof m.rmse === "number" ? m.rmse : parseFloat(m.rmse);
        const mae = typeof m.mae === "number" ? m.mae : parseFloat(m.mae);
        const mase = typeof m.mase === "number" ? m.mase : parseFloat(m.mase);
        const r2 = typeof m.r2 === "number" ? m.r2 : parseFloat(m.r2);

        if (!Number.isNaN(rmse)) rmses.push(rmse);
        if (!Number.isNaN(mae)) maes.push(mae);
        if (!Number.isNaN(mase)) {
          mases.push(mase);
          if (mase < 1.0) beatNaiveCount++;
        }
        if (!Number.isNaN(r2)) r2s.push(r2);
      }
    }

    return {
      id: config.id,
      name: config.name,
      isTrained,
      wins,
      winRate,
      medianRmse: calculateMedian(rmses),
      medianMae: calculateMedian(maes),
      medianMase: calculateMedian(mases),
      medianR2: calculateMedian(r2s),
      beatNaiveCount,
      beatNaivePct: totalCompanies > 0 ? (beatNaiveCount / totalCompanies) * 100 : 0,
    };
  });

  const trainedModels = modelStats.filter((m) => m.isTrained);
  const mostConsistent = trainedModels.reduce(
    (prev, curr) => (curr.wins > prev.wins ? curr : prev),
    trainedModels[0]
  );

  const perCompanyRows = symbols.map((symbol) => {
    const compInfo = metrics.perCompany[symbol];
    const winningModelName = compInfo?.bestModel ?? "—";
    const winningConfig =
      MODEL_CONFIGS.find((m) => m.name === winningModelName) ?? MODEL_CONFIGS[0];
    const m = compInfo?.metrics?.[winningConfig.id];

    const rmse = m ? (typeof m.rmse === "number" ? m.rmse : parseFloat(m.rmse)) : NaN;
    const mae = m ? (typeof m.mae === "number" ? m.mae : parseFloat(m.mae)) : NaN;
    const mase = m ? (typeof m.mase === "number" ? m.mase : parseFloat(m.mase)) : NaN;
    const r2 = m ? (typeof m.r2 === "number" ? m.r2 : parseFloat(m.r2)) : NaN;

    const beatsNaive = !Number.isNaN(mase) && mase < 1.0;

    return {
      symbol,
      bestModel: winningModelName,
      rmse,
      mae,
      mase,
      r2,
      beatsNaive: !Number.isNaN(mase) ? (beatsNaive ? "Yes" : "No") : "—",
      beatsNaiveBool: beatsNaive,
    };
  });

  const rmseChartData = modelStats.map((m) => ({
    model: m.name,
    value: Number(m.medianRmse.toFixed(4)),
  }));
  const maeChartData = modelStats.map((m) => ({
    model: m.name,
    value: Number(m.medianMae.toFixed(4)),
  }));
  const maseChartData = modelStats.map((m) => ({
    model: m.name,
    value: Number(m.medianMase.toFixed(4)),
  }));
  const r2ChartData = modelStats.map((m) => ({
    model: m.name,
    value: Number(m.medianR2.toFixed(4)),
  }));

  const statTests = (metrics.statisticalTests ?? {}) as StatTestsData;

  const totalWinnersBeatNaive = perCompanyRows.filter((r) => r.beatsNaiveBool).length;

  return (
    <div className="space-y-8">
      {/* 1. Title, Subtitle, and Evaluation Note */}
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">Model Performance</h1>
        <p className="text-slate-400 text-sm">
          Backtest performance across the tracked PSE-listed companies.
        </p>
        <div className="mt-3 p-3.5 bg-brand-500/10 border border-brand-500/20 rounded-lg text-xs text-slate-300 leading-relaxed">
          <strong className="text-white font-medium">Evaluation Methodology: </strong>
          Per-company winners are determined by the lowest test-set RMSE among the three forecasting
          models. Cross-company summaries use median error metrics because stock-price scales differ
          substantially across PSE securities.
        </div>
      </div>

      {/* 2. Headline Summary Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Most Consistent Model"
          value={mostConsistent ? mostConsistent.name : "—"}
          sublabel={
            mostConsistent && mostConsistent.winRate !== null
              ? `${mostConsistent.wins} of ${totalCompanies} companies (${mostConsistent.winRate.toFixed(1)}% win rate)`
              : undefined
          }
          accent="text-green-400"
        />
        <StatCard
          label="Trained Models Evaluated"
          value={`${trainedModels.length} Models`}
          sublabel="ARIMA, Lag Regression, LSTM"
        />
        <StatCard
          label="Winners Beating Naive"
          value={`${totalWinnersBeatNaive} / ${totalCompanies}`}
          sublabel={`${((totalWinnersBeatNaive / totalCompanies) * 100).toFixed(1)}% of company winners have MASE < 1`}
          accent="text-brand-400"
        />
        <StatCard
          label="Cross-Company Aggregation"
          value="Median-Based"
          sublabel="Robust to extreme stock price disparities (₱2 to ₱2,000)"
          accent="text-cyan-400"
        />
      </section>

      {/* 3. Model Comparison Table */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6 overflow-x-auto">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4">
          <div>
            <h2 className="text-lg font-semibold text-white">Model Comparison Table</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Ranked by cross-company win frequency (lowest test-set RMSE per company). Error
              metrics summarized as medians.
            </p>
          </div>
        </div>

        <table className="w-full text-sm">
          <thead className="text-xs text-slate-400 uppercase bg-dark-bg/80 border-b border-dark-border">
            <tr>
              <th className="text-left py-2.5 px-3">Model</th>
              <th className="text-center py-2.5 px-3">Companies Won</th>
              <th className="text-center py-2.5 px-3">Win Rate</th>
              <th className="text-right py-2.5 px-3">Median MASE</th>
              <th className="text-right py-2.5 px-3">Median RMSE (₱)</th>
              <th className="text-right py-2.5 px-3">Median MAE (₱)</th>
              <th className="text-right py-2.5 px-3">Median R²</th>
            </tr>
          </thead>
          <tbody>
            {modelStats.map((m) => {
              const isDominant = mostConsistent && m.name === mostConsistent.name;
              return (
                <tr
                  key={m.id}
                  className={`border-b border-dark-border/50 ${isDominant ? "bg-brand-500/5" : ""}`}
                >
                  <td className="py-2.5 px-3 font-medium text-white flex items-center gap-2">
                    <span>{m.name}</span>
                    {isDominant && (
                      <span className="text-[10px] uppercase font-semibold text-green-400 border border-green-500/40 bg-green-500/10 rounded px-1.5 py-0.5">
                        Top Win Rate
                      </span>
                    )}
                    {!m.isTrained && (
                      <span className="text-[10px] uppercase text-slate-400 border border-slate-600 rounded px-1.5 py-0.5">
                        Benchmark
                      </span>
                    )}
                  </td>
                  <td className="text-center py-2.5 px-3 text-slate-300">
                    {m.isTrained ? `${m.wins} / ${totalCompanies}` : "—"}
                  </td>
                  <td className="text-center py-2.5 px-3 text-slate-300">
                    {m.winRate !== null ? `${m.winRate.toFixed(1)}%` : "—"}
                  </td>
                  <td className="text-right py-2.5 px-3 text-slate-300 font-mono">
                    {formatNum(m.medianMase)}
                  </td>
                  <td className="text-right py-2.5 px-3 text-slate-300 font-mono">
                    {formatNum(m.medianRmse)}
                  </td>
                  <td className="text-right py-2.5 px-3 text-slate-300 font-mono">
                    {formatNum(m.medianMae)}
                  </td>
                  <td className="text-right py-2.5 px-3 text-slate-300 font-mono">
                    {formatNum(m.medianR2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {/* 4. Median Performance Charts */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <h2 className="text-sm font-semibold text-white mb-1">Median RMSE (lower is better)</h2>
          <p className="text-xs text-slate-400 mb-3">Median root mean squared error in Pesos (₱)</p>
          <ModelBarChart data={rmseChartData} label="Median RMSE (₱)" />
        </div>
        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <h2 className="text-sm font-semibold text-white mb-1">Median MAE (lower is better)</h2>
          <p className="text-xs text-slate-400 mb-3">Median mean absolute error in Pesos (₱)</p>
          <ModelBarChart data={maeChartData} label="Median MAE (₱)" />
        </div>
        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <h2 className="text-sm font-semibold text-white mb-1">
            Median MASE (lower is better, &lt;1 beats naive)
          </h2>
          <p className="text-xs text-slate-400 mb-3">
            Scaled against in-sample one-step naive error
          </p>
          <ModelBarChart data={maseChartData} label="Median MASE" />
        </div>
        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <h2 className="text-sm font-semibold text-white mb-1">
            Median R² (supplementary variance explained)
          </h2>
          <p className="text-xs text-slate-400 mb-3">
            In-sample/test-set explained variance in price levels
          </p>
          <ModelBarChart data={r2ChartData} label="Median R²" />
        </div>
      </section>

      {/* 5. Naive Baseline Comparison Section */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-white">Naive Baseline Comparison</h2>
          <p className="text-sm text-slate-400 mt-1">
            The Naive baseline serves as a fundamental benchmark by forecasting tomorrow&apos;s
            closing price as today&apos;s closing price. Mean Absolute Scaled Error (MASE) evaluates
            each model relative to this benchmark:
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5 text-xs">
          <div className="bg-dark-bg border border-green-500/30 rounded-lg p-3">
            <span className="font-semibold text-green-400 block mb-1">MASE &lt; 1.0</span>
            <span className="text-slate-300">
              The model outperformed the naive baseline (lower forecast error).
            </span>
          </div>
          <div className="bg-dark-bg border border-slate-600 rounded-lg p-3">
            <span className="font-semibold text-slate-300 block mb-1">MASE = 1.0</span>
            <span className="text-slate-300">
              Approximately equal performance to the naive persistence forecast.
            </span>
          </div>
          <div className="bg-dark-bg border border-amber-500/30 rounded-lg p-3">
            <span className="font-semibold text-amber-400 block mb-1">MASE &gt; 1.0</span>
            <span className="text-slate-300">
              The model performed worse than the naive baseline.
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {trainedModels.map((m) => (
            <div key={m.id} className="bg-dark-bg border border-dark-border rounded-lg p-4">
              <p className="font-semibold text-white text-sm mb-1">{m.name}</p>
              <div className="flex items-baseline gap-2 mb-2">
                <span className="text-2xl font-bold text-brand-400">
                  {m.beatNaiveCount} / {totalCompanies}
                </span>
                <span className="text-xs text-slate-400">({m.beatNaivePct.toFixed(1)}%)</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                {m.beatNaiveCount > 0
                  ? `Outperformed naive persistence on ${m.beatNaiveCount} of ${totalCompanies} evaluated stocks.`
                  : `Did not beat naive persistence on evaluated stocks.`}
              </p>
            </div>
          ))}
        </div>

        <p className="text-xs text-slate-400 mt-4 leading-relaxed bg-dark-bg/60 border border-dark-border/60 rounded-lg p-3">
          <strong className="text-slate-300 font-medium">Market Reality Note: </strong>
          Financial time series frequently exhibit near-random-walk properties. Across all 15
          companies, models beat the naive baseline primarily on high-liquidity large-cap stocks
          (such as ALI, GLO, JFC, MEG, SECB, and SMPH), while exhibiting higher relative error on
          volatile or lower-priced securities.
        </p>
      </section>

      {/* 6. Per-Company Best Model Table */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6 overflow-x-auto">
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-white">Per-Company Best Model</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Per-company winner is determined by the lowest test-set RMSE among Lag-Informed
            Regression, ARIMA, and LSTM.
          </p>
        </div>

        <table className="w-full text-sm">
          <thead className="text-xs text-slate-400 uppercase bg-dark-bg/80 border-b border-dark-border">
            <tr>
              <th className="text-left py-2.5 px-3">Company</th>
              <th className="text-left py-2.5 px-3">Best Model</th>
              <th className="text-right py-2.5 px-3">RMSE (₱)</th>
              <th className="text-right py-2.5 px-3">MAE (₱)</th>
              <th className="text-right py-2.5 px-3">MASE</th>
              <th className="text-right py-2.5 px-3">R²</th>
              <th className="text-center py-2.5 px-3">Beats Naive?</th>
            </tr>
          </thead>
          <tbody>
            {perCompanyRows.map((row) => (
              <tr
                key={row.symbol}
                className="border-b border-dark-border/50 hover:bg-dark-bg/40 transition-colors"
              >
                <td className="py-2.5 px-3 font-semibold text-white">
                  <Link
                    href={`/companies/${row.symbol}`}
                    className="inline-flex items-center gap-2 text-brand-400 hover:text-brand-300 hover:underline"
                  >
                    <CompanyLogo symbol={row.symbol} size="xs" />
                    <span>{row.symbol}</span>
                  </Link>
                </td>
                <td className="py-2.5 px-3 text-slate-200">{row.bestModel}</td>
                <td className="text-right py-2.5 px-3 text-slate-300 font-mono">
                  {formatNum(row.rmse)}
                </td>
                <td className="text-right py-2.5 px-3 text-slate-300 font-mono">
                  {formatNum(row.mae)}
                </td>
                <td className="text-right py-2.5 px-3 text-slate-300 font-mono">
                  {formatNum(row.mase)}
                </td>
                <td className="text-right py-2.5 px-3 text-slate-300 font-mono">
                  {formatNum(row.r2)}
                </td>
                <td className="text-center py-2.5 px-3">
                  {row.beatsNaive === "Yes" ? (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-green-500/10 text-green-400 border border-green-500/30">
                      Yes
                    </span>
                  ) : row.beatsNaive === "No" ? (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-slate-700/40 text-slate-400 border border-slate-600/40">
                      No
                    </span>
                  ) : (
                    <span className="text-slate-500">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* 7. Statistical Significance Suite */}
      {statTests && (statTests.friedman || statTests.wilcoxon_holm_posthoc) && (
        <section className="bg-dark-card border border-dark-border rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-1">
            Statistical Significance &amp; Validation Summary
          </h2>
          <p className="text-xs text-slate-400 mb-4">
            Hypothesis testing conducted across the 15 PSE companies to verify whether performance
            differences are statistically distinguishable from random variation.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {statTests.friedman && (
              <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
                <h3 className="text-sm font-semibold text-white mb-2">Friedman Omnibus Test</h3>
                <p className="text-xs text-slate-400 mb-3">
                  Non-parametric test evaluating overall differences across models across all{" "}
                  {statTests.friedman.n_companies ?? totalCompanies} companies.
                </p>
                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Test Statistic (χ²):</span>
                    <span className="text-white font-mono">
                      {formatNum(statTests.friedman.statistic ?? 0)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">p-value:</span>
                    <span className="text-green-400 font-mono font-medium">
                      {statTests.friedman.p_value !== undefined
                        ? statTests.friedman.p_value < 0.001
                          ? "< 0.001 (Significant)"
                          : formatNum(statTests.friedman.p_value)
                        : "—"}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {statTests.best_model_consistency && (
              <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
                <h3 className="text-sm font-semibold text-white mb-2">
                  Best-Model Consistency Check
                </h3>
                <p className="text-xs text-slate-400 mb-3">
                  Evaluates whether a single forecasting model achieves the lowest test-set RMSE on
                  at least a majority of tracked companies.
                </p>
                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Dominant Model:</span>
                    <span className="text-white font-semibold uppercase">
                      {statTests.best_model_consistency.dominant_model ?? "—"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Consistency Count:</span>
                    <span className="text-green-400 font-medium">
                      {statTests.best_model_consistency.dominant_count} /{" "}
                      {statTests.best_model_consistency.total_companies} companies (
                      {statTests.best_model_consistency.pass ? "Pass" : "Fail"})
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {statTests.wilcoxon_holm_posthoc && (
            <div className="mt-4 bg-dark-bg border border-dark-border rounded-lg p-4">
              <h3 className="text-sm font-semibold text-white mb-2">
                Wilcoxon Signed-Rank Post-Hoc Tests (Holm-Bonferroni Corrected)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs mt-2">
                  <thead className="text-slate-400 uppercase bg-dark-card border-b border-dark-border">
                    <tr>
                      <th className="text-left py-2 px-3">Comparison</th>
                      <th className="text-right py-2 px-3">Statistic (W)</th>
                      <th className="text-right py-2 px-3">Unadjusted p</th>
                      <th className="text-right py-2 px-3">Holm-Adjusted p</th>
                      <th className="text-center py-2 px-3">Significance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(statTests.wilcoxon_holm_posthoc).map(([pair, res]) => {
                      const isSignificant =
                        res.holm_p_value !== undefined && res.holm_p_value < 0.05;
                      return (
                        <tr key={pair} className="border-b border-dark-border/40">
                          <td className="py-2 px-3 font-medium text-white capitalize">{pair}</td>
                          <td className="text-right py-2 px-3 text-slate-300 font-mono">
                            {formatNum(res.statistic ?? 0, 1)}
                          </td>
                          <td className="text-right py-2 px-3 text-slate-300 font-mono">
                            {formatNum(res.p_value ?? 0, 4)}
                          </td>
                          <td className="text-right py-2 px-3 text-slate-300 font-mono">
                            {formatNum(res.holm_p_value ?? 0, 4)}
                          </td>
                          <td className="text-center py-2 px-3">
                            {isSignificant ? (
                              <span className="text-green-400 font-semibold">p &lt; 0.05</span>
                            ) : (
                              <span className="text-slate-500">Not sig.</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      )}

      {/* 8. Methodology Notes */}
      <section className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-3">Methodology &amp; Evaluation Notes</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300 leading-relaxed">
          <div className="space-y-2">
            <p>
              <strong className="text-white font-medium">1. Primary Selection Metric: </strong>
              Per-company model selection is strictly determined by the lowest test-set RMSE
              achieved during chronological backtesting on the held-out test window.
            </p>
            <p>
              <strong className="text-white font-medium">2. Naive Baseline Benchmark: </strong>
              MASE (Mean Absolute Scaled Error) scales forecast errors relative to the in-sample
              one-step naive forecast. Values below 1.0 indicate performance superior to naive
              persistence.
            </p>
          </div>
          <div className="space-y-2">
            <p>
              <strong className="text-white font-medium">3. Scale-Independent Medians: </strong>
              Cross-company RMSE and MAE are summarized using medians because the 15 tracked stocks
              have substantially different price scales (₱2 to ₱2,000). Raw error averages can be
              skewed by higher-priced securities.
            </p>
            <p>
              <strong className="text-white font-medium">4. R² Supplementary Role: </strong>
              R² measures in-sample/test-set explained variance in price levels and should not be
              interpreted alone as evidence that a forecasting model outperforms the naive baseline.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
