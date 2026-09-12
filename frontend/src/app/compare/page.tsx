import Link from "next/link";
import CompanyLogo from "@/components/CompanyLogo";
import StatCard from "@/components/StatCard";
import { getCompanies, getFormalStudy, getLatest, getMetrics } from "@/lib/data";
import { formatDate, formatDateTimePht, formatNum, formatPct, formatPeso } from "@/lib/format";
import type { FormalModelId, FormalStudyData } from "@/lib/types";

const MODELS: FormalModelId[] = ["lag_reg", "arima", "lstm", "naive"];
const PRINCIPAL: Exclude<FormalModelId, "naive">[] = ["lag_reg", "arima", "lstm"];

function label(study: FormalStudyData, model: FormalModelId): string {
  return study.methodology.modelLabels[model];
}

export default async function ComparePage() {
  const [study, operational, companies, latest] = await Promise.all([
    getFormalStudy(), getMetrics(), getCompanies(), getLatest(),
  ]);

  if (!study) {
    return <div className="space-y-3"><h1 className="text-2xl font-bold text-white">Model Results</h1><p className="text-slate-400">The approved formal-study dataset is unavailable. Current forecasts remain available on the <Link href="/companies" className="text-brand-400">Companies page</Link>.</p></div>;
  }

  const significant = study.conclusion.significantVsNaive
    .map((item) => `${item.symbol} (${label(study, item.model)})`).join(", ");

  return (
    <div className="space-y-10">
      <header className="space-y-4">
        <div><h1 className="text-2xl font-bold text-white mb-1">Model Results</h1><p className="text-sm text-slate-400">The frozen capstone experiment and rolling operational forecasts are presented separately.</p></div>
        <nav aria-label="Results sections" className="flex flex-wrap gap-2">
          <a href="#formal-study" className="px-3 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium">Formal Study Results</a>
          <a href="#operational-results" className="px-3 py-2 rounded-lg border border-dark-border bg-dark-card text-slate-200 text-sm font-medium hover:border-brand-500/60">Current Operational Forecasts</a>
        </nav>
      </header>

      <section id="formal-study" className="scroll-mt-28 space-y-7" aria-labelledby="formal-heading">
        <div className="rounded-xl border border-amber-400/30 bg-amber-400/5 p-5">
          <div className="flex flex-wrap items-center gap-2 mb-2"><span className="rounded-md border border-amber-400/30 bg-amber-400/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-300">Approved immutable study</span><span className="text-xs text-slate-400">{study.runId}</span></div>
          <h2 id="formal-heading" className="text-xl font-semibold text-white">Formal Study Results</h2>
          <p className="mt-2 text-sm text-slate-300 leading-relaxed max-w-4xl">{study.conclusion.summary} Lag-Informed Regression and ARIMA each obtained the lowest principal-model RMSE for seven companies; LSTM did so for one. The eight-of-fifteen consistency threshold was not reached.</p>
          <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-xs text-slate-400"><span>Data: {formatDate(study.data.firstDate)}–{formatDate(study.data.cutoffDate)}</span><span>Holdout: {formatDate(study.data.holdoutStart)}–{formatDate(study.data.holdoutEnd)}</span><span>Code: <code>{study.identity.repositoryCommit.slice(0, 8)}</code></span><a href={study.identity.releaseUrl} target="_blank" rel="noreferrer" className="text-brand-400 hover:text-brand-300">Open audited evidence ↗</a></div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard label="Companies" value={String(study.data.companyCount)} sublabel={`${study.data.rowsPerCompany.toLocaleString()} observations each`} />
          <StatCard label="Holdout Forecasts" value={study.data.totalHoldoutPredictions.toLocaleString()} sublabel="243 dates × 4 methods × 15 companies" accent="text-cyan-400" />
          <StatCard label="RMSE Consistency" value="7 / 15" sublabel={`Below the required ${study.conclusion.dominanceThreshold} of 15`} accent="text-amber-300" />
          <StatCard label="Significant vs Naive" value="2 companies" sublabel={significant} accent="text-emerald-400" />
        </div>

        <section className="bg-dark-card border border-dark-border rounded-xl p-6 overflow-x-auto">
          <h3 className="text-lg font-semibold text-white">Cross-company model summary</h3><p className="text-xs text-slate-400 mt-1 mb-4">Medians address price-scale differences. RMSE wins are descriptive, not proof of statistical superiority.</p>
          <table className="w-full min-w-[760px] text-sm"><thead className="text-xs text-slate-400 uppercase bg-dark-bg/80 border-b border-dark-border"><tr><th className="text-left py-2.5 px-3">Model</th><th className="text-center py-2.5 px-3">Principal RMSE wins</th><th className="text-right py-2.5 px-3">Median RMSE</th><th className="text-right py-2.5 px-3">Median MAE</th><th className="text-right py-2.5 px-3">Median MASE</th><th className="text-right py-2.5 px-3">Median R²</th></tr></thead>
            <tbody>{MODELS.map((model) => { const row = study.aggregate[model]; return <tr key={model} className="border-b border-dark-border/50"><td className="py-2.5 px-3 font-medium text-white">{row.label}{model === "naive" && <span className="ml-2 text-[10px] uppercase text-slate-400 border border-dark-border rounded px-1.5 py-0.5">Benchmark</span>}</td><td className="text-center py-2.5 px-3">{row.principalRmseWins ?? "—"}</td><td className="text-right py-2.5 px-3 font-mono">{formatNum(row.medianRMSE, 6)}</td><td className="text-right py-2.5 px-3 font-mono">{formatNum(row.medianMAE, 6)}</td><td className="text-right py-2.5 px-3 font-mono">{formatNum(row.medianMASE, 6)}</td><td className="text-right py-2.5 px-3 font-mono">{formatNum(row.medianR2, 6)}</td></tr>; })}</tbody>
          </table>
        </section>

        <section className="bg-dark-card border border-dark-border rounded-xl p-6 overflow-x-auto">
          <h3 className="text-lg font-semibold text-white">Canonical holdout metrics</h3><p className="text-xs text-slate-400 mt-1 mb-4">All 60 rows use the same 243 target dates. Values are rounded only for display.</p>
          <table className="w-full min-w-[760px] text-sm"><thead className="text-xs text-slate-400 uppercase bg-dark-bg/80 border-b border-dark-border"><tr><th className="text-left py-2.5 px-3">Company</th><th className="text-left py-2.5 px-3">Model</th><th className="text-right py-2.5 px-3">RMSE</th><th className="text-right py-2.5 px-3">MAE</th><th className="text-right py-2.5 px-3">MASE</th><th className="text-right py-2.5 px-3">R²</th></tr></thead>
            <tbody>{study.perCompany.flatMap((company) => MODELS.map((model) => { const metric = company.metrics[model]; return <tr key={`${company.symbol}-${model}`} className="border-b border-dark-border/50"><td className="py-2 px-3"><Link href={`/companies/${company.symbol}`} className="inline-flex items-center gap-2 font-semibold text-white hover:text-brand-300"><CompanyLogo symbol={company.symbol} size="xs" />{company.symbol}</Link></td><td className="py-2 px-3 text-slate-300">{label(study, model)}{model === company.principalWinnerByRmse && <span className="ml-2 text-[10px] uppercase text-brand-300 border border-brand-500/30 rounded px-1.5 py-0.5">Lowest principal RMSE</span>}</td><td className="text-right py-2 px-3 font-mono">{formatNum(metric.rmse, 6)}</td><td className="text-right py-2 px-3 font-mono">{formatNum(metric.mae, 6)}</td><td className="text-right py-2 px-3 font-mono">{formatNum(metric.mase, 6)}</td><td className="text-right py-2 px-3 font-mono">{formatNum(metric.r2, 6)}</td></tr>; }))}</tbody>
          </table>
        </section>

        <section className="bg-dark-card border border-dark-border rounded-xl p-6 overflow-x-auto">
          <h3 className="text-lg font-semibold text-white">Benchmark-first significance tests</h3><p className="text-xs text-slate-400 mt-1 mb-4">Holm-adjusted squared-error Diebold–Mariano p-values. MASE below one alone is not proof of significant improvement.</p>
          <table className="w-full min-w-[760px] text-sm"><thead className="text-xs text-slate-400 uppercase bg-dark-bg/80 border-b border-dark-border"><tr><th className="text-left py-2.5 px-3">Company</th><th className="text-right py-2.5 px-3">LIR adjusted p</th><th className="text-right py-2.5 px-3">ARIMA adjusted p</th><th className="text-right py-2.5 px-3">LSTM adjusted p</th><th className="text-left py-2.5 px-3">Significant improvement</th></tr></thead>
            <tbody>{study.perCompany.map((company) => { const improved = PRINCIPAL.filter((model) => company.dmSquaredErrorVsNaive[model].significantlyBeatsNaive); return <tr key={company.symbol} className="border-b border-dark-border/50"><td className="py-2.5 px-3 font-semibold text-white">{company.symbol}</td>{PRINCIPAL.map((model) => <td key={model} className="text-right py-2.5 px-3 font-mono">{formatNum(company.dmSquaredErrorVsNaive[model].adjustedPValue, 6)}</td>)}<td className={improved.length ? "py-2.5 px-3 text-emerald-400 font-medium" : "py-2.5 px-3 text-slate-400"}>{improved.length ? improved.map((model) => label(study, model)).join(", ") : "None"}</td></tr>; })}</tbody>
          </table>
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="bg-dark-card border border-dark-border rounded-xl p-6"><h3 className="text-lg font-semibold text-white">Across-company inference</h3><dl className="mt-4 space-y-3 text-sm"><div className="flex justify-between gap-4 border-b border-dark-border/50 pb-2"><dt className="text-slate-400">Friedman statistic</dt><dd className="font-mono text-white">{formatNum(study.acrossCompany.friedmanMase.statistic, 6)}</dd></div><div className="flex justify-between gap-4 border-b border-dark-border/50 pb-2"><dt className="text-slate-400">Permutation p-value</dt><dd className="font-mono text-white">{formatNum(study.acrossCompany.friedmanMase.permutation_p_value, 6)}</dd></div><div className="flex justify-between gap-4"><dt className="text-slate-400">Significant Holm-adjusted pairs</dt><dd className="text-white">None</dd></div></dl><p className="mt-4 text-xs text-slate-400 leading-relaxed">The omnibus result shows different rank behavior, but no individual pair survived Holm correction.</p></div>
          <div className="bg-dark-card border border-dark-border rounded-xl p-6"><h3 className="text-lg font-semibold text-white">Methodology controls</h3><ul className="mt-4 space-y-2 text-sm text-slate-300 list-disc pl-5"><li>{study.methodology.lassoAlphaCandidates} LASSO alpha candidates per company.</li><li>{study.methodology.lstmConfigurations} LSTM configurations × {study.methodology.lstmFolds} folds × {study.methodology.lstmTuningSeeds.length} seeds.</li><li>Common target dates and one MASE denominator per company.</li><li>All validated observations retained in the primary analysis.</li><li>Corporate-action dates excluded only in sensitivity analysis.</li></ul></div>
        </section>

        <section className="bg-dark-card border border-dark-border rounded-xl p-6 overflow-x-auto">
          <h3 className="text-lg font-semibold text-white">Selected model configurations</h3><p className="text-xs text-slate-400 mt-1 mb-4">Frozen research configurations; they do not automatically promote deployment models.</p>
          <table className="w-full min-w-[980px] text-sm"><thead className="text-xs text-slate-400 uppercase bg-dark-bg/80 border-b border-dark-border"><tr><th className="text-left py-2.5 px-3">Company</th><th className="text-right py-2.5 px-3">LASSO alpha</th><th className="text-right py-2.5 px-3">Features</th><th className="text-left py-2.5 px-3">ARIMA order / trend</th><th className="text-left py-2.5 px-3">LSTM lookback / hidden / LR / batch</th><th className="text-right py-2.5 px-3">Epochs</th></tr></thead>
            <tbody>{study.perCompany.map((company) => { const c = company.configuration; return <tr key={company.symbol} className="border-b border-dark-border/50"><td className="py-2.5 px-3 font-semibold text-white">{company.symbol}</td><td className="text-right py-2.5 px-3 font-mono">{String(c.lagRegression.alpha)}</td><td className="text-right py-2.5 px-3">{c.lagRegression.selectedFeatureCount}</td><td className="py-2.5 px-3 font-mono">({c.arima.order.join(", ")}) / {c.arima.trend ?? "n"}</td><td className="py-2.5 px-3 font-mono">{c.lstm.lookback} / {c.lstm.hiddenSize} / {c.lstm.learningRate} / {c.lstm.batchSize}</td><td className="text-right py-2.5 px-3">{c.lstm.fixedEpochs}</td></tr>; })}</tbody>
          </table>
        </section>
      </section>

      <section id="operational-results" className="scroll-mt-28 space-y-6 border-t border-dark-border pt-10" aria-labelledby="operational-heading">
        <div><div className="flex flex-wrap items-center gap-2 mb-2"><span className="rounded-md border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-300">Rolling deployment</span><span className="text-xs text-slate-400">Status: {latest?.status ?? operational?.status ?? "unavailable"}</span></div><h2 id="operational-heading" className="text-xl font-semibold text-white">Current and future next-day forecasts</h2><p className="mt-2 text-sm text-slate-400 max-w-4xl">These values come from persisted deployment models and can change as new official PSE data arrive. They never rewrite the approved formal study.</p><div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-xs text-slate-400"><span>Forecast target: {formatDate(latest?.forecastDate ?? operational?.forecastDate)}</span><span>Last pipeline run: {formatDateTimePht(latest?.lastRunAt ?? operational?.lastRunAt)}</span><Link href="/companies" className="text-brand-400 hover:text-brand-300">Open detailed forecasts →</Link></div></div>
        <section className="bg-dark-card border border-dark-border rounded-xl p-6 overflow-x-auto">
          <h3 className="text-lg font-semibold text-white">Operational forecast snapshot</h3><p className="text-xs text-slate-400 mt-1 mb-4">Selected deployment models and the currently published next-session forecasts.</p>
          <table className="w-full min-w-[820px] text-sm"><thead className="text-xs text-slate-400 uppercase bg-dark-bg/80 border-b border-dark-border"><tr><th className="text-left py-2.5 px-3">Company</th><th className="text-left py-2.5 px-3">Deployment model</th><th className="text-right py-2.5 px-3">Latest close</th><th className="text-right py-2.5 px-3">Forecast close</th><th className="text-right py-2.5 px-3">Expected change</th><th className="text-right py-2.5 px-3">Evaluation MASE</th><th className="text-left py-2.5 px-3">Target date</th></tr></thead>
            <tbody>{[...companies].sort((a, b) => a.symbol.localeCompare(b.symbol)).map((company) => { const current = operational?.perCompany[company.symbol]; const modelKey = Object.entries(study.methodology.modelLabels).find(([, value]) => value === company.bestModel)?.[0] as FormalModelId | undefined; const mase = modelKey ? current?.metrics[modelKey]?.mase : undefined; return <tr key={company.symbol} className="border-b border-dark-border/50"><td className="py-2.5 px-3"><Link href={`/companies/${company.symbol}`} className="inline-flex items-center gap-2 font-semibold text-white hover:text-brand-300"><CompanyLogo symbol={company.symbol} size="xs" />{company.symbol}</Link></td><td className="py-2.5 px-3 text-slate-300">{company.bestModel}</td><td className="text-right py-2.5 px-3">{formatPeso(company.latestClose)}</td><td className="text-right py-2.5 px-3 font-semibold text-white">{formatPeso(company.predictedClose)}</td><td className={company.pctChange >= 0 ? "text-right py-2.5 px-3 text-emerald-400" : "text-right py-2.5 px-3 text-rose-400"}>{formatPct(company.pctChange)}</td><td className="text-right py-2.5 px-3 font-mono">{mase === undefined ? "—" : formatNum(mase)}</td><td className="py-2.5 px-3">{formatDate(company.forecastDate ?? latest?.forecastDate)}</td></tr>; })}</tbody>
          </table>
        </section>
      </section>
    </div>
  );
}
