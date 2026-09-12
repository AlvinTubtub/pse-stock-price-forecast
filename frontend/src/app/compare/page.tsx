import ModelsDashboard, {
  type ModelPerformanceRow,
  type OperationalModelId,
} from "./ModelsDashboard";
import { getCompanies, getCompanyDetail, getMetrics } from "@/lib/data";

const PRINCIPAL_MODELS: OperationalModelId[] = ["lag_reg", "arima", "lstm"];
const ALL_MODELS: OperationalModelId[] = [...PRINCIPAL_MODELS, "naive"];
const EVALUATION_PROPORTION = 0.15;

function numericMetric(value: string | number | undefined): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export default async function ComparePage() {
  const [companies, metrics] = await Promise.all([getCompanies(), getMetrics()]);
  const details = await Promise.all(
    companies.map((company) => getCompanyDetail(company.symbol)),
  );

  if (!metrics || companies.length === 0 || details.some((detail) => !detail)) {
    return <UnavailableState />;
  }

  const rows: ModelPerformanceRow[] = [];
  const observationCounts = new Set<number>();
  const evaluationCounts = new Set<number>();
  const evaluationStarts = new Set<string>();
  const evaluationEnds = new Set<string>();

  for (const [index, company] of companies.entries()) {
    const detail = details[index];
    const companyMetrics = metrics.perCompany[company.symbol]?.metrics;
    if (!detail || !companyMetrics || detail.ohlcv.length < 2) return <UnavailableState />;

    const parsedMetrics = Object.fromEntries(
      ALL_MODELS.map((model) => [
        model,
        {
          rmse: numericMetric(companyMetrics[model]?.rmse),
          mae: numericMetric(companyMetrics[model]?.mae),
          mase: numericMetric(companyMetrics[model]?.mase),
          r2: numericMetric(companyMetrics[model]?.r2),
        },
      ]),
    ) as Record<OperationalModelId, Record<"rmse" | "mae" | "mase" | "r2", number | null>>;

    if (ALL_MODELS.some((model) => Object.values(parsedMetrics[model]).some((value) => value === null))) {
      return <UnavailableState />;
    }

    const observationCount = detail.ohlcv.length;
    const evaluationCount = Math.ceil((observationCount - 1) * EVALUATION_PROPORTION);
    observationCounts.add(observationCount);
    evaluationCounts.add(evaluationCount);
    evaluationStarts.add(detail.ohlcv[observationCount - evaluationCount].date);
    evaluationEnds.add(detail.ohlcv[observationCount - 1].date);

    rows.push({
      symbol: company.symbol,
      name: company.name,
      metrics: parsedMetrics as ModelPerformanceRow["metrics"],
    });
  }

  if (
    observationCounts.size !== 1 ||
    evaluationCounts.size !== 1 ||
    evaluationStarts.size !== 1 ||
    evaluationEnds.size !== 1
  ) {
    return <UnavailableState />;
  }

  return (
    <ModelsDashboard
      rows={rows.sort((a, b) => a.symbol.localeCompare(b.symbol))}
      observationCount={[...observationCounts][0]}
      evaluationCount={[...evaluationCounts][0]}
      evaluationStart={[...evaluationStarts][0]}
      evaluationEnd={[...evaluationEnds][0]}
    />
  );
}

function UnavailableState() {
  return (
    <div className="rounded-2xl border border-dark-border bg-dark-card p-6">
      <h1 className="text-2xl font-bold text-white">Models</h1>
      <p className="mt-2 text-sm text-slate-400">
        Model-performance data is unavailable or incomplete. ForecastPH will display this
        dashboard after the next successful fresh training run.
      </p>
    </div>
  );
}
