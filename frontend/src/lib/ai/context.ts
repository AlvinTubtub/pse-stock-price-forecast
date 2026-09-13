import {
  getCompanyDetail,
  getCompanies,
  getDashboard,
  getMetrics,
  getLatest,
} from "@/lib/data";
import { formatPeso, formatPct, formatNum } from "@/lib/format";

export interface ContextOptions {
  route?: string;
  symbol?: string;
  watchlist?: string[];
}

/**
 * Builds a compact, accurate context string for a specific company page (/companies/[symbol]).
 */
export async function buildCompanyContext(symbol: string): Promise<string> {
  const cleanSymbol = symbol.toUpperCase().trim();
  const company = await getCompanyDetail(cleanSymbol);

  if (!company) {
    return `[Context: Company ${cleanSymbol}]
Status: No specific data found for ticker symbol "${cleanSymbol}". Available tracked tickers: ALI, APX, BPI, GLO, ICT, JFC, MBT, MEG, MER, NIKL, PGOLD, SCC, SECB, SHLPH, SMPH.`;
  }

  const modelLabels: Record<string, string> = {
    lag_reg: "Lag-Informed Regression",
    arima: "ARIMA",
    lstm: "LSTM",
    naive: "Naive baseline",
  };

  const metricsLines = Object.entries(company.metrics)
    .map(([key, m]) => {
      const name = modelLabels[key] || key;
      const maseVal = parseFloat(String(m.mase));
      const maseNote =
        !isNaN(maseVal) && maseVal < 1.0
          ? "(beats naive)"
          : !isNaN(maseVal) && maseVal === 1.0
          ? "(equals naive)"
          : "(worse than naive)";
      return `- ${name}: RMSE=₱${formatNum(m.rmse, 4)}, MAE=₱${formatNum(m.mae, 4)}, MASE=${formatNum(m.mase, 4)} ${maseNote}, R²=${formatNum(m.r2, 4)}`;
    })
    .join("\n");

  const nextCloseLines = Object.entries(company.nextClose || {})
    .map(([key, price]) => {
      const predictionLabels: Record<string, string> = {
        lag: "Lag-Informed Regression",
        arima: "ARIMA",
        lstm: "LSTM",
      };
      const name = predictionLabels[key] || key;
      return `- ${name}: ₱${Number(price).toFixed(2)}`;
    })
    .join("\n");

  const selectedModelKey =
    Object.keys(modelLabels).find((k) => modelLabels[k] === company.model) || "arima";
  const selectedMetric = company.metrics[selectedModelKey];
  const selectedMase = selectedMetric ? parseFloat(String(selectedMetric.mase)) : NaN;
  const beatsNaive = !isNaN(selectedMase) && selectedMase < 1.0;

  // Backtest recent summary (last 5 sessions)
  let backtestSummary = "N/A";
  if (
    company.backtestDates &&
    company.backtestDates.length > 0 &&
    company.backtestActual &&
    company.backtestActual.length > 0
  ) {
    const len = company.backtestDates.length;
    const start = Math.max(0, len - 5);
    const recentRows = [];
    for (let i = start; i < len; i++) {
      const d = company.backtestDates[i];
      const act = company.backtestActual[i];
      const pred = company.backtestByModel?.[company.model]?.[i];
      recentRows.push(
        `${d}: Actual=₱${act !== undefined ? act.toFixed(2) : "--"}, Predicted(${company.model})=₱${pred !== undefined ? pred.toFixed(2) : "--"}`
      );
    }
    backtestSummary = recentRows.join("; ");
  }

  return `[Context: Company Detail — ${company.symbol} (${company.name})]
- Sector: ${company.sector}
- Data As Of: ${company.dataAsOf || "Recent"}
- Forecast Target Date: ${company.forecastDate || "Next Trading Session"}
- Previous Close: ${formatPeso(company.previousClose)}
- Forecasted Close: ${formatPeso(company.predictedClose)}
- Expected Change: ${formatPeso(company.pesoChange)} (${formatPct(company.pctChange)}) [${company.direction.toUpperCase()}]
- Selected Model: ${company.model} (Selected based on lowest chronological-evaluation RMSE)
- Selected Model Beats Naive Baseline? ${beatsNaive ? "Yes (MASE < 1.0)" : "No (MASE >= 1.0)"}

Model Performance Metrics on the Held-out Evaluation Set:
${metricsLines}

Next-Day Price Predictions by Model:
${nextCloseLines}

Recent Backtest Observations (Last 5 sessions):
${backtestSummary}`;
}

/**
 * Builds a compact, accurate context string for the Home Dashboard (/).
 */
export async function buildHomeContext(): Promise<string> {
  const [dashboard, companies, latest] = await Promise.all([
    getDashboard(),
    getCompanies(),
    getLatest(),
  ]);

  const topGainerText = dashboard?.topGainer
    ? `${dashboard.topGainer.symbol} (${dashboard.topGainer.name}): ${formatPct(dashboard.topGainer.pctChange)} (Forecast: ₱${dashboard.topGainer.predictedClose.toFixed(2)})`
    : "N/A";

  const topLoserText = dashboard?.topLoser
    ? `${dashboard.topLoser.symbol} (${dashboard.topLoser.name}): ${formatPct(dashboard.topLoser.pctChange)} (Forecast: ₱${dashboard.topLoser.predictedClose.toFixed(2)})`
    : "N/A";

  const sectorsText = dashboard?.sectors
    ? dashboard.sectors.map((s) => `${s.name} (${s.count} stocks)`).join(", ")
    : "N/A";

  const companiesList = companies
    .map(
      (c) =>
        `- ${c.symbol} (${c.name}, ${c.sector}): Last=₱${c.latestClose.toFixed(2)}, Forecast=₱${c.predictedClose.toFixed(2)} (${formatPct(c.pctChange)}), Selected Model=${c.bestModel}`
    )
    .join("\n");

  return `[Context: Home Dashboard / Market Overview]
- Total Tracked PSE Companies: ${companies.length}
- Forecast Target Date: ${latest?.forecastDate || dashboard?.forecastDate || "Next Trading Session"}
- Market Outlook Summary: Gainers=${dashboard?.marketSummary?.gainers ?? 0}, Losers=${dashboard?.marketSummary?.losers ?? 0}, Unchanged=${dashboard?.marketSummary?.unchanged ?? 0}
- Top Forecasted Gainer: ${topGainerText}
- Top Forecasted Loser: ${topLoserText}
- Tracked Sectors: ${sectorsText}

All Tracked Companies Overview:
${companiesList}`;
}

export async function buildCompaniesContext(): Promise<string> {
  const companies = await getCompanies();
  const directory = companies
    .map((company) => `- ${company.symbol}: ${company.name} (${company.sector}), forecast ${formatPeso(company.predictedClose)} (${formatPct(company.pctChange)}), selected model ${company.bestModel}`)
    .join("\n");

  return `[Context: Companies Directory]
- This page lists the PSE companies currently tracked by ForecastPH.
- Users can open a company for its detailed prediction, charts, and evaluation metrics, or add up to five companies to a browser-only watchlist.

Current Company Directory:
${directory}`;
}

export async function buildWatchlistContext(watchlist?: string[]): Promise<string> {
  const [companies, metrics] = await Promise.all([getCompanies(), getMetrics()]);
  const requested = Array.isArray(watchlist) ? watchlist.map((symbol) => symbol.toUpperCase().trim()) : [];
  const selected = companies.filter((company) => requested.includes(company.symbol));

  if (selected.length === 0) {
    return `[Context: My Watchlist]
- The watchlist is stored only in the user's current browser and device, with a maximum of five companies.
- No company is currently selected in the supplied browser watchlist.
- The page can compare expected percentage change and selected-model metrics once companies are added.`;
  }

  const rows = selected.map((company) => {
    const modelMetrics = metrics?.perCompany[company.symbol];
    const modelLabels: Record<string, string> = {
      lag_reg: "Lag-Informed Regression",
      arima: "ARIMA",
      lstm: "LSTM",
      naive: "Naive baseline",
    };
    const selectedMetric = Object.entries(modelMetrics?.metrics ?? {}).find(
      ([key]) => modelLabels[key] === company.bestModel
    )?.[1];
    return `- ${company.symbol}: previous ${formatPeso(company.latestClose)}, forecast ${formatPeso(company.predictedClose)} (${formatPct(company.pctChange)}), selected model ${company.bestModel}, RMSE ${selectedMetric ? formatNum(selectedMetric.rmse) : "unavailable"}, MASE ${selectedMetric ? formatNum(selectedMetric.mase) : "unavailable"}`;
  }).join("\n");

  return `[Context: My Watchlist]
- The watchlist is stored only in the user's current browser and device, with a maximum of five companies.
- Currently watching ${selected.length} company or companies.
- Expected Change comparison uses each selected company's next-session forecast percentage. It is not investment advice.

Selected Watchlist Companies:
${rows}`;
}

export function buildLearnStocksContext(): string {
  return `[Context: Learn Stocks]
- This page is a beginner-focused educational guide to Philippine stock trading and ForecastPH interpretation.
- Topics include Stock Trading 101, PSE trading basics, trading terms, forecast interpretation, RMSE/MAE/MASE/R², chart reading, official PSE educational videos, broker-directory guidance, and ForecastPH research methodology.
- PSE schedules and broker participation may change; users should verify current details directly with the PSE, SEC, and the relevant broker.
- ForecastPH forecasts are educational statistical estimates, not investment advice or buy/sell recommendations.`;
}

export function buildAboutContext(): string {
  return `[Context: About ForecastPH]
- ForecastPH is an educational academic project for next-session Philippine stock price forecasting.
- It compares ARIMA, Lag-Informed Regression, and LSTM against a naive baseline using held-out historical data.
- Company-level model selection uses the lowest held-out evaluation RMSE. Forecasts are not guarantees or investment advice.`;
}

/**
 * Builds a compact context string for Model Performance & Comparison (/compare).
 */
export async function buildCompareContext(): Promise<string> {
  const [metrics, companies, latest] = await Promise.all([
    getMetrics(), getCompanies(), getLatest(),
  ]);
  const details = await Promise.all(
    companies.map((company) => getCompanyDetail(company.symbol)),
  );
  if (!metrics || companies.length === 0) {
    return `[Context: Model Results]\nCurrent operational model-evaluation data is unavailable.`;
  }

  const modelIds = ["lag_reg", "arima", "lstm", "naive"] as const;
  const principalIds = ["lag_reg", "arima", "lstm"] as const;
  const labels: Record<(typeof modelIds)[number], string> = {
    lag_reg: "Lag-Informed Regression",
    arima: "ARIMA",
    lstm: "LSTM",
    naive: "Naive baseline",
  };
  const finiteMetric = (value: string | number | undefined): number | null => {
    const parsed = typeof value === "number" ? value : Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  };
  const median = (values: number[]): number | null => {
    if (values.length === 0) return null;
    const ordered = [...values].sort((left, right) => left - right);
    const middle = Math.floor(ordered.length / 2);
    return ordered.length % 2
      ? ordered[middle]
      : (ordered[middle - 1] + ordered[middle]) / 2;
  };

  const wins = Object.fromEntries(principalIds.map((model) => [model, 0])) as Record<
    (typeof principalIds)[number],
    number
  >;
  const companyRows = companies.map((company) => {
    const companyMetrics = metrics.perCompany[company.symbol]?.metrics;
    const winner = principalIds
      .map((model) => ({ model, rmse: finiteMetric(companyMetrics?.[model]?.rmse) }))
      .filter((row): row is { model: (typeof principalIds)[number]; rmse: number } => row.rmse !== null)
      .sort((left, right) => left.rmse - right.rmse || principalIds.indexOf(left.model) - principalIds.indexOf(right.model))[0];
    if (winner) wins[winner.model] += 1;
    return `${company.symbol}: ${winner ? labels[winner.model] : "unavailable"}`;
  });

  const modelSummary = modelIds.map((model) => {
    const rows = companies.flatMap((company) => {
      const row = metrics.perCompany[company.symbol]?.metrics[model];
      return row ? [row] : [];
    });
    const medianRmse = median(rows.map((row) => finiteMetric(row.rmse)).filter((value): value is number => value !== null));
    const medianMase = median(rows.map((row) => finiteMetric(row.mase)).filter((value): value is number => value !== null));
    const winText = model === "naive" ? "benchmark only" : `${wins[model]} principal-model RMSE wins`;
    return `- ${labels[model]}: ${winText}; median RMSE ${medianRmse === null ? "unavailable" : formatNum(medianRmse, 4)}; median MASE ${medianMase === null ? "unavailable" : formatNum(medianMase, 4)}.`;
  }).join("\n");

  const completeDetails = details.filter((detail) => detail && detail.ohlcv.length > 1);
  const evaluationCounts = completeDetails.map((detail) =>
    Math.ceil(((detail?.ohlcv.length ?? 1) - 1) * 0.15),
  );
  const evaluationCount = new Set(evaluationCounts).size === 1 ? evaluationCounts[0] : null;

  return `[Context: Current Operational Model Results]
- Source: latest fresh-training chronological evaluation.
- Companies evaluated: ${companies.length}.
- Evaluation sessions per company: ${evaluationCount ?? "varies by company"}.
- Forecast target date: ${latest?.forecastDate || metrics.forecastDate || "unavailable"}.
- Metrics generated at: ${metrics.generatedAt || "unavailable"}.
- LIR, ARIMA, and LSTM are the principal deployable models; Naive is an evaluation benchmark only.
- RMSE, MAE, and MASE: lower is better. R²: higher is generally better.

Current model summary:
${modelSummary}

Current best principal model by company:
${companyRows.join(", ")}`;
}

/**
 * Builds context for general pages (/learn, /about, etc.).
 */
export async function buildGeneralContext(): Promise<string> {
  const [dashboard, companies] = await Promise.all([getDashboard(), getCompanies()]);
  const symbols = companies.map((c) => c.symbol).join(", ");

  return `[Context: General PSE Stock Price Forecast Dashboard]
- Scope: Educational forecasting tool tracking ${companies.length} Philippine Stock Exchange (PSE) listed companies: ${symbols}.
- Models Evaluated:
  1. ARIMA (AutoRegressive Integrated Moving Average) - statistical classical time series model.
  2. Lag-Informed Regression - linear model with historical lag price and volume features.
  3. LSTM (Long Short-Term Memory) - recurrent neural network capturing non-linear temporal dynamics.
  4. Naive Baseline - persistence benchmark where tomorrow's forecast equals today's closing price.
- Evaluation Metrics: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), MASE (Mean Absolute Scaled Error), R² (Goodness of Fit).
- Evaluation: The latest fresh-training run compares all four methods on one common chronological holdout and selects each company's principal model by lowest RMSE.
- Deployment: Current next-day forecasts use persisted production models and can change as new official PSE data arrive.`;
}

/**
 * Assembles the full page-aware context payload based on request options.
 */
export async function buildContextForRequest(options?: ContextOptions): Promise<string> {
  const route = options?.route || "";
  const symbol = options?.symbol;

  if (symbol || route.startsWith("/companies/")) {
    const sym = symbol || route.replace("/companies/", "").split("/")[0];
    if (sym && sym !== "undefined") {
      return buildCompanyContext(sym);
    }
  }

  if (route === "/compare") {
    return buildCompareContext();
  }

  if (route === "/" || route === "") {
    return buildHomeContext();
  }

  if (route === "/companies") return buildCompaniesContext();
  if (route === "/watchlist") return buildWatchlistContext(options?.watchlist);
  if (route === "/learn" || route === "/learn-stocks") return buildLearnStocksContext();
  if (route === "/about") return buildAboutContext();
  return buildGeneralContext();
}
