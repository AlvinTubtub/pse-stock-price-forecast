import {
  getCompanyDetail,
  getCompanies,
  getDashboard,
  getFormalStudy,
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
      const name = modelLabels[key] || key;
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
- Selected Model: ${company.model} (Selected based on lowest test-set RMSE)
- Selected Model Beats Naive Baseline? ${beatsNaive ? "Yes (MASE < 1.0)" : "No (MASE >= 1.0)"}

Model Performance Metrics on Held-out Test Set:
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
- Company-level model selection uses lowest test-set RMSE. Forecasts are not guarantees or investment advice.`;
}

export async function buildLiveContext(): Promise<string> {
  const [latest, dashboard] = await Promise.all([getLatest(), getDashboard()]);
  return `[Context: Live Forecast Status]
- Latest forecast target date: ${latest?.forecastDate || dashboard?.forecastDate || "unavailable"}.
- Latest pipeline run: ${latest?.lastRunAt || dashboard?.lastRunAt || "unavailable"}.
- Status: ${latest?.status || dashboard?.status || "unavailable"}.
- Live forecasts supplement the fixed research evaluation; they do not alter the formal backtest metrics.`;
}

/**
 * Builds a compact context string for Model Performance & Comparison (/compare).
 */
export async function buildCompareContext(): Promise<string> {
  const [formal, metrics, companies] = await Promise.all([
    getFormalStudy(), getMetrics(), getCompanies(),
  ]);
  if (!formal) return `[Context: Model Results]\nThe approved formal-study dataset is unavailable.`;

  const modelSummary = formal.methodology.models.map((model) => {
    const row = formal.aggregate[model];
    const wins = row.principalRmseWins === null ? "benchmark" : `${row.principalRmseWins}/15 principal-model RMSE wins`;
    return `- ${row.label}: ${wins}; median RMSE ${formatNum(row.medianRMSE, 6)}; median MASE ${formatNum(row.medianMASE, 6)}.`;
  }).join("\n");
  const winners = formal.perCompany
    .map((row) => `${row.symbol}: ${formal.methodology.modelLabels[row.principalWinnerByRmse]}`)
    .join(", ");
  const significant = formal.conclusion.significantVsNaive
    .map((row) => `${row.symbol}: ${formal.methodology.modelLabels[row.model]} (Holm p=${formatNum(row.adjustedPValue, 6)})`)
    .join("; ");
  const operationalModels = companies.map((company) => `${company.symbol}: ${company.bestModel}`).join(", ");

  return `[Context: Formal Study and Operational Model Results]
Formal study:
- Run ID: ${formal.runId}; status: ${formal.status}; immutable.
- Data cutoff: ${formal.data.cutoffDate}; holdout: ${formal.data.holdoutStart} to ${formal.data.holdoutEnd}; 243 dates per company.
- Conclusion: ${formal.conclusion.summary}
- Descriptive principal-model RMSE wins: Lag-Informed Regression 7, ARIMA 7, LSTM 1. No model reached the required 8/15 threshold.
- Significant squared-error DM improvements over Naive: ${significant}.
- Friedman permutation p=${formatNum(formal.acrossCompany.friedmanMase.permutation_p_value, 6)}, but no Wilcoxon pair survived Holm correction.
- MASE below 1 is a scaling comparison and is not by itself proof of significant improvement over the holdout Naive forecast.

Formal model summary:
${modelSummary}

Formal descriptive principal winners:
${winners}

Operational deployment:
- Current generated metrics timestamp: ${metrics?.generatedAt || "unavailable"}.
- Current deployment model choices: ${operationalModels}.
- Operational forecasts and metrics can change with new PSE data and must not be described as replacements for the fixed formal study.`;
}

/**
 * Builds context for general pages (/learn, /about, /live, etc.).
 */
export async function buildGeneralContext(): Promise<string> {
  const [dashboard, companies] = await Promise.all([getDashboard(), getCompanies()]);
  const symbols = companies.map((c) => c.symbol).join(", ");

  return `[Context: General PSE Stock Price Forecast Dashboard]
- Scope: Educational forecasting tool tracking 15 major Philippine Stock Exchange (PSE) listed companies: ${symbols}.
- Models Evaluated:
  1. ARIMA (AutoRegressive Integrated Moving Average) - statistical classical time series model.
  2. Lag-Informed Regression - linear model with historical lag price and volume features.
  3. LSTM (Long Short-Term Memory) - recurrent neural network capturing non-linear temporal dynamics.
  4. Naive Baseline - persistence benchmark where tomorrow's forecast equals today's closing price.
- Evaluation Metrics: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), MASE (Mean Absolute Scaled Error), R² (Goodness of Fit).
- Formal Study: The immutable Run 02 comparison is reported separately from deployment forecasts. It found a 7–7–1 descriptive RMSE split for Lag-Informed Regression, ARIMA, and LSTM, with no model reaching the 8-of-15 consistency threshold.
- Deployment: Current next-day forecasts use separately approved persisted models and can change as new official PSE data arrive.`;
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
  if (route === "/live") return buildLiveContext();

  return buildGeneralContext();
}
