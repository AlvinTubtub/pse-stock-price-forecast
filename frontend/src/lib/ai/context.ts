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
  const [metrics, companies] = await Promise.all([getMetrics(), getCompanies()]);

  if (!metrics || !metrics.perCompany) {
    return `[Context: Model Performance]
Model evaluation data is currently being generated.`;
  }

  const symbols = Object.keys(metrics.perCompany).sort();
  const totalCompanies = symbols.length;

  const MODEL_CONFIGS = [
    { id: "arima", name: "ARIMA", isTrained: true },
    { id: "lag_reg", name: "Lag-Informed Regression", isTrained: true },
    { id: "lstm", name: "LSTM", isTrained: true },
    { id: "naive", name: "Naive baseline", isTrained: false },
  ];

  const modelSummary = MODEL_CONFIGS.map((config) => {
    const wins = config.isTrained
      ? symbols.filter((s) => metrics.perCompany[s]?.bestModel === config.name).length
      : 0;
    const winRate = config.isTrained && totalCompanies > 0 ? ((wins / totalCompanies) * 100).toFixed(1) : "—";
    
    let beatNaiveCount = 0;
    for (const sym of symbols) {
      const m = metrics.perCompany[sym]?.metrics?.[config.id];
      if (m) {
        const mase = typeof m.mase === "number" ? m.mase : parseFloat(String(m.mase));
        if (!isNaN(mase) && mase < 1.0) beatNaiveCount++;
      }
    }
    const beatNaivePct = totalCompanies > 0 ? ((beatNaiveCount / totalCompanies) * 100).toFixed(1) : "0.0";

    return `- ${config.name}: ${config.isTrained ? `Won ${wins}/${totalCompanies} companies (${winRate}% win rate)` : "Benchmark persistence model"}. Beats Naive on ${beatNaiveCount}/${totalCompanies} stocks (${beatNaivePct}%).`;
  }).join("\n");

  const statTests = (metrics.statisticalTests || {}) as Record<string, any>;
  const friedman = statTests.friedman;
  const consistency = statTests.best_model_consistency;

  let statsText = "Statistical tests conducted:\n";
  if (friedman) {
    statsText += `- Friedman Omnibus Test: Chi-square statistic = ${formatNum(friedman.statistic, 2)}, p-value = ${friedman.p_value < 0.001 ? "< 0.001 (Statistically Significant difference across models)" : formatNum(friedman.p_value, 4)}\n`;
  }
  if (consistency) {
    statsText += `- Best Model Consistency Check: Dominant model = ${consistency.dominant_model}, Won = ${consistency.dominant_count}/${consistency.total_companies} companies (${consistency.pass ? "Pass" : "Fail"})\n`;
  }

  const perCompanyWinners = symbols
    .map((sym) => `${sym}: ${metrics.perCompany[sym]?.bestModel || "N/A"}`)
    .join(", ");

  return `[Context: Model Comparison & Performance]
- Total Tracked Companies: ${totalCompanies}
- Evaluation Rule: Per-company winning model is determined strictly by lowest test-set RMSE on chronological backtest.
- Cross-company aggregation uses median metrics because stock price levels range from ₱2 to ₱2,000 across the PSE.

Cross-Company Model Summary:
${modelSummary}

${statsText}
Per-Company Selected Winners:
${perCompanyWinners}`;
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
- Model Selection: For each company, the model with the lowest test-set RMSE on the held-out test window is automatically selected.`;
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
