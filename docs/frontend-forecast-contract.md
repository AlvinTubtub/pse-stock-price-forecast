# ForecastPH frontend forecast-data contract

## Scope and source of truth

This document records the JSON contract consumed by the current frontend. It is a compatibility contract, not a claim that the checked-in prices, forecasts, metrics, or study results are statistically correct.

The application source is under `frontend/src/app`, `frontend/src/components`, and `frontend/src/lib`. The requested top-level `frontend/app`, `frontend/components`, `frontend/lib`, and `frontend/types` directories do not exist. Types are defined in `frontend/src/lib/types.ts`.

All JSON is read server-side by `frontend/src/lib/data.ts` from `frontend/public/forecasts`. There is no runtime schema validation: TypeScript assertions do not protect against malformed JSON. A missing or invalid whole file is caught and returned as `null`; missing nested required fields can still cause rendering failures.

## 1. Required output files and directories

| Path | Cardinality | Current consumer | Requirement |
|---|---:|---|---|
| `frontend/public/forecasts/companies.json` | One | Home, Companies, Watchlist, Models operational table, navbar, layout, chatbot | Required for normal navigation and company route generation. Missing/invalid becomes an empty list. |
| `frontend/public/forecasts/dashboard.json` | One | Home, About, Live chatbot context | Required for complete dashboard cards and metadata. UI has partial fallbacks. |
| `frontend/public/forecasts/latest.json` | One | Global educational banner, Models page, chatbot | Required for refresh status and target-date metadata. UI degrades if absent. |
| `frontend/public/forecasts/metrics.json` | One | Models operational table, Watchlist comparison, chatbot | Required for operational metric display. UI degrades if absent. |
| `frontend/public/forecasts/company/{SYMBOL}.json` | One per symbol in `companies.json` | `/companies/{SYMBOL}` and company chatbot context | Required for each generated company route; missing/invalid produces a 404. Filename must use the uppercase `symbol`. |
| `frontend/public/forecasts/history/{SYMBOL}.json` | Currently one per symbol | No current frontend reader | Not runtime-required by the current frontend. It is a standalone OHLCV export only. |
| `frontend/public/forecasts/formal/FORMAL_CORRECTED_20260828_02.json` | Exactly this hard-coded path | Models page and chatbot | Required for the formal-study section. Missing/invalid shows a formal-study unavailable message. This is immutable research evidence, not an operational forecast artifact. |

The current universe contains 15 symbols, but most operational UI code is list-driven. The formal-study copy and several research assertions explicitly assume 15 companies.

## 2. JSON schemas

The schemas below use TypeScript notation. Unless marked optional with `?` or nullable with `| null`, a field must be present and non-null for a reliable render.

```ts
type UppercaseSymbol = string;
type DateOnly = string;    // exactly YYYY-MM-DD
type IsoDateTime = string; // ISO 8601 with an explicit UTC offset or Z
type PrincipalModelLabel = "Lag-Informed Regression" | "ARIMA" | "LSTM";
type ModelDisplayLabel = PrincipalModelLabel | "Naive baseline";
```

### `companies.json`

```ts
type Direction = "bullish" | "bearish";

type CompaniesJson = CompanySummary[];

interface CompanySummary {
  symbol: string;
  name: string;
  sector: string;
  latestClose: number;
  predictedClose: number;
  pctChange: number;       // percentage points, e.g. 1.25 means 1.25%
  direction: Direction;
  bestModel: PrincipalModelLabel;
  confidence?: number;     // accepted by the type but not rendered by current UI
  forecastDate?: DateOnly;
}
```

`symbol` is used as a React key, URL segment, watchlist identifier, logo identifier, navbar search value, and lookup key into `metrics.json`. Symbols therefore must be unique, uppercase, and consistent across every artifact. `name` and `sector` are also searched or grouped directly and should not be null.

### `dashboard.json`

```ts
interface DashboardJson {
  generatedAt: IsoDateTime;
  forecastDate: DateOnly;
  lastRunAt: IsoDateTime | null;
  status: string;
  totalCompanies: number;
  missingCompanies: string[];
  sectors: Array<{ name: string; count: number }>;
  marketSummary: {
    gainers: number;
    losers: number;
    unchanged: number;
  };
  topGainer: CompanySummary | null;
  topLoser: CompanySummary | null;
}
```

`topGainer` and `topLoser` are explicitly nullable and have empty states. `lastRunAt` is explicitly nullable. `missingCompanies`, `status`, and `generatedAt` are part of the declared contract; `generatedAt` is used as the About page fallback timestamp, while `missingCompanies` is not currently rendered. Top-mover objects use the same shape as `companies.json`, although only `symbol`, `name`, `sector`, `predictedClose`, and `pctChange` are used by the visible dashboard cards.

### `latest.json`

```ts
interface LatestJson {
  generatedAt: IsoDateTime;
  forecastDate: DateOnly;
  lastRunAt: IsoDateTime | null;
  status: string;
}
```

The global banner only renders refresh metadata when `lastRunAt` is truthy. The Models page and chatbot fall back to `dashboard.json` or `metrics.json` for some missing operational metadata.

### `metrics.json`

```ts
type NumericString = string; // string containing a finite decimal number
type ModelId = "lag_reg" | "arima" | "lstm" | "naive";

interface ModelMetric {
  rmse: number | NumericString;
  mae: number | NumericString;
  mase: number | NumericString;
  r2: number | NumericString;
  ljung_box_pvalue?: number | NumericString;
}

interface MetricsJson {
  generatedAt: IsoDateTime;
  forecastDate?: DateOnly;
  lastRunAt?: IsoDateTime | null;
  status?: string;
  aggregate: Record<ModelDisplayLabel, {
    rmse: number;
    mae: number;
    mase: number;
    r2: number;
  }>;
  bestModel: ModelDisplayLabel;
  worstModel: ModelDisplayLabel;
  perCompany: Record<UppercaseSymbol, {
    metrics: Record<ModelId, ModelMetric>;
    bestModel: PrincipalModelLabel;
  }>;
  statisticalTests: Record<string, unknown>;
}
```

Current visible operational consumers use `perCompany[symbol].metrics[selectedModelId].rmse` and `.mase`. The chatbot also reads `generatedAt`. The other declared fields should remain available for compatibility, though `aggregate`, `bestModel`, `worstModel`, and `statisticalTests` are not currently rendered by application code.

Numeric strings are accepted only for metric leaves because formatting code parses them. New exporters should emit finite JSON numbers consistently. Do not emit `NaN`, infinity, empty strings, or null.

### `company/{SYMBOL}.json`

```ts
interface OhlcvPoint {
  date: DateOnly;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface CompanyDetailJson {
  symbol: string;
  name: string;
  sector: string;
  previousClose: number;
  predictedClose: number;
  pesoChange: number;
  pctChange: number;
  direction: "bullish" | "bearish";
  model: PrincipalModelLabel;
  confidence?: number;
  metrics: Record<ModelId, ModelMetric>;
  nextClose: {
    lag: number;
    arima: number;
    lstm: number;
  };
  ohlcv: OhlcvPoint[];
  backtestDates?: DateOnly[];
  backtestActual: number[];
  backtestByModel: Record<ModelDisplayLabel, number[]>;
  productionBacktestDates?: DateOnly[];
  productionBacktestActual?: number[];
  productionBacktestByModel?: Record<PrincipalModelLabel, number[]>;
  forecastDate?: DateOnly;
  dataAsOf?: DateOnly | null;
  inferenceAt?: IsoDateTime | null;

  // Present in current exports but not read or typed by the frontend:
  backtestMethodology?: {
    source: string;
    alignment: string;
    window: number;
  };
}
```

Cross-field invariants:

- `previousClose` should equal the latest `ohlcv[].close`, and `dataAsOf` should equal that point's date.
- `predictedClose` must be the prediction belonging to `model`; `pesoChange = predictedClose - previousClose`; `pctChange = pesoChange / previousClose * 100`.
- `model` uses the display label, not the ID. It must match the corresponding display-labeled key in `backtestByModel` and the ID-to-label mapping used for `metrics`.
- `backtestDates.length`, `backtestActual.length`, and every `backtestByModel` series length must match. Index `i` describes one trading session.
- If production history is present, `productionBacktestDates.length`, `productionBacktestActual.length`, and each included production model series length should match. These arrays are appended to the stored evaluation arrays by index.
- Arrays must be in ascending trading-session order. The frontend does not sort or realign by date.
- All prices, OHLC values, metric values, predictions, and volumes must be finite numbers. Array elements must not be null.

### `history/{SYMBOL}.json`

```ts
interface HistoryJson {
  symbol: string;
  ohlcv: OhlcvPoint[];
}
```

These files currently duplicate the standalone OHLCV history shape and are not read anywhere in `frontend/src`. Company pages use `company/{SYMBOL}.json.ohlcv` instead. There is no historical-forecast record in this directory despite its name.

### `formal/FORMAL_CORRECTED_20260828_02.json`

The Models page reads this exact filename through a hard-coded approved run ID. The following is the consumed subset and must remain compatible if the file is preserved or deliberately replaced:

```ts
type FormalModelId = "lag_reg" | "arima" | "lstm" | "naive";
type PrincipalModelId = "lag_reg" | "arima" | "lstm";

interface FormalStudyJson {
  schemaVersion: number;
  kind: "immutable_formal_study";
  runId: string;
  status: "complete";
  finalizedAt: IsoDateTime;
  identity: {
    repositoryCommit: string;
    sourceDataCommit: string;
    archiveSha256: string;
    corporateActionRegistrySha256: string;
    releaseUrl: string;
  };
  data: {
    firstDate: DateOnly;
    cutoffDate: DateOnly;
    rowsPerCompany: number;
    totalRows: number;
    forecastPairsPerCompany: number;
    developmentPairsPerCompany: number;
    holdoutPairsPerCompany: number;
    holdoutStart: DateOnly;
    holdoutEnd: DateOnly;
    totalHoldoutPredictions: number;
    companyCount: number;
  };
  methodology: {
    splitBasis: string;
    models: FormalModelId[];
    modelLabels: Record<FormalModelId, ModelDisplayLabel>;
    lassoAlphaCandidates: number;
    lstmConfigurations: number;
    lstmFolds: number;
    lstmTuningSeeds: number[];
    corporateActionPolicy: Record<string, unknown>;
  };
  conclusion: {
    summary: string;
    principalRmseWins: Record<PrincipalModelId, number>;
    dominanceThreshold: number;
    dominantModel: PrincipalModelId | null;
    significantVsNaive: Array<{
      symbol: string;
      model: PrincipalModelId;
      adjustedPValue: number;
      rawPValue: number;
      direction: string;
      significantlyBeatsNaive: boolean;
    }>;
    significantPosthocPairs: string[];
  };
  aggregate: Record<FormalModelId, {
    label: ModelDisplayLabel;
    principalRmseWins: number | null;
    medianRMSE: number;
    medianMAE: number;
    medianMASE: number;
    medianR2: number;
  }>;
  perCompany: Array<{
    symbol: string;
    dataSha256: string;
    principalWinnerByRmse: PrincipalModelId;
    lowestRmseIncludingNaive: FormalModelId[];
    metrics: Record<FormalModelId, ModelMetric>;
    dmSquaredErrorVsNaive: Record<PrincipalModelId, {
      adjustedPValue: number;
      rawPValue: number;
      direction: string;
      significantlyBeatsNaive: boolean;
    }>;
    configuration: {
      lagRegression: {
        alpha: number;
        selectedFeatureCount: number;
        selectedFeatures: string[];
      };
      arima: {
        order: number[];
        trend: string | null;
        converged: boolean;
      };
      lstm: {
        lookback: number;
        hiddenSize: number;
        learningRate: number;
        batchSize: number;
        fixedEpochs: number;
        finalFitSeed: number;
        tuningSeeds: number[];
      };
    };
    corporateActions: {
      verifiedEventCount: number;
      excludedTargetDates: DateOnly[];
      remainingHoldoutCount: number;
      status: string;
    };
  }>;
  acrossCompany: {
    friedmanMase: {
      statistic: number;
      permutation_p_value: number;
      permutation_count: number;
      n_companies: number;
    };
    wilcoxonPosthoc: {
      posthoc_executed: boolean;
      results: Record<string, {
        p_value: number;
        holm_p_value: number;
        statistic: number;
      }>;
    };
    rmseConsistency: {
      counts: Record<PrincipalModelId, number>;
      dominant_count: number;
      dominant_model: PrincipalModelId | null;
      min_required: number;
      pass: boolean;
    };
  };
}
```

Some formal-study fields are retained in the declared type even when the visible page does not currently render them. The page directly requires the identity link/commit, study dates/counts, methodology labels and tuning counts, conclusion summary/threshold/significant results, aggregate metrics, every company's metrics, DM results and selected configurations, and the Friedman statistic/permutation p-value.

## 3. Required model names and naming layers

| Purpose | Lag regression | ARIMA | LSTM | Baseline |
|---|---|---|---|---|
| Metric/formal IDs | `lag_reg` | `arima` | `lstm` | `naive` |
| Display labels and backtest keys | `Lag-Informed Regression` | `ARIMA` | `LSTM` | `Naive baseline` |
| `nextClose` keys | `lag` | `arima` | `lstm` | Not present |

`CompanySummary.bestModel` and `CompanyDetail.model` must use one of the three principal display labels exactly. Matching is string-based and case-sensitive. In particular, preserve lowercase `baseline` in `Naive baseline`. Charts tolerate `Naive Baseline` as a display alias for styling, but selection logic and the canonical mapping use `Naive baseline`.

The company metrics table is ordered by IDs `lag_reg`, `arima`, `lstm`, `naive`. The backtest and error charts dynamically enumerate display-labeled series. The Next-Day Prediction chart is hard-coded to exactly `nextClose.lag`, `.arima`, and `.lstm`.

## 4. Date formats

Use `YYYY-MM-DD` for all market-session dates: `forecastDate`, `dataAsOf`, OHLCV dates, backtest dates, production-backtest dates, and formal-study date ranges. Lexical date filtering and ordering assume this format.

Use a parseable ISO 8601 timestamp with an explicit timezone for `generatedAt`, `lastRunAt`, `inferenceAt`, and `finalizedAt`. The UI displays timestamps in `Asia/Manila`. Do not emit timezone-free timestamps.

## 5. Fields used by every company page

The route exists because `companies.json` supplies `symbol`; the detail payload then supplies:

- Header/profile: `symbol`, `name`, `sector`, `dataAsOf`, `forecastDate`.
- Summary cards: `previousClose`, `predictedClose`, `pesoChange`, `pctChange`, `model`.
- Selected-model evaluation: `model`, `metrics[modelId].rmse`, `.mae`, `.mase`, `.r2`.
- Historical OHLCV: every `ohlcv[].date/open/high/low/close/volume` field.
- Next-day chart: `ohlcv`, `previousClose`, `nextClose.lag/arima/lstm`, `forecastDate`, `dataAsOf`.
- Backtest and error charts: `backtestDates`, `backtestActual`, `backtestByModel`, plus optional production arrays.
- Model table: `metrics.lag_reg`, `.arima`, `.lstm`, and `.naive` when present.
- Chatbot context: all summary fields above, `direction`, all metrics, all `nextClose` entries, and the final five backtest rows for the selected display-label key.

`confidence`, `inferenceAt`, and `backtestMethodology` are currently not rendered or included in chatbot context.

## 6. Fields used by the Models page

The Models page combines four files:

- Formal study file: the consumed fields listed in its schema above.
- `companies.json`: `symbol`, `bestModel`, `latestClose`, `predictedClose`, `pctChange`, `forecastDate`.
- `metrics.json`: `perCompany[symbol].metrics[modelId].mase`, with `forecastDate`, `lastRunAt`, and `status` as fallbacks.
- `latest.json`: preferred `forecastDate`, `lastRunAt`, and `status` for the operational section.

The page uses a fixed four-method formal order and a fixed three-model principal subset. Several explanatory strings also embed current study-specific counts and outcomes rather than deriving all prose from JSON. Replacing the formal file with a different experiment without frontend changes would therefore be unsafe.

## 7. Fields used by dashboard cards

- Companies Tracked: `dashboard.totalCompanies`.
- Sectors Represented: `dashboard.sectors.length`; About also consumes each `sectors[].name` and groups `companies.json` by `sector`.
- Forecasted Gainers/Losers: `dashboard.marketSummary.gainers` and `.losers`. `.unchanged` is used by chatbot context.
- Top Forecasted Gainer/Loser: nullable `topGainer` and `topLoser`, using `symbol`, `name`, `sector`, `predictedClose`, and `pctChange`.
- Hero metadata: `dashboard.forecastDate` and `dashboard.lastRunAt`.

The top-mover objects should be copied from, or generated consistently with, the matching `companies.json` rows. The current type includes additional summary fields even when the visible card does not use them.

## 8. Backtest: Predicted vs Actual

Inputs are `backtestDates`, `backtestActual`, and `backtestByModel`. The frontend creates one row per `backtestActual` element and uses the same index from every date and model series. It does no date join, sorting, length validation, or numeric validation.

Optional realized operational history is appended as:

```ts
chartDates  = [...backtestDates,  ...productionBacktestDates]
chartActual = [...backtestActual, ...productionBacktestActual]
chartModel  = [...backtestByModel[label], ...productionBacktestByModel[label]]
```

`productionBacktestDates[0]` becomes the vertical “Live forecast begins” marker. The production history is considered realized only when the production date and actual arrays are non-empty and equal in length. A naive production series is currently absent, so the naive line ends at the stored evaluation boundary.

## 9. Forecast Error Over Time

This chart consumes the same aligned arrays as the prediction chart. For every available model value at index `i`, it computes:

```ts
error[i] = prediction[i] - actual[i]
```

Positive means overprediction and negative means underprediction. Nulls, strings, non-finite values, or misaligned arrays are unsupported. The chart enumerates all keys in `backtestByModel`, uses known colors for canonical labels, and falls back to gray for unknown labels.

## 10. Next-Day Prediction

The chart consumes the last 15, 25, 40, or 60 points from `ohlcv` and appends one forecast point. It draws exactly three branches from the latest `ohlcv[].close`:

- `nextClose.arima` as ARIMA.
- `nextClose.lag` as Lag-Informed Regression.
- `nextClose.lstm` as LSTM.

The naive baseline is not a next-day branch. Although the chart checks each prediction for `undefined`, the company-detail contract should provide all three finite numbers to preserve the existing three-model UI. `forecastDate` labels the appended point; if missing, the chart displays “Next Day”.

## 11. Last-60-session assumptions

- The heading explicitly promises “Last 60 Sessions”. Current checked-in company files contain exactly 60 dates, 60 actuals, and 60 values for each of four backtest series.
- The component does not slice these arrays to 60. If the exporter emits more or fewer points, the chart displays that count while the heading still says 60.
- “60” means trading sessions, not calendar days.
- All arrays must be chronologically ascending and index-aligned.
- Production history is appended after the 60 stored evaluation points, so the rendered total can exceed 60 even though the heading remains unchanged.
- The Next-Day Prediction chart separately offers a 60-point OHLCV view; it slices the last 60 elements itself.

The exporter should continue emitting exactly 60 stored evaluation rows unless the frontend heading and semantics are intentionally revised in a later phase.

## 12. Historical forecast structures

There are two distinct historical structures:

1. `backtest*` is a fixed stored deployment-evaluation window with actual prices and predictions for three principal models plus the naive baseline.
2. `productionBacktest*` is realized prospective operational history. It contains dates, actual closes, and predictions for the three principal models. The UI appends it after the fixed evaluation window and marks its first date.

`history/{SYMBOL}.json` is not forecast history. It contains only `{ symbol, ohlcv }` and is unused by current frontend code.

## 13. Nullability and failure expectations

- Whole-file failure: all readers return `null` on missing files, invalid JSON, or read errors. `companies.json` uniquely converts failure to `[]`.
- Explicitly nullable: `dashboard.topGainer`, `dashboard.topLoser`, `dashboard.lastRunAt`, `latest.lastRunAt`, optional `metrics.lastRunAt`, company `dataAsOf`, company `inferenceAt`, formal `conclusion.dominantModel`, formal `acrossCompany.rmseConsistency.dominant_model`, formal naive `principalRmseWins`, and formal ARIMA `trend`.
- Optional with UI fallback: company `confidence`, `forecastDate`, `backtestDates`, all three `productionBacktest*` fields, `dataAsOf`, `inferenceAt`; summary `confidence` and `forecastDate`; metrics operational date/status fields.
- Structurally required and non-null: identifiers, names, sectors, numeric prices/changes, OHLCV fields, base actual/model arrays, model metrics, and `nextClose` values.
- Empty production arrays are supported. Empty base backtest arrays show an empty-state chart. Empty OHLCV makes the next-day chart empty and weakens other company-page assumptions.
- `direction` has no `unchanged` value. Current exports classify zero change as `bullish` while `dashboard.marketSummary.unchanged` counts it as unchanged. Preserve the two-value type unless a frontend change is approved.

## 14. Does the UI assume exactly three principal models?

Yes. The principal models are exactly Lag-Informed Regression, ARIMA, and LSTM. This is hard-coded in:

- The Next-Day Prediction chart's three keys, branches, legends, and cards.
- The company model order and label map.
- The Models page's `PRINCIPAL` constant and four-method `MODELS` constant.
- Watchlist model-ID/display-label matching.
- Chatbot context mappings and explanatory copy.
- About and Learn page prose.

The naive method is a fourth evaluation benchmark, not a principal deployable model. Adding, removing, or renaming a principal model requires frontend changes. Merely adding another JSON key will not produce complete UI support.

## 15. Outputs safe to regenerate from a new backend

The new backend can safely regenerate these operational outputs as long as it preserves this contract and cross-file consistency:

- `companies.json`.
- `dashboard.json`.
- `latest.json`.
- `metrics.json`.
- Every `company/{SYMBOL}.json`.
- Every `history/{SYMBOL}.json`, although these files are currently unused and could be omitted only after confirming no external consumer depends on them.

The backend may also regenerate the operational `backtest*` and realized `productionBacktest*` structures inside company files. It must preserve the fixed 60-session base window, index alignment, model naming layers, chronological ordering, and the distinction between retrospective evaluation and prospective realized forecasts.

Do not automatically regenerate or overwrite `formal/FORMAL_CORRECTED_20260828_02.json`. The frontend treats it as an approved immutable study tied to a hard-coded run ID, evidence hashes, commit, release URL, fixed research design, and study-specific prose. Preserve it verbatim unless a separate, explicitly approved formal-study migration updates both evidence and frontend assumptions.

## Exporter compatibility checklist

- Emit valid strict JSON with finite numbers and no `NaN`/infinity.
- Emit all symbols consistently and create one uppercase company filename per `companies.json` row.
- Preserve all three naming layers exactly.
- Emit date-only and timestamp values in the formats above.
- Keep all indexed arrays aligned, ascending, and null-free.
- Emit exactly 60 base backtest sessions and three next-day principal predictions.
- Derive summaries, top movers, directions, changes, and metrics from the same underlying run so duplicated values agree.
- Keep operational artifacts separate from the immutable formal-study file.
