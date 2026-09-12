export type Direction = "bullish" | "bearish";

export interface CompanySummary {
  symbol: string;
  name: string;
  sector: string;
  latestClose: number;
  predictedClose: number;
  pctChange: number;
  direction: Direction;
  bestModel: string;
  confidence?: number;
  forecastDate?: string;
}

export interface OhlcvPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ModelMetric {
  rmse: string | number;
  mae: string | number;
  mase: string | number;
  r2: string | number;
  ljung_box_pvalue?: string | number;
}

export interface CompanyDetail {
  symbol: string;
  name: string;
  sector: string;
  previousClose: number;
  predictedClose: number;
  pesoChange: number;
  pctChange: number;
  direction: Direction;
  model: string;
  confidence?: number;
  metrics: Record<string, ModelMetric>;
  nextClose: Record<string, number>;
  ohlcv: OhlcvPoint[];
  backtestDates?: string[];
  backtestActual: number[];
  backtestByModel: Record<string, number[]>;
  productionBacktestDates?: string[];
  productionBacktestActual?: number[];
  productionBacktestByModel?: Record<string, number[]>;
  forecastDate?: string;
  dataAsOf?: string | null;
  inferenceAt?: string | null;
}

export interface DashboardData {
  generatedAt: string;
  forecastDate: string;
  lastRunAt: string | null;
  status: string;
  totalCompanies: number;
  missingCompanies: string[];
  sectors: { name: string; count: number }[];
  marketSummary: { gainers: number; losers: number; unchanged: number };
  topGainer: CompanySummary | null;
  topLoser: CompanySummary | null;
}

export interface MetricsData {
  generatedAt: string;
  forecastDate?: string;
  lastRunAt?: string | null;
  status?: string;
  aggregate: Record<string, { rmse: number; mae: number; mase: number; r2: number }>;
  bestModel: string;
  worstModel: string;
  perCompany: Record<string, { metrics: Record<string, ModelMetric>; bestModel: string }>;
  statisticalTests: Record<string, unknown>;
}

export type FormalModelId = "lag_reg" | "arima" | "lstm" | "naive";

export interface FormalModelSummary {
  label: string;
  principalRmseWins: number | null;
  medianRMSE: number;
  medianMAE: number;
  medianMASE: number;
  medianR2: number;
}

export interface FormalCompanyResult {
  symbol: string;
  dataSha256: string;
  principalWinnerByRmse: Exclude<FormalModelId, "naive">;
  lowestRmseIncludingNaive: FormalModelId[];
  metrics: Record<FormalModelId, ModelMetric>;
  dmSquaredErrorVsNaive: Record<
    Exclude<FormalModelId, "naive">,
    {
      adjustedPValue: number;
      rawPValue: number;
      direction: string;
      significantlyBeatsNaive: boolean;
    }
  >;
  configuration: {
    lagRegression: {
      alpha: number;
      selectedFeatureCount: number;
      selectedFeatures: string[];
    };
    arima: { order: number[]; trend: string | null; converged: boolean };
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
    excludedTargetDates: string[];
    remainingHoldoutCount: number;
    status: string;
  };
}

export interface FormalStudyData {
  schemaVersion: number;
  kind: "immutable_formal_study";
  runId: string;
  status: "complete";
  finalizedAt: string;
  identity: {
    repositoryCommit: string;
    sourceDataCommit: string;
    archiveSha256: string;
    corporateActionRegistrySha256: string;
    releaseUrl: string;
  };
  data: {
    firstDate: string;
    cutoffDate: string;
    rowsPerCompany: number;
    totalRows: number;
    forecastPairsPerCompany: number;
    developmentPairsPerCompany: number;
    holdoutPairsPerCompany: number;
    holdoutStart: string;
    holdoutEnd: string;
    totalHoldoutPredictions: number;
    companyCount: number;
  };
  methodology: {
    splitBasis: string;
    models: FormalModelId[];
    modelLabels: Record<FormalModelId, string>;
    lassoAlphaCandidates: number;
    lstmConfigurations: number;
    lstmFolds: number;
    lstmTuningSeeds: number[];
    corporateActionPolicy: Record<string, unknown>;
  };
  conclusion: {
    summary: string;
    principalRmseWins: Record<Exclude<FormalModelId, "naive">, number>;
    dominanceThreshold: number;
    dominantModel: null;
    significantVsNaive: Array<{
      symbol: string;
      model: Exclude<FormalModelId, "naive">;
      adjustedPValue: number;
    }>;
    significantPosthocPairs: string[];
  };
  aggregate: Record<FormalModelId, FormalModelSummary>;
  perCompany: FormalCompanyResult[];
  acrossCompany: {
    friedmanMase: {
      statistic: number;
      permutation_p_value: number;
      permutation_count: number;
      n_companies: number;
    };
    wilcoxonPosthoc: {
      posthoc_executed: boolean;
      results: Record<string, { p_value: number; holm_p_value: number; statistic: number }>;
    };
    rmseConsistency: {
      counts: Record<Exclude<FormalModelId, "naive">, number>;
      dominant_count: number;
      dominant_model: null;
      min_required: number;
      pass: false;
    };
  };
}

export interface LatestData {
  generatedAt: string;
  forecastDate: string;
  lastRunAt: string | null;
  status: string;
}
