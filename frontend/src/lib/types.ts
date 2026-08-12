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
  confidence: number;
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
  confidence: number;
  metrics: Record<string, ModelMetric>;
  nextClose: Record<string, number>;
  ohlcv: OhlcvPoint[];
  backtestActual: number[];
  backtestByModel: Record<string, number[]>;
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
  aggregate: Record<string, { rmse: number; mae: number; mase: number; r2: number }>;
  bestModel: string;
  worstModel: string;
  perCompany: Record<string, { metrics: Record<string, ModelMetric>; bestModel: string }>;
  statisticalTests: Record<string, unknown>;
}

export interface LatestData {
  generatedAt: string;
  forecastDate: string;
  lastRunAt: string | null;
  status: string;
}
