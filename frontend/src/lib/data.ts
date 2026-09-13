import { promises as fs } from "fs";
import path from "path";
import type {
  CompanyDetail,
  CompanySummary,
  DashboardData,
  LatestData,
  MetricsData,
} from "./types";

// Server-side readers load the operational JSON published atomically by the
// backend exporter. The frontend never loads fitted models or runs inference.
const FORECASTS_DIR = path.join(process.cwd(), "public", "forecasts");

async function readJson<T>(relativePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(path.join(FORECASTS_DIR, relativePath), "utf-8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export async function getDashboard(): Promise<DashboardData | null> {
  return readJson<DashboardData>("dashboard.json");
}

export async function getLatest(): Promise<LatestData | null> {
  return readJson<LatestData>("latest.json");
}

export async function getMetrics(): Promise<MetricsData | null> {
  return readJson<MetricsData>("metrics.json");
}

export async function getCompanies(): Promise<CompanySummary[]> {
  return (await readJson<CompanySummary[]>("companies.json")) ?? [];
}

export async function getCompanyDetail(symbol: string): Promise<CompanyDetail | null> {
  return readJson<CompanyDetail>(`company/${symbol.toUpperCase()}.json`);
}

export async function getAllSymbols(): Promise<string[]> {
  const companies = await getCompanies();
  return companies.map((c) => c.symbol);
}
