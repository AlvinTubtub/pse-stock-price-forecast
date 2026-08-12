import { promises as fs } from "fs";
import path from "path";
import type {
  CompanyDetail,
  CompanySummary,
  DashboardData,
  LatestData,
  MetricsData,
} from "./types";

// Every reader here does a plain fs.readFile against the JSON files that
// backend/scripts/export_forecast_artifacts.py writes straight into this
// repo's frontend/public/forecasts/ (monorepo — frontend and backend are
// sibling directories in the same repo, so there is no cross-repo fetch
// here). There is no database, no API route, no Python — this is the
// entire "backend" of the deployed app; the real backend/ pipeline that
// produces the JSON runs entirely inside GitHub Actions, never on Vercel.
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
