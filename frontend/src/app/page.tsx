import Link from "next/link";
import StatCard from "@/components/StatCard";
import CompanyCard from "@/components/CompanyCard";
import ChangeBadge from "@/components/ChangeBadge";
import { getCompanies, getDashboard } from "@/lib/data";
import { formatDate, formatPeso } from "@/lib/format";

export default async function HomePage() {
  const [dashboard, companies] = await Promise.all([getDashboard(), getCompanies()]);

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold text-white mb-1">Forecast Dashboard</h1>
        <p className="text-slate-400">
          Next-session closing price forecasts for {dashboard?.totalCompanies ?? "--"} PSE-listed companies,
          generated {dashboard?.forecastDate ? `for ${formatDate(dashboard.forecastDate)}` : ""} by the
          automated pipeline.
        </p>
      </section>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Companies Tracked" value={String(dashboard?.totalCompanies ?? "--")} />
        <StatCard label="Sectors" value={String(dashboard?.sectors.length ?? "--")} />
        <StatCard
          label="Gainers / Losers"
          value={`${dashboard?.marketSummary.gainers ?? 0} / ${dashboard?.marketSummary.losers ?? 0}`}
          accent="text-white"
        />
        <StatCard label="Pipeline Status" value={dashboard?.status ?? "unknown"} accent={dashboard?.status === "ok" ? "text-green-400" : "text-amber-400"} />
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <p className="text-xs uppercase tracking-wide text-slate-400 mb-3">Top Predicted Gainer</p>
          {dashboard?.topGainer ? (
            <div className="flex items-center justify-between">
              <div>
                <Link href={`/companies/${dashboard.topGainer.symbol}`} className="text-xl font-bold text-white hover:text-brand-400">
                  {dashboard.topGainer.symbol}
                </Link>
                <p className="text-sm text-slate-400">{dashboard.topGainer.name}</p>
              </div>
              <div className="text-right">
                <p className="text-lg font-semibold text-white">{formatPeso(dashboard.topGainer.predictedClose)}</p>
                <ChangeBadge pctChange={dashboard.topGainer.pctChange} />
              </div>
            </div>
          ) : (
            <p className="text-slate-500 text-sm">No data yet.</p>
          )}
        </div>
        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <p className="text-xs uppercase tracking-wide text-slate-400 mb-3">Top Predicted Loser</p>
          {dashboard?.topLoser ? (
            <div className="flex items-center justify-between">
              <div>
                <Link href={`/companies/${dashboard.topLoser.symbol}`} className="text-xl font-bold text-white hover:text-brand-400">
                  {dashboard.topLoser.symbol}
                </Link>
                <p className="text-sm text-slate-400">{dashboard.topLoser.name}</p>
              </div>
              <div className="text-right">
                <p className="text-lg font-semibold text-white">{formatPeso(dashboard.topLoser.predictedClose)}</p>
                <ChangeBadge pctChange={dashboard.topLoser.pctChange} />
              </div>
            </div>
          ) : (
            <p className="text-slate-500 text-sm">No data yet.</p>
          )}
        </div>
      </section>

      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-white">All Companies</h2>
          <Link href="/companies" className="text-sm text-brand-400 hover:text-brand-300">
            View full list →
          </Link>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {companies.slice(0, 6).map((c) => (
            <CompanyCard key={c.symbol} company={c} />
          ))}
        </div>
      </section>
    </div>
  );
}
