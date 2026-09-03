import Link from "next/link";
import StatCard from "@/components/StatCard";
import CompanyCard from "@/components/CompanyCard";
import CompanyLogo from "@/components/CompanyLogo";
import ChangeBadge from "@/components/ChangeBadge";
import { getCompanies, getDashboard } from "@/lib/data";
import { formatDate, formatDateTimePht, formatPeso } from "@/lib/format";

export default async function HomePage() {
  const [dashboard, companies] = await Promise.all([getDashboard(), getCompanies()]);

  return (
    <div className="space-y-8">
      {/* 1. Hero Section */}
      <section className="bg-dark-card border border-dark-border rounded-2xl p-6 sm:p-8 shadow-sm">
        <span className="inline-block px-3 py-1 text-xs font-semibold text-brand-400 bg-brand-500/10 border border-brand-500/30 rounded-full mb-3">
          Educational Dashboard
        </span>
        <h1 className="text-2xl sm:text-4xl font-bold text-white mb-3 tracking-tight">
          Cross-Sector Next-Day Stock Price Forecasting
        </h1>
        <p className="text-slate-300 text-sm sm:text-base max-w-3xl leading-relaxed">
          Explore historical Philippine stock market data, compare machine learning and statistical
          models, and understand next-day price prediction techniques.
        </p>

        {(dashboard?.forecastDate || dashboard?.lastRunAt) && (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 mt-5 pt-4 border-t border-dark-border/60 text-xs text-slate-400">
            {dashboard?.forecastDate && (
              <span className="text-brand-300 font-medium">
                Forecast for: {formatDate(dashboard.forecastDate)}
              </span>
            )}
            {dashboard?.forecastDate && dashboard?.lastRunAt && <span>&middot;</span>}
            {dashboard?.lastRunAt && (
              <span>Last pipeline run: {formatDateTimePht(dashboard.lastRunAt)}</span>
            )}
          </div>
        )}
      </section>

      {/* 2. Summary Grid */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Companies Tracked" value={String(dashboard?.totalCompanies ?? "--")} />
        <StatCard label="Sectors Represented" value={String(dashboard?.sectors.length ?? "--")} />
        <StatCard
          label="Forecasted Gainers / Losers"
          value={`${dashboard?.marketSummary.gainers ?? 0} / ${dashboard?.marketSummary.losers ?? 0}`}
          accent="text-white"
        />
        <div className="bg-dark-card border border-dark-border rounded-xl p-5 shadow-sm flex flex-col justify-between">
          <p className="text-xs uppercase tracking-wide text-slate-400 mb-2">Data Source</p>
          <p className="text-xs font-medium text-slate-200 leading-relaxed">
            Official PSE Daily Quotations Reports
          </p>
        </div>
      </section>

      {/* 3. Top Forecasted Gainer & Loser */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <p className="text-xs uppercase tracking-wide text-slate-400 mb-3">
            Top Forecasted Gainer
          </p>
          {dashboard?.topGainer ? (
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <CompanyLogo symbol={dashboard.topGainer.symbol} name={dashboard.topGainer.name} size="md" />
                <div className="min-w-0">
                  <Link
                    href={`/companies/${dashboard.topGainer.symbol}`}
                    className="text-xl font-bold text-white hover:text-brand-400 transition-colors leading-tight block"
                  >
                    {dashboard.topGainer.symbol}
                  </Link>
                  <p className="text-sm text-slate-400 truncate max-w-[12rem]">{dashboard.topGainer.name}</p>
                  <p className="text-xs text-slate-500 mt-0.5">{dashboard.topGainer.sector}</p>
                </div>
              </div>
              <div className="text-right shrink-0">
                <p className="text-lg font-semibold text-white">
                  {formatPeso(dashboard.topGainer.predictedClose)}
                </p>
                <ChangeBadge pctChange={dashboard.topGainer.pctChange} />
              </div>
            </div>
          ) : (
            <p className="text-slate-500 text-sm">No data yet.</p>
          )}
        </div>

        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
          <p className="text-xs uppercase tracking-wide text-slate-400 mb-3">
            Top Forecasted Loser
          </p>
          {dashboard?.topLoser ? (
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <CompanyLogo symbol={dashboard.topLoser.symbol} name={dashboard.topLoser.name} size="md" />
                <div className="min-w-0">
                  <Link
                    href={`/companies/${dashboard.topLoser.symbol}`}
                    className="text-xl font-bold text-white hover:text-brand-400 transition-colors leading-tight block"
                  >
                    {dashboard.topLoser.symbol}
                  </Link>
                  <p className="text-sm text-slate-400 truncate max-w-[12rem]">{dashboard.topLoser.name}</p>
                  <p className="text-xs text-slate-500 mt-0.5">{dashboard.topLoser.sector}</p>
                </div>
              </div>
              <div className="text-right shrink-0">
                <p className="text-lg font-semibold text-white">
                  {formatPeso(dashboard.topLoser.predictedClose)}
                </p>
                <ChangeBadge pctChange={dashboard.topLoser.pctChange} />
              </div>
            </div>
          ) : (
            <p className="text-slate-500 text-sm">No data yet.</p>
          )}
        </div>
      </section>

      {/* 4. Forecast Disclaimer */}
      <section className="p-3.5 bg-dark-bg/60 border border-dark-border/60 rounded-xl text-xs text-slate-400 leading-relaxed">
        <p>
          <strong className="text-slate-300 font-medium">Disclaimer: </strong>
          Forecasts are model-generated estimates for the next trading session and are not investment
          advice.
        </p>
      </section>

      {/* 5. Beginner Guide Section */}
      <section className="bg-dark-card border border-dark-border rounded-2xl p-6 sm:p-7 shadow-sm space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <span className="inline-block px-2.5 py-0.5 text-[11px] font-semibold text-brand-400 bg-brand-500/10 border border-brand-500/20 rounded-full mb-1">
              Beginner Guide
            </span>
            <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              New to Stock Forecasting?
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 mt-1">
              Three quick rules for interpreting automated machine learning stock predictions.
            </p>
          </div>
          <Link
            href="/learn-stocks#trading-101"
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-white text-xs sm:text-sm font-semibold transition-all shrink-0 self-start sm:self-auto shadow-md shadow-brand-500/20"
          >
            Start with Stock Trading 101 →
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-dark-bg/80 border border-dark-border/80 rounded-xl p-4.5 space-y-2">
            <div className="flex items-center gap-2.5 text-brand-400 text-base font-semibold">
              <span className="w-7 h-7 rounded-lg bg-brand-500/15 border border-brand-500/30 flex items-center justify-center text-xs font-bold">
                1
              </span>
              <h3 className="text-sm font-bold text-white">Understand the Forecast</h3>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              ForecastPH predicts the estimated next trading-day closing price based on numerical historical market data. Predictions are statistical estimates and are not guaranteed.
            </p>
          </div>

          <div className="bg-dark-bg/80 border border-dark-border/80 rounded-xl p-4.5 space-y-2">
            <div className="flex items-center gap-2.5 text-amber-400 text-base font-semibold">
              <span className="w-7 h-7 rounded-lg bg-amber-500/15 border border-amber-500/30 flex items-center justify-center text-xs font-bold">
                2
              </span>
              <h3 className="text-sm font-bold text-white">Check Historical Accuracy</h3>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Always review the Backtest, Forecast Error, RMSE, MAE, MASE, and R² to see how accurately each model has performed historically before interpreting a forecast.
            </p>
          </div>

          <div className="bg-dark-bg/80 border border-dark-border/80 rounded-xl p-4.5 space-y-2">
            <div className="flex items-center gap-2.5 text-emerald-400 text-base font-semibold">
              <span className="w-7 h-7 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-xs font-bold">
                3
              </span>
              <h3 className="text-sm font-bold text-white">Learn Before You Trade</h3>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Build a strong foundation in Philippine stock market fundamentals, risk management, and order types before making financial decisions.
            </p>
          </div>
        </div>
      </section>

      {/* 6. Company Preview Section */}
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
