import Link from "next/link";
import ChangeBadge from "./ChangeBadge";
import CompanyLogo from "./CompanyLogo";
import WatchlistStar from "./watchlist/WatchlistStar";
import { formatDate, formatPeso } from "@/lib/format";
import type { CompanySummary } from "@/lib/types";

export default function CompanyCard({ company }: { company: CompanySummary }) {
  return (
    <article className="bg-dark-card border border-dark-border rounded-xl p-5 hover:border-brand-500/50 transition-colors shadow-sm space-y-3">
      <div className="flex items-start justify-between gap-3">
        <Link
          href={`/companies/${company.symbol}`}
          className="flex items-center gap-3 min-w-0 rounded-md focus:outline-none focus:ring-2 focus:ring-brand-500"
        >
          <CompanyLogo symbol={company.symbol} name={company.name} size="md" />
          <p className="font-bold text-white text-lg leading-tight">{company.symbol}</p>
        </Link>
        <div className="flex items-center gap-1.5 shrink-0">
          <WatchlistStar symbol={company.symbol} showLabel size="sm" />
          <ChangeBadge pctChange={company.pctChange} />
        </div>
      </div>

      <Link
        href={`/companies/${company.symbol}`}
        className="block min-h-10 text-sm leading-snug text-slate-400 rounded-md focus:outline-none focus:ring-2 focus:ring-brand-500"
      >
        {company.name}
      </Link>

      <Link
        href={`/companies/${company.symbol}`}
        className="block rounded-md focus:outline-none focus:ring-2 focus:ring-brand-500"
      >
        <div className="pt-3 border-t border-dark-border/50 grid grid-cols-2 gap-2">
        <div>
          <p className="text-xs text-slate-500">Forecasted Close</p>
          <p className="text-lg font-semibold text-white">{formatPeso(company.predictedClose)}</p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Selected Model</p>
          <p className="text-sm font-medium text-brand-400 truncate">{company.bestModel}</p>
        </div>
      </div>

        <div className="flex items-center justify-between text-xs text-slate-400 pt-1">
        {company.forecastDate ? (
          <span className="text-slate-400">Forecast for {formatDate(company.forecastDate)}</span>
        ) : (
          <span />
        )}
        <span className="text-[11px] px-2 py-0.5 rounded-md bg-dark-bg border border-dark-border text-slate-400">
          {company.sector}
        </span>
        </div>
      </Link>
    </article>
  );
}
