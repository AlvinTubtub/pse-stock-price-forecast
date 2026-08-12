import Link from "next/link";
import ChangeBadge from "./ChangeBadge";
import { formatPeso } from "@/lib/format";
import type { CompanySummary } from "@/lib/types";

export default function CompanyCard({ company }: { company: CompanySummary }) {
  return (
    <Link
      href={`/companies/${company.symbol}`}
      className="block bg-dark-card border border-dark-border rounded-xl p-5 hover:border-brand-500/50 transition-colors shadow-sm"
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="font-bold text-white text-lg">{company.symbol}</p>
          <p className="text-sm text-slate-400 truncate max-w-[16rem]">{company.name}</p>
        </div>
        <ChangeBadge pctChange={company.pctChange} />
      </div>
      <div className="flex items-end justify-between">
        <div>
          <p className="text-xs text-slate-500">Predicted next close</p>
          <p className="text-xl font-semibold text-white">{formatPeso(company.predictedClose)}</p>
        </div>
        <span className="text-xs px-2 py-1 rounded-md bg-dark-bg border border-dark-border text-slate-400">
          {company.sector}
        </span>
      </div>
    </Link>
  );
}
