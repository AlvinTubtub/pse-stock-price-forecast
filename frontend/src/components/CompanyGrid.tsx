"use client";

import { useMemo, useState } from "react";
import CompanyCard from "./CompanyCard";
import type { CompanySummary } from "@/lib/types";

export default function CompanyGrid({ companies }: { companies: CompanySummary[] }) {
  const [sector, setSector] = useState("All");
  const sectors = useMemo(() => ["All", ...Array.from(new Set(companies.map((c) => c.sector))).sort()], [companies]);

  const filtered = sector === "All" ? companies : companies.filter((c) => c.sector === sector);

  return (
    <div>
      <div className="mb-6 max-w-xs">
        <select
          value={sector}
          onChange={(e) => setSector(e.target.value)}
          className="bg-dark-card border border-dark-border text-sm rounded-lg focus:ring-brand-500 focus:border-brand-500 block w-full p-2.5 text-white"
        >
          {sectors.map((s) => (
            <option key={s} value={s}>
              {s === "All" ? "All Sectors" : s}
            </option>
          ))}
        </select>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((c) => (
          <CompanyCard key={c.symbol} company={c} />
        ))}
      </div>
    </div>
  );
}
