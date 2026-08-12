import CompanyGrid from "@/components/CompanyGrid";
import { getCompanies } from "@/lib/data";

export default async function CompaniesPage() {
  const companies = await getCompanies();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">Company List</h1>
        <p className="text-slate-400 text-sm">
          {companies.length} PSE-listed companies with automated next-session forecasts.
        </p>
      </div>
      <CompanyGrid companies={companies} />
    </div>
  );
}
