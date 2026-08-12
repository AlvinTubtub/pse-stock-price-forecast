export default function StatCard({
  label,
  value,
  sublabel,
  accent = "text-white",
}: {
  label: string;
  value: string;
  sublabel?: string;
  accent?: string;
}) {
  return (
    <div className="bg-dark-card border border-dark-border rounded-xl p-5 shadow-sm">
      <p className="text-xs uppercase tracking-wide text-slate-400 mb-2">{label}</p>
      <p className={`text-3xl font-bold mb-1 ${accent}`}>{value}</p>
      {sublabel && <p className="text-sm text-slate-400">{sublabel}</p>}
    </div>
  );
}
