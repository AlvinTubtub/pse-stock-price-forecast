import { formatPct } from "@/lib/format";

export default function ChangeBadge({ pctChange }: { pctChange: number }) {
  const positive = pctChange >= 0;
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold ${
        positive ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"
      }`}
    >
      {positive ? "▲" : "▼"} {formatPct(pctChange)}
    </span>
  );
}
