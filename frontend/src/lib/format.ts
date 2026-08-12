export function formatPeso(value: number | string): string {
  const num = typeof value === "string" ? parseFloat(value) : value;
  if (Number.isNaN(num)) return "--";
  return `₱${num.toLocaleString("en-PH", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function formatPct(value: number | string): string {
  const num = typeof value === "string" ? parseFloat(value) : value;
  if (Number.isNaN(num)) return "--";
  const sign = num > 0 ? "+" : "";
  return `${sign}${num.toFixed(2)}%`;
}

export function formatNum(value: number | string, digits = 4): string {
  const num = typeof value === "string" ? parseFloat(value) : value;
  if (Number.isNaN(num)) return "--";
  return num.toFixed(digits);
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "--";
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-PH", { year: "numeric", month: "short", day: "numeric" });
  } catch {
    return iso;
  }
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "--";
  try {
    const d = new Date(iso);
    return `${d.toLocaleDateString("en-PH", { year: "numeric", month: "short", day: "numeric" })} ${d.toLocaleTimeString("en-PH", { hour: "2-digit", minute: "2-digit" })} UTC`;
  } catch {
    return iso;
  }
}
