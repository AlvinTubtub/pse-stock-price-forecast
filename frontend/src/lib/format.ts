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
    // Avoid UTC timezone shifts on date-only YYYY-MM-DD strings
    if (/^\d{4}-\d{2}-\d{2}$/.test(iso)) {
      const [y, m, d] = iso.split("-").map(Number);
      const date = new Date(y, m - 1, d);
      return date.toLocaleDateString("en-PH", { year: "numeric", month: "short", day: "numeric" });
    }
    const d = new Date(iso);
    return d.toLocaleDateString("en-PH", {
      year: "numeric",
      month: "short",
      day: "numeric",
      timeZone: "Asia/Manila",
    });
  } catch {
    return iso;
  }
}

export function formatDateTimePht(iso: string | null | undefined): string {
  if (!iso) return "--";
  try {
    const d = new Date(iso);
    const dateStr = d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      timeZone: "Asia/Manila",
    });
    const timeStr = d.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
      timeZone: "Asia/Manila",
    });
    return `${dateStr}, ${timeStr} PHT`;
  } catch {
    return iso;
  }
}

export function formatDateTime(iso: string | null | undefined): string {
  return formatDateTimePht(iso);
}
