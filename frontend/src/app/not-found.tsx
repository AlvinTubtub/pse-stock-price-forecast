import Link from "next/link";

export default function NotFound() {
  return (
    <div className="text-center py-20">
      <p className="text-6xl mb-4">📉</p>
      <h1 className="text-2xl font-bold text-white mb-2">Not found</h1>
      <p className="text-slate-400 mb-6">That page or ticker doesn&apos;t exist in the forecast data.</p>
      <Link href="/" className="text-brand-400 hover:text-brand-300 text-sm">
        ← Back home
      </Link>
    </div>
  );
}
