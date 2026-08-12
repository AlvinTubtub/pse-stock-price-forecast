import Link from "next/link";

export default function LivePredictionPage() {
  return (
    <div className="max-w-2xl space-y-6">
      <h1 className="text-2xl font-bold text-white">Live Prediction</h1>
      <div className="bg-dark-card border border-dark-border rounded-xl p-6 space-y-4 text-sm text-slate-300 leading-relaxed">
        <p>
          The original prototype let a visitor upload a CSV and run a prediction directly in the browser. That
          flow depended on a Python backend (ARIMA/LSTM training and inference), which this deployment doesn&apos;t
          have — the frontend is 100% static and never executes Python, by design.
        </p>
        <p>
          Every forecast on this site is instead produced ahead of time by the scheduled GitHub Actions pipeline
          and published as JSON, so results are consistent, reviewable, and don&apos;t require running machine
          learning code on someone else&apos;s browser or a paid server.
        </p>
        <p>
          Browse the companies the pipeline already covers, or check when data was last refreshed on the
          <Link href="/" className="text-brand-400 hover:text-brand-300"> home page</Link>.
        </p>
      </div>
      <Link href="/companies" className="inline-block px-4 py-2 bg-brand-600 hover:bg-brand-500 text-white rounded-lg text-sm font-medium transition-colors">
        Browse Companies →
      </Link>
    </div>
  );
}
