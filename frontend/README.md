# ForecastPH — Frontend (Next.js, Vercel)

A static, frontend-only Next.js 14 (App Router) dashboard for PSE stock forecasts. It contains **no
Python, no backend, no database, and no API routes** — every page is a React Server Component that
reads JSON files from `public/forecasts/` at build/request time.

This is one half of a monorepo — see the [repo root README](../README.md) for the overall layout.
Those JSON files are produced and committed by the sibling `../backend/` pipeline's GitHub Actions
workflows (see `../backend/README.md` and `../backend/scripts/export_forecast_artifacts.py`). This
app never fetches from a remote API and never runs model inference — it's a plain `fs.readFile`
against files already sitting in this same repo by the time Vercel builds it.

## Local development

```bash
npm install
npm run dev
```

Open http://localhost:3000. The app reads whatever JSON currently exists under `public/forecasts/`
— that's committed to the repo, so a normal `git pull` keeps it current; no separate copy step is
needed (unlike a two-repo setup).

## Data contract

```
public/forecasts/
  dashboard.json        # home page summary: totals, sectors, top gainer/loser, pipeline status
  latest.json           # lightweight freshness indicator (forecast date, last run, status)
  metrics.json          # aggregate + per-company model performance (RMSE/MAE/MASE/R²)
  companies.json        # flat list for the Company List page + search
  company/<SYMBOL>.json # full detail: OHLCV, backtest series, per-model metrics
  history/<SYMBOL>.json # full OHLCV history (superset of company/<SYMBOL>.json's trimmed series)
```

If a file or ticker is missing, the relevant page renders a "not available yet" state instead of
throwing — a ticker the pipeline hasn't processed yet simply won't appear in `companies.json`.

## Deploying to Vercel

1. Import the repo root into Vercel. Framework preset: **Next.js** (auto-detected).
2. **Root Directory**: Project Settings → Root Directory → `frontend/`. This is the only
   monorepo-specific setting needed.
3. No environment variables are required; there's nothing to configure secrets for, since there's no
   API this app calls.
4. Every commit that touches `frontend/public/forecasts/**` (i.e. every pipeline run in
   `../backend/`) triggers a new Vercel deployment automatically via Vercel's normal Git
   integration — no deploy hook, no cross-repo wiring, since it's all one push to one repo now.
5. Free tier is sufficient: this is a static/SSR-light site with no serverless functions, cron jobs,
   or image optimization pipeline in use (`next.config.js` sets `images.unoptimized = true`).

## What was intentionally left out

The original HTML prototype included a "Live Prediction" flow (upload a CSV, run a prediction in the
browser). That required Python model inference at request time, which conflicts directly with the
"no backend, no Python on Vercel" requirement. `/live` now explains this trade-off instead of silently
breaking; see the repo root README's architecture notes for the reasoning.
