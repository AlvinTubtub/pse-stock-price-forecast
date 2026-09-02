# ForecastPH — Backend (Data/ML Pipeline)

ForecastPH forecasts next-session closing prices for selected PSE-listed companies. **This directory
is the data/ML pipeline only** — it scrapes, cleans, engineers features, trains, evaluates, and
produces forecasts, then exports the results as static JSON consumed by the frontend.

This is one half of a monorepo — see the [repo root README](../README.md) for the overall layout.
The UI lives in the sibling `../frontend/` directory (Next.js, deployed on Vercel). It contains no
Python and calls no backend — it reads the JSON files this pipeline's GitHub Actions workflows
commit straight to `../frontend/public/forecasts/`. Because both live in the same repo, that commit
*is* the frontend's data update, and Vercel's own Git integration (Root Directory: `frontend/`)
redeploys automatically on every push — no deploy hook or cross-repo wiring required.

**This directory is a pure, read-only-to-the-frontend data pipeline.** There is no server, no API,
and no upload/retraining capability triggered from the UI — every number the frontend shows comes
from files already committed here by the automated pipeline described below.
`scripts/export_forecast_artifacts.py` is the only piece written specifically for the frontend: it
reshapes this pipeline's existing outputs (`data/raw/`, `prediction_cache/`, `best_models.json`,
`latest_processed.json`, `statistical_tests.json`) into the flat JSON contract the frontend expects,
and runs as the last step of both GitHub Actions workflows below.

> The previous Streamlit UI (`app.py`, `ui/`, `pages_app/`) has been removed — it's fully superseded
> by `../frontend/`.

## Features

- Home dashboard with project overview
- Company list and sector browsing
- Company details with historical charts, next-day forecast, and actual-vs-predicted backtests
- Model comparison across forecasting methods (RMSE/MAE/MASE/R²)
- Educational section explaining OHLCV and forecasting models
- About page for project context and capstone background
- A "data last refreshed" indicator sourced directly from the automated pipeline's own run metadata

## Architecture

```text
Cron-job.org (Mon–Fri, 4:00 PM PHT)         GitHub Actions cron (Sun, 8:00 AM PHT)
        │  POST repository_dispatch                  │  schedule trigger
        ▼                                             ▼
.github/workflows/update_pipeline.yml       .github/workflows/train_models.yml
        │  ("Fast Pipeline")                          │  ("Heavy Training")
        ▼                                             ▼
run_pipeline.py --no-train                  model_selector --mode deployment-refresh --strict
  1. Download latest PSE EDGE disclosures      1. Load approved configurations
  2. Extract and validate PDF tables           2. Refit Lag Regression, ARIMA, and LSTM
  3. Update OHLCV datasets                        without retuning or formal evaluation
     (data/raw/<SYMBOL>.csv)                   3. Preserve approved model families and metrics
  4. Update latest_processed.json              4. Update deployment models + operational cache
        │
        ▼
Commit changed artifacts only
(idempotent — no-op if nothing changed)               │
        │                                             ▼
        │                                    Commit changed artifacts only
        │                                    (idempotent — no-op if nothing changed)
        ▼                                             │
        └──────────────────┬──────────────────────────┘
                            ▼
        Vercel (Root Directory: frontend/) auto-redeploys from the new commit
        via its normal Git integration — no deploy hook needed
                            │
                            ▼
        Dashboard reflects the latest data/models — no user interaction required
```

Data refreshes daily (Mon-Fri); models retrain weekly (Sun) — see
"Automated Pipeline" below for why, and the runtime numbers behind that
split.

The frontend (`../frontend/`) only ever:
- reads the JSON `scripts/export_forecast_artifacts.py` exports from `data/raw/*.csv`, `prediction_cache/`, `best_models.json`, `latest_processed.json`
- displays the Company List, Company Details, Forecast Results, Model Performance, charts, and dashboard metrics built from what it read

The frontend never downloads PDFs, processes data, retrains models, executes any forecasting pipeline, or writes anything back to the repository. There is no "Update Data" page, no upload widget, and no button anywhere in the app that triggers processing — the only way data changes is a commit from the automated pipeline landing in the repo.

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Statsmodels (ADF test, CV-scored ARIMA order search, Ljung-Box diagnostics)
- Torch
- Joblib (model persistence)
- pdfplumber, requests (PDF ingestion pipeline)

## Model Training Pipeline

Training happens exclusively inside the automated pipeline (never in the frontend):

```text
PSE EDGE PDF -> PDF Extraction -> CSV Generation -> Data Validation
    -> Feature Engineering -> Model Training (x3) -> Model Evaluation
    -> Statistical Significance Tests -> Best Model Selection
    -> Saved Models -> export_forecast_artifacts.py -> Frontend Dashboard (read-only)
```

All three models predict next-day **ΔClose** = Close(t+1) − Close(t)
rather than the raw Close level, and reconstruct
`Predicted Close(t+1) = Close(t) + Predicted ΔClose(t+1)` before any
metric is computed — this matches the capstone paper's methodology and
keeps RMSE/MAE/MASE/R² reported in peso terms, not on an internal
differenced/scaled target.

- `services/feature_engineering.py` — lag features + technical indicators
  (EMA 10/20, RSI 14, MACD/Signal, Bollinger Bands, daily return, rolling
  volatility, High-Low and Open-Close spreads) plus the expanded return
  feature set (lagged returns 1-20, rolling return mean/volatility at
  5/10/20, high-low range %, log volume, rolling volume means), shared by
  every model.
- `services/time_series_cv.py` — shared expanding-window rolling-origin
  CV splitter (5 folds, shrinking only for short series), used by both
  the regression's lambda selection and the ARIMA order search.
- `services/forecasting/lag_regression.py` — training-only StandardScaler
  -> PACF-assisted lag selection -> LASSO as the *final* estimator (no
  secondary OLS refit), lambda chosen by 5-fold expanding-window CV.
- `services/forecasting/arima_model.py` — ADF stationarity test +
  (p, d, q) search restricted to p<=3, d<=2, q<=3, scored by
  expanding-window rolling-origin CV with walk-forward one-step
  forecasting (not AIC), plus Ljung-Box residual diagnostics.
- `services/forecasting/lstm_model.py` — single LSTM layer + linear head,
  predicting scaled ΔClose (Min-Max scaler fit on the training split
  only), hyperparameters chosen by grid search over lookback
  (5/10/20/30), hidden units (25/50/100), learning rate (0.01/0.001), and
  batch size (16/32) — 48 configurations, each trained with mini-batches,
  up to 200 epochs, early stopping (patience 10), seed 42.
- `services/evaluation.py` — shared RMSE/MAE/MASE/R² metrics (computed on
  reconstructed peso prices; MASE is scaled against the in-sample naive
  one-step forecast), plus the cross-model statistical-significance
  suite: Diebold-Mariano with Newey-West HAC variance and the
  Harvey-Leybourne-Newbold small-sample correction (within each company),
  Holm-Bonferroni correction, a Friedman rank test and Holm-adjusted
  Wilcoxon signed-rank post-hoc tests (across companies), and a
  best-model consistency check (lowest RMSE on >=8 of 15 companies).
- `services/model_selector.py` — keeps formal evaluation, scheduled
  deployment refresh, and manual challenger retuning as explicit operations.
  Refresh reads approved configuration metadata and never reruns formal tests.

Deployment refresh is decoupled from data ingestion:
`refresh_deployment_all()` is called directly by
`.github/workflows/train_models.yml` (weekly), *not* by every run of
`services/pdf_pipeline/pipeline.py`/`run_pipeline.py` (Fast Pipeline,
Monday-Friday — see "Automated Pipeline" below for the full split and
why). `run_pipeline.py` still supports training inline via its
`train_models=True` default, for local/manual use:

```bash
python -m services.model_selector --mode deployment-refresh  # refit approved configurations
python -m services.model_selector --mode deployment-retune --symbols BPI  # manual challenger only
python run_pipeline.py                     # ingest + train in one go (local/dev; CI never does both together)
python run_pipeline.py --no-train           # ingest new data only, skip retraining (what the Fast Pipeline runs)
```

The expensive ARIMA and LSTM searches run only during explicitly requested
manual challenger retuning or formal research. The Sunday workflow performs
refresh only and fails if approved configuration metadata is unavailable.

## Project Structure

This directory is one half of the monorepo — see the [repo root README](../README.md) for
`../frontend/`'s structure.

```text
backend/
├── run_pipeline.py            # headless CLI entrypoint for the Fast Pipeline (and local dev)
├── requirements-fast.txt      # deps for the Fast Pipeline (no ML stack)
├── requirements-pipeline.txt  # deps for the Heavy Training pipeline / local full-pipeline dev
├── README.md
├── data/
│   ├── raw/                # <TICKER>.csv — the data the frontend's exported JSON is built from
│   ├── pdf_reports/        # staged PSE EDGE EOD PDFs (gitignored, except bundled samples)
│   └── pdf_pipeline/       # intermediate ETL artifacts + pipeline.log (gitignored)
├── models/
│   ├── lag_regression/     # <TICKER>.pkl
│   ├── arima/              # <TICKER>.pkl
│   └── lstm/                # <TICKER>.pth
├── prediction_cache/        # <TICKER>.json — cached metrics/predictions, read by the export script
├── best_models.json         # {"<TICKER>": "<best model label>"} per ticker, lowest RMSE
├── statistical_tests.json   # cross-model significance tests (DM/HLN, Friedman, Wilcoxon-Holm, consistency check)
├── latest_processed.json    # metadata about the most recent automated pipeline run
├── scripts/
│   └── export_forecast_artifacts.py   # reshapes the above into ../frontend/public/forecasts/*.json
└── services/
    ├── data_loader.py
    ├── data_validator.py
    ├── feature_engineering.py   # lag + technical-indicator + return features, shared by all models
    ├── time_series_cv.py        # shared expanding-window rolling-origin CV splitter
    ├── evaluation.py            # shared RMSE/MAE/MASE/R² metrics + statistical significance suite
    ├── model_selector.py        # trains all 3 models per ticker, saves them, picks the best
    ├── forecasting/
    │   ├── __init__.py
    │   ├── lag_regression.py
    │   ├── arima_model.py
    │   └── lstm_model.py
    └── pdf_pipeline/       # PDF ingestion pipeline (download, parser, cleaner, validator, merge)
        ├── config.py
        ├── downloader.py
        ├── parser.py
        ├── cleaner.py
        ├── validator.py
        ├── merge.py
        └── pipeline.py     # single orchestration layer: ingestion -> training -> metadata
```

The root `.github/workflows/` (not inside `backend/`) runs everything below against these paths —
see the repo root README for exactly how.

## Requirements

- Python 3.10 or newer
- pip

Running the actual dashboard UI is a separate, Node-based step — see
[`../frontend/README.md`](../frontend/README.md). This directory only produces the data it reads.

## Data Format

Each company CSV in `data/raw/` contains: `Date, Open, High, Low, Close, Volume`, named by ticker symbol (e.g. `ALI.csv`, `BPI.csv`, ...). The full list of 15 tracked tickers: ALI, APX, BPI, GLO, ICT, JFC, MBT, MEG, MER, NIKL, PGOLD, SCC, SECB, SHLPH, SMPH.

## Notes

- `SECB.csv` replaces `BDO.csv` throughout the project.
- `services/data_loader.py` maps ticker symbols to the company metadata used in the dashboard.
- If a company CSV or trained model is missing, the app shows a placeholder or an in-app "not processed yet" message instead of failing — see [Architecture](#architecture).

## Automated Pipeline

The pipeline is split across two workflows so the expensive part doesn't
run daily:

| | `.github/workflows/update_pipeline.yml` ("Fast Pipeline") | `.github/workflows/train_models.yml` ("Heavy Training") |
|---|---|---|
| Does | PDF ingestion -> `backend/data/raw/` CSVs only (`python backend/run_pipeline.py --no-train`) | Refits approved configurations (`python -m services.model_selector --mode deployment-refresh --strict`) |
| Schedule | Monday-Friday, 4:00 PM Philippine Time | Sunday, 8:00 AM Philippine Time |
| Trigger | External: [Cron-job.org](https://cron-job.org) `repository_dispatch` (no GitHub-native cron) | GitHub Actions' own `schedule: cron` |
| Dependencies | `backend/requirements-fast.txt` (pandas/numpy/pdfplumber/requests) | `backend/requirements-pipeline.txt` (adds scikit-learn/statsmodels/torch) |
| Typical runtime | A couple of minutes | Refit-dependent; no scheduled ARIMA/LSTM grid search |
| Commits | `backend/data/raw/`, `backend/latest_processed.json`, `frontend/public/forecasts/` | `backend/models/deployment/current/`, `backend/prediction_cache/`, `frontend/public/forecasts/` |

Both share the `pse-pipeline` concurrency group, so they queue instead of
racing each other if a run overlaps. Since PSE doesn't trade weekends,
Heavy Training doesn't re-fetch PDFs itself — by Sunday, `data/raw/` is
already current through Friday's close via the week's Fast Pipeline runs.

### Setting up the Cron-job.org trigger (Fast Pipeline only)

Heavy Training needs no external setup — it's a native GitHub Actions
`schedule: cron` trigger, already configured in `train_models.yml`. Only
the Fast Pipeline needs Cron-job.org:

1. Create a GitHub Personal Access Token with `repo` + `workflow` scope (a fine-grained token scoped to just this repo's contents+actions permissions also works).
2. In Cron-job.org, create a new job with:
   - **Schedule**: Monday–Friday, 16:00 (4:00 PM) — set the job's timezone to `Asia/Manila`.
   - **Request type**: Custom HTTP request (`POST`)
   - **URL**: `https://api.github.com/repos/<OWNER>/<REPO>/dispatches`
   - **Headers**:
     - `Accept: application/vnd.github+json`
     - `Authorization: Bearer <YOUR_GITHUB_PAT>`
     - `X-GitHub-Api-Version: 2022-11-28`
   - **Body**: `{"event_type": "run-pipeline"}`
3. Save. Cron-job.org will now POST to GitHub on that schedule, which fires the `repository_dispatch` trigger and starts the Fast Pipeline workflow — no polling, no GitHub Actions schedule minute-drift.

**Never commit the PAT to this repository.** Store it only in Cron-job.org's own encrypted request-header field.

GitHub Actions' own cron (used by Heavy Training) can be delayed by a few
minutes during periods of high platform load — not a concern for a
weekly, non-latency-sensitive job, which is why it's only used there and
not for the Fast Pipeline's tighter Monday-Friday schedule.

### What each workflow does

**Fast Pipeline** (Monday-Friday):

1. Checks out the repo and installs `backend/requirements-fast.txt`.
2. Runs `python backend/run_pipeline.py --no-train`, which downloads new EOD reports, extracts, cleans, validates, and merges them into `backend/data/raw/`, then writes `backend/latest_processed.json`. No model retraining.
3. Verifies at least one non-empty CSV exists in `backend/data/raw/` — if not, the job fails loudly instead of silently pushing nothing.
4. Runs `python backend/scripts/export_forecast_artifacts.py`, which writes `frontend/public/forecasts/*.json`.
5. Stages `backend/data/raw/`, `backend/latest_processed.json`, and `frontend/public/forecasts/`, then checks `git diff --cached`. If nothing changed (e.g. a market holiday, or the pipeline already ran for that data), the job **finishes successfully without committing** — idempotent, no empty commits ever.
6. If something changed, commits and pushes.
7. Uploads `backend/data/pdf_pipeline/pipeline.log` as a build artifact either way, for troubleshooting.
8. Vercel (Root Directory: `frontend/`) picks up the new commit and redeploys automatically via its Git integration — no separate step needed on this repo's side.

**Heavy Training** (Sunday):

1. Checks out the repo (already current through Friday, via the week's Fast Pipeline commits) and installs `backend/requirements-pipeline.txt`.
2. Runs `python -m services.model_selector --mode deployment-refresh --strict` (from `backend/`), which refits each approved configuration and preserves existing formal metrics and model-family choices.
3. Verifies `backend/best_models.json` and `backend/prediction_cache/` were actually populated.
4. Runs `python backend/scripts/export_forecast_artifacts.py`, which writes `frontend/public/forecasts/*.json`.
5. Stages `backend/models/deployment/current/`, `backend/prediction_cache/`, and `frontend/public/forecasts/`, then checks `git diff --cached` — same idempotency guarantee as the Fast Pipeline.
6. If something changed, commits and pushes; Vercel redeploys automatically.

Both workflows are granted only `contents: write` — nothing else.

### Manual / ops trigger (workflow_dispatch)

For a maintainer testing or backfilling outside the scheduled runs:
**Actions → PSE Fast Data Pipeline → Run workflow** (optional `start_date`/`end_date` inputs, YYYY-MM-DD) or **Actions → PSE Weekly Model Training → Run workflow** (no inputs — trains on whatever `data/raw/` currently has). Both are operator actions taken directly in GitHub, entirely outside the deployed frontend.

### Running it locally

From the repo root:

```bash
pip install -r backend/requirements-fast.txt       # PDF ingestion only
pip install -r backend/requirements-pipeline.txt   # adds the ML stack, for training

python backend/run_pipeline.py                     # fetch new reports, process, train, evaluate, select
python backend/run_pipeline.py --no-download       # only process what's already in backend/data/pdf_reports/
python backend/run_pipeline.py --no-train          # skip retraining (only refresh backend/data/raw/ CSVs)
python backend/run_pipeline.py --start-date 2026-07-01 --end-date 2026-07-27
cd backend && python -m services.model_selector --mode deployment-refresh  # refresh approved models
python backend/scripts/export_forecast_artifacts.py  # refresh frontend/public/forecasts/ from whatever's on disk
```

Exit code `0` means success (including "nothing new to do"); exit code `1` means a real failure — check `backend/data/pdf_pipeline/pipeline.log`.

### Disabling automation

- In GitHub: **Actions → \<workflow name\> → ⋯ → Disable workflow** (do this for each of the two workflows independently).
- Or delete/rename the corresponding file under `.github/workflows/`.
- Independently, pause or delete the Cron-job.org job — that alone stops new Fast Pipeline runs from being triggered, without touching anything in this repo (Heavy Training is unaffected, since it doesn't depend on Cron-job.org).

### Idempotency / duplicate-run protection

Re-running either workflow on data/models it already has is safe and a
no-op at the commit layer: `merge_into_raw()` upserts by date (identical
rows produce an identical file), retraining on unchanged data reproduces
bit-for-bit-equivalent models, and each workflow's `git diff --cached`
check means an unchanged working tree never produces a commit —
including two accidental triggers on the same day.

## About the Forecasting Models

The dashboard compares three forecasting approaches, each predicting
next-day ΔClose and reconstructing a peso price from it (see "Model
Training Pipeline" above for each model's methodology):

- Lag-Informed Regression
- ARIMA
- LSTM

against a naive (yesterday's close) baseline, using:

- RMSE
- MAE
- MASE
- R²

Cross-model significance is assessed with Diebold-Mariano (Newey-West HAC
variance, HLN small-sample correction) within each company, and a
Friedman rank test with Holm-adjusted Wilcoxon signed-rank post-hoc tests
across all companies — see `statistical_tests.json`, written only by an
explicit formal-evaluation run.

## Disclaimer

This dashboard is intended for academic, educational, and analytical decision-support purposes only. It is not financial advice and should not be used as the sole basis for investment decisions.

## License

For academic and internal project use.
