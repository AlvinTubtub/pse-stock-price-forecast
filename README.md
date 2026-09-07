# ForecastPH

**Cross-Sector Next-Day Stock Price Forecasting of Selected PSE-Listed Companies Using Lag-Informed Regression, ARIMA, and LSTM**

ForecastPH is a full-stack stock forecasting and research platform for selected companies listed on the Philippine Stock Exchange (PSE). It compares three forecasting approaches—**Lag-Informed Regression**, **ARIMA**, and **LSTM**—using a leakage-controlled, chronological, out-of-sample evaluation methodology.

The system provides next-trading-session forecasts, model evaluation metrics, interactive historical charts, 60-session backtests, forecast-error visualization, model comparison, an AI-assisted explanation layer, and an administrative management interface.

The repository contains both the **forecasting backend** and the **Next.js frontend** used for the deployed ForecastPH dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Objective](#research-objective)
- [Tracked Companies](#tracked-companies)
- [Forecasting Models](#forecasting-models)
- [Corrected Research Methodology](#corrected-research-methodology)
- [Model Evaluation](#model-evaluation)
- [Statistical Testing](#statistical-testing)
- [Backtest Methodology](#backtest-methodology)
- [Formal Research vs Deployment](#formal-research-vs-deployment)
- [System Architecture](#system-architecture)
- [Frontend Features](#frontend-features)
- [Admin Features](#admin-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Backend](#running-the-backend)
- [Running the Frontend](#running-the-frontend)
- [Testing](#testing)
- [Forecast Artifact Generation](#forecast-artifact-generation)
- [Deployment Workflow](#deployment-workflow)
- [Scheduled Model Lifecycle](#scheduled-model-lifecycle)
- [Vercel Deployment](#vercel-deployment)
- [Important Methodology Notes](#important-methodology-notes)
- [Research Interpretation](#research-interpretation)
- [Technology Stack](#technology-stack)
- [Disclaimer](#disclaimer)

---

## Project Overview

ForecastPH is designed to support beginner and intermediate users who want to understand how different forecasting models behave across multiple sectors of the Philippine stock market.

The system separates computationally intensive forecasting from frontend delivery:

```text
PSE Market Data
      │
      ▼
Data Processing / Feature Engineering
      │
      ├── Lag-Informed Regression
      ├── ARIMA
      └── LSTM
      │
      ▼
Evaluation / Model Selection
      │
      ▼
Deployment Forecast Artifacts
      │
      ▼
frontend/public/forecasts/
      │
      ▼
Next.js Dashboard
      │
      ▼
Vercel
```

The frontend primarily consumes generated forecast artifacts rather than executing model training inside Vercel.

---

## Research Objective

The project evaluates whether different forecasting approaches provide different levels of next-day closing-price forecasting accuracy across selected PSE-listed companies and sectors.

The three principal forecasting models are:

1. **Lag-Informed Regression**
2. **ARIMA**
3. **Long Short-Term Memory (LSTM)**

A **Naive Baseline** is also included as a benchmark.

The primary forecasting target is:

```text
Next trading session closing price
```

The project does not assume that one model must be superior for every company.

---

## Tracked Companies

ForecastPH currently covers **15 PSE-listed companies across five sectors**.

| Sector | Companies |
|---|---|
| Financials | BPI, MBT, SECB |
| Industrial | MER, JFC, SHLPH |
| Property | MEG, ALI, SMPH |
| Services | GLO, PGOLD, ICT |
| Mining & Oil | APX, NIKL, SCC |

Total:

```text
15 companies
5 sectors
3 principal forecasting models
1 naive benchmark
```

---

## Forecasting Models

### Lag-Informed Regression

The regression pipeline uses lag-based features derived from historical observations.

The corrected methodology includes:

- chronological training and validation;
- PACF applied to training-period daily returns;
- fold-specific PACF during rolling validation;
- feature selection without future leakage;
- `StandardScaler` fitted independently inside each rolling fold;
- no future information used in preprocessing;
- genuine out-of-sample prediction generation.

The model is designed to provide a comparatively interpretable statistical forecasting approach.

---

### ARIMA

ForecastPH uses AutoRegressive Integrated Moving Average models for univariate time-series forecasting.

The corrected ARIMA methodology includes:

- bounded ARIMA candidate search;
- chronological expanding-window validation;
- candidate completeness across validation folds;
- finite prediction validation;
- ADF-supported differencing prioritization;
- hold-out evaluation on unseen observations;
- convergence metadata recording;
- Ljung-Box diagnostics on formal hold-out forecast errors.

For walk-forward forecasting, the deployed implementation uses:

```python
append(actual, refit=False)
```

This updates the ARIMA model state using newly observed values while preserving the estimated coefficients.

Therefore, the methodology should be described as:

> One-step-ahead walk-forward forecasting with fixed estimated ARIMA parameters. At each forecast origin, the newly observed value updates the model state without re-estimating the ARIMA coefficients.

---

### LSTM

The formal LSTM implementation predicts **daily changes in closing price**, rather than directly fitting raw closing-price levels.

Key methodology:

- univariate `ΔClose` target;
- Min-Max scaling fitted using training data only;
- five-fold expanding-window cross-validation;
- fold-specific scaling;
- early stopping based only on an internal tail of the fold-training data;
- validation folds remain untouched by early stopping;
- original-price reconstruction before evaluation;
- hyperparameter selection using mean reconstructed-price RMSE;
- fresh final model refit after hyperparameter selection;
- frozen hold-out evaluation.

Hyperparameter grid:

| Parameter | Values |
|---|---|
| Lookback | 5, 10, 20, 30 |
| Hidden units | 25, 50, 100 |
| Learning rate | 0.01, 0.001 |
| Batch size | 16, 32 |

Total configurations:

```text
4 × 3 × 2 × 2 = 48
```

With five validation folds:

```text
48 × 5 = 240 cross-validation fits per company
```

A fresh final LSTM is then fitted after hyperparameter selection.

Maximum epochs:

```text
200
```

Early-stopping patience:

```text
10
```

Random seed:

```text
42
```

---

## Corrected Research Methodology

ForecastPH separates:

```text
development data
```

from:

```text
formal hold-out data
```

using chronological splitting.

The formal methodology uses approximately:

```text
85% development
15% hold-out
```

No random train/test shuffling is used.

The final audited formal evaluation uses **target-date alignment**, ensuring that all forecasting models are evaluated against the same observations.

For each company:

```text
Lag-Informed Regression
ARIMA
LSTM
Naive Baseline
```

must share the same formal hold-out target dates.

This prevents misleading comparisons caused by evaluating models on different subsets of the test period.

---

## Model Evaluation

ForecastPH reports the following forecasting metrics.

### RMSE

Root Mean Squared Error:

```text
Lower = better
```

RMSE gives greater weight to relatively large forecast errors.

---

### MAE

Mean Absolute Error:

```text
Lower = better
```

MAE represents the average absolute difference between predicted and actual closing prices.

---

### MASE

Mean Absolute Scaled Error compares model forecast error with a naive benchmark.

Interpretation:

```text
MASE < 1  → better than naive benchmark
MASE = 1  → approximately equal to naive
MASE > 1  → worse than naive benchmark
```

A common development-period naive MAE denominator is used per company so models remain comparable.

---

### R²

R² is provided as a supplementary goodness-of-fit measure.

It must **not** be interpreted as forecast confidence or prediction probability.

---

## Statistical Testing

ForecastPH includes formal statistical testing in addition to descriptive error metrics.

### Level 1: Model vs Naive Baseline

Each principal model is tested against the naive benchmark.

Primary loss:

```text
Squared forecast error
```

Robustness loss:

```text
Absolute forecast error
```

Testing includes:

- Diebold-Mariano testing;
- Newey-West HAC variance;
- Harvey-Leybourne-Newbold correction;
- Holm multiple-comparison correction.

Only models that significantly outperform the Naive Baseline in the predefined favorable direction can proceed to the principal pairwise comparison stage.

---

### Level 2: Principal Model Comparison

Pairwise principal-model comparisons are performed only when at least two models pass the Naive gate for a company.

This prevents declaring one sophisticated model superior to another when neither can establish evidence of improvement over the baseline.

---

### Moving-Block Bootstrap

ForecastPH also applies moving-block bootstrap procedures to preserve temporal dependence.

Configuration:

```text
Bootstrap repetitions: 5000
Random seed: 42
```

---

### Across-Company Comparison

Unrounded company-level MASE values are used for overall model comparison.

Models included:

```text
Lag-Informed Regression
ARIMA
LSTM
Naive Baseline
```

The analysis uses:

1. Friedman rank test;
2. fixed-seed permutation robustness testing;
3. Wilcoxon post-hoc testing only when the overall test is significant;
4. Holm correction for multiple comparisons.

This avoids unnecessary post-hoc inference when the global null hypothesis is not rejected.

---

## Backtest Methodology

The dashboard contains:

### Backtest: Predicted vs. Actual — Last 60 Sessions

The chart must display genuine historical out-of-sample forecasts rather than reconstructed or artificially shifted values.

Correct data flow:

```text
chronological historical observations
        │
        ▼
genuine OOS / walk-forward forecasts
        │
        ▼
target-date alignment
        │
        ▼
common sessions across models
        │
        ▼
latest 60 common trading sessions
        │
        ▼
frontend JSON
        │
        ▼
Vercel chart
```

The frontend consumes:

```text
backtestDates
backtestActual
backtestByModel
```

from:

```text
frontend/public/forecasts/company/<SYMBOL>.json
```

These fields should be generated automatically by the backend artifact exporter.

They should **not be maintained through manual JSON editing**.

---

### Forecast Error Over Time

Forecast Error Over Time must use the exact same aligned observations used by the 60-session backtest.

The error definition is:

```text
Forecast Error = Actual Close - Predicted Close
```

Therefore:

```text
positive error → model predicted below actual
negative error → model predicted above actual
zero error     → exact prediction
```

Both charts must share:

- identical dates;
- identical actual closing prices;
- identical model prediction arrays;
- identical out-of-sample methodology.

---

## Formal Research vs Deployment

ForecastPH deliberately separates immutable research evidence from operational forecasting.

### Formal Research Artifacts

Formal research runs are stored under:

```text
backend/results/formal/<RUN_ID>/
```

Example structure:

```text
backend/results/formal/<RUN_ID>/
├── split_manifest.json
├── methodology_manifest.json
├── data_manifest.json
├── statistical_tests.json
├── finalized.json
└── per_company/
    └── <SYMBOL>/
        ├── holdout_predictions.csv
        ├── metrics.json
        └── diagnostics.json
```

Formal runs are intended to be:

```text
immutable
reproducible
auditable
```

Once finalized, they should not be modified by normal deployment operations.

Finalization also requires complete tuning and policy evidence: all 48 LSTM
configurations across five folds and three seeds, the complete expanded LASSO
alpha grid with a non-boundary winner, and the raw-Close corporate-action
policy. Valid observations remain in the primary analysis. Verified event
dates, when available, are flagged by forecast target date and used only in a
separate sensitivity analysis; observations are never removed merely because
the model error or price movement is large.

---

### Deployment Artifacts

Operational models are separate from formal research artifacts.

Expected layout:

```text
backend/models/deployment/current/
├── lag_regression/
├── arima/
├── lstm/
└── deployment_manifest.json
```

Deployment models may be retrained according to the production schedule.

Formal artifacts are evidence for the research study.

Deployment artifacts are used for live forecasts.

These two concerns must not be mixed.

---

## System Architecture

```text
                     ┌─────────────────────┐
                     │   PSE Market Data   │
                     └─────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Ingestion       │
                    │ & Preprocessing      │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      ┌────────────┐    ┌────────────┐    ┌────────────┐
      │ Lag-Reg    │    │   ARIMA    │    │    LSTM    │
      └──────┬─────┘    └──────┬─────┘    └──────┬─────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Evaluation           │
                    │ RMSE / MAE / MASE    │
                    │ R² / Statistical     │
                    │ Testing              │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Deployment Artifacts │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ JSON Export          │
                    └──────────┬───────────┘
                               │
                               ▼
               frontend/public/forecasts/
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Next.js Frontend     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Vercel         │
                    └──────────────────────┘
```

---

## Frontend Features

ForecastPH includes:

- company overview pages;
- sector pages;
- model-performance comparison;
- historical OHLCV visualization;
- next-day prediction chart;
- Backtest: Predicted vs. Actual;
- Forecast Error Over Time;
- latest 60-session visualization;
- Naive Baseline comparison;
- RMSE, MAE, MASE, and R²;
- selected-model indicators;
- interactive chart controls;
- dark and light themes;
- responsive dashboard layout;
- beginner-oriented explanations;
- Learn Stocks page;
- About page;
- AI assistant;
- starter questions.

Chart interactions include:

```text
+             Zoom In
−             Zoom Out
Box Zoom      Select chart range
Pan           Move horizontally
Reset         Restore full view
Mouse Wheel   Zoom
Mouse Drag    Pan
Double Click  Reset view
```

---

## Admin Features

The ForecastPH administrative interface provides management functionality separate from the public dashboard.

Administrative areas may include:

```text
/admin
/admin/models
/admin/audit
/admin/calendar
/admin/ai
/admin/pipeline
```

The admin system supports configuration such as:

- site configuration;
- content configuration;
- navigation;
- frontend feature toggles;
- AI assistant configuration;
- pipeline status;
- model information;
- PSE calendar management;
- health monitoring;
- audit information;
- manual pipeline controls.

Configuration may be persisted in PostgreSQL rather than static frontend configuration files.

---

## Project Structure

A simplified repository structure:

```text
pse-stock-price-forecast/
│
├── backend/
│   ├── data/
│   ├── models/
│   │   └── deployment/
│   │       └── current/
│   ├── results/
│   │   └── formal/
│   ├── scripts/
│   │   └── export_forecast_artifacts.py
│   ├── services/
│   │   ├── forecasting/
│   │   ├── evaluation.py
│   │   └── ...
│   ├── tests/
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   │   └── forecasts/
│   │       └── company/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   └── lib/
│   ├── package.json
│   └── next.config.*
│
├── .github/
│   └── workflows/
│
├── README.md
└── ...
```

Exact internal paths may vary as the project evolves.

---

## Installation

### Prerequisites

Recommended:

```text
Python 3.11
Node.js
npm
Git
```

Clone the repository:

```bash
git clone https://github.com/AlvinTubtub/pse-stock-price-forecast.git
cd pse-stock-price-forecast
```

---

## Running the Backend

Move to the backend:

```bash
cd backend
```

Create a virtual environment:

```bash
python3.11 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Frontend

From the project root:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Run development mode:

```bash
npm run dev
```

Then open:

```text
http://localhost:3000
```

---

## Testing

Backend tests should be executed before deployment or methodology changes are merged.

From:

```text
backend/
```

run:

```bash
.venv/bin/python -m pytest -q
```

or, with the virtual environment activated:

```bash
python -m pytest -q
```

The full automated test suite covers forecasting, artifact handling, statistical evaluation, diagnostics, and methodology-related safeguards.

Also useful:

```bash
python -m compileall -q services scripts
```

From the repository root:

```bash
git diff --check
```

---

## Forecast Artifact Generation

The Vercel frontend does not independently calculate the forecasting results shown in company pages.

Forecast artifacts are generated by the backend and exported to:

```text
frontend/public/forecasts/
```

Company-level output:

```text
frontend/public/forecasts/company/<SYMBOL>.json
```

Typical company JSON fields may include:

```json
{
  "symbol": "BPI",
  "dataAsOf": "YYYY-MM-DD",
  "forecastDate": "YYYY-MM-DD",
  "previousClose": 0,
  "predictedClose": 0,
  "model": "ARIMA",
  "metrics": {},
  "nextClose": {},
  "backtestDates": [],
  "backtestActual": [],
  "backtestByModel": {}
}
```

The exact structure is determined by the exporter.

The export path includes:

```text
backend/scripts/export_forecast_artifacts.py
```

Generated frontend JSON should be treated as a **presentation artifact**, not the authoritative model-training source.

Do not manually edit forecast values as a long-term solution because generated files may be overwritten by subsequent pipeline runs.

---

## Deployment Workflow

The intended live forecasting workflow is:

```text
1. Obtain newest PSE market data
2. Validate and preprocess data
3. Update deployment dataset
4. Use current deployment models or retrain when scheduled
5. Generate next-session predictions
6. Generate deployment backtests
7. Align backtest target dates
8. Select latest 60 common sessions
9. Export frontend JSON
10. Publish updated artifacts
11. Redeploy / refresh Vercel
```

This ensures the dashboard reflects backend-generated results rather than manually modified frontend data.

---

## Scheduled Model Lifecycle

ForecastPH separates model retraining from normal daily forecasting.

### Weekly Model Retraining

Scheduled:

```text
Sunday — 8:00 AM PHT
```

Weekly retraining should:

```text
latest main branch
→ corrected deployment methodology
→ fresh deployment models
→ updated deployment metadata
```

---

### Daily Market Processing / Inference

The forecasting pipeline may run after the relevant PSE trading-session data becomes available.

The production process should use the newest validated market observations and the latest deployment models.

Daily inference should not modify finalized formal research runs.

---

## Vercel Deployment

Vercel serves the Next.js frontend.

A typical update cycle is:

```text
backend pipeline
→ updated frontend-readable artifacts
→ GitHub main
→ Vercel deployment
→ updated public dashboard
```

A Vercel redeployment alone does **not** recalculate forecasts.

If the generated forecast JSON has not changed, a new Vercel build can still display the same forecast values.

Therefore:

```text
corrected source code
≠ automatically corrected live forecast artifacts
```

The full live update requires:

```text
corrected code
+ regenerated deployment results
+ regenerated frontend JSON
+ published artifacts
+ Vercel refresh/deployment
```

---

## Important Methodology Notes

### No Random Train/Test Shuffle

Financial time series must maintain chronological order.

The methodology therefore avoids random train/test splitting.

---

### No Future Leakage

Preprocessing that learns parameters from data must be fitted only using information available at the corresponding forecast origin.

Examples include:

```text
scalers
PACF-based lag selection
feature selection
LSTM normalization
```

---

### No Backfilling From Future Observations

Formal backtests must not use future observations to fill unavailable historical feature values.

Backtests should be genuine historical forecasts.

---

### Common Target Dates

All models being compared must predict the same target sessions.

Model arrays must not simply be truncated to the same length without checking dates.

Correct comparison requires:

```text
date-indexed alignment
```

---

### Naive Baseline

The Naive Baseline assumes:

```text
Tomorrow's close = Today's close
```

It is included as a meaningful forecasting benchmark.

A complex forecasting model should not automatically be considered useful simply because it produces predictions.

---

### Model Selection

The lowest RMSE principal model can be reported descriptively for each company.

However, descriptive RMSE ranking must be distinguished from formal statistical evidence.

A model can have the lowest RMSE without demonstrating statistically significant superiority.

---

### Full Precision

Evaluation calculations should preserve full numeric precision.

Rounding belongs in the frontend presentation layer.

---

### RSI Edge Cases

RSI feature calculation should correctly handle:

```text
gain-only periods → near 100
loss-only periods → 0
flat periods      → neutral 50
```

---

## Research Interpretation

Correcting the methodology improves the **validity, reproducibility, and comparability** of the forecasting experiment.

It can also improve model selection because hyperparameters and models are chosen using more defensible out-of-sample procedures.

However:

> Methodological correction does not guarantee that every corrected forecast will have a lower error than every forecast generated under the previous implementation.

Forecast accuracy remains an empirical result.

The correct interpretation is:

```text
Better methodology
→ more trustworthy evaluation
→ better protection against leakage and optimistic bias
→ more defensible model selection
→ potentially better generalization
```

not:

```text
Better methodology
→ guaranteed perfect or universally superior forecasts
```

Financial markets remain noisy and affected by information unavailable to purely historical forecasting models.

---

## Formal Audit Principles

The audited implementation is designed around the following principles:

```text
✓ chronological evaluation
✓ common hold-out dates
✓ genuine out-of-sample predictions
✓ no array-length truncation as alignment
✓ fold-specific preprocessing
✓ training-only scaling
✓ fold-specific PACF
✓ proper LSTM validation separation
✓ fresh LSTM final refit
✓ original-scale model selection
✓ strict ARIMA candidate validation
✓ ARIMA convergence recording
✓ date-indexed Naive baseline
✓ Naive-first statistical gatekeeping
✓ DM testing with HAC / HLN correction
✓ Holm multiple-testing correction
✓ moving-block bootstrap
✓ Friedman overall comparison
✓ conditional Wilcoxon post-hoc testing
✓ unrounded statistical calculations
✓ residual diagnostics
✓ formal/deployment artifact separation
✓ immutable finalized research runs
✓ reproducible metadata and manifests
```

---

## Reproducibility

Formal evaluation runs should contain enough information to identify:

- dataset version;
- data cutoff;
- company universe;
- train/hold-out split;
- model configuration;
- dependencies;
- source-code commit;
- statistical settings;
- generated artifacts;
- artifact SHA-256 hashes.

A finalized formal run should therefore represent a reproducible research snapshot rather than a continuously changing deployment output.

---

## Technology Stack

### Backend

```text
Python
Pandas
NumPy
scikit-learn
statsmodels
PyTorch
SciPy
```

### Forecasting

```text
Lag-Informed Regression
ARIMA
LSTM
Naive Baseline
```

### Frontend

```text
Next.js
React
TypeScript
Tailwind CSS
Recharts
```

### Infrastructure

```text
GitHub
GitHub Actions
Vercel
PostgreSQL / Neon
```

---

## Research Scope

ForecastPH is designed for next-trading-session closing-price forecasting using historical market data.

The project does not attempt to predict every possible market-moving event.

Unexpected factors may include:

- macroeconomic announcements;
- company disclosures;
- earnings surprises;
- political developments;
- geopolitical shocks;
- regulatory actions;
- natural disasters;
- changes in investor sentiment;
- abnormal liquidity;
- extraordinary corporate events.

These limitations should be considered when interpreting any model forecast.

---

## Disclaimer

ForecastPH is an academic and educational forecasting system.

The forecasts, model rankings, backtests, statistical results, and dashboard outputs are provided for:

```text
research
education
model comparison
data-analysis demonstration
```

They are **not financial advice**, investment recommendations, trading signals, or guarantees of future market performance.

Stock prices are inherently uncertain, and historical predictive performance does not guarantee future results.

Users should conduct independent research and consult qualified financial professionals before making investment decisions.

---

## Repository

```text
https://github.com/AlvinTubtub/pse-stock-price-forecast
```

---

## Project

**ForecastPH**

> Cross-Sector Next-Day Stock Price Forecasting of Selected PSE-Listed Companies: A Comparative Study of Lag-Informed Regression, ARIMA, and LSTM.
