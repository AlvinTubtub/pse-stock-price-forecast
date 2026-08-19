# ForecastPH — Completion Report

## 1. Project Overview

**Project:** ForecastPH — PSE Stock Price Forecast Dashboard  
**Repository:** https://github.com/AlvinTubtub/pse-stock-price-forecast.git  
**Live Dashboard:** https://pse-stock-price-forecast.vercel.app/

ForecastPH is an educational Philippine Stock Exchange (PSE) next-session stock-price forecasting dashboard. It combines a Python-based forecasting and evaluation pipeline with a Next.js frontend deployed through Vercel.

The system tracks 15 PSE-listed companies across five sectors and compares:

- Lag-Informed Regression
- ARIMA
- LSTM
- Naive previous-close baseline

The forecasting target is next-session closing price, reconstructed from predicted next-session price change (ΔClose).

---

# 2. Final System Architecture

```text
                         PSE EOD DISCLOSURES
                                │
                                ▼
                    ┌────────────────────────┐
                    │    Python Backend      │
                    │                        │
                    │ PDF Download           │
                    │ PDF Extraction          │
                    │ Data Cleaning           │
                    │ Data Validation         │
                    │ OHLCV Updates           │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   Model Training       │
                    │                        │
                    │ Lag Regression          │
                    │ ARIMA                  │
                    │ LSTM                   │
                    │ Naive Baseline         │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   Model Evaluation     │
                    │                        │
                    │ RMSE / MAE / MASE / R²│
                    │ Statistical Testing    │
                    │ Best Model Selection   │
                    └───────────┬────────────┘
                                │
                                ▼
                ┌────────────────────────────────┐
                │ Forecast Artifact Export        │
                │ frontend/public/forecasts/     │
                └───────────────┬────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │     Next.js Frontend   │
                    │                        │
                    │ Home Dashboard         │
                    │ Company List           │
                    │ Company Details        │
                    │ Historical OHLCV        │
                    │ Next-Day Prediction    │
                    │ Backtest                │
                    │ Forecast Error         │
                    │ Model Performance       │
                    │ Learn / About           │
                    │ AI Assistant            │
                    └───────────┬────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
             Vercel                         Gemini API
                                              │
                                              ▼
                                      PSE Forecast Assistant
3. Forecasting Models

The project compares the following forecasting approaches:

Lag-Informed Regression

Uses engineered lagged and technical features to model next-session ΔClose.

ARIMA

A statistical time-series model used for next-session forecasting.

LSTM

A recurrent neural-network model for sequential time-series forecasting.

Naive Baseline

Uses the previous closing price as the benchmark forecast.

Forecast reconstruction:

ΔClose(t+1) = Close(t+1) − Close(t)


Predicted Close(t+1)
    = Close(t) + Predicted ΔClose(t+1)

The Naive baseline is used as the benchmark for scale-free comparison.

4. Model Evaluation

The forecasting models are evaluated using:

Metric	Purpose
RMSE	Measures overall magnitude of prediction errors, penalizing larger errors more heavily
MAE	Measures average absolute prediction error
MASE	Compares model error against the naive baseline
R²	Measures explained variance on the evaluation/test set

MASE interpretation:

MASE < 1.0  → better than Naive baseline
MASE = 1.0  → approximately equal to Naive baseline
MASE > 1.0  → worse than Naive baseline

R² is treated as a supplementary evaluation metric and is not interpreted as forecast confidence or probability.

The system also performs statistical testing, including:

Friedman rank testing
Holm-adjusted Wilcoxon signed-rank tests
Diebold-Mariano testing
Harvey-Leybourne-Newbold correction
Best-model consistency analysis
5. Supported Companies

The final dashboard tracks 15 PSE-listed companies:

ALI   Ayala Land, Inc.
APX   Apex Mining Co., Inc.
BPI   Bank of the Philippine Islands
GLO   Globe Telecom, Inc.
ICT   Intl. Container Terminal Services
JFC   Jollibee Foods Corporation
MBT   Metropolitan Bank & Trust Co.
MEG   Megaworld Corporation
MER   Manila Electric Company
NIKL  Nickel Asia Corporation
PGOLD Puregold Price Club, Inc.
SCC   Semirara Mining and Power Corp.
SECB  Security Bank Corporation
SHLPH Pilipinas Shell Petroleum Corp.
SMPH  SM Prime Holdings, Inc.

Five represented sectors:

Financials
Industrial
Property
Services
Mining and Oil
6. Historical OHLCV Visualization

The Company Details page includes a stakeholder-approved line-based Historical OHLCV chart.

The chart displays:

Open
High
Low
Close
Volume

It supports:

Historical date-range selection
Start date selection
End date selection
Day/month/year selection
1 Month quick range
3 Months quick range
6 Months quick range
1 Year quick range
Update Chart
Zoom In
Zoom Out
Pan
Box Zoom
Reset
Responsive rendering
Dark Mode / Light Mode

The selected date range controls the base historical dataset, while zoom/pan operate within the selected range.

The chart remains a line graph by stakeholder requirement.

7. Historical Chart Interaction Fix

The interactive chart state management was corrected to prevent the previously observed Zoom In (+) / Zoom Out (-) corruption.

The chart viewport now maintains a consistent start/end range and prevents:

Invalid indexes
Empty chart states
Start index exceeding end index
Ranges exceeding dataset bounds
Incorrect reset behavior

Reset now restores the currently selected historical date range rather than unexpectedly returning to the entire dataset.

8. Next-Day Prediction Chart

A dedicated Next-Day Prediction chart was added below Historical OHLCV.

The chart shows:

Actual Close as one continuous historical line
ARIMA next-session prediction
Lag-Informed Regression next-session prediction
LSTM next-session prediction

The three model forecasts are displayed as separate dashed segments originating from the latest actual close and ending at the next PSE trading session.

The chart does not display historical model predictions across the entire historical series.

9. Backtest Chart

The Backtest: Predicted vs. Actual (Last 60 Sessions) chart was improved to use actual trading-session dates rather than generic Day 1–Day 60 labels.

The chart includes:

Actual
ARIMA
Lag-Informed Regression
LSTM
Naive Baseline

The selected/best model is visually emphasized.

The Naive baseline is visually distinguished from forecasting models.

The chart retains interactive zoom, pan, and reset behavior.

10. Forecast Error Chart

The Forecast Error Over Time chart uses:

Forecast Error = Predicted Price − Actual Price

Interpretation:

Positive error → overprediction
Negative error → underprediction
Zero           → perfect prediction

The chart includes a clear zero reference line and uses actual backtest trading-session dates.

The selected model is visually emphasized and the Naive baseline is visually distinguished.

11. Philippine Trading Calendar

The system now contains a centralized PSE trading calendar:

backend/services/pse_calendar.py

The calendar handles:

Weekends
Philippine holidays
Special non-working holidays
PSE non-trading dates represented by the repository

The system specifically recognizes:

Friday, August 21, 2026
Ninoy Aquino Day
PSE CLOSED

The next valid trading session is:

Monday, August 24, 2026

The calendar is used as the single source of truth for next-session determination.

12. Automated Holiday Protection

A dedicated calendar check was added:

backend/scripts/check_pse_trading_day.py

The GitHub Actions Fast Pipeline now checks the PSE calendar before performing the normal data update.

On a PSE holiday:

PSE CLOSED
    ↓
Skip data ingestion
    ↓
Skip daily inference
    ↓
Skip forecast artifact export
    ↓
Skip validation
    ↓
Skip generated-data commit
    ↓
Exit successfully

The holiday condition is treated as a normal operational state rather than a pipeline failure.

Example:

PSE Trading Calendar Check
Date: 2026-08-21
PSE Status: CLOSED
Reason: Ninoy Aquino Day
Action: SKIP daily trading-data update
Next PSE Trading Session: 2026-08-24
13. Forecast-Date Fallback Protection

The forecast artifact exporter was corrected so that its fallback date logic uses the PSE calendar.

The system no longer relies on:

data_as_of + 1 calendar day

when determining the next forecast date.

Instead it uses the next valid PSE trading session.

This prevents the system from incorrectly targeting:

weekends
Philippine holidays
other configured PSE non-trading dates
14. Philippine Time (PHT)

All user-facing operational timestamps use Philippine Time:

Asia/Manila
UTC+8

The dashboard displays:

PHT

instead of UTC for user-facing timestamps.

Forecast dates are handled as trading dates and are not incorrectly shifted by timezone conversion.

15. Home Dashboard Corrections

The Home page was updated to present the project accurately.

The hero section uses:

Cross-Sector Next-Day Stock Price Forecasting

with the description:

Explore historical Philippine stock market data, compare machine learning and statistical models, and understand next-day price prediction techniques.

The following hero actions were removed:

Explore Forecasts
Upload & Predict
Learn About Forecasting

The previous Pipeline Status section was replaced with:

Data Source


Official PSE Daily Quotations Reports

The page also includes an educational forecast disclaimer.

Forecast summaries are presented as model-generated estimates and are not presented as investment recommendations.

16. Dark Mode / Light Mode

A global Dark Mode / Light Mode theme toggle was added to the navigation/header.

Features:

Dark Mode
Light Mode
Theme persistence
Responsive behavior
Keyboard accessibility
Theme-aware charts and controls

The existing dark dashboard appearance is preserved as the primary visual design.

17. AI Chatbot — PSE Forecast Assistant

A floating AI chatbot was added to the lower-right of the dashboard:

PSE Forecast Assistant

The assistant is designed for educational explanations of:

Forecasts
ARIMA
LSTM
Lag-Informed Regression
Naive baseline
RMSE
MAE
MASE
R²
Backtest charts
Forecast Error charts
Dashboard data

The assistant is context-aware.

Company context

On:

/companies/[symbol]

the assistant can use company-specific generated forecast artifacts.

Market overview context

The assistant can use:

dashboard.json
companies.json
latest.json
Model Performance context

The assistant can use:

metrics.json
per-model metrics
median metrics
win rates
Naive comparisons
statistical test results
18. Gemini Integration

The chatbot uses the Google Gemini API through the official:

@google/genai

SDK.

Server-side integration:

frontend/src/app/api/chat/route.ts

The API key is stored as:

GEMINI_API_KEY

The key is intentionally server-side only.

The project does not use:

NEXT_PUBLIC_GEMINI_API_KEY

The local development environment uses:

frontend/.env.local

while:

frontend/.env.example

documents the required variable without storing the real secret.

19. Gemini Model Resilience

The chatbot uses:

Primary:
gemini-3.5-flash-lite


Fallback:
gemini-3.5-flash

Resilience behavior:

Primary request
      ↓
Transient 503 / UNAVAILABLE?
      ↓
Retry primary once
      ↓
Still unavailable?
      ↓
Fallback model
      ↓
User response

The system also handles unavailable-model errors without exposing raw Gemini errors to users.

Sampling parameters deprecated by the Gemini 3.5 model family were removed.

Final user-facing failure behavior uses a safe message such as:

Gemini is temporarily busy. Please try again in a moment.
20. AI Safety and Educational Guardrails

The PSE Forecast Assistant is explicitly educational.

It must not:

Recommend buying a stock
Recommend selling a stock
Recommend holding a stock
Provide personalized investment advice
Claim that a forecast is guaranteed
Claim that a prediction is certain
Invent prices, metrics, or forecasts

The assistant uses supplied dashboard JSON artifacts as its source of truth for company-specific information.

It also correctly explains:

MASE < 1.0 → better than Naive
MASE = 1.0 → approximately equal to Naive
MASE > 1.0 → worse than Naive

R² is not presented as confidence or probability.

21. Automated Data Pipeline

The Fast Pipeline is responsible for daily market-data updates and inference.

Current operational flow:

PSE EOD Reports
      ↓
Download
      ↓
Extract
      ↓
Clean
      ↓
Validate
      ↓
Update OHLCV
      ↓
Daily inference
      ↓
Export JSON artifacts
      ↓
Validate exports
      ↓
Commit changed artifacts
      ↓
Vercel redeployment

The Fast Pipeline does not retrain models.

Weekly model training remains a separate workflow.

22. Automated Schedules
Daily Inference
Monday:
5:30 PM Philippine Time


Tuesday–Friday:
4:00 PM Philippine Time

The daily inference uses persisted weekly models and generates next-session forecasts from the latest validated OHLCV data.

Weekly Training
Sunday:
8:00 AM Philippine Time

The weekly training workflow retrains:

Lag-Informed Regression
ARIMA
LSTM

and refreshes model evaluation and model-selection artifacts.

23. Data Integrity

The pipeline includes safeguards for:

Duplicate data
OHLCV validation
Empty data
Artifact validation
Date-based updates
No-op runs
PSE holiday handling
Forecast-date validation
Generated artifact integrity

Generated artifacts are committed only when actual changes exist.

24. Frontend Data Architecture

The frontend is read-oriented.

It consumes generated artifacts under:

frontend/public/forecasts/

Including:

dashboard.json
latest.json
metrics.json
companies.json
company/<SYMBOL>.json
history/<SYMBOL>.json

The frontend does not:

Train models
Run Python
Perform inference
Access a database
Process raw PSE PDFs
Modify forecast artifacts

Generated data is produced by the backend pipeline and exported to the frontend data contract.

25. Security

Security measures include:

API keys stored as environment variables
Gemini API key restricted to server-side code
No NEXT_PUBLIC_* Gemini key
.env.local ignored by Git
.env.example provided for setup documentation
Sensitive environment variables configured in Vercel
AI responses restricted by educational guardrails
No investment recommendations

Secrets must never be committed to GitHub.

26. Validation Results

The following validation has been completed during development:

Backend
PSE Calendar tests:
21 tests passed


Full backend test suite:
111 tests passed
0 failures
0 errors
Frontend
npx tsc --noEmit
0 errors
Production build
npm run build
Successful

The production frontend build includes the dynamic:

/api/chat

route.

27. Final Technology Stack
Backend
Python
Pandas
NumPy
scikit-learn
Statsmodels
TensorFlow/Keras
Joblib
Custom PSE PDF/data pipeline
Frontend
Next.js 14
React
TypeScript
Tailwind CSS
Recharts
Responsive UI
AI
Google Gemini API
@google/genai
Gemini 3.5 Flash-Lite
Gemini 3.5 Flash fallback
Automation
GitHub Actions
Cron-job.org trigger for weekday pipeline
Vercel deployment
PSE trading-calendar guard
28. Deployment

The frontend is deployed to:

https://pse-stock-price-forecast.vercel.app/

Vercel configuration:

Framework:
Next.js


Root Directory:
frontend/


Environment Variable:
GEMINI_API_KEY

The Gemini API key is configured as a Vercel sensitive environment variable for the required deployment environments.

29. Current Production Capabilities

The completed ForecastPH system now provides:

Automated PSE data processing
PSE holiday-aware data updates
Automated daily inference
Weekly model training
Cross-model evaluation
Statistical significance testing
Interactive historical OHLCV exploration
Historical date-range selection
Working Zoom In / Zoom Out
Pan / Box Zoom / Reset
Next-Day Prediction visualization
Backtest visualization
Forecast Error visualization
Model Performance comparison
Dark Mode / Light Mode
Philippine Time display
Educational forecast disclaimer
PSE Forecast Assistant AI chatbot
Gemini retry/fallback resilience
Vercel production deployment
30. Limitations

The system remains an educational forecasting application.

Forecasts are generated from historical data and machine-learning/statistical models and do not guarantee future market outcomes.

The AI assistant is intended for educational explanations and dashboard interpretation and is not a financial-advisory system.

The PSE calendar is based on the holidays currently represented in the repository's maintained trading calendar and should be updated when official future PSE non-trading schedules change.

Gemini API usage is subject to Google's active model and project rate limits.

31. Final Status

ForecastPH is operational and production-ready for its intended educational and academic use case.

Core components validated:

✅ PSE data pipeline
✅ PSE trading calendar
✅ Holiday-aware automation
✅ Daily inference
✅ Weekly model training
✅ Forecast generation
✅ Model evaluation
✅ Statistical testing
✅ Historical OHLCV chart
✅ Interactive chart controls
✅ Next-Day Prediction chart
✅ Backtest chart
✅ Forecast Error chart
✅ Model Performance page
✅ Dark / Light Mode
✅ Philippine Time
✅ Gemini AI Assistant
✅ Gemini fallback handling
✅ Vercel deployment
✅ TypeScript validation
✅ Backend test suite
✅ Production build

Repository:
https://github.com/AlvinTubtub/pse-stock-price-forecast.git

Live Dashboard:
https://pse-stock-price-forecast.vercel.app/
