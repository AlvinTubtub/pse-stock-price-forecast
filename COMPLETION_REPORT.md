# ForecastPH — PSE Stock Price Forecast Dashboard

## Completion Report

**Project:** ForecastPH — PSE Stock Price Forecast Dashboard
**Repository:** `AlvinTubtub/pse-stock-price-forecast`
**Frontend:** Next.js
**Backend:** Python
**Deployment:** Vercel
**Automation:** GitHub Actions + Cron-job.org
**Market:** Philippine Stock Exchange (PSE)
**Project Status:** **IN PROGRESS — Predictive Accuracy Integration and Final Verification**

---

# 1. Project Status

The ForecastPH system has substantial completed functionality across the data pipeline, forecasting models, evaluation framework, historical OHLCV data, interactive charts, dashboard UI, automated workflows, and deployment architecture.

However, the project is **not yet considered fully completed**.

The remaining completion requirement is the final integration and verification of the **predictive-accuracy evaluation system** with the production forecasting codebase.

The predictive-accuracy work includes a dedicated evaluation suite designed to verify:

* Model computations
* Training workflows
* Algorithmic accuracy
* Data preprocessing
* Feature engineering
* Evaluation metrics
* Final unseen-test performance
* Cross-model statistical comparisons
* Reproducibility of model results
* Correctness of the forecasting workflow

The predictive-accuracy suite has already been implemented and verified with:

**63/63 tests passing**

The final project status therefore remains **IN PROGRESS** until the predictive-accuracy implementation is fully integrated and confirmed against the final `pse-stock-price-forecast` codebase.

---

# 2. Completed System Components

The following major components have been implemented.

### Data Pipeline

* PSE end-of-day data acquisition
* Market-data extraction
* Data cleaning
* Data validation
* Historical OHLCV updates
* Duplicate prevention
* Pipeline metadata
* JSON artifact generation

### Forecasting

* Lag-Informed Regression
* ARIMA
* LSTM
* Naive previous-close baseline
* Next-session closing-price forecasting
* Backtesting

### Model Evaluation

* RMSE
* MAE
* MASE
* R²
* Cross-model comparison
* Statistical significance testing
* Best-model selection

### Dashboard

* Main dashboard
* Company list
* Company detail pages
* Interactive historical stock charts
* Full historical OHLCV visualization
* Forecast visualization
* Backtesting visualization
* Model-performance dashboard
* Responsive UI
* Pipeline-status information

### Automation

* Fast Pipeline
* Heavy Training Pipeline
* GitHub Actions
* Cron-based execution
* Generated-artifact commits
* Vercel deployment

These components are functional project infrastructure, but final project completion remains dependent on predictive-accuracy verification.

---

# 3. Predictive Accuracy Validation

## 3.1 Purpose

The predictive-accuracy system was created to independently verify that the forecasting system produces technically correct and statistically defensible results.

The purpose is not simply to confirm that the dashboard displays forecasts.

It verifies the complete computational path:

```text
Raw Data
    ↓
Data Loading
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Training
    ↓
Prediction
    ↓
Evaluation
    ↓
Statistical Testing
    ↓
Predictive Accuracy
```

This provides an additional validation layer around the production forecasting implementation.

---

# 4. Predictive Accuracy Test Suite

The predictive-accuracy evaluation suite is located under:

```text
backend/tests/predictive_accuracy/
```

The suite was designed to reuse the existing project implementation rather than creating an independent replacement forecasting system.

The evaluation suite reuses the existing:

* Data loaders
* Data-processing logic
* Feature-engineering implementation
* Model-training code
* Forecasting models
* Evaluation functionality

This is important because the objective is to test the actual forecasting implementation used by the project.

---

# 5. Predictive Accuracy Test Coverage

The predictive-accuracy suite covers the major computational stages of the forecasting system.

### Data Processing

Validation includes:

* Input-data loading
* Required fields
* Data structure
* Date handling
* Numeric conversion
* Missing-data behavior
* Historical ordering
* Dataset integrity

### Feature Engineering

Validation includes:

* Feature generation
* Feature availability
* Lag construction
* Technical indicators
* Return calculations
* Rolling features
* Feature consistency

### Model Computation

Validation includes:

* Model initialization
* Training behavior
* Prediction behavior
* Output shape
* Numerical validity
* Reproducibility where applicable

### Training Workflow

Validation includes:

* Training-data construction
* Train/test separation
* Feature preparation
* Model fitting
* Prediction generation
* Evaluation workflow

### Evaluation Metrics

Validation includes:

* RMSE
* MAE
* MASE
* R²
* Metric consistency
* Numerical correctness

### Statistical Testing

Validation includes the project's cross-model statistical evaluation procedures.

---

# 6. Final Unseen-Test Evaluation

A critical part of the predictive-accuracy framework is the **final unseen-test evaluation**.

The purpose is to prevent the final accuracy assessment from being based on data that was already used during model development.

The intended evaluation structure is:

```text
Historical Dataset
       │
       ├── Training Data
       │
       ├── Validation / Cross-Validation
       │
       └── Final Unseen Test Data
                    │
                    ▼
             Final Accuracy
```

The final unseen-test period is reserved for evaluating generalization.

This provides a more meaningful estimate of how the trained models perform on previously unseen observations.

---

# 7. Predictive Accuracy Verification Status

The predictive-accuracy suite has reached the following verification state:

```text
Predictive Accuracy Test Suite
        │
        ├── Test implementation       COMPLETE
        ├── Test coverage             COMPLETE
        ├── Existing code reuse       COMPLETE
        ├── Test execution            COMPLETE
        └── Tests passing             63 / 63
```

### Current Result

**63/63 predictive-accuracy tests passing**

This establishes that the implemented validation suite itself is functioning correctly.

However, passing the validation suite does not by itself mean that the entire ForecastPH project can be marked complete.

The remaining requirement is to ensure that the predictive-accuracy validation is fully integrated with and confirmed against the final production codebase and model outputs.

---

# 8. Predictive Accuracy vs. Model Performance

The project distinguishes between **model-performance reporting** and **predictive-accuracy verification**.

### Model Performance

The dashboard reports metrics such as:

* RMSE
* MAE
* MASE
* R²

These describe model performance on the evaluation datasets.

### Predictive Accuracy Validation

The predictive-accuracy suite verifies that:

* The calculations are implemented correctly.
* The correct datasets are used.
* Data preprocessing is correct.
* Features are generated correctly.
* Models are trained correctly.
* Predictions are generated correctly.
* Metrics are calculated correctly.
* Unseen-test evaluation is performed correctly.
* Cross-model statistical testing is implemented correctly.

Therefore:

```text
Dashboard Metrics
        +
Predictive Accuracy Tests
        +
Final Unseen-Test Evaluation
        =
Complete Model Validation
```

---

# 9. Historical OHLCV Data

Full historical OHLCV support has been implemented.

OHLCV consists of:

| Field  | Description    |
| ------ | -------------- |
| Date   | Trading date   |
| Open   | Opening price  |
| High   | Highest price  |
| Low    | Lowest price   |
| Close  | Closing price  |
| Volume | Trading volume |

Backend historical datasets are stored under:

```text
backend/data/raw/<TICKER>.csv
```

Frontend historical artifacts are generated under:

```text
frontend/public/forecasts/history/<TICKER>.json
```

The historical datasets support both forecasting and interactive visualization.

---

# 10. Interactive Stock Charts

Interactive historical stock charts have been implemented in the frontend.

The charts are designed to allow users to inspect:

* Historical price movement
* OHLC information
* Trading volume
* Long-term price history
* Individual company performance
* Historical time ranges
* Price behavior around forecasting periods

The charts consume generated historical JSON artifacts rather than independently running the forecasting pipeline in the browser.

---

# 11. Forecasting Models

The project evaluates multiple forecasting approaches:

### Lag-Informed Regression

Uses historical and engineered features to estimate next-session closing-price change.

### ARIMA

Provides a statistical time-series forecasting approach.

### LSTM

Provides a neural-network sequence forecasting approach.

### Naive Previous-Close Baseline

Provides a baseline for determining whether more complex models provide meaningful predictive improvement.

---

# 12. Forecast Target

The primary forecasting target is next-session closing-price change:

```text
ΔClose(t+1) = Close(t+1) − Close(t)
```

The forecasted closing price is reconstructed as:

```text
Predicted Close(t+1)
=
Close(t) + Predicted ΔClose(t+1)
```

This formulation is used throughout the forecasting workflow.

---

# 13. Feature Engineering

The forecasting system includes multiple price, return, technical, volatility, and volume features.

Current feature categories include:

* Lagged prices
* Lagged returns
* EMA 10
* EMA 20
* RSI 14
* MACD
* MACD signal
* Bollinger Bands
* Daily return
* Rolling volatility
* High-Low spread
* Open-Close spread
* Rolling return statistics
* High-Low range percentage
* Log volume
* Rolling volume statistics

The predictive-accuracy test suite also validates the feature-engineering workflow.

---

# 14. Model Evaluation

The forecasting models are evaluated using:

| Metric | Purpose                            |
| ------ | ---------------------------------- |
| RMSE   | Average squared-error magnitude    |
| MAE    | Average absolute prediction error  |
| MASE   | Error relative to a naive baseline |
| R²     | Explained variance                 |

The metrics are generated as structured artifacts for use by the dashboard and validation workflows.

---

# 15. Statistical Model Comparison

The project includes cross-model statistical testing.

The evaluation framework includes:

* Diebold-Mariano testing
* Harvey-Leybourne-Newbold correction
* Friedman rank testing
* Holm-adjusted Wilcoxon signed-rank tests
* Best-model consistency analysis

The objective is to determine whether differences between models are statistically meaningful rather than relying only on the lowest numerical error.

---

# 16. Backtesting

Historical backtesting has been implemented.

Backtesting allows the system to compare:

```text
Actual Historical Result
          vs.
Model Prediction
```

The resulting information is used for model-performance analysis and dashboard visualization.

Backtesting is separate from the final unseen-test evaluation.

---

# 17. Frontend

The Next.js frontend provides:

* Dashboard
* Company List
* Company Details
* Interactive historical charts
* OHLCV visualization
* Forecast results
* Backtesting visualization
* Model Performance
* Search and navigation
* Pipeline status
* Educational information
* Responsive UI

The frontend is read-only with respect to the forecasting engine.

It consumes generated JSON artifacts.

---

# 18. Frontend Data Contract

The main generated artifacts are:

```text
frontend/public/forecasts/
├── dashboard.json
├── latest.json
├── metrics.json
├── companies.json
├── company/
│   └── <SYMBOL>.json
└── history/
    └── <SYMBOL>.json
```

### `dashboard.json`

Dashboard-level summary data.

### `latest.json`

Latest pipeline and freshness information.

### `metrics.json`

Model-performance metrics.

### `companies.json`

Tracked-company metadata.

### `company/<SYMBOL>.json`

Company-specific forecasts and evaluation data.

### `history/<SYMBOL>.json`

Full historical OHLCV data.

---

# 19. Automated Fast Pipeline

The Fast Pipeline is responsible for routine market-data updates without model retraining.

Execution path:

```text
PSE EOD Data
    ↓
Download
    ↓
Extraction
    ↓
Cleaning
    ↓
Validation
    ↓
OHLCV Update
    ↓
Artifact Export
    ↓
Git Commit
    ↓
Vercel Deployment
```

The no-training execution mode is:

```bash
python backend/run_pipeline.py --no-train
```

---

# 20. Heavy Training Pipeline

The Heavy Training workflow performs model retraining and comprehensive evaluation.

Execution path:

```text
Latest OHLCV
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Evaluation
    ↓
Statistical Testing
    ↓
Best Model Selection
    ↓
Forecast Generation
    ↓
Artifact Export
    ↓
Git Commit
    ↓
Vercel Deployment
```

This workflow is intentionally separated from the routine Fast Pipeline.

---

# 21. GitHub Actions

The repository contains:

```text
.github/workflows/
├── update_pipeline.yml
└── train_models.yml
```

The workflows automate:

* Market-data updates
* Model training
* Forecast generation
* Statistical evaluation
* Artifact export
* Repository updates

---

# 22. Deployment

The frontend is designed for Vercel deployment.

Primary configuration:

```text
Framework: Next.js
Root Directory: frontend/
```

The deployment flow is:

```text
GitHub
   ↓
Generated Artifacts
   ↓
Vercel Build
   ↓
Next.js Dashboard
```

Live dashboard:

https://pse-stock-price-forecast.vercel.app/

---

# 23. Repository Structure

```text
pse-stock-price-forecast/
│
├── .github/
│   └── workflows/
│       ├── update_pipeline.yml
│       └── train_models.yml
│
├── backend/
│   ├── data/
│   ├── models/
│   ├── prediction_cache/
│   ├── services/
│   ├── scripts/
│   ├── tests/
│   │   └── predictive_accuracy/
│   ├── best_models.json
│   ├── latest_processed.json
│   ├── statistical_tests.json
│   ├── run_pipeline.py
│   ├── requirements-fast.txt
│   └── requirements-pipeline.txt
│
├── frontend/
│   ├── public/
│   │   └── forecasts/
│   │       ├── dashboard.json
│   │       ├── latest.json
│   │       ├── metrics.json
│   │       ├── companies.json
│   │       ├── company/
│   │       └── history/
│   │
│   ├── src/
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.ts
│   └── tsconfig.json
│
├── README.md
├── COMPLETION_REPORT.md
└── .gitignore
```

---

# 24. Verification Status

| Component                                                          | Status            |
| ------------------------------------------------------------------ | ----------------- |
| PSE data ingestion                                                 | Completed         |
| Data cleaning                                                      | Completed         |
| Data validation                                                    | Completed         |
| Historical OHLCV storage                                           | Completed         |
| Historical OHLCV frontend artifacts                                | Completed         |
| Feature engineering                                                | Completed         |
| Lag-Informed Regression                                            | Completed         |
| ARIMA                                                              | Completed         |
| LSTM                                                               | Completed         |
| Naive baseline                                                     | Completed         |
| Forecast generation                                                | Completed         |
| RMSE                                                               | Completed         |
| MAE                                                                | Completed         |
| MASE                                                               | Completed         |
| R²                                                                 | Completed         |
| Backtesting                                                        | Completed         |
| Cross-model statistical testing                                    | Completed         |
| Best-model selection                                               | Completed         |
| Interactive stock charts                                           | Completed         |
| Model Performance dashboard                                        | Completed         |
| UI enhancements                                                    | Completed         |
| Fast Pipeline                                                      | Completed         |
| Heavy Training Pipeline                                            | Completed         |
| GitHub Actions                                                     | Completed         |
| Vercel deployment architecture                                     | Completed         |
| Predictive-accuracy test suite                                     | Completed         |
| Predictive-accuracy test execution                                 | **63/63 passing** |
| Final unseen-test predictive accuracy integration                  | **In Progress**   |
| Final predictive-accuracy confirmation against production codebase | **In Progress**   |
| Overall project completion                                         | **In Progress**   |

---

# 25. Remaining Completion Requirements

The project should not be marked fully complete until the following items are confirmed:

### 1. Predictive Accuracy Integration

Confirm that:

```text
backend/tests/predictive_accuracy/
```

is integrated with the final production repository and tests the exact forecasting implementation used by the application.

### 2. Final Unseen-Test Evaluation

Run the final predictive-accuracy evaluation against the designated unseen-test period.

### 3. Accuracy Results Confirmation

Confirm the final:

* RMSE
* MAE
* MASE
* R²
* Model rankings
* Statistical comparisons

for the final unseen-test data.

### 4. Artifact Synchronization

Ensure that the final validated model-performance and predictive-accuracy results are correctly reflected in the generated frontend artifacts.

### 5. Dashboard Verification

Confirm that the Model Performance page displays the validated results rather than stale or independently calculated values.

### 6. End-to-End Verification

Verify the complete workflow:

```text
Market Data
    ↓
OHLCV
    ↓
Features
    ↓
Training
    ↓
Prediction
    ↓
Predictive Accuracy
    ↓
Statistical Testing
    ↓
JSON Artifacts
    ↓
Dashboard
    ↓
Vercel
```

---

# 26. Definition of Final Completion

ForecastPH should be considered **fully completed** only when all of the following are true:

```text
[✓] Data pipeline operational
[✓] Historical OHLCV operational
[✓] Forecasting models operational
[✓] Model evaluation operational
[✓] Statistical testing operational
[✓] Interactive charts operational
[✓] Dashboard UI operational
[✓] Fast Pipeline operational
[✓] Heavy Training operational
[✓] Vercel deployment operational
[✓] Predictive-accuracy test suite implemented
[✓] Predictive-accuracy suite passes 63/63 tests
[ ] Final unseen-test accuracy confirmed
[ ] Predictive-accuracy results integrated into final production artifacts
[ ] Final Model Performance dashboard verified against validated results
[ ] End-to-end final verification completed
```

Therefore, the correct current project state is:

# **IN PROGRESS**

The predictive-accuracy validation framework is substantially complete and has passed **63/63 tests**, but the overall ForecastPH project should remain open until the final unseen-test accuracy and production integration are confirmed.

---

# 27. Final Architecture

The target completed architecture is:

```text
                  PSE EOD MARKET DATA
                           │
                           ▼
                 ┌───────────────────┐
                 │ Data Pipeline     │
                 │ Cleaning          │
                 │ Validation        │
                 │ OHLCV Updates     │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Feature           │
                 │ Engineering       │
                 └─────────┬─────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │ Forecasting Models         │
              │                            │
              │ Regression                 │
              │ ARIMA                      │
              │ LSTM                       │
              │ Naive Baseline             │
              └─────────────┬──────────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ Model Evaluation           │
              │                            │
              │ RMSE / MAE / MASE / R²    │
              │ Backtesting                │
              │ Statistical Tests          │
              └─────────────┬──────────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ Predictive Accuracy        │
              │ Validation Suite            │
              │                            │
              │ 63/63 Tests Passing        │
              │ Final Unseen Test          │
              └─────────────┬──────────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ JSON Artifact Export       │
              └─────────────┬──────────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ Next.js Dashboard           │
              │                            │
              │ Historical OHLCV            │
              │ Interactive Charts          │
              │ Forecasts                   │
              │ Backtesting                 │
              │ Model Performance           │
              └─────────────┬──────────────┘
                            │
                            ▼
                          Vercel
```

---

# 28. Conclusion

ForecastPH has reached a mature implementation stage with the core forecasting platform, automated pipelines, historical OHLCV data, interactive stock charts, model evaluation, statistical testing, dashboard UI, and deployment architecture implemented.

The dedicated predictive-accuracy validation suite is also implemented and has passed:

**63/63 tests**

However, this result should not be interpreted as final project completion.

The remaining work is specifically focused on validating the final unseen-test predictive accuracy and confirming that the predictive-accuracy results are fully integrated with the production forecasting codebase, generated artifacts, and Model Performance dashboard.

Until those steps are completed, the authoritative project status is:

**IN PROGRESS — Predictive Accuracy Integration and Final Verification**
