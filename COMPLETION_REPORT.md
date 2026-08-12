# ForecastPH — PSE Stock Price Forecast Dashboard

# Completion Report

**Project:** ForecastPH — PSE Stock Price Forecast Dashboard
**Repository:** `AlvinTubtub/pse-stock-price-forecast`
**Frontend:** Next.js
**Backend:** Python
**Deployment:** Vercel
**Automation:** GitHub Actions + Cron-job.org
**Market:** Philippine Stock Exchange (PSE)
**Current Status:** **IN PROGRESS — Final Production Integration and End-to-End Verification**

---

# 1. Project Status

ForecastPH is an integrated PSE stock-data, forecasting, model-evaluation, predictive-accuracy, statistical-analysis, and interactive-dashboard system.

The project currently includes:

* Automated PSE end-of-day data processing
* Historical OHLCV data
* Feature engineering
* Multiple forecasting models
* Model evaluation
* Cross-model statistical testing
* Final unseen-test predictive-accuracy evaluation
* Leakage detection
* Walk-forward final-test prediction
* Predictive-accuracy metrics
* Bootstrap confidence intervals
* Interactive historical stock charts
* Forecast and backtesting visualization
* Model Performance dashboard
* Automated Fast Pipeline
* Automated Heavy Training Pipeline
* GitHub Actions
* Vercel deployment
* A dedicated predictive-accuracy test suite

The predictive-accuracy evaluation has now been expanded substantially and has produced actual final unseen-test results for all 15 supported PSE tickers.

The project is **not yet marked fully complete** because final production integration and end-to-end verification still need to be confirmed.

---

# 2. Latest Repository Update

The latest major predictive-accuracy update is:

**Commit:** `bdb41a9`
**Commit message:** `Update predictive accuracy statistical analysis`

This update added:

* Predictive-accuracy evaluation configuration
* Final unseen-test splitting
* Leakage checks
* Predictive metrics
* Model runners
* Statistical-testing implementation
* Evaluation runner
* Evaluation results
* Model comparison results
* Predictive-accuracy tests
* Predictive-accuracy documentation

The commit contains **22 changed files and 5,029 added lines**.

The predictive-accuracy implementation is located at:

```text
backend/tests/predictive_accuracy/
```

The repository now contains:

```text
backend/tests/predictive_accuracy/
├── README.md
├── __init__.py
├── config.py
├── leakage_checks.py
├── metrics.py
├── run_evaluation.py
├── runners.py
├── splits.py
├── statistical_tests.py
├── results/
│   ├── evaluation_summary.md
│   ├── metrics.csv
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── statistical_tests.json
└── tests/
    ├── __init__.py
    ├── _helpers.py
    ├── test_integration.py
    ├── test_leakage_checks.py
    ├── test_metrics.py
    ├── test_naive_baseline.py
    ├── test_splits.py
    └── test_statistical_tests.py
```

This structure is now present in the repository.

---

# 3. Predictive Accuracy Evaluation

## 3.1 Purpose

The predictive-accuracy suite was created to determine how well each forecasting model generalizes to data that was not available during model training, tuning, scaling, or model selection.

It evaluates:

* Lag-Informed Regression/LASSO
* ARIMA
* LSTM
* Naive previous-close baseline

The suite is additive.

It does not replace the production forecasting implementation.

It reuses the existing:

* Data loading
* Feature engineering
* Forecasting models
* Model-training functions
* Evaluation functions
* Statistical-testing functions

The suite therefore evaluates the actual forecasting implementation used by ForecastPH rather than constructing a separate experimental model.

---

# 4. Final Unseen-Test Methodology

The final predictive-accuracy evaluation uses a chronological split.

```text
┌──────────────────────────────────────────────────────┐
│                  TRAIN + VALIDATION                   │
│                                                      │
│ Used for model training, tuning, scaling and         │
│ model selection                                      │
└──────────────────────────────┬───────────────────────┘
                               │
                               │ strict chronological boundary
                               ▼
┌──────────────────────────────────────────────────────┐
│                    FINAL TEST                         │
│                                                      │
│ Never available during training, tuning, scaling,    │
│ or model selection                                   │
└──────────────────────────────────────────────────────┘
```

The default configuration uses:

**Final-test fraction: 15%**

**Random seed: 42**

The split is configurable and can also use an explicit date range.

The final-test observations are withheld before model training begins.

This prevents the final test period from influencing fitted model parameters.

---

# 5. Frozen-Model Evaluation

After each model is trained, its fitted artifact is treated as frozen.

The evaluation process does not refit the model on final-test observations.

This applies to:

* LASSO coefficients
* Selected features
* Feature scalers
* ARIMA/SARIMAX parameters
* LSTM weights
* LSTM input/output scalers

The final-test process may use realized historical market values that occurred before the prediction date.

This is intentional.

It represents normal walk-forward forecasting.

For example:

```text
Actual Close on Day 1
        ↓
Used to predict Day 2

Actual Close on Day 2
        ↓
Used to predict Day 3

Actual Close on Day 3
        ↓
Used to predict Day 4
```

The model itself remains frozen.

---

# 6. Leakage Controls

The predictive-accuracy framework contains explicit leakage checks.

The following checks are implemented:

| Leakage Check                              | Purpose                                                         |
| ------------------------------------------ | --------------------------------------------------------------- |
| `assert_no_date_overlap`                   | Ensures training/validation and final-test dates do not overlap |
| `assert_fit_input_excludes_dates`          | Ensures final-test dates are not passed into model training     |
| `assert_model_selection_before_final_test` | Ensures model selection occurs before final-test evaluation     |
| `assert_scaler_fit_row_count`              | Ensures scalers were not fitted on final-test observations      |
| `assert_no_future_values_in_features`      | Prevents future observations from entering model input windows  |
| `assert_naive_uses_prior_close_only`       | Ensures the naive model uses only the prior close               |
| `assert_identical_test_dates`              | Ensures models are evaluated on identical final-test dates      |

Each leakage check raises a specific error and aborts the evaluation if the condition fails.

This makes leakage a hard failure rather than a warning.

---

# 7. Supported Predictive-Accuracy Metrics

The final evaluation includes the project's standard forecasting metrics:

* RMSE
* MAE
* MASE
* R²

It also adds predictive-accuracy-specific measures.

### Directional Accuracy

Measures how often the predicted market direction matches the actual direction.

The direction is classified relative to the previous actual closing price.

### Mean Directional Error

Measures systematic directional bias.

The direction values are:

```text
-1 = downward
 0 = flat
+1 = upward
```

A mean directional error near zero indicates less systematic bullish or bearish directional bias.

### Hit Rate vs. Naive

Measures the fraction of final-test observations where a model's absolute error is less than or equal to the naive baseline's error for the same date.

### Prediction Error Statistics

The suite records:

* Mean signed error
* Error standard deviation
* Minimum error
* Maximum error
* Median absolute error

### Confidence Intervals

The suite calculates 95% bootstrap confidence intervals for:

* RMSE
* MAE

The default bootstrap configuration uses:

**2,000 resamples**

Confidence intervals are omitted when there are insufficient final-test observations rather than being fabricated.

These metrics and methodology are documented in the predictive-accuracy suite.

---

# 8. Final Unseen-Test Evaluation Coverage

The current evaluation covers:

**15 PSE tickers**

```text
ALI
APX
BPI
GLO
ICT
JFC
MBT
MEG
MER
NIKL
PGOLD
SCC
SECB
SHLPH
SMPH
```

Every supported model is evaluated against the same final-test period for each ticker.

This provides consistent cross-model comparison.

The generated evaluation summary confirms all 15 tickers were evaluated using the 15% final-test fraction and seed 42.

---

# 9. Final Predictive-Accuracy Results

The latest generated evaluation results show the following aggregate ranking by mean RMSE across the 15 tickers:

| Rank | Model                           | Mean RMSE | Selection Frequency | Mean RMSE Improvement vs. Naive |
| ---: | ------------------------------- | --------: | ------------------: | ------------------------------: |
|    1 | ARIMA                           |    4.3895 |                 40% |                          +0.20% |
|    2 | Naive Baseline                  |    4.4131 |                   — |                               — |
|    3 | Lag-Informed Regression (LASSO) |    7.8471 |                 47% |                         -19.41% |
|    4 | LSTM                            |   14.7503 |                 13% |                         -85.94% |

These are the current final unseen-test aggregate results committed to the repository.

---

# 10. Interpretation of Aggregate Results

The final unseen-test results provide several important findings.

### ARIMA

ARIMA has the lowest aggregate mean RMSE:

**4.3895**

It beats the naive baseline on:

**9 of 15 tickers**

or:

**60% of tickers**

Its mean RMSE improvement versus the naive baseline is:

**+0.20%**

Therefore, ARIMA is the strongest model by aggregate mean RMSE, but its advantage over the naive baseline is small.

### Naive Baseline

The naive previous-close baseline has:

**4.4131 mean RMSE**

This is extremely close to ARIMA.

This is an important result because it establishes a strong benchmark for determining whether model complexity provides meaningful predictive improvement.

### Lag-Informed Regression / LASSO

Lag-Informed Regression has:

**7.8471 mean RMSE**

It is selected as the lowest-RMSE model on:

**7 of 15 tickers**

or:

**46.67%**

However, its aggregate mean RMSE is substantially higher than both ARIMA and the naive baseline.

### LSTM

LSTM has:

**14.7503 mean RMSE**

It is selected as the lowest-RMSE model on:

**2 of 15 tickers**

or:

**13.33%**

Its aggregate RMSE is substantially higher than the other approaches in the current final-test evaluation.

These results are descriptive of the current final unseen-test period and should not be interpreted as a universal ranking of these model classes.

---

# 11. Statistical Significance Analysis

The latest predictive-accuracy update added a complete statistical-analysis layer.

The analysis uses:

* Diebold-Mariano testing
* Harvey-Leybourne-Newbold correction
* Holm correction
* Friedman testing
* Pairwise Wilcoxon signed-rank testing
* Best-model consistency analysis

The statistical tests are calculated from the frozen final-test predictions.

The evaluation reuses the statistical-testing implementations already used by the production model-selection system.

This avoids maintaining separate statistical-testing implementations that could produce inconsistent results.

---

# 12. Friedman Test

For the three tuned forecasting models:

```text
Statistic = 10.1333
p-value   = 0.006303
n         = 15 tickers
```

The result is statistically significant at the 0.05 level.

Therefore, the final-test results provide evidence that the three tuned models do not have identical performance distributions across the evaluated tickers.

When the naive baseline is also included:

```text
Statistic = 15.3446
p-value   = 0.001545
n         = 15 tickers
```

This also indicates a statistically significant difference across the model set.

The values are taken from the committed `statistical_tests.json` result.

---

# 13. Wilcoxon-Holm Post-Hoc Analysis

The pairwise post-hoc results are:

| Comparison              | Holm-adjusted p-value | Result          |
| ----------------------- | --------------------: | --------------- |
| ARIMA vs LSTM           |              0.000916 | Significant     |
| Lag Regression vs LSTM  |              0.030151 | Significant     |
| Lag Regression vs ARIMA |              0.488709 | Not significant |

At the 0.05 level:

* ARIMA differs significantly from LSTM.
* Lag Regression differs significantly from LSTM.
* Lag Regression and ARIMA do not show a statistically significant difference in this post-hoc test.

These results are important because the aggregate RMSE ranking alone could otherwise suggest a stronger ARIMA-vs-LASSO difference than the statistical test supports.

---

# 14. Diebold-Mariano Testing

The evaluation also performs pairwise Diebold-Mariano tests for model errors.

The analysis is performed:

* Per ticker
* For model pairs
* Including the naive baseline
* With Holm correction

The latest statistical-analysis artifact contains the complete pairwise results for all evaluated tickers.

Examples from the current result set show statistically significant ARIMA-vs-LSTM and Lag-Regression-vs-LSTM differences for some tickers, while other model comparisons are not statistically significant.

The results therefore support a ticker-specific interpretation rather than assuming one model universally dominates.

---

# 15. Model Selection Frequency

The final-test best-model counts are:

| Model                   | Tickers Selected | Frequency |
| ----------------------- | ---------------: | --------: |
| Lag-Informed Regression |                7 |    46.67% |
| ARIMA                   |                6 |    40.00% |
| LSTM                    |                2 |    13.33% |

This means Lag-Informed Regression wins the most individual ticker RMSE comparisons, even though ARIMA has the best aggregate mean RMSE.

This distinction is important.

```text
Most ticker-level wins
        ≠
Best aggregate mean RMSE
```

The current results demonstrate why model selection should be reported using both aggregate metrics and per-ticker results.

The committed statistical results confirm the 7/6/2 model-selection distribution.

---

# 16. Best-Model Consistency

The current best-model consistency check reports:

```text
Lag Regression lowest RMSE:
7 / 15 tickers

Required threshold:
8 / 15 tickers

Result:
PASS = False
```

Therefore, no single tuned model currently satisfies the configured dominance threshold.

This is an important validation result.

The system should not claim that one model universally dominates across the complete PSE ticker universe.

---

# 17. Naive Baseline Comparison

The naive baseline is an essential component of the evaluation.

The current final-test results show:

### ARIMA

```text
Tickers beating naive: 9 / 15
Fraction: 60%
Mean RMSE improvement: +0.20%
```

### Lag-Informed Regression

```text
Tickers beating naive: 6 / 15
Fraction: 40%
Mean RMSE improvement: -19.41%
```

### LSTM

```text
Tickers beating naive: 1 / 15
Fraction: 6.67%
Mean RMSE improvement: -85.94%
```

This indicates that ARIMA provides only a modest aggregate advantage over the naive benchmark in the current unseen-test period.

The results also show that the more complex models do not automatically outperform a simple baseline.

---

# 18. Predictive-Accuracy Test Suite

The predictive-accuracy suite contains tests covering:

### Integration

* Existing production forecasting functions
* End-to-end evaluation workflow
* Model execution

### Leakage

* Temporal separation
* Training-data isolation
* Model-selection isolation
* Scaler isolation
* Future-feature prevention
* Naive-baseline correctness
* Identical final-test dates

### Metrics

* RMSE
* MAE
* MASE
* R²
* Directional accuracy
* Mean directional error
* Hit rate
* Error statistics
* Confidence intervals

### Statistical Tests

* Diebold-Mariano
* Friedman
* Wilcoxon
* Holm correction
* Best-model consistency

The repository's predictive-accuracy directory contains dedicated automated tests for these areas.

---

# 19. Predictive-Accuracy Test Result

The predictive-accuracy test suite previously reached:

**63/63 tests passing**

The test suite itself is therefore implemented and passing.

The current evaluation artifacts add a second layer beyond unit/integration testing:

```text
Automated Tests
      +
Final Unseen-Test Evaluation
      +
Leakage Checks
      +
Statistical Analysis
      =
Predictive-Accuracy Validation
```

The 63/63 result confirms the validation code is functioning.

The generated evaluation results provide empirical model-performance evidence on unseen data.

---

# 20. Historical OHLCV Data

Full historical OHLCV data is implemented.

Each ticker contains:

* Date
* Open
* High
* Low
* Close
* Volume

Backend data:

```text
backend/data/raw/<TICKER>.csv
```

Frontend artifacts:

```text
frontend/public/forecasts/history/<TICKER>.json
```

The historical datasets support:

* Forecasting
* Backtesting
* Historical visualization
* Interactive charts
* Predictive-accuracy evaluation

---

# 21. Interactive Stock Charts

The dashboard includes interactive historical stock charts.

The charts support:

* Historical price movement
* OHLC visualization
* Volume visualization
* Time-range exploration
* Hover information
* Company-specific history
* Responsive rendering

The charts consume generated historical artifacts rather than directly running the Python forecasting engine.

---

# 22. Forecasting Models

The production forecasting system currently evaluates:

### Lag-Informed Regression / LASSO

A feature-based forecasting model using lagged and engineered market information.

### ARIMA

A statistical time-series model.

### LSTM

A neural-network sequence model.

### Naive Baseline

A previous-close benchmark:

```text
Tomorrow = Today
```

The naive baseline is included in the predictive-accuracy evaluation to provide a meaningful benchmark for model complexity.

---

# 23. Forecast Target

The production forecasting target is next-session closing-price change:

```text
ΔClose(t+1)
=
Close(t+1) − Close(t)
```

The predicted closing price is reconstructed as:

```text
Predicted Close(t+1)
=
Close(t) + Predicted ΔClose(t+1)
```

This target formulation is shared by the forecasting and evaluation workflows where applicable.

---

# 24. Feature Engineering

The production feature-engineering system includes:

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

The predictive-accuracy suite reuses the production feature-engineering implementation.

---

# 25. Production Model Evaluation

The production forecasting pipeline evaluates models using:

* RMSE
* MAE
* MASE
* R²

The predictive-accuracy suite additionally evaluates:

* Directional accuracy
* Mean directional error
* Hit rate vs. naive
* Prediction-error statistics
* Bootstrap confidence intervals

This separates normal model-performance reporting from the more comprehensive final unseen-test validation.

---

# 26. Backtesting

Historical backtesting remains part of the production forecasting workflow.

Backtesting compares:

```text
Historical Actual
       vs.
Model Prediction
```

Backtesting should not be confused with the final unseen-test evaluation.

The final unseen-test evaluation has a stricter purpose:

```text
Training / Validation
       ↓
Frozen Model
       ↓
Previously Unseen Test Data
       ↓
Final Predictive Accuracy
```

---

# 27. Frontend

The Next.js frontend provides:

* Dashboard
* Company List
* Company Details
* Historical stock charts
* OHLCV visualization
* Forecast results
* Backtesting visualization
* Model Performance
* Search and navigation
* Pipeline status
* Responsive UI
* Educational/project information

The frontend consumes generated JSON artifacts.

It does not:

* Train models
* Run Python
* Perform model fitting
* Process PSE PDFs
* Run predictive-accuracy evaluation
* Directly access the forecasting engine

---

# 28. Frontend Data Contract

The primary artifacts are:

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

The predictive-accuracy evaluation results are separately maintained under:

```text
backend/tests/predictive_accuracy/results/
```

This separation prevents experimental validation artifacts from being silently confused with production dashboard artifacts.

---

# 29. Automated Fast Pipeline

The Fast Pipeline processes routine PSE data updates without retraining the models.

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

Execution:

```bash
python backend/run_pipeline.py --no-train
```

---

# 30. Heavy Training Pipeline

The Heavy Training workflow performs:

* Model retraining
* Evaluation
* Model selection
* Statistical testing
* Forecast generation
* Artifact generation

```text
Latest OHLCV
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Model Evaluation
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

---

# 31. Predictive Accuracy Evaluation Workflow

The final unseen-test evaluation is separate from the routine production pipeline.

Its workflow is:

```text
Historical OHLCV
        ↓
Chronological Split
        ↓
Train + Validation
        │
        ├── Model Training
        ├── Feature Engineering
        ├── Scaling
        └── Model Selection
        │
        ▼
     Freeze Models
        │
        ▼
   Final Unseen Test
        │
        ├── Walk-Forward Prediction
        ├── Leakage Checks
        ├── RMSE
        ├── MAE
        ├── MASE
        ├── R²
        ├── Directional Accuracy
        ├── Hit Rate
        └── Error Statistics
        │
        ▼
Statistical Analysis
        │
        ├── Diebold-Mariano
        ├── Friedman
        ├── Wilcoxon-Holm
        └── Best-Model Consistency
        │
        ▼
Evaluation Results
```

---

# 32. Predictive Accuracy Results Artifacts

The final evaluation produces:

```text
backend/tests/predictive_accuracy/results/
├── evaluation_summary.md
├── metrics.csv
├── metrics.json
├── model_comparison.csv
└── statistical_tests.json
```

The `evaluation_summary.md` contains the high-level final-test results.

The `metrics.csv` and `metrics.json` contain per-ticker model metrics.

The `model_comparison.csv` contains model-comparison results.

The `statistical_tests.json` contains the detailed statistical-analysis output.

These artifacts are committed to the repository.

---

# 33. Current Verification Status

| Component                                                             | Status      |
| --------------------------------------------------------------------- | ----------- |
| PSE data ingestion                                                    | Completed   |
| Data cleaning                                                         | Completed   |
| Data validation                                                       | Completed   |
| Historical OHLCV storage                                              | Completed   |
| Historical OHLCV frontend artifacts                                   | Completed   |
| Feature engineering                                                   | Completed   |
| Lag-Informed Regression / LASSO                                       | Completed   |
| ARIMA                                                                 | Completed   |
| LSTM                                                                  | Completed   |
| Naive baseline                                                        | Completed   |
| Forecast generation                                                   | Completed   |
| RMSE                                                                  | Completed   |
| MAE                                                                   | Completed   |
| MASE                                                                  | Completed   |
| R²                                                                    | Completed   |
| Backtesting                                                           | Completed   |
| Production statistical testing                                        | Completed   |
| Best-model selection                                                  | Completed   |
| Interactive stock charts                                              | Completed   |
| Model Performance dashboard                                           | Completed   |
| UI enhancements                                                       | Completed   |
| Fast Pipeline                                                         | Completed   |
| Heavy Training Pipeline                                               | Completed   |
| GitHub Actions                                                        | Completed   |
| Vercel deployment architecture                                        | Completed   |
| Predictive-accuracy evaluation framework                              | Completed   |
| Final unseen-test split                                               | Completed   |
| Leakage checks                                                        | Completed   |
| Predictive-accuracy metrics                                           | Completed   |
| Statistical predictive-accuracy analysis                              | Completed   |
| Final unseen-test results for 15 tickers                              | Completed   |
| 63/63 predictive-accuracy tests                                       | Passing     |
| Predictive-accuracy result artifacts                                  | Completed   |
| Production dashboard integration of final predictive-accuracy results | In Progress |
| Final end-to-end verification                                         | In Progress |
| Overall project completion                                            | In Progress |

---

# 34. What Has Changed Since the Previous Completion Report

The previous report stated that the project was waiting for final unseen-test predictive-accuracy evaluation.

That is no longer accurate.

The repository now contains the completed evaluation framework and generated final-test results.

The following items have moved from pending to completed:

```text
[✓] Final unseen-test evaluation framework
[✓] Temporal final-test split
[✓] Model freezing
[✓] Leakage detection
[✓] Predictive-accuracy metrics
[✓] Bootstrap confidence intervals
[✓] Statistical testing
[✓] Model-comparison artifacts
[✓] Final evaluation summary
[✓] 15-ticker evaluation
```

The remaining work is now primarily:

```text
[ ] Integrate validated predictive-accuracy results into the production Model Performance presentation
[ ] Confirm dashboard values correspond exactly to validated artifacts
[ ] Complete final end-to-end production verification
```

---

# 35. Important Predictive-Accuracy Finding

The latest evaluation demonstrates an important result:

**The most complex model is not necessarily the most accurate model.**

The current aggregate final-test ranking is:

```text
1. ARIMA                  RMSE 4.3895
2. Naive baseline        RMSE 4.4131
3. Lag Regression        RMSE 7.8471
4. LSTM                   RMSE 14.7503
```

ARIMA has only a small aggregate improvement over the naive baseline.

Therefore, the system should not claim that machine learning automatically provides superior forecasting accuracy.

The predictive-accuracy evaluation instead provides evidence-based model comparison.

---

# 36. Important Model-Selection Finding

The current results also show a difference between aggregate and per-ticker model selection.

```text
Aggregate mean RMSE winner:
ARIMA

Most individual ticker wins:
Lag-Informed Regression
```

The counts are:

```text
Lag Regression    7 / 15
ARIMA             6 / 15
LSTM              2 / 15
```

This supports the project's per-company model-selection architecture rather than assuming one model is universally optimal.

---

# 37. Statistical-Analysis Finding

The Friedman test indicates a significant difference across the tuned models:

```text
Friedman statistic = 10.1333
p-value            = 0.006303
```

Including the naive baseline:

```text
Friedman statistic = 15.3446
p-value            = 0.001545
```

However, the Holm-adjusted pairwise comparison between Lag Regression and ARIMA is not statistically significant:

```text
Holm-adjusted p = 0.488709
```

Therefore, the report should not claim that ARIMA is statistically proven superior to Lag Regression based solely on the current final-test evaluation.

The strongest statistically supported differences in the current post-hoc analysis involve LSTM.

---

# 38. Current Project Architecture

The current target architecture is:

```text
                 PSE EOD MARKET DATA
                          │
                          ▼
               ┌─────────────────────┐
               │ Data Pipeline       │
               │                     │
               │ Download            │
               │ Extraction          │
               │ Cleaning            │
               │ Validation          │
               │ OHLCV Update        │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ Feature Engineering │
               └──────────┬──────────┘
                          │
                          ▼
          ┌─────────────────────────────────┐
          │ Forecasting Models              │
          │                                 │
          │ Lag Regression / LASSO          │
          │ ARIMA                           │
          │ LSTM                            │
          │ Naive Baseline                  │
          └───────────────┬─────────────────┘
                          │
                          ▼
          ┌─────────────────────────────────┐
          │ Production Evaluation           │
          │                                 │
          │ RMSE / MAE / MASE / R²         │
          │ Backtesting                     │
          │ Statistical Testing             │
          │ Best Model Selection             │
          └───────────────┬─────────────────┘
                          │
                          ▼
          ┌─────────────────────────────────┐
          │ Predictive Accuracy Validation   │
          │                                 │
          │ Temporal Holdout                │
          │ Frozen Models                   │
          │ Leakage Checks                  │
          │ Directional Accuracy             │
          │ Hit Rate                        │
          │ Confidence Intervals            │
          │ Final Unseen-Test Metrics       │
          └───────────────┬─────────────────┘
                          │
                          ▼
          ┌─────────────────────────────────┐
          │ Statistical Accuracy Analysis    │
          │                                 │
          │ Diebold-Mariano                 │
          │ Friedman                        │
          │ Wilcoxon-Holm                  │
          │ Best-Model Consistency          │
          └───────────────┬─────────────────┘
                          │
                          ▼
                JSON / CSV / MD Artifacts
                          │
                          ▼
          ┌─────────────────────────────────┐
          │ Next.js Dashboard               │
          │                                 │
          │ Historical OHLCV                │
          │ Interactive Charts              │
          │ Forecasts                       │
          │ Backtesting                     │
          │ Model Performance               │
          └───────────────┬─────────────────┘
                          │
                          ▼
                        Vercel
```

---

# 39. Remaining Completion Requirements

The project should remain **IN PROGRESS** until these final steps are completed.

## 39.1 Production Model Performance Integration

The validated predictive-accuracy results should be clearly incorporated into the appropriate production-facing Model Performance presentation.

The dashboard should distinguish:

* Production model metrics
* Final unseen-test predictive accuracy
* Statistical significance results
* Per-ticker model selection

These should not be mixed together in a way that could cause users to interpret training/validation performance as final unseen-test accuracy.

---

## 39.2 Artifact Verification

Verify that the frontend consumes the intended validated artifacts.

Confirm:

```text
Backend Results
      ↓
Generated Artifacts
      ↓
Frontend
```

No stale model-performance artifact should remain in the dashboard.

---

## 39.3 End-to-End Verification

The final verification should cover:

```text
Raw PSE Data
      ↓
Historical OHLCV
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Selection
      ↓
Forecast
      ↓
Predictive Accuracy
      ↓
Statistical Testing
      ↓
Generated Artifacts
      ↓
Next.js Dashboard
      ↓
Vercel
```

---

# 40. Definition of Final Completion

The project should be marked **COMPLETED** only after:

```text
[✓] PSE data pipeline operational
[✓] Historical OHLCV operational
[✓] Forecasting models operational
[✓] Production model evaluation operational
[✓] Statistical testing operational
[✓] Interactive charts operational
[✓] Dashboard UI operational
[✓] Fast Pipeline operational
[✓] Heavy Training operational
[✓] Vercel deployment architecture operational

[✓] Predictive-accuracy framework implemented
[✓] Final unseen-test temporal split implemented
[✓] Leakage checks implemented
[✓] Frozen-model evaluation implemented
[✓] Predictive-accuracy metrics implemented
[✓] Bootstrap confidence intervals implemented
[✓] Statistical predictive-accuracy analysis implemented
[✓] Final unseen-test evaluation completed for 15 tickers
[✓] Evaluation results committed
[✓] Statistical results committed
[✓] 63/63 predictive-accuracy tests passing

[ ] Production Model Performance integration verified
[ ] Final frontend artifact synchronization verified
[ ] Final end-to-end production verification completed
```

---

# 41. Current Authoritative Status

## IN PROGRESS

The project has passed the major predictive-accuracy development milestone.

The final unseen-test evaluation framework is implemented, the statistical analysis is implemented, the evaluation results are committed, and the predictive-accuracy test suite is passing.

The latest final-test results cover all 15 supported PSE tickers and provide:

* Model-level RMSE rankings
* Per-ticker rankings
* Naive-baseline comparisons
* Directional-accuracy metrics
* Error statistics
* Confidence intervals
* Diebold-Mariano comparisons
* Friedman tests
* Wilcoxon-Holm post-hoc tests
* Best-model consistency analysis

The current final-test aggregate result places ARIMA first by mean RMSE, narrowly ahead of the naive baseline, while Lag-Informed Regression has the highest number of individual ticker wins.

The statistical analysis confirms significant differences across the model set but does not establish a statistically significant ARIMA-vs-Lag-Regression difference in the current Holm-adjusted post-hoc comparison.

Therefore, the correct project state is:

**Predictive-Accuracy Evaluation: COMPLETED**

**Predictive-Accuracy Statistical Analysis: COMPLETED**

**Predictive-Accuracy Test Suite: 63/63 PASSING**

**Production Dashboard Integration: IN PROGRESS**

**Overall Project: IN PROGRESS**

---

# 42. Conclusion

ForecastPH has progressed from a forecasting dashboard into a substantially validated analytical forecasting system.

The repository now combines:

* Automated PSE market-data ingestion
* Full historical OHLCV data
* Feature engineering
* Multiple forecasting models
* Production model evaluation
* Backtesting
* Interactive stock charts
* Model selection
* Statistical model comparison
* Final unseen-test evaluation
* Leakage detection
* Directional-accuracy analysis
* Naive-baseline comparison
* Bootstrap confidence intervals
* Diebold-Mariano testing
* Friedman testing
* Wilcoxon-Holm post-hoc testing
* Automated pipelines
* Next.js visualization
* Vercel deployment

The latest predictive-accuracy implementation provides a reproducible evaluation of model generalization on data withheld from training and model selection.

The current evidence shows that:

* ARIMA has the lowest aggregate final-test RMSE.
* The naive baseline is extremely competitive.
* Lag-Informed Regression wins the largest number of individual ticker comparisons.
* LSTM performs substantially worse on the current aggregate final-test RMSE.
* The tuned models have statistically different performance distributions.
* ARIMA and Lag Regression are not statistically distinguishable in the current Holm-adjusted post-hoc comparison.
* LSTM is significantly different from both ARIMA and Lag Regression in the current post-hoc analysis.

These findings are now documented in committed evaluation artifacts rather than being treated as hypothetical future work.

The remaining work is limited to final production integration, dashboard synchronization, and end-to-end verification.

**Current project status: IN PROGRESS — FINAL PRODUCTION INTEGRATION AND END-TO-END VERIFICATION**
