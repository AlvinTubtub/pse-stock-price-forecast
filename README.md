Cross-Sector Next-Day Stock Price Forecasting
Research-oriented stock forecasting and decision-support platform for selected Philippine Stock Exchange (PSE) companies.
The system compares three forecasting approaches:
- Lag-Informed Regression
- ARIMA
- LSTM
A Naive baseline is also evaluated as a benchmark.
ForecastPH is designed to support the capstone study:
Cross-Sector Next-Day Stock Price Forecasting of Selected PSE-Listed Companies: A Comparative Study of Lag-Informed Regression, ARIMA, and LSTM
The platform combines a Python forecasting backend, a Next.js frontend, automated data-processing workflows, model evaluation, statistical testing, deployment-oriented forecasting, and an administrative interface.
Current Status
The forecasting and evaluation methodology has been audited and corrected.
The corrected implementation is now merged into main.
Major methodology corrections include:
- Common hold-out target dates across all models
- Genuine out-of-sample evaluation
- No future-data leakage or backfilling
- Walk-forward one-step forecasting
- Proper development and hold-out separation
- Expanding-window cross-validation
- Fold-specific preprocessing
- Fold-specific PACF lag selection
- Training-only scaling
- Correct LSTM validation and refitting procedure
- Original-price-scale LSTM model selection
- Date-aligned Naive baseline
- Naive-first Diebold-Mariano testing
- Holm multiple-testing correction
- Moving-block bootstrap robustness testing
- Friedman and conditional Wilcoxon testing
- Full-precision backend metrics
- Formal research artifact separation from deployment artifacts
- ARIMA fixed-parameter state updating using append(..., refit=False)
- Formal ARIMA candidate completeness checks
- Hold-out residual diagnostics
- Correct RSI zero-gain / zero-loss handling
- Univariate ΔClose LSTM methodology
- Reproducible seeds and provenance tracking
The finalized 15-company formal evaluation was successfully completed using the audited implementation.
Companies Covered
ForecastPH currently evaluates 15 selected PSE-listed companies across five sectors.
Sector	Companies
Financials	BPI, MBT, SECB
Industrial	MER, JFC, SHLPH
Property	MEG, ALI, SMPH
Services	GLO, PGOLD, ICT
Mining & Oil	APX, NIKL, SCC


Forecasting Models
1. Lag-Informed Regression
The regression pipeline uses lag-informed predictors selected using training data only.
Key safeguards include:
- PACF calculated from training-period daily returns
- PACF recalculated within each cross-validation fold
- Fold-specific StandardScaler
- No scaling leakage from validation or hold-out observations
- Chronological expanding-window validation
- Genuine out-of-sample evaluation
The deployment and formal evaluation implementations preserve chronological forecasting behavior.
2. ARIMA
ARIMA uses bounded candidate selection and walk-forward forecasting.
The corrected formal procedure includes:
- Candidate grid bounded to:
  - p <= 3
  - d <= 2
  - q <= 3
- ADF-informed differencing priority
- Candidate must successfully complete all required folds with finite predictions
- Candidate convergence metadata recorded separately
- No out-of-range formal fallback order
- Final model trained on the complete development period
- Hold-out forecasts generated one step at a time
During walk-forward hold-out evaluation, the observed value is appended using:
append(actual, refit=False)
This updates the ARIMA model state without re-estimating coefficients at every forecast origin.
The methodology therefore represents:
One-step-ahead walk-forward forecasting with fixed estimated ARIMA parameters and sequential state updating.

3. LSTM
The formal LSTM implementation predicts daily price changes rather than raw closing prices.
Target:
ΔClose_t = Close_t - Close_(t-1)
The predicted change is reconstructed into price form:
Predicted Close_t = Close_(t-1) + Predicted ΔClose_t
The corrected LSTM methodology includes:
- Univariate ΔClose input
- Min-Max scaling fitted on training data only
- 5-fold expanding-window cross-validation
- Validation data never used for early-stopping training
- Early stopping uses an internal tail from the fold-training portion
- Maximum 200 epochs
- Early stopping patience of 10
- Random seed 42
- Fresh final model refit after hyperparameter selection
- Hold-out period remains untouched during model selection
Hyperparameter grid:
Parameter	Values
Lookback	5, 10, 20, 30
Hidden units	25, 50, 100
Learning rate	0.01, 0.001
Batch size	16, 32


This produces 48 candidate configurations.
With 5 cross-validation folds:
48 configurations × 5 folds = 240 CV fits per company
A fresh final LSTM is then trained using the selected configuration.
The winning configuration is selected using mean reconstructed-price RMSE across validation folds rather than scaled validation loss.
Data Splitting
The corrected formal evaluation uses a chronological split.
Development period: 85%
Hold-out period: 15%
The development period is used for:
- Hyperparameter selection
- Cross-validation
- PACF selection
- Scaling
- Model fitting decisions
The hold-out period is reserved strictly for final out-of-sample evaluation.
All models are evaluated on the same target dates.
No model is allowed to use:
- Future observations
- Hold-out information during tuning
- Backfilled future values
- Array truncation to artificially align results
Alignment is performed explicitly by target date.
Walk-Forward Evaluation
ForecastPH uses one-step-ahead chronological forecasting.
Conceptually:
Training / development history
        ↓
Forecast next trading session
        ↓
Observe actual closing price
        ↓
Update model state/history
        ↓
Forecast following trading session
This produces genuine out-of-sample predictions suitable for evaluating next-day forecasting performance.
Evaluation Metrics
ForecastPH evaluates:
- RMSE
- MAE
- MASE
- R²
RMSE
Root Mean Squared Error:
RMSE = sqrt(mean((Actual - Predicted)^2))
Lower values indicate smaller forecast errors.
MAE
Mean Absolute Error:
MAE = mean(abs(Actual - Predicted))
Lower values are better.
MASE
Mean Absolute Scaled Error compares a model against the Naive benchmark.
MASE < 1 → model outperforms Naive on average
MASE = 1 → approximately equal to Naive
MASE > 1 → worse than Naive
A common company-level scaling denominator is calculated from the development-period Naive forecast error.
R²
R² is included as a supplementary goodness-of-fit metric.
It is not interpreted as a forecast probability or forecast confidence score.
Naive Baseline
ForecastPH includes a one-step Naive benchmark:
Forecast_t = Actual_(t-1)
The Naive model is treated as a full evaluation model rather than only as a MASE denominator.
It participates in:
- Common target-date alignment
- RMSE
- MAE
- MASE
- R²
- Diebold-Mariano testing
- Across-company Friedman analysis
Statistical Evaluation
ForecastPH uses a hierarchical statistical-testing framework.
Stage 1 — Principal Models vs Naive
Each principal model is first compared against the Naive benchmark using the Diebold-Mariano test.
Principal models:
- Lag-Informed Regression
- ARIMA
- LSTM
Primary loss:
Squared forecast error
Robustness loss:
Absolute forecast error
Multiple comparisons are controlled using Holm correction.
Only models that significantly outperform Naive in the favorable direction are eligible for principal pairwise testing.
Diebold-Mariano Test
The implementation includes:
- Newey-West HAC variance
- Harvey-Leybourne-Newbold correction
- Holm-adjusted significance testing
- Date-aligned forecast errors
The primary analysis uses squared-error loss.
Absolute-error DM tests are retained as robustness evidence.
Moving-Block Bootstrap
ForecastPH includes moving-block bootstrap robustness analysis.
Configuration:
Replications: 5000
Random seed: 42
The block bootstrap preserves short-range temporal dependence better than independent resampling.
Across-Company Statistical Testing
Across the 15 companies, ForecastPH compares unrounded MASE values for:
- Lag-Informed Regression
- ARIMA
- LSTM
- Naive
The workflow is:
Friedman test
        ↓
Fixed-seed permutation robustness test
        ↓
If significant:
    Holm-adjusted Wilcoxon pairwise tests
Wilcoxon post-hoc testing is not performed if the omnibus test is not significant.
Final Formal 15-Company Evaluation
The definitive formal run was:
FORMAL_15COMP_20260828_01
The run covered all 15 companies.
Each company contained:
243 hold-out trading sessions
4 evaluated models
972 prediction rows per company
Models:
Lag-Informed Regression
ARIMA
LSTM
Naive
All companies used the same chronological development/hold-out structure.
The final hold-out period covered:
2025-09-01 → 2026-08-27
Final Principal-Model RMSE Winners
The lowest-RMSE principal model by company produced:
ARIMA                  6 companies
Lag-Informed Regression 5 companies
LSTM                   4 companies
No principal model reached the predefined consistency threshold of:
8 out of 15 companies
Therefore, the study did not identify one universally dominant forecasting model across the complete company sample.
Across-Company Final Statistical Result
The final Friedman test on company-level MASE values produced:
Friedman statistic ≈ 0.44
p-value ≈ 0.932
Permutation robustness result:
Permutation p-value ≈ 0.944
Permutations = 10000
Seed = 42
The omnibus result was not statistically significant.
Therefore:
Wilcoxon post-hoc tests were not executed.
The final evidence does not support claiming that one forecasting method is statistically superior across all 15 companies.
Interpretation of Forecast Accuracy
The corrected methodology improves the validity and reliability of the forecasting experiment.
It reduces methodological problems such as:
- Data leakage
- Inconsistent model dates
- Improper validation
- Scaling leakage
- Incorrect LSTM model selection
- Misaligned baseline comparison
- Invalid statistical comparisons
However, methodological correction does not guarantee that every future forecast will have a lower numerical error.
The appropriate interpretation is:
The corrected implementation provides a more reliable and reproducible estimate of true out-of-sample forecasting performance.

Forecast accuracy still depends on:
- Company-specific price behavior
- Market volatility
- Structural changes
- Unexpected economic events
- Company disclosures
- Political and macroeconomic shocks
- Liquidity
- Non-stationarity
Residual Diagnostics
Formal evaluation includes diagnostic information such as:
- Ljung-Box testing
- Shapiro-Wilk testing
- ARCH / volatility diagnostics
- Regression residual autocorrelation
- LSTM training and stopping information
- Naive skill indicators
- ARIMA convergence metadata
For ARIMA, Ljung-Box diagnostics are evaluated using formal hold-out forecast errors.
Formal vs Deployment Artifacts
ForecastPH separates research evidence from production forecasting artifacts.
Formal Research Artifacts
Formal evaluation outputs are immutable once finalized.
Structure:
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
Formal artifacts are used for:
- Capstone evidence
- Reproducibility
- Methodology verification
- Statistical analysis
- Research reporting
They are not intended to serve directly as live Vercel forecasting data.
Deployment Artifacts
Live forecasting models are stored separately from the formal research experiment.
Example structure:
backend/models/deployment/current/
├── lag_regression/
│   └── <SYMBOL>.pkl
├── arima/
│   └── <SYMBOL>.pkl
├── lstm/
│   └── <SYMBOL>.pth
└── deployment_manifest.json
Deployment models may be refreshed according to the production schedule.
Formal research artifacts must remain unchanged.
Production Forecasting Schedule
ForecastPH separates model retraining from daily inference.
Weekly Model Retraining
Every Sunday
8:00 AM Philippine Time
The weekly process refreshes deployment-oriented model artifacts using the latest approved methodology.
Daily Forecasting / Inference
The production pipeline performs daily data processing and next-session forecasting according to the configured PSE schedule.
Inference uses deployment models and the latest available market information.
PSE holidays and non-trading days are handled by the trading-calendar logic.
Vercel Frontend
The frontend is built using Next.js and deployed through Vercel.
The website displays pre-generated forecasting artifacts rather than training machine-learning models inside Vercel.
Conceptually:
PSE data
   ↓
Python backend
   ↓
Feature engineering
   ↓
Deployment models
   ↓
Next-day inference
   ↓
Artifact exporter
   ↓
frontend/public/forecasts/
   ↓
Vercel
Company Forecast JSON
Company pages consume files located under:
frontend/public/forecasts/company/
Example:
frontend/public/forecasts/company/BPI.json
These files may contain:
previousClose
predictedClose
forecastDate
dataAsOf
selected model
metrics
nextClose
OHLCV
backtestDates
backtestActual
backtestByModel
These JSON files are deployment/frontend outputs.
They should not be manually edited as a permanent forecasting solution.
The backend exporter should regenerate them.
Correct 60-Session Backtest Pipeline
The following charts are driven by company forecast JSON:
- Backtest: Predicted vs. Actual (Last 60 Sessions)
- Forecast Error Over Time
The intended corrected pipeline is:
Corrected deployment methodology
        ↓
Genuine out-of-sample predictions
        ↓
Align Lag-Reg, ARIMA, LSTM and Naive by target date
        ↓
Retain common valid trading sessions
        ↓
Select latest 60 sessions
        ↓
Export:
    backtestDates
    backtestActual
    backtestByModel
        ↓
frontend/public/forecasts/company/*.json
        ↓
Vercel charts
The exporter responsible for these frontend fields is:
backend/scripts/export_forecast_artifacts.py
The frontend should display the exported backtest instead of recalculating forecasts independently.
Backtest: Predicted vs. Actual
The chart compares actual closing prices with genuine historical one-step-ahead model predictions.
For every plotted session:
Target date = same for all models
Actual price = observed close on target date
Predicted price = forecast generated without seeing target-date actual price
The chart should display the latest 60 valid common out-of-sample sessions.
It must not use:
- Fitted training values
- Future information
- Backfilled predictions
- Independently truncated arrays
- Artificially reconstructed predictions
- Different target dates across models
Forecast Error Over Time
Forecast Error Over Time uses exactly the same dates and predictions as the Backtest chart.
Error is defined as:
Forecast Error = Predicted Close - Actual Close
Therefore:
Positive error → model overpredicted
Negative error → model underpredicted
Zero → exact prediction
The chart should not use a separately generated residual dataset if that dataset can become misaligned with the displayed backtest.
Both charts should be derived from the same aligned 60-session source.
Frontend Chart Controls
ForecastPH charts support interactive inspection.
Controls include:
+       Zoom in
−       Zoom out
Box     Select / zoom range
Pan     Move horizontally
Reset   Restore full range
Mouse controls may include:
Mouse wheel → Zoom
Mouse drag  → Pan
Double-click → Reset
Next-Day Prediction Chart
The Next-Day Prediction chart displays:
- Latest actual close
- ARIMA next-session forecast
- Lag-Informed Regression next-session forecast
- LSTM next-session forecast
The next-day forecast is separate from the historical 60-session backtest.
Historical backtests evaluate previous forecasts.
The next-day chart represents the currently generated future forecast.
Model Selection
Principal model selection is based on out-of-sample RMSE.
The selected model should be chosen from:
Lag-Informed Regression
ARIMA
LSTM
The Naive baseline remains a benchmark rather than the production forecasting model.
A lower RMSE indicates better historical out-of-sample accuracy for that company over the evaluated period.
Statistical significance should be interpreted separately from descriptive RMSE ranking.
Repository Structure
Typical project structure:
pse-stock-price-forecast/
├── backend/
│   ├── data/
│   ├── models/
│   ├── results/
│   ├── scripts/
│   ├── services/
│   └── tests/
│
├── frontend/
│   ├── public/
│   │   └── forecasts/
│   │       └── company/
│   ├── src/
│   └── package.json
│
├── .github/
│   └── workflows/
│
└── README.md
Local Backend Setup
From the project root:
cd ~/Downloads/pse-stock-price-forecast-main/backend
Create a virtual environment if needed:
python3.11 -m venv .venv
Activate it:
source .venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Run Backend Tests
From:
cd ~/Downloads/pse-stock-price-forecast-main/backend
Run:
.venv/bin/python -m pytest -q
The audited implementation has passed the complete backend test suite.
Run Frontend Locally
From the project root:
cd frontend
Install dependencies:
npm install
Run development server:
npm run dev
Then open:
http://localhost:3000
Build Frontend
cd frontend
npm run build
A successful production build should complete without TypeScript or Next.js build errors.
Deployment
The frontend is deployed through Vercel.
A push to the configured production branch may trigger a Vercel deployment.
However:
New Git commit
≠ automatically new forecast values
Forecast values change only after the backend generates and publishes updated deployment artifacts.
The expected production sequence is:
Latest PSE data
        ↓
Corrected backend pipeline
        ↓
Weekly model retraining when scheduled
        ↓
Deployment models
        ↓
Daily inference
        ↓
Frontend forecast JSON generation
        ↓
Git / deployment publication
        ↓
Vercel redeployment
Important Deployment Rule
Do not use the immutable formal research run as the live production forecast source.
Correct separation:
Formal artifacts
→ research evidence

Deployment artifacts
→ live forecasting
This separation prevents scheduled production changes from modifying the finalized research evidence.
Reproducibility
The corrected research pipeline records provenance information including:
- Run ID
- Git commit
- Branch
- Working-tree state
- Dataset information
- Data cutoff
- Company universe
- Dependency versions
- Split definition
- Methodology configuration
- Artifact hashes
Formal runs can therefore be traced back to the exact implementation used to generate them.
Research Integrity
ForecastPH follows these core principles:
No leakage
No future-data backfilling
Chronological evaluation
Common target dates
Training-only preprocessing
Untouched formal hold-out
Explicit Naive comparison
Reproducible statistical testing
Formal/deployment artifact separation
Immutable finalized evidence
Limitations
ForecastPH is a next-day price forecasting research system.
It does not guarantee:
- Future profits
- Correct directional movement every day
- Protection from market shocks
- Stable accuracy under structural market changes
- Investment suitability for a particular user
Historical predictive accuracy is not a guarantee of future performance.
Forecasts should be treated as analytical and educational outputs rather than personalized financial advice.
Intended Users
ForecastPH is designed primarily for:
- Beginner traders
- Intermediate market learners
- Students
- Researchers
- Stakeholders interested in comparative forecasting methods
The dashboard emphasizes interpretability, model comparison, historical performance, and educational context.
Key Research Conclusion
The corrected ForecastPH implementation provides a methodologically aligned framework for comparing Lag-Informed Regression, ARIMA, and LSTM on next-day PSE closing-price forecasting.
The final formal evaluation shows that model performance varies by company.
No single principal forecasting method demonstrated statistically significant overall superiority across the full 15-company sample.
Therefore, ForecastPH uses company-specific evaluation rather than assuming that one forecasting algorithm is universally best.
Disclaimer
ForecastPH is developed for academic, educational, and research purposes.
Forecasted stock prices are statistical estimates based on historical data and model assumptions.
They should not be interpreted as guaranteed future prices, trading instructions, investment recommendations, or financial advice.
