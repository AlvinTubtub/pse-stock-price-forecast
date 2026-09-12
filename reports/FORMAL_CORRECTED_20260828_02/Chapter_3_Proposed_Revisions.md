# Chapter 3 Proposed Methodology Revisions

These passages describe the corrected formal experiment. They are proposed replacements for the corresponding methodology sections, subject to reconciliation with the authoritative manuscript and adviser review. They do not establish when adviser approval occurred. Preserve the original manuscript until the current version is identified. The companion Chapter 4 draft covers forecasting results only, not surveys, usability testing, or other capstone outcomes.

## Data scope and chronological split

The formal study used 15 company series: ALI, APX, BPI, GLO, ICT, JFC, MBT, MEG, MER, NIKL, PGOLD, SCC, SECB, SHLPH, and SMPH. Each series contained 1,624 validated OHLCV observations from January 2, 2020 through August 28, 2026. The complete dataset contained 24,360 company-date observations. Each forecast used information available after the origin trading day's close to predict the next observed trading day's closing price.

The split was defined over 1,623 origin–target pairs per company. The development portion comprised 1,380 target dates from January 3, 2020 through September 1, 2025. The final holdout comprised 243 target dates from September 2, 2025 through August 28, 2026. The same holdout dates and actual closing prices were used for all four methods. Earlier holdout actual prices could become available as inputs at later forecast origins without retuning the models on holdout outcomes.

## Model specific inputs and regression selection

Lag-Informed Regression and the formal LSTM predicted the next-day change in closing price. Their predicted changes were added to the known origin-day Close to obtain next-day closing-price forecasts. ARIMA modeled closing-price levels and applied internal differencing according to its selected order. Fair comparison required identical forecast horizons, information availability, target dates, and final evaluated prices, rather than identical internal model representations.

Regression candidate predictors comprised lagged closing prices, moving averages, technical indicators, returns, volatility, price-range measures, and volume measures. The implemented feature contract contained price lags 1, 2, 3, and 5; moving averages 5 and 10; volume moving average 5; EMA 10 and 20; RSI 14; MACD and its signal; Bollinger upper and lower bands; daily return; rolling volatility; high–low and open–close relative measures; normalized EMA gap and MACD; Bollinger width and percent B; return lags 1 through 20; return means and volatilities over 5, 10, and 20 observations; log volume; and volume means over 10 and 20 observations. PACF selected among candidate return lags. Exact feature formulas and shifts should be reproduced from backend/services/feature_engineering.py in the manuscript's feature appendix.

PACF selection used fold-training daily returns. A new StandardScaler was fitted within each fold-training portion. LASSO regularization was selected from 36 logarithmically spaced alpha values spanning 0.0001 through 1000 using five expanding development folds and mean original-scale validation RMSE. Candidates with unconfirmed fold convergence were excluded. Boundary selections were checked before accepting the formal search. The selected configuration was fitted using development data, and its feature coefficients were recorded. A model with no retained feature coefficients remained an intercept-only change predictor.

## ARIMA selection and evaluation

ADF was recorded as a stationarity diagnostic. Candidate orders covered p from 0 to 3, d from 0 to 2, and q from 0 to 3, including orders with p and q both zero. For d equal to zero, the trend choices were none and constant; for d equal to one, none and linear trend; for d equal to two, no trend. The resulting grid contained 80 order–trend configurations.

Selection minimized full-precision mean validation RMSE across five chronological development folds. Accepted configurations required finite forecasts and confirmed optimizer convergence in every fold. Optimizer retries followed the recorded deterministic policy, and unsuccessful candidates were excluded. Final development fitting also required confirmed convergence. During holdout evaluation, each newly revealed actual price updated the fitted ARIMA state with append(actual, refit=False); parameters were not re-estimated at every origin. ADF did not uniquely determine d, and ACF/PACF did not define the search bounds.

## LSTM selection and complete development refit

The formal LSTM used a one-layer architecture with univariate closing-price changes. Its 48 configurations combined lookbacks of 5, 10, 20, and 30; hidden sizes of 25, 50, and 100; learning rates of 0.01 and 0.001; and batch sizes of 16 and 32. Five expanding development folds used common validation target dates constructed with maximum lookback 30. All configurations were scored on those dates using the predeclared seeds 42, 123, and 2026.

Within each fold, a chronological stopping tail was separated from the training block. Scaling excluded that tail, and outer validation targets were used only for scoring. Configurations were ranked by mean original-scale RMSE across all folds and seeds, with deterministic configuration tie-breaking. Per-seed scores, epochs, means, and standard deviations were retained. The final seed was fixed at 42 rather than selected by performance.

After configuration selection, a preliminary development fit used a chronological stopping tail to determine the epoch count. A fresh scaler was then fitted to all development changes, and a fresh model trained on every available development sequence for that fixed count. No stopping tail was retained during this complete-development refit. The final holdout was scored afterward. Five folds and three seeds were computational design choices, not empirically demonstrated universal optima.

## Evaluation metrics and benchmark

RMSE, MAE, MASE, and R² were computed on reconstructed closing-price forecasts at full precision. Rounding was reserved for presentation. One company-level MASE denominator was calculated as the mean absolute successive difference of the complete development Close series and shared by all models. MASE below one refers to that development scaling reference and does not alone establish superiority over holdout Naive.

The Naive benchmark predicted the origin-day Close for the next target date. Forecast errors were defined as actual Close minus predicted Close. All models used the same 243 target dates, origin dates, and actual values. Date mismatches were errors rather than grounds for truncating arrays.

## Corporate action policy

The primary analysis retained all validated raw quoted closing-price observations. Verified corporate-action dates were flagged using the frozen registry, and a separate sensitivity analysis recalculated metrics after excluding registered event target dates identically across models. Observations were not removed merely because their errors or movements were large. This policy does not describe a dividend-adjusted total-return target.

## Hierarchical statistical treatment

Within each company, the three principal models were first compared with Naive using paired Diebold–Mariano tests on squared-error loss, with HAC variance estimation, the HLN correction, and Holm adjustment across the three benchmark comparisons. A model passed the improvement gate only if its mean loss was lower and its adjusted p-value was below 0.05. Comparisons among principal models were performed only if at least two passed the gate. Absolute-error comparisons supplied robustness evidence. Moving-block bootstrap results used 5,000 replications, seed 42, and seven-observation blocks for the 243-day holdout.

Across companies, Friedman analysis used MASE for all four methods, with 10,000 permutations and seed 42. Wilcoxon post-hoc comparisons with Holm adjustment were conditional on a significant permutation Friedman result. The eight-of-fifteen RMSE consistency criterion considered principal models only and was descriptive. Shared market dependence limits claims based on treating companies as independent datasets.

Proposed research question for this design: Which principal forecasting models improve on the Naive benchmark for each company under the declared hierarchical tests, and how do the four methods compare descriptively and in supporting across-company analysis? A broader claim that every within-company principal-model pair was tested would be inaccurate. Changing to an all-pairs design after observing results must be labeled as an additional analysis, not retroactively described as predeclared.

## Formal evidence and deployment separation

The formal study preserved its source hashes, split, predictions, configurations, diagnostics, statistical results, dependency versions, and code identity in run FORMAL_CORRECTED_20260828_02 at commit bfb33b8c184c87cc8828af5529410da94addd71c. Deployment retraining and model promotion are separate operational decisions. A deployment model selected using these historical rankings requires prospective evaluation; the same holdout is not an independent estimate of the selected policy's future performance.

## Reconciliation checklist

| Manuscript claim to locate | Required treatment | Status |
|---|---|---|
| June 30 cutoff or 1582 rows per company | Replace with audited dates and counts | Proposed wording ready |
| ARIMA parameters re-estimated daily | Describe fixed parameters and state updates | Proposed wording ready |
| All models learn the same change target | Distinguish ARIMA levels from LIR and LSTM changes | Proposed wording ready |
| LSTM single seed or partial final fit | Describe three-seed tuning and complete refit | Proposed wording ready |
| Corporate-action observations excluded in primary results | Describe retain-and-flag plus sensitivity metrics | Proposed wording ready |
| All principal-model pairs tested | Align research question with benchmark-first gate | Proposed wording ready; adviser review deferred |
| Indicators improved all final regressions | Report nine intercept-only selected models | Results draft addresses this |
| High R² proves superior forecasting | Require direct Naive comparison | Results draft addresses this |
| One best model across all companies | Report no 8-of-15 dominant principal model | Results draft addresses this |
| Current manuscript version and section numbering | Identify authoritative file before integration | Pending user identification |
| Prior adviser approval of protocol changes | Confirm from dated project records | Not established by artifact audit |
| External corporate-action disclosure completeness | Review registry source records if required | Not independently revalidated |

## Source requirements

Reference documents: Models Check.docx and Model_Development_Audit_and_Improvement_Report.docx in Downloads. Their requirements inform these revisions; they do not establish approval or authorize changes to frozen experiment evidence. Code references: backend/services/feature_engineering.py; backend/services/forecasting/lag_regression.py; backend/services/forecasting/arima_model.py; backend/services/forecasting/lstm_model.py; backend/services/evaluation.py. Artifact references are listed in the companion Chapter 4 draft.
