# Chapter 4 Results and Discussion

## Forecast comparison

The corrected formal experiment compared Lag-Informed Regression, ARIMA, LSTM, and the Naive benchmark for 15 Philippine stocks. Each company contributed 243 one-step-ahead forecasts over September 2, 2025 through August 28, 2026. All methods were evaluated against the same actual closing prices on identical target dates. The results did not establish a consistently dominant model. Lag Regression significantly improved on Naive for ICT and MBT under the predeclared within-company benchmark-first tests; no ARIMA or LSTM model passed that improvement gate.

## Data and experiment integrity

Each frozen company series contained 1,624 observations from January 2, 2020 through August 28, 2026, giving 24,360 company-date observations. Splitting the 1,623 next-day forecast pairs produced 1,380 development pairs and 243 holdout pairs per company. Development target dates ended on September 1, 2025. The experiment generated 14,580 holdout predictions across all companies and methods.

The local acceptance audit verified all 155 protected artifact hashes, the 15 completed checkpoints, source-data hashes, forecast-date alignment, and Naive predictions. All 60 sets of RMSE, MAE, MASE, and R² reproduced from the stored forecasts within numerical tolerance. The statistical outputs also reproduced. These checks establish consistency of the saved evidence; they do not independently establish completeness of external corporate-action disclosures.

## Table 4 1 Canonical holdout metrics

RMSE and MAE are in pesos. MASE and R² are dimensionless. Values are rounded to six decimal places for presentation; comparisons use unrounded evidence. MASE uses one development-only scaling denominator per company. MASE below one does not itself imply improvement over holdout Naive.

| Company | Model | RMSE | MAE | MASE | R² |
|---|---|---:|---:|---:|---:|
| ALI | Lag Regression | 0.461824 | 0.358838 | 0.647994 | 0.985940 |
| ALI | ARIMA | 0.462202 | 0.359466 | 0.649126 | 0.985917 |
| ALI | LSTM | 0.463660 | 0.362114 | 0.653910 | 0.985828 |
| ALI | Naive | 0.462790 | 0.360082 | 0.650240 | 0.985881 |
| APX | Lag Regression | 0.540661 | 0.398352 | 7.445837 | 0.963121 |
| APX | ARIMA | 0.547549 | 0.407867 | 7.623689 | 0.962176 |
| APX | LSTM | 0.542305 | 0.402417 | 7.521822 | 0.962897 |
| APX | Naive | 0.540953 | 0.398519 | 7.448944 | 0.963082 |
| BPI | Lag Regression | 2.326041 | 1.689330 | 1.193689 | 0.937897 |
| BPI | ARIMA | 2.314899 | 1.682982 | 1.189204 | 0.938491 |
| BPI | LSTM | 2.435520 | 1.835551 | 1.297010 | 0.931914 |
| BPI | Naive | 2.363644 | 1.698354 | 1.200066 | 0.935873 |
| GLO | Lag Regression | 26.297302 | 18.043018 | 0.590130 | 0.943033 |
| GLO | ARIMA | 26.290010 | 18.037037 | 0.589935 | 0.943065 |
| GLO | LSTM | 27.573587 | 19.268610 | 0.630215 | 0.937369 |
| GLO | Naive | 26.290010 | 18.037037 | 0.589935 | 0.943065 |
| ICT | Lag Regression | 17.648428 | 12.874807 | 3.292667 | 0.988293 |
| ICT | ARIMA | 17.435691 | 12.857918 | 3.288348 | 0.988574 |
| ICT | LSTM | 17.407644 | 12.687147 | 3.244674 | 0.988610 |
| ICT | Naive | 17.732978 | 12.963786 | 3.315423 | 0.988181 |
| JFC | Lag Regression | 4.037364 | 2.854610 | 0.872969 | 0.982646 |
| JFC | ARIMA | 4.051790 | 2.857434 | 0.873833 | 0.982522 |
| JFC | LSTM | 4.152090 | 2.956039 | 0.903987 | 0.981646 |
| JFC | Naive | 4.034716 | 2.851852 | 0.872126 | 0.982669 |
| MBT | Lag Regression | 1.155558 | 0.852592 | 1.054091 | 0.860259 |
| MBT | ARIMA | 1.146625 | 0.848532 | 1.049073 | 0.862411 |
| MBT | LSTM | 1.154074 | 0.863362 | 1.067407 | 0.860618 |
| MBT | Naive | 1.165899 | 0.860082 | 1.063352 | 0.857747 |
| MEG | Lag Regression | 0.027705 | 0.019861 | 0.528699 | 0.922204 |
| MEG | ARIMA | 0.027725 | 0.019980 | 0.531887 | 0.922090 |
| MEG | LSTM | 0.028421 | 0.021143 | 0.562846 | 0.918128 |
| MEG | Naive | 0.027644 | 0.019465 | 0.518166 | 0.922545 |
| MER | Lag Regression | 10.089510 | 7.158533 | 1.502795 | 0.938409 |
| MER | ARIMA | 10.276873 | 7.187860 | 1.508952 | 0.936100 |
| MER | LSTM | 10.285031 | 7.213746 | 1.514386 | 0.935999 |
| MER | Naive | 10.153955 | 7.261728 | 1.524459 | 0.937620 |
| NIKL | Lag Regression | 0.151742 | 0.112017 | 1.232136 | 0.944437 |
| NIKL | ARIMA | 0.151775 | 0.112183 | 1.233960 | 0.944413 |
| NIKL | LSTM | 0.152097 | 0.112813 | 1.240887 | 0.944177 |
| NIKL | Naive | 0.151744 | 0.112016 | 1.232127 | 0.944435 |
| PGOLD | Lag Regression | 0.860882 | 0.634098 | 1.273827 | 0.897928 |
| PGOLD | ARIMA | 0.859415 | 0.632908 | 1.271435 | 0.898276 |
| PGOLD | LSTM | 0.864189 | 0.637594 | 1.280849 | 0.897143 |
| PGOLD | Naive | 0.860872 | 0.633951 | 1.273531 | 0.897931 |
| SCC | Lag Regression | 0.790923 | 0.464243 | 1.207325 | 0.970833 |
| SCC | ARIMA | 0.790114 | 0.462305 | 1.202284 | 0.970892 |
| SCC | LSTM | 0.791401 | 0.462398 | 1.202527 | 0.970797 |
| SCC | Naive | 0.790114 | 0.462305 | 1.202284 | 0.970892 |
| SECB | Lag Regression | 0.990407 | 0.725871 | 0.533501 | 0.897281 |
| SECB | ARIMA | 0.979271 | 0.709388 | 0.521387 | 0.899578 |
| SECB | LSTM | 0.984843 | 0.722544 | 0.531056 | 0.898432 |
| SECB | Naive | 0.979680 | 0.709053 | 0.521141 | 0.899494 |
| SHLPH | Lag Regression | 0.311960 | 0.184146 | 1.005264 | 0.976441 |
| SHLPH | ARIMA | 0.310804 | 0.186799 | 1.019752 | 0.976616 |
| SHLPH | LSTM | 0.311097 | 0.186933 | 1.020480 | 0.976572 |
| SHLPH | Naive | 0.310919 | 0.181893 | 0.992968 | 0.976598 |
| SMPH | Lag Regression | 0.415493 | 0.299057 | 0.557137 | 0.955926 |
| SMPH | ARIMA | 0.416930 | 0.299587 | 0.558124 | 0.955621 |
| SMPH | LSTM | 0.427910 | 0.306745 | 0.571459 | 0.953252 |
| SMPH | Naive | 0.422738 | 0.303128 | 0.564720 | 0.954375 |

## Table 4 2 Company comparisons with Naive

Adjusted p-values refer to squared-error Diebold–Mariano tests with Holm correction within each company. A significant improvement also requires lower mean loss. The results do not apply a single correction across all 15 companies.

| Company | Lowest RMSE including Naive | LIR p | ARIMA p | LSTM p | Significant improvement |
|---|---|---:|---:|---:|---|
| ALI | Lag Regression | 0.161469 | 0.101285 | 0.353499 | None |
| APX | Lag Regression | 0.694464 | 0.694464 | 0.747060 | None |
| BPI | ARIMA | 0.181543 | 0.247957 | 0.247957 | None |
| GLO | ARIMA/Naive | 1.000000 | 1.000000 | 0.016720 | None |
| ICT | LSTM | 0.041762 | 0.285980 | 0.157693 | Lag Regression |
| JFC | Naive | 0.324091 | 0.432730 | 0.027423 | None |
| MBT | ARIMA | 0.002594 | 0.095311 | 0.408407 | Lag Regression |
| MEG | Naive | 0.783467 | 0.783467 | 0.024175 | None |
| MER | Lag Regression | 1.000000 | 1.000000 | 1.000000 | None |
| NIKL | Lag Regression | 1.000000 | 1.000000 | 0.885210 | None |
| PGOLD | ARIMA | 1.000000 | 1.000000 | 1.000000 | None |
| SCC | ARIMA/Naive | 0.411111 | 1.000000 | 1.000000 | None |
| SECB | ARIMA | 0.649872 | 0.900440 | 0.900440 | None |
| SHLPH | ARIMA | 1.000000 | 1.000000 | 1.000000 | None |
| SMPH | Lag Regression | 0.265450 | 0.284759 | 0.457792 | None |

Among the three principal models, Lag Regression and ARIMA each obtained the lowest RMSE for seven companies, while LSTM ranked first for ICT. Neither model reached the predeclared eight-of-fifteen consistency threshold. Including Naive changes the counts to five outright wins each for Lag Regression and ARIMA, one for LSTM, two for Naive, and two ARIMA–Naive ties. The selected ARIMA(0,1,0) models for GLO and SCC generated the same forecasts as Naive.

Only Lag Regression passed the squared-error improvement gate for ICT (adjusted p = 0.041762) and MBT (adjusted p = 0.002594). Absolute-error robustness tests supported these two improvements. LSTM had significantly higher squared-error loss than Naive for GLO, JFC, and MEG. No company had two eligible principal models, so the conditional second-stage comparisons were correctly skipped. Their absence is not evidence that all principal-model pairs are equivalent.

## Table 4 3 Across company comparisons

| Comparison | Result |
|---|---|
| Friedman statistic on MASE | 13.013514 |
| Permutation p-value with 10000 permutations | 0.004300 |
| Conditional Wilcoxon post-hoc | Executed |
| Significant Holm-adjusted pairwise comparisons | None |

| Pair | Holm-adjusted p-value |
|---|---:|
| arima vs lstm | 0.254883 |
| arima vs naive | 1.000000 |
| lag_reg vs arima | 1.000000 |
| lag_reg vs lstm | 0.061523 |
| lag_reg vs naive | 1.000000 |
| lstm vs naive | 0.127869 |

The significant omnibus result indicates that the four methods do not have identical rank behavior across this sample. However, no individual post-hoc pair survived Holm correction. The smallest adjusted p-value was 0.061523 for Lag Regression versus LSTM. The findings therefore do not support a statistically established pairwise winner from the across-company analysis. Shared market conditions also limit the independence of the 15 companies.

## Table 4 4 Selected configurations

LSTM entries show lookback, hidden units, learning rate, and batch size. ARIMA trend n denotes none, c denotes constant, and t denotes linear trend. Epochs are the fixed complete-development refit counts.

| Company | LASSO alpha | Features retained | ARIMA order and trend | LSTM configuration | Epochs |
|---|---:|---:|---|---|---:|
| ALI | 0.1 | 0 | (1, 0, 0) n | 10 / 25 / 0.01 / 32 | 4 |
| APX | 0.01 | 0 | (1, 1, 1) n | 5 / 100 / 0.01 / 32 | 21 |
| BPI | 0.1 | 2 | (1, 1, 1) n | 30 / 25 / 0.01 / 16 | 8 |
| GLO | 15.848932 | 0 | (0, 1, 0) n | 10 / 25 / 0.001 / 32 | 10 |
| ICT | 0.39810717 | 3 | (1, 1, 1) t | 30 / 25 / 0.001 / 16 | 47 |
| JFC | 1.5848932 | 0 | (1, 0, 0) c | 20 / 25 / 0.001 / 32 | 65 |
| MBT | 0.1 | 1 | (2, 1, 0) n | 30 / 25 / 0.001 / 16 | 51 |
| MEG | 0.015848932 | 0 | (1, 0, 0) n | 30 / 25 / 0.001 / 16 | 2 |
| MER | 0.39810717 | 3 | (3, 1, 2) t | 10 / 50 / 0.01 / 16 | 43 |
| NIKL | 0.025118864 | 0 | (1, 0, 0) n | 10 / 25 / 0.001 / 32 | 8 |
| PGOLD | 0.15848932 | 0 | (1, 1, 0) n | 5 / 25 / 0.01 / 16 | 5 |
| SCC | 0.15848932 | 0 | (0, 1, 0) n | 30 / 25 / 0.001 / 32 | 14 |
| SECB | 0.25118864 | 1 | (1, 0, 0) n | 20 / 25 / 0.001 / 16 | 2 |
| SHLPH | 0.15848932 | 0 | (0, 1, 3) t | 10 / 25 / 0.001 / 16 | 11 |
| SMPH | 0.039810717 | 3 | (1, 0, 1) n | 10 / 25 / 0.01 / 16 | 11 |

All 48 LSTM configurations retained five-fold, three-seed evidence, totaling 720 tuning fits per company and 10,800 across the study. Validation dates were identical across lookbacks, and final fitting used the declared seed 42. Every LASSO search recorded 36 alpha candidates from 0.0001 to 1000; no winner lay at a boundary. Nine company regressions retained zero nonzero feature coefficients and thus predicted a constant change through the fitted intercept. This is a substantive finding, not evidence that technical indicators contributed to every final regression.

## Corporate actions and error diagnostics

The primary analysis retained every validated holdout observation. A separate sensitivity calculation excluded 26 company-specific registered event dates. Recomputed sensitivity metrics matched the stored outputs, and the lowest-RMSE model or tie was unchanged for all companies. This check supports stability of the descriptive ranking to those registered dates; it does not establish that all significance conclusions remain unchanged after exclusion.

Shapiro–Wilk tests rejected normality for all principal models in 14 companies; ALI was the exception. ARCH tests detected changing error variance for all principal models in ICT, PGOLD, and SCC, and for LSTM in BPI. No recorded Lag Regression or ARIMA Ljung–Box test rejected at the five-percent level. Failure to reject is not proof of independence. The declared HAC-adjusted comparisons and moving-block bootstrap address aspects of serial dependence, but the study remains a historical comparison with limited cross-company independence.

## Implications and limits

Complexity alone did not produce stronger forecasts. Naive remained competitive, and selected models often differed only slightly in error. High closing-price R² should not be interpreted as evidence of market-beating predictive skill: the benchmark also obtained high values. The valid conclusion is that improvement was company-specific and limited under the declared tests.

These findings concern frozen models evaluated on the stated historical holdout. Selecting a deployment model using these rankings does not create an independent performance estimate for that selected live policy. Prospective forecasts are needed to assess deployment performance. Further changes chosen after viewing these results would require a clearly separated analysis and must not be presented as predeclared choices.

## Evidence references

Run: FORMAL_CORRECTED_20260828_02. Code commit: bfb33b8c184c87cc8828af5529410da94addd71c. Source-data commit: 2e72058057f5ba2ef903147c8390c3f05f41ffe3.

All tables derive from formal_evidence/FORMAL_CORRECTED_20260828_02: per_company/*/metrics.json, per_company/*/diagnostics.json, statistical_tests.json, split_manifest.json, data_manifest.json, methodology_manifest.json, and finalized.json. Table 4 1 contains 60 metric rows; Tables 4 2 and 4 4 each contain 15 company rows. The local audit recomputed metrics and statistics without training models.

Archive SHA-256: 2b2ed0ca6b88ea6cfef5ac14013440da1c7c55c1d9f9640e04a595fdafca5d24.

Corporate-action registry SHA-256: 98199732d5372683c256d1e7c9b9ab1b9987b9f2a7240dbe0c43d00370d508eb.
