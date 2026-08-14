# Final Unseen-Test Evaluation Summary

- Tickers evaluated: 15 (ALI, APX, BPI, GLO, ICT, JFC, MBT, MEG, MER, NIKL, PGOLD, SCC, SECB, SHLPH, SMPH)
- Final-test fraction: 0.15
- Random seed: 42

## Aggregate ranking (mean RMSE across tickers, ascending)

| Rank | Model | Mean RMSE | Selection frequency | Mean RMSE improvement vs naive |
|---|---|---|---|---|
| 1 | ARIMA | 4.3929 | 47% | -0.26%
| 2 | Naive baseline | 4.4140 | - | - |
| 3 | LSTM | 8.5076 | 13% | -32.80%
| 4 | Lag-Informed Regression (LASSO) | 8.7642 | 40% | -24.67%

## Statistical significance

- Friedman test across tickers (tuned models only): statistic=8.4000, p=0.015, n_tickers=15
- Best-model consistency: arima is lowest-RMSE on 7/15 tickers (pass=False, threshold=8)

See `metrics.csv`/`metrics.json` for per-ticker detail and `statistical_tests.json` for the full pairwise Diebold-Mariano, Wilcoxon-Holm, and Friedman output.
