# Final Unseen-Test Evaluation Summary

- Tickers evaluated: 15 (ALI, APX, BPI, GLO, ICT, JFC, MBT, MEG, MER, NIKL, PGOLD, SCC, SECB, SHLPH, SMPH)
- Final-test fraction: 0.15
- Random seed: 42

## Aggregate ranking (mean RMSE across tickers, ascending)

| Rank | Model | Mean RMSE | Selection frequency | Mean RMSE improvement vs naive |
|---|---|---|---|---|
| 1 | ARIMA | 4.3895 | 40% | +0.20%
| 2 | Naive baseline | 4.4131 | - | - |
| 3 | Lag-Informed Regression (LASSO) | 7.8471 | 47% | -19.41%
| 4 | LSTM | 14.7503 | 13% | -85.94%

## Statistical significance

- Friedman test across tickers (tuned models only): statistic=10.1333, p=0.006303, n_tickers=15
- Best-model consistency: lag_reg is lowest-RMSE on 7/15 tickers (pass=False, threshold=8)

See `metrics.csv`/`metrics.json` for per-ticker detail and `statistical_tests.json` for the full pairwise Diebold-Mariano, Wilcoxon-Holm, and Friedman output.
