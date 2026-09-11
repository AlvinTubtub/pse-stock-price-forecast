# ForecastPH backend

This directory is the clean foundation for the ForecastPH forecasting pipeline.

## Layout

- `data/raw/`: preserved source market data. Pipeline code must treat these files as immutable inputs.
- `config/`: pipeline configuration added in later phases.
- `src/data/`: data loading and validation.
- `src/features/`: feature engineering.
- `src/models/`: model definitions.
- `src/training/`: training orchestration.
- `src/evaluation/`: backtesting and metrics.
- `src/inference/`: next-session inference.
- `src/export/`: frontend-contract export logic.
- `artifacts/`: ignored runtime outputs such as trained models, metrics, and intermediate data.
- `scripts/`: command-line entry points added in later phases.
- `tests/`: backend tests added with each implementation phase.

## Current status

Phase 4 adds a fresh Lag-Informed Regression implementation. It predicts next-session close deltas with causal OHLCV features, fold-local PACF lag selection, fold-local standardization, expanding-window alpha tuning, and LASSO as the final estimator. ARIMA and LSTM are intentionally not implemented yet.

Lag-Informed Regression keeps evaluation and production fitting separate. Reproducibility metadata can be written only under the ignored `artifacts/lir/` directory.

## Development setup

Create a Python 3.11 or newer virtual environment, then install the requirements from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run tests from `backend/` after tests are introduced:

```bash
python -m pytest
```
