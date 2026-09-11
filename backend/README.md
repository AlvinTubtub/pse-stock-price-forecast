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

Phase 5 adds a fresh statsmodels ARIMA implementation alongside Lag-Informed Regression. ARIMA models the chronological Close series, searches an explicit order/trend grid with expanding-window one-step validation, requires convergence evidence, and updates state without refitting coefficients during evaluation. LSTM is intentionally not implemented yet.

Lag-Informed Regression keeps evaluation and production fitting separate. Reproducibility metadata can be written only under the ignored `artifacts/lir/` directory.

ARIMA likewise keeps evaluation and production fitting separate. Its reproducibility metadata can be written only under the ignored `artifacts/arima/` directory.

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
