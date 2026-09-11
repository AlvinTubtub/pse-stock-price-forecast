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

Phase 6 adds one fresh univariate PyTorch LSTM alongside Lag-Informed Regression and ARIMA. It consumes only historical daily Close deltas and predicts the next daily Close delta. Every lookback is compared on validation target dates derived from the maximum configured lookback.

Lag-Informed Regression keeps evaluation and production fitting separate. Reproducibility metadata can be written only under the ignored `artifacts/lir/` directory.

ARIMA likewise keeps evaluation and production fitting separate. Its reproducibility metadata can be written only under the ignored `artifacts/arima/` directory.

LSTM early stopping uses a chronological tail inside each training block. Each final fit first selects an epoch count, then creates a fresh scaler and model and trains on the entire development or production block. Metadata and PyTorch state are written only under the ignored `artifacts/lstm/` directory.

PyTorch deterministic algorithms and fixed seeds are enabled. Exact floating-point results can still vary across PyTorch releases, operating systems, CPU architectures, and other hardware/software differences.

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
