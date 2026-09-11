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

Phase 7 adds unified chronological evaluation for Lag-Informed Regression, ARIMA, LSTM, and a date-indexed naive persistence benchmark. The split is rebuilt from current raw history, all model outputs must cover the exact same evaluation target dates, and malformed or incomplete model/date combinations fail before metrics are calculated.

Lag-Informed Regression keeps evaluation and production fitting separate. Reproducibility metadata can be written only under the ignored `artifacts/lir/` directory.

ARIMA likewise keeps evaluation and production fitting separate. Its reproducibility metadata can be written only under the ignored `artifacts/arima/` directory.

LSTM early stopping uses a chronological tail inside each training block. Each final fit first selects an epoch count, then creates a fresh scaler and model and trains on the entire development or production block. Metadata and PyTorch state are written only under the ignored `artifacts/lstm/` directory.

PyTorch deterministic algorithms and fixed seeds are enabled. Exact floating-point results can still vary across PyTorch releases, operating systems, CPU architectures, and other hardware/software differences.

Unified evaluation emits canonical one-step-ahead records and computes full-precision RMSE, MAE, MASE, and R-squared. Every model uses the same MASE denominator calculated once from the company's development Close series. Principal models are ranked by evaluation RMSE; production refitting remains a separate later operation.

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
