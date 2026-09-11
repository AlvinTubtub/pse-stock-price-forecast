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
- `artifacts/`: ignored generated outputs split into `models/`, `evaluations/`, `forecasts/`, and `logs/`.
- `scripts/`: command-line entry points added in later phases.
- `tests/`: backend tests added with each implementation phase.

## Current status

Phase 9 adds one lightweight generated-artifact layout. Ordinary training-run manifests in `artifacts/logs/` record timing, environment versions, the source-data boundary, processed symbols, status, and errors. They are operational records only, not immutable formal experiments.

Phase 8 adds fresh production refitting and next-PSE-session inference. Selected hyperparameters are carried forward from chronological evaluation, but every principal model, scaler, and fitted state is rebuilt using all currently available raw history.

Lag-Informed Regression keeps evaluation and production fitting separate. Generated evaluation metadata is written under `artifacts/evaluations/lir/`.

ARIMA likewise keeps evaluation and production fitting separate. Generated evaluation metadata is written under `artifacts/evaluations/arima/`.

LSTM early stopping uses a chronological tail inside each training block. Each final fit first selects an epoch count, then creates a fresh scaler and model and trains on the entire development or production block. Generated evaluation metadata and state are written under `artifacts/evaluations/lstm/`.

PyTorch deterministic algorithms and fixed seeds are enabled. Exact floating-point results can still vary across PyTorch releases, operating systems, CPU architectures, and other hardware/software differences.

Unified evaluation emits canonical one-step-ahead records and computes full-precision RMSE, MAE, MASE, and R-squared. Every model uses the same MASE denominator calculated once from the company's development Close series. Principal models are ranked by evaluation RMSE; production refitting remains a separate later operation.

Production model files and compatibility metadata live only under `artifacts/models/{SYMBOL}/`. Fresh refits never load existing files. Inference validates schema, implementation version, model identity, artifact format, SHA-256, training boundary, row count, and family-specific model state before predicting. A configured PSE calendar determines the next trading session.

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

Safely clear generated artifacts and recreate the four artifact directories:

```bash
python -m scripts.reset_artifacts
```

The reset command validates that its target is exactly `backend/artifacts`. It never deletes or rewrites `data/raw`.
