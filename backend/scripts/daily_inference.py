"""Daily inference runner: loads persisted weekly models and generates
fresh next-trading-day forecasts using the latest validated OHLCV data.

This module is deliberately separate from training:
  - It NEVER fits, retrains, or refits any model.
  - It ONLY loads already-persisted artifacts (from weekly training)
    and runs predict_next() on the newest data.
  - It updates prediction_cache/<TICKER>.json with fresh predictions
    while preserving all training-time metadata (metrics, backtests,
    model-selection results, statistical tests).

Intended to run Monday-Friday after new PSE EOD data is ingested,
*before* export_forecast_artifacts.py, so the frontend always serves
forecasts computed on the most recent close.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from services.data_validator import CSVValidationError, validate_ohlcv_csv
from services.forecasting import arima_model, lag_regression, lstm_model
from services.pdf_pipeline.config import TARGET_COMPANIES
from services.pse_calendar import get_calendar

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
# New weekly deployment artifacts are written here.  The three legacy
# directories remain readable while existing repositories are migrated.
DEPLOYMENT_CURRENT_DIR = MODELS_DIR / "deployment" / "current"
LAG_MODELS_DIR = MODELS_DIR / "lag_regression"
ARIMA_MODELS_DIR = MODELS_DIR / "arima"
LSTM_MODELS_DIR = MODELS_DIR / "lstm"
PREDICTION_CACHE_DIR = BASE_DIR / "prediction_cache"

PHT = timezone(timedelta(hours=8))  # Philippine Time (UTC+8, no DST)
MODEL_SOURCE = "weekly_persisted_artifacts"

# The 15-ticker universe the dashboard tracks. Sourced from the pdf
# pipeline's config (which is itself kept in sync with
# services/data_loader.py:COMPANY_META) so there is exactly one place
# that defines "the 15 companies" for the whole backend.
EXPECTED_TICKERS: list[str] = sorted(TARGET_COMPANIES.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cache(symbol: str) -> dict[str, Any]:
    path = PREDICTION_CACHE_DIR / f"{symbol}.json"
    if not path.exists():
        raise FileNotFoundError(f"Prediction cache missing for {symbol}: {path}")
    return json.loads(path.read_text())


def _write_cache(symbol: str, payload: dict) -> None:
    PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTION_CACHE_DIR / f"{symbol}.json"
    path.write_text(json.dumps(payload, indent=2))


def _deployment_artifact_path(model_directory: str, filename: str, legacy_dir: Path) -> Path:
    """Prefer deployment-current, then safely read the pre-Phase-6 layout."""
    current = DEPLOYMENT_CURRENT_DIR / model_directory / filename
    if current.is_file():
        return current
    return legacy_dir / filename


# ---------------------------------------------------------------------------
# Per-model inference helpers (no training, no refitting)
# ---------------------------------------------------------------------------


def _infer_lasso(symbol: str, df: pd.DataFrame) -> float:
    """Load persisted LASSO artifact and predict next Close.

    Uses the persisted StandardScaler and LassoCV model directly.
    Does NOT fit the scaler or model.
    """
    artifact_path = _deployment_artifact_path("lag_regression", f"{symbol}.pkl", LAG_MODELS_DIR)
    if not artifact_path.exists():
        raise FileNotFoundError(f"LASSO artifact missing for {symbol}: {artifact_path}")

    artifact = lag_regression.load(artifact_path)
    # predict_next uses the artifact scaler + model directly; no fit.
    return lag_regression.predict_next(artifact, df)


def _infer_lstm(symbol: str, df: pd.DataFrame) -> float:
    """Load persisted LSTM artifact and predict next Close.

    Uses the persisted MinMaxScaler, model weights, and lookback window.
    Does NOT retrain or refit.
    """
    artifact_path = _deployment_artifact_path("lstm", f"{symbol}.pth", LSTM_MODELS_DIR)
    if not artifact_path.exists():
        raise FileNotFoundError(f"LSTM artifact missing for {symbol}: {artifact_path}")

    artifact = lstm_model.load(artifact_path)
    # predict_next uses the artifact scaler + model weights directly; no fit.
    return lstm_model.predict_next(artifact, df)


def _infer_arima(symbol: str, df: pd.DataFrame) -> float:
    """Load persisted ARIMA model, validate lineage, append new observations,
    and forecast one step ahead.

    Lineage validation compares the current historical Close series against
    the original training observations stored in the persisted model
    (``model.model.endog``), NOT ``fittedvalues``.

    For genuinely new observations, updates the persisted ARIMA state with
    ``model.append(new_values, refit=False)``.  Never performs daily ARIMA
    parameter refitting.
    """
    model_path = _deployment_artifact_path("arima", f"{symbol}.pkl", ARIMA_MODELS_DIR)
    if not model_path.exists():
        raise FileNotFoundError(f"ARIMA artifact missing for {symbol}: {model_path}")

    model = arima_model.load(model_path)

    # ------------------------------------------------------------------
    # Lineage check: compare current historical data against the
    # original training observations (model.model.endog), NOT fittedvalues.
    # ------------------------------------------------------------------
    close = df["Close"].astype(float)
    endog = pd.Series(model.model.endog.flatten(), name="Close")
    n_endog = len(endog)

    if n_endog > len(close):
        raise ValueError(
            f"{symbol} ARIMA model was trained on {n_endog} obs, "
            f"but current data only has {len(close)} rows. "
            f"Model/data lineage is inconsistent — weekly retraining required."
        )

    historical_close = close.iloc[:n_endog].reset_index(drop=True)
    endog_aligned = endog.reset_index(drop=True)

    # Allow small numerical tolerance for floating-point differences
    diff = np.abs(historical_close.values - endog_aligned.values)
    max_diff = float(np.max(diff))
    if max_diff > 1e-3:
        raise ValueError(
            f"{symbol} ARIMA model/data lineage mismatch: "
            f"max historical Close diff = {max_diff:.4f} against "
            f"model.model.endog. The persisted model was not trained "
            f"on this data series — weekly retraining required."
        )

    # ------------------------------------------------------------------
    # Append new observations (if any) without refitting
    # ------------------------------------------------------------------
    new_close = close.iloc[n_endog:]
    if len(new_close) > 0:
        log.info("%s ARIMA: appending %d new observation(s)", symbol, len(new_close))
        model = model.append(new_close.values, refit=False)

    # Forecast exactly one step ahead
    return arima_model.predict_next(model)


# ---------------------------------------------------------------------------
# Per-symbol inference
# ---------------------------------------------------------------------------


def infer_symbol(
    symbol: str, df: pd.DataFrame, inference_at: datetime | None = None
) -> dict[str, Any]:
    """Run daily inference for one ticker and return the updated cache payload.

    Preserves all existing metadata (metrics, backtest series, statistical
    tests) and only updates:
      - next_close predictions from each model
      - inference metadata (data_as_of, inference_at, forecast_for, etc.)
    """
    inference_at = inference_at or datetime.now(PHT)

    # data_as_of = latest actual PSE trading date in the raw data
    data_as_of_ts = df["Date"].max()
    if hasattr(data_as_of_ts, "to_pydatetime"):
        data_as_of = data_as_of_ts.to_pydatetime().date()
    elif isinstance(data_as_of_ts, pd.Timestamp):
        data_as_of = data_as_of_ts.date()
    else:
        data_as_of = pd.to_datetime(data_as_of_ts).date()

    # forecast_for = next actual PSE trading session (handles PH holidays)
    calendar = get_calendar()
    forecast_for = calendar.next_trading_day(data_as_of)

    cache = _load_cache(symbol)

    # Run inference for each model — fail the whole symbol if any model fails
    next_lag = _infer_lasso(symbol, df)
    next_arima = _infer_arima(symbol, df)
    next_lstm = _infer_lstm(symbol, df)

    # Preserve existing metrics/backtest/model-selection metadata;
    # overwrite only the prediction values and inference metadata.
    cache["next_close"] = {
        "lag": round(float(next_lag), 2),
        "arima": round(float(next_arima), 2),
        "lstm": round(float(next_lstm), 2),
    }

    cache["inference_metadata"] = {
        "data_as_of": data_as_of.isoformat(),
        "forecast_for": forecast_for.isoformat(),
        "inference_at": inference_at.isoformat(timespec="seconds"),
        "models_retrained": False,
        "model_source": MODEL_SOURCE,
    }

    return cache


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------


def run_daily_inference(
    raw_dir: Path = RAW_DIR,
    symbols: list[str] | None = None,
    enforce_universe: bool | None = None,
) -> dict[str, Any]:
    """Run daily inference for all (or specified) tickers.

    Returns a result dict with status, per-symbol outcomes, and any failures.
    On any failure the workflow should halt — never silently fall back
    to stale predictions.

    When ``symbols`` is not given (the production default — infer for
    every ticker), the full 15-ticker universe (EXPECTED_TICKERS) is
    enforced: any ticker with no raw CSV at all is recorded as a failure
    rather than silently skipped, so a run can never report success with
    fewer than 15/15 tickers. Pass ``enforce_universe=False`` to opt out
    (used by tests that intentionally exercise a smaller ticker set).
    """
    inference_at = datetime.now(PHT)
    results: dict[str, Any] = {
        "status": "ok",
        "inference_at": inference_at.isoformat(timespec="seconds"),
        "symbols_processed": [],
        "symbols_failed": {},
    }

    if symbols is None:
        symbols = list(EXPECTED_TICKERS)
        if enforce_universe is None:
            enforce_universe = True
    elif enforce_universe is None:
        enforce_universe = False

    csv_paths = [raw_dir / f"{s}.csv" for s in symbols]

    for csv_path in csv_paths:
        symbol = csv_path.stem
        if not csv_path.exists():
            if enforce_universe and symbol in EXPECTED_TICKERS:
                results["symbols_failed"][symbol] = (
                    f"Expected ticker missing entirely — no raw data at {csv_path}"
                )
            else:
                results["symbols_failed"][symbol] = "CSV not found"
            continue

        try:
            df = validate_ohlcv_csv(csv_path)
        except CSVValidationError as exc:
            results["symbols_failed"][symbol] = f"CSV validation failed: {exc}"
            continue
        except Exception as exc:
            results["symbols_failed"][symbol] = f"Unexpected CSV load error: {exc}"
            continue

        try:
            updated_cache = infer_symbol(symbol, df, inference_at=inference_at)
            _write_cache(symbol, updated_cache)
            results["symbols_processed"].append(symbol)
            log.info(
                "%s inference OK → lag=%.2f arima=%.2f lstm=%.2f (data_as_of=%s forecast_for=%s)",
                symbol,
                updated_cache["next_close"]["lag"],
                updated_cache["next_close"]["arima"],
                updated_cache["next_close"]["lstm"],
                updated_cache["inference_metadata"]["data_as_of"],
                updated_cache["inference_metadata"]["forecast_for"],
            )
        except Exception as exc:
            log.exception("Daily inference failed for %s", symbol)
            results["symbols_failed"][symbol] = str(exc)

    if results["symbols_failed"]:
        results["status"] = (
            "partial_failure" if results["symbols_processed"] else "failure"
        )

    return results


def main() -> int:
    """CLI entrypoint for the daily inference step."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = run_daily_inference()

    print("=" * 60)
    print(f"Daily inference: {result['status']}")
    print(f"Processed : {len(result['symbols_processed'])} symbol(s)")
    if result["symbols_failed"]:
        print(f"Failed    : {len(result['symbols_failed'])} symbol(s)")
        for sym, err in result["symbols_failed"].items():
            print(f"  {sym}: {err}")
    print("=" * 60)

    if result["status"] in ("failure", "partial_failure"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
