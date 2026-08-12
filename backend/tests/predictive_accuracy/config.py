"""Configuration for the final unseen-test predictive-accuracy evaluation suite.

Nothing about *what* gets evaluated (the final-test window, the ticker
universe, the random seed) is hard-coded into the evaluation logic itself —
everything here is a constant that can be overridden with an environment
variable, so the same code runs a smoke-test subset in CI and the full
15-ticker canonical universe for a release evaluation.

See ``README.md`` in this directory for the full list of overrides and
example commands.
"""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BACKEND_DIR / "data" / "raw"
SUITE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SUITE_DIR / "results"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


# --------------------------------------------------------------------------
# Final-test window (configurable — never hard-coded downstream)
# --------------------------------------------------------------------------
# Fraction of each ticker's chronological history reserved as the FINAL
# unseen test set (applied after everything upstream — training,
# validation, hyperparameter/model selection, scaler fitting — has already
# been restricted to what comes before it). Override with
# PA_FINAL_TEST_FRACTION.
FINAL_TEST_FRACTION = _env_float("PA_FINAL_TEST_FRACTION", 0.15)

# Optional explicit final-test window (YYYY-MM-DD). When
# PA_FINAL_TEST_START_DATE is set it takes precedence over
# PA_FINAL_TEST_FRACTION for every ticker's split. PA_FINAL_TEST_END_DATE
# defaults to each ticker's most recent available date when unset.
FINAL_TEST_START_DATE = os.environ.get("PA_FINAL_TEST_START_DATE") or None
FINAL_TEST_END_DATE = os.environ.get("PA_FINAL_TEST_END_DATE") or None

MIN_TRAINVAL_ROWS = _env_int("PA_MIN_TRAINVAL_ROWS", 60)
MIN_FINAL_TEST_ROWS = _env_int("PA_MIN_FINAL_TEST_ROWS", 5)

# --------------------------------------------------------------------------
# Ticker universe
# --------------------------------------------------------------------------
# Defaults to every CSV present in data/raw/ (i.e. every currently supported
# PSE ticker). Override with a comma-separated PA_TICKERS, e.g.
# "PA_TICKERS=BPI,ALI" for a fast local smoke test.
_tickers_env = os.environ.get("PA_TICKERS")
TICKERS: list[str] = (
    [t.strip().upper() for t in _tickers_env.split(",") if t.strip()]
    if _tickers_env
    else sorted(p.stem.upper() for p in RAW_DATA_DIR.glob("*.csv"))
)

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
MODEL_KEYS = ("lag_reg", "arima", "lstm")
ALL_MODEL_KEYS = ("naive",) + MODEL_KEYS  # naive always evaluated alongside

MODEL_LABELS = {
    "lag_reg": "Lag-Informed Regression (LASSO)",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "naive": "Naive baseline",
}

# --------------------------------------------------------------------------
# Statistics / reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = _env_int("PA_SEED", 42)

# Minimum number of tickers a model must dominate (lowest RMSE) on to be
# called "consistently best" by the best-model-consistency check. Defaults
# to a majority of whatever ticker universe is configured.
MIN_CONSISTENCY_COMPANIES = _env_int(
    "PA_MIN_CONSISTENCY_COMPANIES", max(1, (len(TICKERS) // 2) + 1) if TICKERS else 1
)

BOOTSTRAP_ITERATIONS = _env_int("PA_BOOTSTRAP_ITERATIONS", 2000)
CONFIDENCE_LEVEL = _env_float("PA_CONFIDENCE_LEVEL", 0.95)
MIN_ROWS_FOR_CI = 8  # below this, a bootstrap CI isn't meaningful — omitted, not fabricated
