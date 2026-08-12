"""Shared helpers for the predictive_accuracy test suite."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # backend/

import numpy as np
import pandas as pd


def make_ohlcv(n: int = 250, seed: int = 42, start: str = "2022-01-03") -> pd.DataFrame:
    """A small deterministic synthetic OHLCV dataframe shaped like
    services/data_validator.py's output (business-day dates, sorted,
    Close-derived High/Low/Open, positive Volume) — big enough to exercise
    every model's minimum-history requirements without the cost of a full
    real ticker."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)
    steps = rng.normal(loc=0.02, scale=0.6, size=n)
    close = 50.0 + np.cumsum(steps)
    close = np.maximum(close, 1.0)
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.15, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.15, n))
    volume = rng.integers(10_000, 500_000, n).astype(float)

    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
