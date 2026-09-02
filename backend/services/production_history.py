"""Immutable-issued production forecast ledger and leakage-safe reconciliation."""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

MODEL_LABELS = ("Lag-Informed Regression", "ARIMA", "LSTM")


class ProductionHistoryError(ValueError):
    """Raised when a production forecast record cannot be trusted."""


def _path(history_dir: Path, symbol: str) -> Path:
    return history_dir / f"{symbol.upper()}.json"


def _empty(symbol: str) -> dict:
    return {"schema_version": 1, "symbol": symbol.upper(), "records": []}


def load_history(history_dir: Path, symbol: str) -> dict:
    path = _path(history_dir, symbol)
    if not path.exists():
        return _empty(symbol)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ProductionHistoryError(f"{symbol}: invalid production history JSON.") from exc
    if payload.get("schema_version") != 1 or payload.get("symbol") != symbol.upper() or not isinstance(payload.get("records"), list):
        raise ProductionHistoryError(f"{symbol}: malformed production history schema.")
    _validate_records(payload["records"], symbol)
    return payload


def _validate_records(records: list[dict], symbol: str) -> None:
    dates: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise ProductionHistoryError(f"{symbol}: production record must be an object.")
        date = record.get("target_date")
        if not isinstance(date, str):
            raise ProductionHistoryError(f"{symbol}: production record has no target_date.")
        pd.to_datetime(date, format="%Y-%m-%d", errors="raise")
        predictions = record.get("predictions")
        if not isinstance(predictions, dict) or set(predictions) != set(MODEL_LABELS):
            raise ProductionHistoryError(f"{symbol}/{date}: predictions must contain exactly the three production models.")
        if not all(math.isfinite(float(value)) for value in predictions.values()):
            raise ProductionHistoryError(f"{symbol}/{date}: prediction contains a non-finite value.")
        actual = record.get("actual")
        if actual is not None and not math.isfinite(float(actual)):
            raise ProductionHistoryError(f"{symbol}/{date}: actual contains a non-finite value.")
        dates.append(date)
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ProductionHistoryError(f"{symbol}: records must have unique chronological target dates.")


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def record_issued_forecast(
    history_dir: Path,
    symbol: str,
    *,
    target_date: str,
    issued_at: str,
    data_as_of: str,
    predictions: dict[str, float],
) -> bool:
    """Append one issued forecast, or reject a conflicting duplicate target."""
    record = {
        "target_date": target_date,
        "issued_at": issued_at,
        "data_as_of": data_as_of,
        "predictions": {label: float(predictions[label]) for label in MODEL_LABELS},
        "actual": None,
        "reconciled_at": None,
    }
    _validate_records([record], symbol)
    history = load_history(history_dir, symbol)
    existing = next((row for row in history["records"] if row["target_date"] == target_date), None)
    if existing is not None:
        immutable = ("target_date", "issued_at", "data_as_of", "predictions")
        if any(existing.get(key) != record[key] for key in immutable):
            raise ProductionHistoryError(
                f"{symbol}/{target_date}: refusing to replace an already-issued production forecast."
            )
        return False
    history["records"].append(record)
    history["records"].sort(key=lambda row: row["target_date"])
    _atomic_write(_path(history_dir, symbol), history)
    return True


def reconcile_history(history_dir: Path, symbol: str, df: pd.DataFrame, reconciled_at: str) -> int:
    """Fill actual closes only for forecasts issued before their target closes."""
    history = load_history(history_dir, symbol)
    actual_by_date = {
        pd.Timestamp(row.Date).strftime("%Y-%m-%d"): float(row.Close)
        for row in df[["Date", "Close"]].itertuples(index=False)
    }
    changed = 0
    for record in history["records"]:
        if record["actual"] is None and record["target_date"] in actual_by_date:
            record["actual"] = actual_by_date[record["target_date"]]
            record["reconciled_at"] = reconciled_at
            changed += 1
    if changed:
        _atomic_write(_path(history_dir, symbol), history)
    return changed
