"""Predeclared corporate-action policy evidence for formal evaluation.

The capstone target is the raw quoted closing price.  Valid observations are
therefore retained in the primary analysis.  Verified corporate-action target
dates may be supplied for a secondary sensitivity analysis, but large errors
or price jumps are never treated as proof of a corporate action.
"""
from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path

import pandas as pd

from services.evaluation import compute_canonical_formal_metrics


POLICY_ID = "raw_close_retain_and_flag_v1"


def load_verified_event_registry(
    path: Path | None,
    symbols: Iterable[str],
    *,
    required_start_date: object | None = None,
    required_end_date: object | None = None,
) -> tuple[dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """Load an optional registry and prove its symbol/date review coverage."""
    requested = tuple(str(symbol).upper() for symbol in symbols)
    empty = {symbol: () for symbol in requested}
    if path is None:
        return empty, {"status": "not_supplied", "path": None, "sha256": None}
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read corporate-action registry {path}.") from exc
    if payload.get("schema_version") != 1 or payload.get("review_status") != "complete":
        raise ValueError("Corporate-action registry must have schema_version=1 and review_status='complete'.")
    reviewed_symbols_raw = payload.get("reviewed_symbols")
    if not isinstance(reviewed_symbols_raw, list) or not reviewed_symbols_raw:
        raise ValueError("Corporate-action registry reviewed_symbols must be a non-empty list.")
    reviewed_symbols = tuple(str(symbol).upper() for symbol in reviewed_symbols_raw)
    if len(set(reviewed_symbols)) != len(reviewed_symbols):
        raise ValueError("Corporate-action registry reviewed_symbols contains duplicates.")
    missing_symbols = sorted(set(requested) - set(reviewed_symbols))
    if missing_symbols:
        raise ValueError(
            "Corporate-action registry does not document review coverage for: "
            + ", ".join(missing_symbols)
        )
    review_period = payload.get("review_period")
    if not isinstance(review_period, dict):
        raise ValueError("Corporate-action registry review_period must be an object.")
    try:
        review_start = pd.Timestamp(review_period["start_date"]).normalize()
        review_end = pd.Timestamp(review_period["end_date"]).normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Corporate-action review_period requires valid start_date and end_date.") from exc
    if review_start > review_end:
        raise ValueError("Corporate-action registry review_period start_date must not follow end_date.")
    if required_start_date is not None and review_start > pd.Timestamp(required_start_date).normalize():
        raise ValueError("Corporate-action registry starts after the formal holdout period.")
    if required_end_date is not None and review_end < pd.Timestamp(required_end_date).normalize():
        raise ValueError("Corporate-action registry ends before the formal holdout period.")
    review_scope = payload.get("review_scope")
    if not isinstance(review_scope, list):
        raise ValueError("Corporate-action registry review_scope must be a list.")
    scope_text = " ".join(str(item).lower() for item in review_scope)
    missing_scope = [term for term in ("dividend", "rights", "split") if term not in scope_text]
    if missing_scope:
        raise ValueError(
            "Corporate-action registry review_scope is missing: " + ", ".join(missing_scope)
        )
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("Corporate-action registry events must be a list.")
    grouped: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in requested}
    seen_event_dates: set[tuple[str, pd.Timestamp]] = set()
    for event in raw_events:
        if not isinstance(event, dict):
            raise ValueError("Every corporate-action registry event must be an object.")
        symbol = str(event.get("symbol", "")).upper()
        if symbol not in reviewed_symbols:
            raise ValueError(f"Corporate-action event symbol {symbol!r} was not documented as reviewed.")
        try:
            event_date = pd.Timestamp(event["event_date"]).normalize()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Every corporate-action event requires a valid event_date.") from exc
        if event_date < review_start or event_date > review_end:
            raise ValueError(f"Corporate-action event {symbol} {event_date.date()} is outside review_period.")
        event_type = str(event.get("event_type", "")).strip()
        source = str(event.get("source", "")).strip()
        if not event_type or not source:
            raise ValueError("Every corporate-action event requires non-empty event_type and source.")
        event_key = (symbol, event_date)
        if event_key in seen_event_dates:
            raise ValueError(f"Duplicate corporate-action event date for {symbol}: {event_date.date()}.")
        seen_event_dates.add(event_key)
        if symbol in grouped:
            grouped[symbol].append(event)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (
        {symbol: tuple(events) for symbol, events in grouped.items()},
        {
            "status": "complete",
            "path": str(path),
            "sha256": digest,
            "event_count": len(raw_events),
            "selected_event_count": sum(len(events) for events in grouped.values()),
            "reviewed_symbols": list(reviewed_symbols),
            "review_period": {
                "start_date": str(review_start.date()),
                "end_date": str(review_end.date()),
            },
        },
    )


def build_corporate_action_evidence(
    forecasts: dict[str, pd.DataFrame],
    development_close: Iterable[float],
    *,
    verified_events: Iterable[dict[str, object]] = (),
) -> dict[str, object]:
    """Record the primary policy and optional event-exclusion sensitivity.

    Each verified event must contain ``event_date``, ``event_type``, and
    ``source``.  Event dates are interpreted as forecast target dates.
    """
    events: list[dict[str, str]] = []
    seen: set[pd.Timestamp] = set()
    for raw in verified_events:
        try:
            date = pd.Timestamp(raw["event_date"]).normalize()
            event_type = str(raw["event_type"]).strip()
            source = str(raw["source"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Corporate-action events require event_date, event_type, and source.") from exc
        if not event_type or not source:
            raise ValueError("Corporate-action event_type and source must be non-empty.")
        if date in seen:
            raise ValueError(f"Duplicate verified corporate-action date: {date.date()}.")
        seen.add(date)
        events.append({
            "event_date": str(date.date()),
            "event_type": event_type,
            "source": source,
        })

    excluded_dates = {pd.Timestamp(event["event_date"]) for event in events}
    filtered = {
        model: frame.loc[~pd.to_datetime(frame["target_date"]).dt.normalize().isin(excluded_dates)].copy()
        for model, frame in forecasts.items()
    }
    holdout_dates = set(pd.to_datetime(next(iter(forecasts.values()))["target_date"]).dt.normalize())
    relevant_events = sorted(str(date.date()) for date in excluded_dates & holdout_dates)

    sensitivity: dict[str, object]
    if not relevant_events:
        sensitivity = {
            "status": "not_applicable_no_verified_holdout_events",
            "excluded_target_dates": [],
            "remaining_holdout_count": len(next(iter(forecasts.values()))),
            "metrics": None,
        }
    elif len(next(iter(filtered.values()))) < 2:
        sensitivity = {
            "status": "not_computable_insufficient_remaining_rows",
            "excluded_target_dates": relevant_events,
            "remaining_holdout_count": len(next(iter(filtered.values()))),
            "metrics": None,
        }
    else:
        _denominator, metrics = compute_canonical_formal_metrics(
            filtered,
            development_close,
        )
        sensitivity = {
            "status": "complete",
            "excluded_target_dates": relevant_events,
            "remaining_holdout_count": len(next(iter(filtered.values()))),
            "metrics": metrics,
        }

    return {
        "policy_id": POLICY_ID,
        "target_series": "raw_quoted_close",
        "primary_analysis": "retain_all_validated_observations",
        "automatic_outlier_or_error_exclusion": False,
        "event_date_semantics": "forecast_target_date",
        "verified_events": events,
        "primary_rows_excluded": 0,
        "sensitivity_analysis": sensitivity,
    }
