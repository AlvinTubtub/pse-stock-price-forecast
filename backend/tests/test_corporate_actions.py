"""Corporate-action policy and sensitivity evidence tests."""
from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.corporate_actions import build_corporate_action_evidence, load_verified_event_registry


def _forecasts() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2026-01-02", periods=4, freq="B")
    result = {}
    for model in ("lag_reg", "arima", "lstm", "naive"):
        result[model] = pd.DataFrame({
            "target_date": dates,
            "actual_close": [10.0, 12.0, 11.0, 13.0],
            "predicted_close": [10.1, 11.5, 11.2, 12.8],
        })
    return result


def test_primary_policy_never_removes_rows_and_empty_registry_is_explicit():
    evidence = build_corporate_action_evidence(_forecasts(), [8.0, 9.0, 10.0])
    assert evidence["primary_rows_excluded"] == 0
    assert evidence["automatic_outlier_or_error_exclusion"] is False
    assert evidence["verified_events"] == []
    assert evidence["sensitivity_analysis"]["status"] == "not_applicable_no_verified_holdout_events"


def test_verified_holdout_event_produces_separate_sensitivity_metrics():
    evidence = build_corporate_action_evidence(
        _forecasts(),
        [8.0, 9.0, 10.0],
        verified_events=[{
            "event_date": "2026-01-05",
            "event_type": "stock_split",
            "source": "verified exchange disclosure",
        }],
    )
    sensitivity = evidence["sensitivity_analysis"]
    assert sensitivity["status"] == "complete"
    assert sensitivity["excluded_target_dates"] == ["2026-01-05"]
    assert sensitivity["remaining_holdout_count"] == 3
    assert set(sensitivity["metrics"]) == {"lag_reg", "arima", "lstm", "naive"}


def test_reviewed_registry_is_hashed_and_grouped(tmp_path):
    registry = tmp_path / "corporate_actions.json"
    registry.write_text(json.dumps({
        "schema_version": 1,
        "review_status": "complete",
        "reviewed_symbols": ["BPI", "ALI"],
        "review_period": {"start_date": "2025-09-02", "end_date": "2026-08-28"},
        "review_scope": ["dividends", "rights", "stock splits"],
        "events": [{
            "symbol": "BPI",
            "event_date": "2026-01-05",
            "event_type": "stock_split",
            "source": "exchange disclosure",
        }],
    }))
    grouped, provenance = load_verified_event_registry(
        registry,
        ["BPI", "ALI"],
        required_start_date="2025-09-02",
        required_end_date="2026-08-28",
    )
    assert len(grouped["BPI"]) == 1 and grouped["ALI"] == ()
    assert provenance["status"] == "complete"
    assert provenance["event_count"] == 1
    assert provenance["reviewed_symbols"] == ["BPI", "ALI"]
    assert len(provenance["sha256"]) == 64


def test_registry_rejects_missing_symbol_or_holdout_coverage(tmp_path):
    registry = tmp_path / "corporate_actions.json"
    registry.write_text(json.dumps({
        "schema_version": 1,
        "review_status": "complete",
        "reviewed_symbols": ["BPI"],
        "review_period": {"start_date": "2025-10-01", "end_date": "2026-08-01"},
        "review_scope": ["dividends", "rights", "stock splits"],
        "events": [],
    }))
    try:
        load_verified_event_registry(registry, ["BPI", "ALI"])
    except ValueError as exc:
        assert "ALI" in str(exc)
    else:
        raise AssertionError("Missing reviewed symbol must fail.")

    try:
        load_verified_event_registry(
            registry,
            ["BPI"],
            required_start_date="2025-09-02",
            required_end_date="2026-08-28",
        )
    except ValueError as exc:
        assert "formal holdout period" in str(exc)
    else:
        raise AssertionError("Incomplete review period must fail.")


def test_frozen_formal_registry_covers_all_companies_and_holdout_dates():
    registry = Path(__file__).resolve().parent.parent / "config" / (
        "formal_corporate_actions_20250902_20260828.json"
    )
    symbols = [
        "ALI", "APX", "BPI", "GLO", "ICT", "JFC", "MBT", "MEG",
        "MER", "NIKL", "PGOLD", "SCC", "SECB", "SHLPH", "SMPH",
    ]
    grouped, provenance = load_verified_event_registry(
        registry,
        symbols,
        required_start_date="2025-09-02",
        required_end_date="2026-08-28",
    )
    assert set(grouped) == set(symbols)
    assert provenance["event_count"] == 26
    assert provenance["selected_event_count"] == 26
