"""Production forecast history must remain issued-first and leakage-safe."""
from __future__ import annotations

import pandas as pd
import pytest

from services.production_history import ProductionHistoryError, load_history, record_issued_forecast, reconcile_history


def _predictions() -> dict[str, float]:
    return {"Lag-Informed Regression": 65.41, "ARIMA": 65.44, "LSTM": 65.40}


def test_issued_forecast_is_immutable_and_reconciled_only_after_target_close(tmp_path):
    history = tmp_path / "production_history"
    assert record_issued_forecast(
        history, "BPI", target_date="2026-09-03", issued_at="2026-09-02T16:01:42+08:00",
        data_as_of="2026-09-02", predictions=_predictions(),
    )
    frame = pd.DataFrame({"Date": pd.to_datetime(["2026-09-02", "2026-09-03"]), "Close": [64.9, 65.2]})
    assert reconcile_history(history, "BPI", frame, "2026-09-03T16:00:00+08:00") == 1
    record = load_history(history, "BPI")["records"][0]
    assert record["predictions"] == _predictions()
    assert record["actual"] == 65.2
    assert record["reconciled_at"] == "2026-09-03T16:00:00+08:00"


def test_duplicate_target_is_idempotent_but_conflicting_forecast_is_rejected(tmp_path):
    history = tmp_path / "production_history"
    kwargs = dict(target_date="2026-09-03", issued_at="2026-09-02T16:01:42+08:00", data_as_of="2026-09-02", predictions=_predictions())
    assert record_issued_forecast(history, "BPI", **kwargs)
    assert not record_issued_forecast(history, "BPI", **kwargs)
    changed = _predictions(); changed["ARIMA"] = 66.0
    with pytest.raises(ProductionHistoryError, match="refusing to replace"):
        record_issued_forecast(history, "BPI", **{**kwargs, "predictions": changed})


def test_unresolved_records_are_not_exported_or_filled_without_their_date(tmp_path):
    history = tmp_path / "production_history"
    record_issued_forecast(history, "BPI", target_date="2026-09-03", issued_at="2026-09-02T16:01:42+08:00", data_as_of="2026-09-02", predictions=_predictions())
    frame = pd.DataFrame({"Date": pd.to_datetime(["2026-09-02"]), "Close": [64.9]})
    assert reconcile_history(history, "BPI", frame, "2026-09-02T16:00:00+08:00") == 0
    assert load_history(history, "BPI")["records"][0]["actual"] is None
