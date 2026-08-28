from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.artifact_runs import FormalRunWriter, RunMode, deployment_current_dir
from services.time_series_cv import create_formal_evaluation_plan

def test_formal_run_is_versioned_immutable_and_writes_evidence(tmp_path):
    df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=12), "Close": range(12)})
    plan = create_formal_evaluation_plan(df, "BPI")
    writer = FormalRunWriter(tmp_path, "controlled_run"); writer.create(); writer.write_split_manifest({"BPI": plan})
    dates = plan.holdout_target_dates
    frame = pd.DataFrame({"symbol":"BPI","model":"naive","origin_date":plan.holdout_origin_dates,"target_date":dates,"actual_close":[1.]*len(dates),"predicted_close":[1.]*len(dates),"error":[0.]*len(dates)})
    writer.write_company("BPI", {"naive": frame}, {"naive": {"rmse": 0.0}}, {})
    writer.write_methodology_manifest("2025-01-12", ["BPI"]); writer.write_statistics({})
    assert (writer.path / "per_company/BPI/holdout_predictions.csv").exists()
    assert deployment_current_dir(tmp_path) == tmp_path / "models/deployment/current"
    assert RunMode.FORMAL == "formal"
    with pytest.raises(FileExistsError): writer.create()
