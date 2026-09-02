"""Fast tests for explicit manual deployment challenger approval."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import model_selector
from services.artifact_runs import (
    DeploymentConfigurationError,
    load_approved_deployment_configurations,
    write_deployment_manifest,
)


APPROVED = {
    "lag_reg": {
        "alpha": 0.1,
        "candidate_features": ["rsi_14"],
        "pacf_selected_lags": [1],
    },
    "arima": {"order": [1, 1, 0], "trend": "n"},
    "lstm": {
        "lookback": 30,
        "hidden_size": 50,
        "learning_rate": 0.001,
        "batch_size": 16,
    },
}


def _configure_paths(monkeypatch, root: Path) -> None:
    current = root / "models" / "deployment" / "current"
    monkeypatch.setattr(model_selector, "BASE_DIR", root)
    monkeypatch.setattr(model_selector, "DEPLOYMENT_CURRENT_DIR", current)
    monkeypatch.setattr(model_selector, "LAG_MODELS_DIR", current / "lag_regression")
    monkeypatch.setattr(model_selector, "ARIMA_MODELS_DIR", current / "arima")
    monkeypatch.setattr(model_selector, "LSTM_MODELS_DIR", current / "lstm")
    monkeypatch.setattr(model_selector, "BEST_MODELS_PATH", root / "best_models.json")


def _write_challenger(
    root: Path,
    *,
    run_id: str = "CHALLENGER_001",
    symbols: tuple[str, ...] = ("BPI",),
    manifest_updates: dict | None = None,
    omit_artifact: tuple[str, str] | None = None,
) -> Path:
    challenger = root / "models" / "deployment" / "challengers" / run_id
    configurations = {symbol: APPROVED for symbol in symbols}
    manifest = {
        "mode": "deployment",
        "operation": "retune",
        "status": "challenger_only",
        "automatic_promotion": False,
        "configurations": configurations,
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    challenger.mkdir(parents=True)
    (challenger / "challenger_manifest.json").write_text(json.dumps(manifest))
    suffixes = {"lag_regression": ".pkl", "arima": ".pkl", "lstm": ".pth"}
    for symbol in symbols:
        for model, suffix in suffixes.items():
            if omit_artifact == (symbol, model):
                continue
            path = challenger / model / f"{symbol}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"challenger:{run_id}:{symbol}:{model}".encode())
    return challenger


def _write_current(root: Path, symbols: tuple[str, ...]) -> dict[str, bytes]:
    artifacts = {}
    contents = {}
    suffixes = {"lag_regression": ".pkl", "arima": ".pkl", "lstm": ".pth"}
    for symbol in symbols:
        artifacts[symbol] = {}
        for model, suffix in suffixes.items():
            path = root / "models" / "deployment" / "current" / model / f"{symbol}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"current:{symbol}:{model}".encode()
            path.write_bytes(content)
            contents[str(path)] = content
            artifacts[symbol][model] = str(path.relative_to(root))
    write_deployment_manifest(
        root,
        artifacts,
        approved_configurations={symbol: APPROVED for symbol in symbols},
    )
    (root / "best_models.json").write_text(
        json.dumps({symbol: "ARIMA" for symbol in symbols})
    )
    return contents


def test_approval_requires_explicit_confirmation(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)

    with pytest.raises(DeploymentConfigurationError, match="--confirm-approved"):
        model_selector.approve_deployment_challenger("CHALLENGER_001")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "formal"),
        ("operation", "refresh"),
        ("status", "approved"),
        ("automatic_promotion", True),
    ],
)
def test_malformed_challenger_manifest_is_rejected(tmp_path, monkeypatch, field, value):
    _configure_paths(monkeypatch, tmp_path)
    _write_challenger(tmp_path, manifest_updates={field: value})

    with pytest.raises(DeploymentConfigurationError, match="requires"):
        model_selector.approve_deployment_challenger("CHALLENGER_001", confirmed=True)


def test_incomplete_model_configuration_is_rejected(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)
    _write_challenger(
        tmp_path,
        manifest_updates={"configurations": {"BPI": {"lag_reg": APPROVED["lag_reg"]}}},
    )

    with pytest.raises(DeploymentConfigurationError, match="configuration metadata is malformed"):
        model_selector.approve_deployment_challenger("CHALLENGER_001", confirmed=True)


def test_duplicate_and_unexpected_challenger_symbols_are_rejected(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)
    challenger = _write_challenger(tmp_path)
    manifest_path = challenger / "challenger_manifest.json"
    duplicate_config = json.dumps(APPROVED)
    manifest_path.write_text(
        '{"mode":"deployment","operation":"retune","status":"challenger_only",'
        '"automatic_promotion":false,"configurations":{"BPI":'
        + duplicate_config
        + ',"BPI":'
        + duplicate_config
        + "}}"
    )
    with pytest.raises(DeploymentConfigurationError, match="duplicate key"):
        model_selector.approve_deployment_challenger("CHALLENGER_001", confirmed=True)

    challenger = _write_challenger(tmp_path, run_id="CHALLENGER_002", symbols=("XYZ",))
    (tmp_path / "best_models.json").write_text(json.dumps({"XYZ": "ARIMA"}))
    with pytest.raises(DeploymentConfigurationError, match="unexpected deployment symbol"):
        model_selector.approve_deployment_challenger("CHALLENGER_002", confirmed=True)


def test_missing_artifact_fails_before_current_is_modified(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)
    before = _write_current(tmp_path, ("BPI",))
    _write_challenger(tmp_path, omit_artifact=("BPI", "lstm"))
    manifest_path = model_selector.DEPLOYMENT_CURRENT_DIR / "deployment_manifest.json"
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(DeploymentConfigurationError, match="lstm artifact is missing"):
        model_selector.approve_deployment_challenger("CHALLENGER_001", confirmed=True)

    assert manifest_path.read_bytes() == manifest_before
    assert {path: Path(path).read_bytes() for path in before} == before


def test_manifest_failure_rolls_back_all_promoted_artifacts(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)
    before = _write_current(tmp_path, ("BPI",))
    challenger = _write_challenger(tmp_path)

    def fail_manifest_write(*_args, **_kwargs):
        for model, suffix in {"lag_regression": ".pkl", "arima": ".pkl", "lstm": ".pth"}.items():
            current = model_selector.DEPLOYMENT_CURRENT_DIR / model / f"BPI{suffix}"
            source = challenger / model / f"BPI{suffix}"
            assert current.read_bytes() == source.read_bytes()
        raise RuntimeError("manifest write failed")

    monkeypatch.setattr(model_selector, "write_deployment_manifest", fail_manifest_write)

    with pytest.raises(RuntimeError, match="manifest write failed"):
        model_selector.approve_deployment_challenger("CHALLENGER_001", confirmed=True)

    assert {path: Path(path).read_bytes() for path in before} == before
    assert not list((tmp_path / "models" / "deployment").glob(".approval-*"))


@pytest.mark.parametrize("run_id", ("../escape", "nested/run", "/absolute", ".."))
def test_path_traversal_in_run_id_is_rejected(tmp_path, monkeypatch, run_id):
    _configure_paths(monkeypatch, tmp_path)

    with pytest.raises(DeploymentConfigurationError, match="safe path component"):
        model_selector.approve_deployment_challenger(run_id, confirmed=True)


def test_requested_symbols_must_exist_in_challenger_manifest(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)
    _write_challenger(tmp_path)

    with pytest.raises(DeploymentConfigurationError, match="absent from the challenger manifest"):
        model_selector.approve_deployment_challenger(
            "CHALLENGER_001", symbols=["ALI"], confirmed=True
        )


def test_successful_approval_promotes_all_artifacts_and_metadata_without_formal_calls(
    tmp_path, monkeypatch
):
    _configure_paths(monkeypatch, tmp_path)
    _write_current(tmp_path, ("BPI",))
    challenger = _write_challenger(tmp_path)
    challenger_before = {
        str(path.relative_to(challenger)): path.read_bytes()
        for path in challenger.rglob("*") if path.is_file()
    }
    formal_file = tmp_path / "results" / "formal" / "FINAL" / "finalized.json"
    formal_file.parent.mkdir(parents=True)
    formal_file.write_bytes(b"immutable-formal-evidence")
    formal_before = formal_file.read_bytes()
    monkeypatch.setattr(
        model_selector, "evaluate_formal_symbol", Mock(side_effect=AssertionError("formal called"))
    )
    monkeypatch.setattr(
        model_selector, "run_formal_evaluation", Mock(side_effect=AssertionError("formal called"))
    )
    monkeypatch.setattr(
        model_selector, "run_formal_statistical_tests", Mock(side_effect=AssertionError("formal called"))
    )
    monkeypatch.setattr(
        model_selector, "FormalRunWriter", Mock(side_effect=AssertionError("formal writer called"))
    )

    manifest_path = model_selector.approve_deployment_challenger(
        "CHALLENGER_001", confirmed=True
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["operation"] == "approval"
    assert manifest["source_challenger_run_id"] == "CHALLENGER_001"
    assert manifest["approved_symbols"] == ["BPI"]
    assert manifest["manually_invoked"] is True
    assert manifest["approval_timestamp"]
    assert manifest["approved_configurations"]["BPI"] == APPROVED
    for model, suffix in {"lag_regression": ".pkl", "arima": ".pkl", "lstm": ".pth"}.items():
        source = challenger / model / f"BPI{suffix}"
        current = model_selector.DEPLOYMENT_CURRENT_DIR / model / f"BPI{suffix}"
        assert current.read_bytes() == source.read_bytes()
    challenger_after = {
        str(path.relative_to(challenger)): path.read_bytes()
        for path in challenger.rglob("*") if path.is_file()
    }
    assert challenger_after == challenger_before
    assert formal_file.read_bytes() == formal_before
    assert json.loads((tmp_path / "best_models.json").read_text()) == {"BPI": "ARIMA"}
    assert load_approved_deployment_configurations(tmp_path, ["BPI"])["BPI"] == APPROVED
    assert not list((tmp_path / "models" / "deployment").glob(".approval-*"))


def test_partial_approval_preserves_other_approved_symbols_and_artifacts(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)
    current_before = _write_current(tmp_path, ("BPI", "ALI"))
    challenger = _write_challenger(tmp_path, symbols=("BPI", "ALI"))

    manifest_path = model_selector.approve_deployment_challenger(
        "CHALLENGER_001", symbols=["BPI"], confirmed=True
    )

    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["approved_configurations"]) == {"BPI", "ALI"}
    assert set(manifest["artifacts"]) == {"BPI", "ALI"}
    assert manifest["approved_symbols"] == ["BPI"]
    for model, suffix in {"lag_regression": ".pkl", "arima": ".pkl", "lstm": ".pth"}.items():
        ali_path = model_selector.DEPLOYMENT_CURRENT_DIR / model / f"ALI{suffix}"
        bpi_path = model_selector.DEPLOYMENT_CURRENT_DIR / model / f"BPI{suffix}"
        assert ali_path.read_bytes() == current_before[str(ali_path)]
        assert bpi_path.read_bytes() == (challenger / model / f"BPI{suffix}").read_bytes()


def test_scheduled_workflow_remains_refresh_only():
    workflow = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "train_models.yml"
    ).read_text()
    scheduled_command = "python -m services.model_selector --mode deployment-refresh --strict"

    assert workflow.count(scheduled_command) == 1
    assert "--mode deployment-approve" not in workflow
    assert "--mode deployment-retune" not in workflow
    assert "--mode formal" not in workflow
