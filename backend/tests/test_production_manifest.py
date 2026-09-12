"""Phase 14 production package and manifest regression tests."""

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from config.companies import COMPANIES
from src.artifacts.production_manifest import (
    MANIFEST_MODEL_FAMILIES,
    PRODUCTION_ARTIFACT_GENERATION,
    PRODUCTION_ARTIFACT_NAME,
    PRODUCTION_MANIFEST_SCHEMA_ID,
    PRODUCTION_MANIFEST_SCHEMA_VERSION,
    PRODUCTION_WORKFLOW_FILE,
    ProductionArtifactReport,
    ProductionArtifactValidationError,
    configured_symbols,
    create_production_manifest,
    expected_artifact_paths,
    load_production_manifest,
    parse_production_manifest,
    validate_production_artifacts,
)


MANILA = ZoneInfo("Asia/Manila")
SOURCE_SHA = "a" * 40


def valid_payload() -> dict[str, object]:
    symbols = configured_symbols()
    return {
        "schema_id": PRODUCTION_MANIFEST_SCHEMA_ID,
        "schema_version": PRODUCTION_MANIFEST_SCHEMA_VERSION,
        "artifact_generation": PRODUCTION_ARTIFACT_GENERATION,
        "artifact_name": PRODUCTION_ARTIFACT_NAME,
        "workflow_file": PRODUCTION_WORKFLOW_FILE,
        "run_id": "123456789",
        "source_commit": SOURCE_SHA,
        "source_branch": "refactor/backend-from-scratch",
        "created_at": "2026-09-12T08:00:00+08:00",
        "training_data_through": "2026-09-11",
        "symbols": list(symbols),
        "symbol_count": len(symbols),
        "model_families": list(MANIFEST_MODEL_FAMILIES),
    }


def test_manifest_accepts_only_exact_phase14_identity() -> None:
    manifest = parse_production_manifest(valid_payload())

    assert manifest.artifact_generation == "phase14"
    assert manifest.symbols == tuple(sorted(company.symbol for company in COMPANIES))
    assert manifest.model_families == ("lag_regression", "arima", "lstm")


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("artifact_generation", "legacy", "identity"),
        ("schema_version", 0, "identity"),
        ("artifact_name", "old-models", "identity"),
        ("workflow_file", "legacy.yml", "identity"),
        ("symbols", ["BPI"], "symbol universe"),
        ("model_families", ["arima"], "model families"),
    ),
)
def test_manifest_rejects_legacy_or_incomplete_identity(
    field: str,
    replacement: object,
    message: str,
) -> None:
    payload = valid_payload()
    payload[field] = replacement

    with pytest.raises(ProductionArtifactValidationError, match=message):
        parse_production_manifest(payload)


def test_manifest_is_bound_to_selected_actions_run(tmp_path: Path) -> None:
    report = ProductionArtifactReport(
        artifacts_root=tmp_path,
        symbols=configured_symbols(),
        training_data_through=date(2026, 9, 11),
        validated_file_count=180,
        manifest=None,
    )
    path = create_production_manifest(
        report,
        run_id="123456789",
        source_commit=SOURCE_SHA,
        source_branch="refactor/backend-from-scratch",
        created_at=datetime(2026, 9, 12, 8, tzinfo=MANILA),
    )

    loaded = load_production_manifest(
        tmp_path,
        expected_run_id="123456789",
        expected_source_commit=SOURCE_SHA,
        expected_source_branch="refactor/backend-from-scratch",
    )

    assert path == tmp_path / "production_manifest.json"
    assert loaded.training_data_through == date(2026, 9, 11)
    with pytest.raises(ProductionArtifactValidationError, match="selected training run"):
        load_production_manifest(tmp_path, expected_run_id="987654321")


def test_expected_inventory_uses_current_configured_universe(tmp_path: Path) -> None:
    without_forecasts = expected_artifact_paths(tmp_path, require_forecasts=False)
    with_forecasts = expected_artifact_paths(tmp_path, require_forecasts=True)

    assert len(without_forecasts) == len(COMPANIES) * 12
    assert len(with_forecasts) == len(COMPANIES) * 13
    assert tmp_path / "models" / "BPI" / "lag_reg.joblib" in without_forecasts
    assert tmp_path / "evaluations" / "lstm" / "BPI.pt" in without_forecasts
    assert tmp_path / "forecasts" / "BPI.json" in with_forecasts


def test_incomplete_package_fails_before_any_model_can_be_used(tmp_path: Path) -> None:
    with pytest.raises(ProductionArtifactValidationError, match="incomplete"):
        validate_production_artifacts(
            tmp_path,
            require_manifest=False,
            require_forecasts=True,
        )
