"""Generated-artifact layout, run metadata, and reset safety tests."""

from datetime import date, datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.artifacts.manager import (
    ARTIFACT_DIRECTORY_NAMES,
    RUN_METADATA_SCHEMA_ID,
    ArtifactManager,
    ArtifactSafetyError,
    RunMetadataError,
    RunStatus,
    reset_generated_artifacts,
)


MANILA = ZoneInfo("Asia/Manila")


def backend_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    backend_root = tmp_path / "backend"
    artifacts = backend_root / "artifacts"
    raw = backend_root / "data" / "raw"
    raw.mkdir(parents=True)
    return backend_root, artifacts, raw


def test_artifact_manager_records_run_lifecycle_and_environment(tmp_path: Path) -> None:
    backend_root, artifacts, _ = backend_paths(tmp_path)
    manager = ArtifactManager(artifacts)
    directories = manager.ensure_directories()

    assert directories.root == artifacts
    assert {path.name for path in directories.all_generated_directories()} == set(
        ARTIFACT_DIRECTORY_NAMES
    )
    assert all(path.is_dir() for path in directories.all_generated_directories())

    started_at = datetime(2026, 9, 11, 9, 30, tzinfo=MANILA)
    run = manager.start_run(
        symbols_processed=("BPI", "ALI", "ALI"),
        raw_data_latest_date=date(2026, 9, 10),
        started_at=started_at,
    )
    manifest_path = directories.logs / f"{run.run_id}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert run.run_id.startswith("run-")
    assert "formal" not in run.run_id
    assert payload["schema_id"] == RUN_METADATA_SCHEMA_ID
    assert payload["started_at"] == started_at.isoformat()
    assert payload["completed_at"] is None
    assert payload["raw_data_latest_date"] == "2026-09-10"
    assert payload["symbols_processed"] == ["ALI", "BPI"]
    assert payload["status"] == "running"
    assert payload["errors"] == []
    assert payload["python_version"]
    assert set(payload["dependency_versions"]) == {
        "numpy",
        "scikit-learn",
        "statsmodels",
        "torch",
    }
    assert payload["git_commit_sha"] is None or len(payload["git_commit_sha"]) == 40

    completed = manager.complete_run(
        run.run_id,
        completed_at=datetime(2026, 9, 11, 9, 45, tzinfo=MANILA),
    )
    assert completed.status is RunStatus.COMPLETED
    assert completed.completed_at is not None
    assert manager.load_run(run.run_id) == completed
    with pytest.raises(RunMetadataError, match="running"):
        manager.fail_run(run.run_id, ["late failure"])


def test_artifact_manager_records_failed_run_errors(tmp_path: Path) -> None:
    _, artifacts, _ = backend_paths(tmp_path)
    manager = ArtifactManager(artifacts)
    run = manager.start_run(
        symbols_processed=("ALI",),
        raw_data_latest_date=date(2026, 9, 10),
    )

    failed = manager.fail_run(run.run_id, ["ALI: model fitting failed"])

    assert failed.status is RunStatus.FAILED
    assert failed.completed_at is not None
    assert failed.errors == ("ALI: model fitting failed",)
    assert manager.load_run(run.run_id) == failed


def test_reset_removes_only_artifacts_and_preserves_raw_bytes(tmp_path: Path) -> None:
    backend_root, artifacts, raw = backend_paths(tmp_path)
    raw_file = raw / "ALI.csv"
    original_raw = b"Date,Open,High,Low,Close,Volume\n2026-09-10,1,2,0.5,1.5,100\n"
    raw_file.write_bytes(original_raw)
    manager = ArtifactManager(artifacts)
    directories = manager.ensure_directories()
    (directories.models / "model.bin").write_bytes(b"generated model")
    (directories.evaluations / "metrics.json").write_text("{}", encoding="utf-8")
    nested = directories.forecasts / "ALI"
    nested.mkdir()
    (nested / "next.json").write_text("{}", encoding="utf-8")
    (artifacts / "unexpected-generated.tmp").write_text("temporary", encoding="utf-8")
    raw_link = directories.logs / "raw-link"
    raw_link.symlink_to(raw, target_is_directory=True)

    reset = reset_generated_artifacts(
        backend_root=backend_root,
        artifacts_dir=artifacts,
        raw_data_dir=raw,
    )

    assert raw_file.read_bytes() == original_raw
    assert raw.is_dir()
    assert {path.name for path in artifacts.iterdir()} == set(ARTIFACT_DIRECTORY_NAMES)
    assert all(not any(path.iterdir()) for path in reset.all_generated_directories())


def test_reset_rejects_raw_data_as_artifact_target(tmp_path: Path) -> None:
    backend_root, _, raw = backend_paths(tmp_path)
    raw_file = raw / "ALI.csv"
    original_raw = b"immutable raw bytes\x00\xff"
    raw_file.write_bytes(original_raw)

    with pytest.raises(ArtifactSafetyError, match="backend/artifacts"):
        reset_generated_artifacts(
            backend_root=backend_root,
            artifacts_dir=raw,
            raw_data_dir=raw,
        )

    assert raw_file.read_bytes() == original_raw
