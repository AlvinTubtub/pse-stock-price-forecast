"""Generated-artifact directories, run manifests, and safe reset support."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import StrEnum
from importlib import metadata as importlib_metadata
import json
import logging
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any, Final
from uuid import uuid4

from config.settings import BACKEND_ROOT, SETTINGS, as_manila_time, manila_now


LOGGER = logging.getLogger(__name__)
ARTIFACT_DIRECTORY_NAMES: Final[tuple[str, ...]] = (
    "models",
    "evaluations",
    "forecasts",
    "logs",
)
RUN_METADATA_SCHEMA_ID: Final[str] = "forecastph.training-run"
RUN_METADATA_SCHEMA_VERSION: Final[int] = 1
DEPENDENCY_DISTRIBUTIONS: Final[tuple[str, ...]] = (
    "numpy",
    "scikit-learn",
    "statsmodels",
    "torch",
)
RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^run-[0-9]{8}T[0-9]{12}-[0-9a-f]{8}$"
)


class ArtifactSafetyError(RuntimeError):
    """Raised when an artifact reset target is not provably safe."""


class RunMetadataError(ValueError):
    """Raised for malformed run metadata or an invalid lifecycle transition."""


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ArtifactDirectories:
    root: Path
    models: Path
    evaluations: Path
    forecasts: Path
    logs: Path

    @classmethod
    def from_root(cls, root: Path) -> "ArtifactDirectories":
        root = Path(root)
        return cls(
            root=root,
            models=root / "models",
            evaluations=root / "evaluations",
            forecasts=root / "forecasts",
            logs=root / "logs",
        )

    def all_generated_directories(self) -> tuple[Path, ...]:
        return (self.models, self.evaluations, self.forecasts, self.logs)


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """One mutable-in-storage, non-research training-run manifest."""

    run_id: str
    started_at: datetime
    completed_at: datetime | None
    git_commit_sha: str | None
    python_version: str
    dependency_versions: Mapping[str, str | None]
    raw_data_latest_date: date
    symbols_processed: tuple[str, ...]
    status: RunStatus
    errors: tuple[str, ...]
    schema_id: str = RUN_METADATA_SCHEMA_ID
    schema_version: int = RUN_METADATA_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at is not None else None
            ),
            "git_commit_sha": self.git_commit_sha,
            "python_version": self.python_version,
            "dependency_versions": dict(self.dependency_versions),
            "raw_data_latest_date": self.raw_data_latest_date.isoformat(),
            "symbols_processed": list(self.symbols_processed),
            "status": self.status.value,
            "errors": list(self.errors),
        }


def collect_dependency_versions(
    distributions: Iterable[str] = DEPENDENCY_DISTRIBUTIONS,
) -> dict[str, str | None]:
    """Return installed versions, retaining a null entry for missing dependencies."""

    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def current_git_commit(repository_root: Path) -> str | None:
    """Return the current commit SHA, or null outside an available Git checkout."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        return None
    return sha


def _aware_manila_time(value: datetime | None) -> datetime:
    return manila_now() if value is None else as_manila_time(value)


def _normalize_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    symbols = tuple(symbols)
    if any(not isinstance(symbol, str) for symbol in symbols):
        raise RunMetadataError("Processed symbols must be strings")
    normalized = tuple(sorted({symbol.strip().upper() for symbol in symbols}))
    if not normalized or any(not symbol for symbol in normalized):
        raise RunMetadataError("At least one non-empty processed symbol is required")
    if any(re.fullmatch(r"[A-Z0-9.-]+", symbol) is None for symbol in normalized):
        raise RunMetadataError("Processed symbols contain unsupported characters")
    return normalized


def _run_id(started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%f")
    return f"run-{timestamp}-{uuid4().hex[:8]}"


def _parse_run_metadata(payload: Mapping[str, Any]) -> RunMetadata:
    if payload.get("schema_id") != RUN_METADATA_SCHEMA_ID:
        raise RunMetadataError("Unsupported training-run schema_id")
    if payload.get("schema_version") != RUN_METADATA_SCHEMA_VERSION:
        raise RunMetadataError("Unsupported training-run schema_version")
    try:
        run_id = payload["run_id"]
        started_at = datetime.fromisoformat(payload["started_at"])
        completed_value = payload["completed_at"]
        completed_at = (
            datetime.fromisoformat(completed_value)
            if completed_value is not None
            else None
        )
        raw_data_latest_date = date.fromisoformat(payload["raw_data_latest_date"])
        status = RunStatus(payload["status"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunMetadataError("Malformed training-run metadata") from exc
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise RunMetadataError("Invalid training run ID")
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise RunMetadataError("started_at must be timezone-aware")
    if completed_at is not None and (
        completed_at.tzinfo is None or completed_at.utcoffset() is None
    ):
        raise RunMetadataError("completed_at must be timezone-aware")
    if status is RunStatus.RUNNING and completed_at is not None:
        raise RunMetadataError("A running training run cannot have completed_at")
    if status is not RunStatus.RUNNING and completed_at is None:
        raise RunMetadataError("A finished training run requires completed_at")
    if completed_at is not None and completed_at < started_at:
        raise RunMetadataError("completed_at cannot precede started_at")
    git_commit_sha = payload.get("git_commit_sha")
    if git_commit_sha is not None and (
        not isinstance(git_commit_sha, str)
        or re.fullmatch(r"[0-9a-f]{40}", git_commit_sha) is None
    ):
        raise RunMetadataError("Invalid Git commit SHA")
    python_version = payload.get("python_version")
    dependency_versions = payload.get("dependency_versions")
    symbols = payload.get("symbols_processed")
    errors = payload.get("errors")
    if not isinstance(python_version, str) or not python_version:
        raise RunMetadataError("Missing Python version")
    if not isinstance(dependency_versions, dict) or any(
        not isinstance(name, str)
        or (version is not None and not isinstance(version, str))
        for name, version in dependency_versions.items()
    ):
        raise RunMetadataError("Invalid dependency versions")
    if not isinstance(symbols, list) or any(not isinstance(item, str) for item in symbols):
        raise RunMetadataError("Invalid processed symbols")
    normalized_symbols = _normalize_symbols(symbols)
    if list(normalized_symbols) != symbols:
        raise RunMetadataError("Processed symbols must be sorted and unique")
    if not isinstance(errors, list) or any(not isinstance(item, str) for item in errors):
        raise RunMetadataError("Invalid training-run errors")
    if status is RunStatus.FAILED and not errors:
        raise RunMetadataError("A failed training run requires at least one error")
    if status is not RunStatus.FAILED and errors:
        raise RunMetadataError("Only failed training runs may contain errors")
    return RunMetadata(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        git_commit_sha=git_commit_sha,
        python_version=python_version,
        dependency_versions=dict(dependency_versions),
        raw_data_latest_date=raw_data_latest_date,
        symbols_processed=normalized_symbols,
        status=status,
        errors=tuple(errors),
    )


class ArtifactManager:
    """Own the four generated directories and lightweight run manifests."""

    def __init__(self, artifacts_root: Path = SETTINGS.artifacts_dir) -> None:
        self.directories = ArtifactDirectories.from_root(Path(artifacts_root))

    def ensure_directories(self) -> ArtifactDirectories:
        self.directories.root.mkdir(parents=True, exist_ok=True)
        for directory in self.directories.all_generated_directories():
            directory.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Ensured generated artifact directories root=%s", self.directories.root)
        return self.directories

    def _manifest_path(self, run_id: str) -> Path:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise RunMetadataError("Invalid training run ID")
        logs = self.ensure_directories().logs.resolve()
        path = (logs / f"{run_id}.json").resolve()
        if path.parent != logs:
            raise RunMetadataError("Unsafe training run manifest path")
        return path

    def _write_run(self, run: RunMetadata) -> None:
        path = self._manifest_path(run.run_id)
        temporary_path = path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(run.as_dict(), output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        temporary_path.replace(path)
        LOGGER.info("Recorded training run run_id=%s status=%s", run.run_id, run.status)

    def start_run(
        self,
        *,
        symbols_processed: Iterable[str],
        raw_data_latest_date: date,
        started_at: datetime | None = None,
    ) -> RunMetadata:
        """Create a running manifest for one ordinary fresh training run."""

        if isinstance(raw_data_latest_date, datetime) or not isinstance(
            raw_data_latest_date, date
        ):
            raise RunMetadataError("raw_data_latest_date must be a date")
        started = _aware_manila_time(started_at)
        run = RunMetadata(
            run_id=_run_id(started),
            started_at=started,
            completed_at=None,
            git_commit_sha=current_git_commit(BACKEND_ROOT.parent),
            python_version=platform.python_version(),
            dependency_versions=collect_dependency_versions(),
            raw_data_latest_date=raw_data_latest_date,
            symbols_processed=_normalize_symbols(symbols_processed),
            status=RunStatus.RUNNING,
            errors=(),
        )
        self._write_run(run)
        return run

    def load_run(self, run_id: str) -> RunMetadata:
        path = self._manifest_path(run_id)
        try:
            with path.open("r", encoding="utf-8") as source:
                payload = json.load(source)
        except (OSError, json.JSONDecodeError) as exc:
            raise RunMetadataError(f"Cannot load training run {run_id}") from exc
        if not isinstance(payload, dict):
            raise RunMetadataError("Training-run manifest must be a JSON object")
        run = _parse_run_metadata(payload)
        if run.run_id != run_id:
            raise RunMetadataError("Training-run manifest identity mismatch")
        return run

    def complete_run(
        self,
        run_id: str,
        *,
        completed_at: datetime | None = None,
    ) -> RunMetadata:
        run = self.load_run(run_id)
        if run.status is not RunStatus.RUNNING:
            raise RunMetadataError("Only a running training run can be completed")
        completed = _aware_manila_time(completed_at)
        finished = replace(
            run,
            completed_at=completed,
            status=RunStatus.COMPLETED,
        )
        _parse_run_metadata(finished.as_dict())
        self._write_run(finished)
        return finished

    def fail_run(
        self,
        run_id: str,
        errors: Iterable[str],
        *,
        completed_at: datetime | None = None,
    ) -> RunMetadata:
        run = self.load_run(run_id)
        if run.status is not RunStatus.RUNNING:
            raise RunMetadataError("Only a running training run can be failed")
        supplied_errors = tuple(errors)
        if any(not isinstance(message, str) for message in supplied_errors):
            raise RunMetadataError("Training-run errors must be strings")
        error_messages = tuple(
            message.strip() for message in supplied_errors if message.strip()
        )
        if not error_messages:
            raise RunMetadataError("A failed training run requires at least one error")
        failed = replace(
            run,
            completed_at=_aware_manila_time(completed_at),
            status=RunStatus.FAILED,
            errors=error_messages,
        )
        _parse_run_metadata(failed.as_dict())
        self._write_run(failed)
        return failed


def reset_generated_artifacts(
    *,
    backend_root: Path = BACKEND_ROOT,
    artifacts_dir: Path = SETTINGS.artifacts_dir,
    raw_data_dir: Path = SETTINGS.raw_data_dir,
) -> ArtifactDirectories:
    """Remove only the contents of backend/artifacts, then restore its four folders."""

    backend_root = Path(backend_root).resolve()
    configured_artifacts = Path(artifacts_dir)
    configured_raw = Path(raw_data_dir)
    expected_artifacts = backend_root / "artifacts"
    expected_raw = backend_root / "data" / "raw"
    if configured_artifacts.absolute() != expected_artifacts.absolute():
        raise ArtifactSafetyError("Artifact reset target must be backend/artifacts")
    if configured_raw.absolute() != expected_raw.absolute():
        raise ArtifactSafetyError("Raw-data path must be backend/data/raw")
    if configured_artifacts.is_symlink():
        raise ArtifactSafetyError("Artifact reset target cannot be a symbolic link")
    resolved_artifacts = configured_artifacts.resolve()
    resolved_raw = configured_raw.resolve()
    if resolved_artifacts == resolved_raw or resolved_raw.is_relative_to(resolved_artifacts):
        raise ArtifactSafetyError("Artifact reset target cannot contain raw data")
    if resolved_artifacts.parent != backend_root:
        raise ArtifactSafetyError("Artifact reset target escaped the backend directory")

    LOGGER.info("Resetting generated artifacts root=%s", resolved_artifacts)
    resolved_artifacts.mkdir(parents=True, exist_ok=True)
    for child in resolved_artifacts.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            raise ArtifactSafetyError(f"Unsupported artifact entry: {child}")
    return ArtifactManager(resolved_artifacts).ensure_directories()
