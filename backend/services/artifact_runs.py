"""Immutable formal-run evidence and separate deployment artifact locations."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from services.formal_evaluation import FORMAL_FORECAST_COLUMNS, FORMAL_MODEL_KEYS, validate_formal_holdout_alignment
from services.time_series_cv import FormalEvaluationPlan


class RunMode(StrEnum):
    FORMAL = "formal"
    DEPLOYMENT = "deployment"


FORMAL_VERSION = "1.0"


class FormalRunIntegrityError(ValueError):
    """Raised when a formal run is incomplete or its evidence is inconsistent."""


def _json_default(value: object) -> str:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _json_safe(value: object) -> object:
    """Convert dataframe/numpy values and non-finite floats into strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float,)) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except ValueError:
            pass
    return value


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of an immutable source or evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_data_manifest(path: Path, df: pd.DataFrame) -> dict[str, object]:
    """Build reproducible source-data provenance for one formal company."""
    dates = pd.to_datetime(df["Date"], errors="raise")
    return {
        "source_path": str(path),
        "sha256": file_sha256(path),
        "row_count": int(len(df)),
        "first_date": dates.iloc[0],
        "last_date": dates.iloc[-1],
    }


def git_worktree_is_dirty() -> bool:
    """Return whether tracked or untracked repository state is dirty."""
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception as exc:
        raise FormalRunIntegrityError("Cannot determine Git worktree state for formal run.") from exc


def create_run_id(run_id: str | None = None) -> str:
    """Return a caller ID or a timestamped, collision-resistant ID."""
    if run_id:
        if Path(run_id).name != run_id or not run_id.strip():
            raise ValueError("Formal run ID must be a non-empty path component.")
        return run_id
    now = datetime.now(timezone.utc)
    digest = hashlib.sha256(now.isoformat().encode()).hexdigest()[:8]
    return f"{now:%Y%m%dT%H%M%SZ}_{digest}"


def deployment_current_dir(base_dir: Path) -> Path:
    """Return the sole writable deployment-model location."""
    return base_dir / "models" / "deployment" / "current"


def write_deployment_manifest(base_dir: Path, artifacts: dict[str, dict[str, str]]) -> Path:
    """Write the mutable deployment manifest without touching formal evidence."""
    target = deployment_current_dir(base_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "deployment_manifest.json"
    payload = {
        "mode": RunMode.DEPLOYMENT,
        "created_at": datetime.now(timezone.utc),
        "artifact_layout": "deployment/current",
        "artifacts": artifacts,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return path


@dataclass
class FormalRunWriter:
    """Write and finalize one immutable formal-evaluation evidence package."""

    base_dir: Path
    run_id: str

    @property
    def path(self) -> Path:
        return self.base_dir / "results" / "formal" / self.run_id

    @property
    def finalized_path(self) -> Path:
        return self.path / "finalized.json"

    def create(self) -> None:
        """Create an empty run directory; an existing ID is never reusable."""
        if self.path.exists():
            raise FileExistsError(f"Formal run {self.run_id} already exists and cannot be overwritten.")
        self.path.mkdir(parents=True)

    def write_split_manifest(self, plans: dict[str, FormalEvaluationPlan]) -> None:
        self._assert_mutable()
        self._write("split_manifest.json", {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc),
            "split_definition": "target_date_based_85_15",
            "formal_evaluation_version": FORMAL_VERSION,
            "companies": {symbol: self._plan_payload(plan) for symbol, plan in plans.items()},
        })

    def write_data_manifest(self, companies: dict[str, dict[str, object]]) -> None:
        """Record hashed raw-source provenance before formal finalization."""
        self._assert_mutable()
        self._write("data_manifest.json", {"run_id": self.run_id, "companies": companies})

    def write_company(self, symbol: str, forecasts: dict[str, pd.DataFrame], metrics: dict, diagnostics: dict) -> None:
        """Write complete, date-indexed holdout evidence for one company."""
        self._assert_mutable()
        target = self.path / "per_company" / symbol.upper()
        target.mkdir(parents=True, exist_ok=True)
        rows = pd.concat(forecasts.values(), ignore_index=True).sort_values(["model", "target_date"])
        missing = set(FORMAL_FORECAST_COLUMNS) - set(rows.columns)
        if missing:
            raise FormalRunIntegrityError(f"{symbol}: formal predictions lack required columns: {sorted(missing)}.")
        rows.to_csv(target / "holdout_predictions.csv", index=False)
        self._write(target / "metrics.json", metrics)
        self._write(target / "diagnostics.json", diagnostics)

    def write_methodology_manifest(
        self,
        data_cutoff: str,
        companies: list[str],
        *,
        git_dirty: bool | None = None,
    ) -> None:
        self._assert_mutable()
        self._write("methodology_manifest.json", {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc),
            "git_branch": self._git("branch", "--show-current"),
            "repository_commit": self._git("rev-parse", "HEAD"),
            "git_worktree_dirty": git_worktree_is_dirty() if git_dirty is None else git_dirty,
            "data_cutoff": data_cutoff,
            "company_universe": sorted(companies),
            "company_count": len(companies),
            "research_model_set": list(FORMAL_MODEL_KEYS),
            "formal_split": {"development_ratio": .85, "holdout_ratio": .15, "split_basis": "target_date"},
            "artifact_policy": {"formal_immutable": True, "deployment_separate": True},
            "dependencies": {name: self._version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "torch")},
        })

    def write_statistics(self, statistics: dict) -> None:
        self._assert_mutable()
        self._write("statistical_tests.json", statistics)

    def finalize(self) -> Path:
        """Validate all required evidence, then permanently mark the run complete."""
        self._assert_mutable()
        plans = self._read_plans()
        required_root = ("methodology_manifest.json", "split_manifest.json", "data_manifest.json", "statistical_tests.json")
        absent = [name for name in required_root if not (self.path / name).is_file()]
        if absent:
            raise FormalRunIntegrityError(f"{self.run_id}: missing required run artifact(s): {absent}.")
        for symbol, plan in plans.items():
            company_dir = self.path / "per_company" / symbol.upper()
            required_company = ("holdout_predictions.csv", "metrics.json", "diagnostics.json")
            absent = [name for name in required_company if not (company_dir / name).is_file()]
            if absent:
                raise FormalRunIntegrityError(f"{symbol}: incomplete formal evidence: {absent}.")
            try:
                validate_formal_holdout_alignment(self._read_company_forecasts(company_dir / "holdout_predictions.csv"), plan)
            except Exception as exc:
                raise FormalRunIntegrityError(f"{symbol}: invalid formal holdout evidence: {exc}") from exc
            self._validate_arima_diagnostics(symbol, company_dir / "diagnostics.json")
        evidence_files = [path for path in self.path.rglob("*") if path.is_file() and path.name != "finalized.json"]
        self._write("finalized.json", {
            "run_id": self.run_id,
            "finalized_at": datetime.now(timezone.utc),
            "status": "complete",
            "artifact_sha256": {str(path.relative_to(self.path)): file_sha256(path) for path in sorted(evidence_files)},
        })
        return self.finalized_path

    def _read_plans(self) -> dict[str, FormalEvaluationPlan]:
        manifest = self.path / "split_manifest.json"
        if not manifest.is_file():
            raise FormalRunIntegrityError(f"{self.run_id}: split_manifest.json is required before finalization.")
        payload = json.loads(manifest.read_text())
        try:
            return {symbol: FormalEvaluationPlan(**{
                **data,
                "development_origin_dates": tuple(pd.Timestamp(d) for d in data["development_origin_dates"]),
                "development_target_dates": tuple(pd.Timestamp(d) for d in data["development_target_dates"]),
                "holdout_origin_dates": tuple(pd.Timestamp(d) for d in data["holdout_origin_dates"]),
                "holdout_target_dates": tuple(pd.Timestamp(d) for d in data["holdout_target_dates"]),
                "development_start_date": pd.Timestamp(data["development_start_date"]),
                "development_end_date": pd.Timestamp(data["development_end_date"]),
                "holdout_start_date": pd.Timestamp(data["holdout_start_date"]),
                "holdout_end_date": pd.Timestamp(data["holdout_end_date"]),
            }) for symbol, data in payload["companies"].items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise FormalRunIntegrityError(f"{self.run_id}: malformed split manifest.") from exc

    @staticmethod
    def _read_company_forecasts(path: Path) -> dict[str, pd.DataFrame]:
        rows = pd.read_csv(path)
        missing = set(FORMAL_FORECAST_COLUMNS) - set(rows.columns)
        if missing:
            raise FormalRunIntegrityError(f"{path}: missing prediction columns: {sorted(missing)}.")
        return {model: frame.copy() for model, frame in rows.groupby("model", sort=False)}

    @staticmethod
    def _validate_arima_diagnostics(symbol: str, path: Path) -> None:
        """Require auditable convergence evidence before formal finalization."""
        try:
            diagnostics = json.loads(path.read_text())["arima"]
            selected = diagnostics["selected_configuration"]
            selected_order = selected["order"]
            selected_trend = selected["trend"]
            folds = diagnostics["cv_fold_results"]
            final_attempts = diagnostics["final_fit_attempts"]
            retry_policy = diagnostics["optimizer_retry_policy"]
            candidates = diagnostics["candidate_cv"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FormalRunIntegrityError(
                f"{symbol}: ARIMA diagnostics lack required configuration or convergence evidence."
            ) from exc

        if selected_order != diagnostics.get("selected_order") or selected_trend != diagnostics.get("selected_trend"):
            raise FormalRunIntegrityError(f"{symbol}: ARIMA selected-configuration diagnostics disagree.")
        if diagnostics.get("all_cv_folds_converged") is not True:
            raise FormalRunIntegrityError(f"{symbol}: ARIMA does not confirm convergence for every CV fold.")
        if diagnostics.get("final_fit_converged") is not True:
            raise FormalRunIntegrityError(f"{symbol}: ARIMA final development fit is not confirmed converged.")
        if not isinstance(folds, list) or len(folds) != 5:
            raise FormalRunIntegrityError(f"{symbol}: ARIMA must contain exactly five CV fold diagnostics.")
        for expected_fold, fold in enumerate(folds, start=1):
            try:
                rmse = float(fold["validation_rmse"])
                attempts = fold["optimizer_attempts"]
            except (KeyError, TypeError, ValueError) as exc:
                raise FormalRunIntegrityError(
                    f"{symbol}: ARIMA fold {expected_fold} diagnostics are incomplete."
                ) from exc
            if (
                fold.get("fold_number") != expected_fold
                or fold.get("successful") is not True
                or fold.get("converged") is not True
                or not math.isfinite(rmse)
                or not isinstance(attempts, list)
                or not attempts
                or attempts[-1].get("converged") is not True
            ):
                raise FormalRunIntegrityError(
                    f"{symbol}: ARIMA fold {expected_fold} lacks finite, confirmed-converged evidence."
                )
        if not isinstance(final_attempts, list) or not final_attempts or final_attempts[-1].get("converged") is not True:
            raise FormalRunIntegrityError(f"{symbol}: ARIMA final-fit attempts do not confirm convergence.")
        if not isinstance(retry_policy, list) or not retry_policy:
            raise FormalRunIntegrityError(f"{symbol}: ARIMA optimizer retry policy is missing.")
        maxiters = [attempt.get("maxiter") for attempt in retry_policy]
        if (
            any(not isinstance(maxiter, int) or maxiter <= 0 for maxiter in maxiters)
            or maxiters != sorted(set(maxiters))
        ):
            raise FormalRunIntegrityError(f"{symbol}: ARIMA optimizer retry policy is not deterministic and increasing.")
        if not isinstance(candidates, list) or not candidates:
            raise FormalRunIntegrityError(f"{symbol}: ARIMA candidate diagnostics are missing.")
        selected_candidates = [
            candidate for candidate in candidates
            if candidate.get("configuration") == selected and candidate.get("valid") is True
        ]
        if len(selected_candidates) != 1:
            raise FormalRunIntegrityError(
                f"{symbol}: ARIMA selected configuration lacks one valid candidate record."
            )

    @staticmethod
    def _plan_payload(plan: FormalEvaluationPlan) -> dict[str, Any]:
        payload = asdict(plan)
        return {key: [str(d) for d in value] if isinstance(value, tuple) else str(value) if isinstance(value, pd.Timestamp) else value for key, value in payload.items()}

    def _assert_mutable(self) -> None:
        if not self.path.is_dir():
            raise FormalRunIntegrityError(f"Formal run {self.run_id} has not been created.")
        if self.finalized_path.exists():
            raise FormalRunIntegrityError(f"Formal run {self.run_id} is finalized and immutable.")

    def _write(self, relative: str | Path, payload: object) -> None:
        path = relative if isinstance(relative, Path) else self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, default=_json_default, allow_nan=False))

    @staticmethod
    def _version(name: str) -> str | None:
        try:
            return version(name)
        except Exception:
            return None

    @staticmethod
    def _git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except Exception:
            return None
