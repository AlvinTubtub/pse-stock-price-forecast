"""Immutable formal-run artifacts and separate deployment-current paths."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from services.time_series_cv import FormalEvaluationPlan


class RunMode(StrEnum):
    FORMAL = "formal"
    DEPLOYMENT = "deployment"


FORMAL_VERSION = "1.0"


def _json_default(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def create_run_id(run_id: str | None = None) -> str:
    if run_id:
        return run_id
    now = datetime.now(timezone.utc)
    digest = hashlib.sha256(now.isoformat().encode()).hexdigest()[:8]
    return f"{now:%Y%m%dT%H%M%SZ}_{digest}"


def deployment_current_dir(base_dir: Path) -> Path:
    return base_dir / "models" / "deployment" / "current"


@dataclass
class FormalRunWriter:
    base_dir: Path
    run_id: str

    @property
    def path(self) -> Path:
        return self.base_dir / "results" / "formal" / self.run_id

    def create(self) -> None:
        if self.path.exists():
            raise FileExistsError(f"Formal run {self.run_id} already exists and cannot be overwritten.")
        self.path.mkdir(parents=True)

    def write_split_manifest(self, plans: dict[str, FormalEvaluationPlan]) -> None:
        companies = {symbol: {**asdict(plan), **{key: [str(d) for d in value] if isinstance(value, tuple) else str(value) if isinstance(value, pd.Timestamp) else value for key, value in asdict(plan).items()}} for symbol, plan in plans.items()}
        self._write("split_manifest.json", {"run_id": self.run_id, "created_at": datetime.now(timezone.utc), "split_definition": "target_date_based_85_15", "formal_evaluation_version": FORMAL_VERSION, "companies": companies})

    def write_company(self, symbol: str, forecasts: dict[str, pd.DataFrame], metrics: dict, diagnostics: dict) -> None:
        target = self.path / "per_company" / symbol; target.mkdir(parents=True, exist_ok=True)
        rows = pd.concat(forecasts.values(), ignore_index=True).sort_values(["model", "target_date"])
        required = {"symbol", "model", "origin_date", "target_date", "actual_close", "predicted_close", "error"}
        if not required.issubset(rows.columns): raise ValueError(f"{symbol}: formal predictions lack required evidence columns.")
        rows.to_csv(target / "holdout_predictions.csv", index=False)
        self._write(target / "metrics.json", metrics); self._write(target / "diagnostics.json", diagnostics)

    def write_methodology_manifest(self, data_cutoff: str, companies: list[str]) -> None:
        self._write("methodology_manifest.json", {"run_id": self.run_id, "created_at": datetime.now(timezone.utc, ), "git_branch": self._git("branch", "--show-current"), "repository_commit": self._git("rev-parse", "HEAD"), "data_cutoff": data_cutoff, "company_universe": companies, "company_count": len(companies), "research_model_set": ["lag_reg", "arima", "lstm", "naive"], "formal_split": {"development_ratio": .85, "holdout_ratio": .15, "split_basis": "target_date"}, "lstm": {"input_design": "univariate_delta_close", "folds": 5, "max_epochs": 200, "patience": 10, "seed": 42}, "artifact_policy": {"formal_immutable": True, "deployment_separate": True}, "dependencies": {name: self._version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "torch")}})

    def write_statistics(self, statistics: dict) -> None: self._write("statistical_tests.json", statistics)
    def _write(self, relative, payload) -> None:
        path = relative if isinstance(relative, Path) else self.path / relative
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    @staticmethod
    def _version(name):
        try: return version(name)
        except Exception: return None
    @staticmethod
    def _git(*args):
        try: return subprocess.check_output(["git", *args], text=True).strip()
        except Exception: return None
