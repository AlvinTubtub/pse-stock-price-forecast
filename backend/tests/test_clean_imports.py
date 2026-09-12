"""Fresh-process import regression tests for package dependency boundaries."""

from pathlib import Path
import subprocess
import sys

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_name",
    (
        "src.models.arima",
        "src.models.lag_regression",
        "src.models.lstm",
        "src.training.train_arima",
        "src.training.train_lir",
        "src.training.train_lstm",
        "src.training.orchestration",
    ),
)
def test_authoritative_modules_import_in_fresh_process(module_name: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
