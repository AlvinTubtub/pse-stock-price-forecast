"""Post-export validation for the frontend JSON artifacts.

Runs as the last check before the Fast Pipeline commits/pushes anything —
after export_forecast_artifacts.py has written frontend/public/forecasts/.
Confirms the export actually produced a complete, consistent 15-ticker
dashboard rather than silently committing a partial/broken data set.

This does NOT re-validate OHLCV data (that already happens earlier, both
inside services/pdf_pipeline/pipeline.py post-merge and via
services/data_validator.py) — it only checks the shape of what
export_forecast_artifacts.py just wrote.

Usage:
    python backend/scripts/validate_exports.py

Exit codes:
    0  all checks passed
    1  one or more checks failed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
REPO_ROOT = BASE_DIR.parent
FORECASTS_DIR = REPO_ROOT / "frontend" / "public" / "forecasts"

sys.path.insert(0, str(BASE_DIR))
from services.pdf_pipeline.config import TARGET_COMPANIES  # noqa: E402

EXPECTED_TICKERS = sorted(TARGET_COMPANIES.keys())
BACKTEST_MODELS = {"Lag-Informed Regression", "ARIMA", "LSTM", "Naive baseline"}


def _load(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise AssertionError(f"{path} is not valid JSON: {exc}") from exc


def main() -> int:
    errors: list[str] = []

    companies = _load(FORECASTS_DIR / "companies.json")
    if companies is None:
        errors.append(f"Missing {FORECASTS_DIR / 'companies.json'}")
    else:
        found_symbols = sorted(c["symbol"] for c in companies)
        missing = sorted(set(EXPECTED_TICKERS) - set(found_symbols))
        extra = sorted(set(found_symbols) - set(EXPECTED_TICKERS))
        if missing:
            errors.append(f"companies.json is missing ticker(s): {', '.join(missing)}")
        if extra:
            errors.append(f"companies.json has unexpected ticker(s): {', '.join(extra)}")
        if len(found_symbols) != len(EXPECTED_TICKERS):
            errors.append(
                f"companies.json has {len(found_symbols)} companies, expected {len(EXPECTED_TICKERS)}"
            )

    dashboard = _load(FORECASTS_DIR / "dashboard.json")
    if dashboard is None:
        errors.append(f"Missing {FORECASTS_DIR / 'dashboard.json'}")
    else:
        if dashboard.get("missingCompanies"):
            errors.append(f"dashboard.json reports missingCompanies: {dashboard['missingCompanies']}")
        if dashboard.get("totalCompanies") != len(EXPECTED_TICKERS):
            errors.append(
                f"dashboard.json totalCompanies={dashboard.get('totalCompanies')}, "
                f"expected {len(EXPECTED_TICKERS)}"
            )

    latest = _load(FORECASTS_DIR / "latest.json")
    if latest is None:
        errors.append(f"Missing {FORECASTS_DIR / 'latest.json'}")
    elif latest.get("status") == "error":
        errors.append("latest.json status is 'error' — refusing to publish a failed run's artifacts")

    metrics = _load(FORECASTS_DIR / "metrics.json")
    if metrics is None:
        errors.append(f"Missing {FORECASTS_DIR / 'metrics.json'}")

    for symbol in EXPECTED_TICKERS:
        company_path = FORECASTS_DIR / "company" / f"{symbol}.json"
        history_path = FORECASTS_DIR / "history" / f"{symbol}.json"
        if not company_path.exists():
            errors.append(f"Missing {company_path}")
        if not history_path.exists():
            errors.append(f"Missing {history_path}")
        detail = _load(company_path)
        if not isinstance(detail, dict):
            continue
        dates = detail.get("backtestDates")
        actual = detail.get("backtestActual")
        by_model = detail.get("backtestByModel")
        methodology = detail.get("backtestMethodology")
        if not isinstance(dates, list) or len(dates) != 60 or dates != sorted(dates) or len(set(dates)) != 60:
            errors.append(f"{symbol}: backtestDates must contain exactly 60 unique chronological sessions")
            continue
        if not isinstance(actual, list) or len(actual) != len(dates):
            errors.append(f"{symbol}: backtestActual does not align with backtestDates")
        if not isinstance(by_model, dict) or set(by_model) != BACKTEST_MODELS:
            errors.append(f"{symbol}: backtestByModel must contain exactly the four audited models")
        elif any(not isinstance(values, list) or len(values) != len(dates) for values in by_model.values()):
            errors.append(f"{symbol}: a model prediction series does not align with backtestDates")
        if methodology != {"source": "audited_oos_holdout", "alignment": "common_target_date", "window": 60}:
            errors.append(f"{symbol}: backtest methodology metadata is missing or invalid")

    print("=" * 60)
    print("Validating exported frontend artifacts...")
    print(f"Directory: {FORECASTS_DIR}")
    print("=" * 60)

    if errors:
        print(f"FAILED — {len(errors)} issue(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"OK — {len(EXPECTED_TICKERS)}/{len(EXPECTED_TICKERS)} companies present, "
          "dashboard/metrics/latest all consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
