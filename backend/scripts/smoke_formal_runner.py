#!/usr/bin/env python3
"""Development-only smoke test for formal tuning paths.

This command never evaluates the formal holdout and never creates a formal
run directory.  It uses at most two real company datasets, the full LASSO
grid, and one representative LSTM configuration across the five common folds
and three required seeds.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.data_validator import validate_ohlcv_csv
from services.forecasting import lag_regression, lstm_model
from services.time_series_cv import (
    create_development_cv_date_plan,
    create_formal_evaluation_plan,
    development_ohlcv_for_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a non-formal development tuning smoke test.")
    parser.add_argument("--symbols", nargs="+", default=["BPI"])
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "raw")
    parser.add_argument("--epochs", type=int, default=2, help="Smoke-only LSTM epoch cap.")
    args = parser.parse_args()
    symbols = [symbol.upper() for symbol in args.symbols]
    if not 1 <= len(symbols) <= 2 or len(set(symbols)) != len(symbols):
        parser.error("--symbols requires one or two unique company symbols.")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = {"mode": "development_only_smoke", "formal_artifacts_written": False, "companies": {}}
    original_epochs = lstm_model.EPOCHS
    try:
        lstm_model.EPOCHS = args.epochs
        for symbol in symbols:
            logging.info("Smoke stage 1/3 %s: validating source and development dates.", symbol)
            frame = validate_ohlcv_csv(args.raw_dir / f"{symbol}.csv")
            plan = create_formal_evaluation_plan(frame, symbol)
            development = development_ohlcv_for_plan(frame, plan)

            logging.info("Smoke stage 2/3 %s: exercising the complete expanded LASSO grid.", symbol)
            alpha, alpha_evidence = lag_regression._select_alpha_with_evidence(
                lag_regression._usable_features(development)
            )

            logging.info("Smoke stage 3/3 %s: exercising common LSTM folds and required seeds.", symbol)
            cv_plan = create_development_cv_date_plan(plan, maximum_lookback=max(lstm_model.LOOKBACK_GRID))
            config = lstm_model.LSTMConfig(
                lookback=max(lstm_model.LOOKBACK_GRID),
                hidden_size=min(lstm_model.HIDDEN_UNITS_GRID),
                learning_rate=min(lstm_model.LEARNING_RATE_GRID),
                batch_size=max(lstm_model.BATCH_SIZE_GRID),
            )
            mean_rmse, rmse_std, folds = lstm_model._evaluate_formal_config(development, config, cv_plan)
            if mean_rmse is None or rmse_std is None or len(folds) != 5:
                raise RuntimeError(f"{symbol}: LSTM smoke did not complete all five development folds.")
            if any([row["seed"] for row in fold["seed_results"]] != [42, 123, 2026] for fold in folds):
                raise RuntimeError(f"{symbol}: LSTM smoke did not execute all required seeds.")

            summary["companies"][symbol] = {
                "source_rows": len(frame),
                "development_target_count": plan.development_count,
                "lasso_grid_count": alpha_evidence["grid_count"],
                "lasso_selected_alpha": alpha,
                "lasso_selected_at_boundary": alpha_evidence["selected_at_boundary"],
                "lstm_smoke_configuration": config.__dict__,
                "lstm_fold_count": len(folds),
                "lstm_seed_count_per_fold": [len(fold["seed_results"]) for fold in folds],
                "lstm_mean_validation_rmse": mean_rmse,
                "lstm_validation_rmse_std": rmse_std,
            }
    finally:
        lstm_model.EPOCHS = original_epochs

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
