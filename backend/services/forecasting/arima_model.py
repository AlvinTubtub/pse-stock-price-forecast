"""Strict formal ARIMA evaluation and separate deployment ARIMA fitting."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from services.evaluation import compute_metrics, run_formal_residual_diagnostics
from services.time_series_cv import FormalEvaluationPlan, development_ohlcv_for_plan, expanding_window_splitter

log = logging.getLogger(__name__)

try:
    import joblib
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover - optional dependency unavailable/incompatible
    HAS_STATSMODELS = False

MAX_P, MAX_D, MAX_Q = 3, 2, 3
DEPLOYMENT_FALLBACK_ORDER = (3, 1, 0)
LJUNG_BOX_LAGS = 10


class ARIMAFormalSelectionError(RuntimeError):
    """Raised when no bounded candidate completes every required CV fold."""


@dataclass(frozen=True)
class ARIMACandidateResult:
    order: tuple[int, int, int]
    required_fold_count: int
    successful_fold_count: int
    valid: bool
    mean_validation_rmse: float | None
    failure_reasons: tuple[str, ...] = ()
    cv_fold_convergence: tuple[bool | None, ...] = ()


@dataclass
class FormalARIMAResult:
    model: object
    order: tuple[int, int, int]
    metrics: dict[str, float]
    forecasts: pd.DataFrame
    backtest: list[float]
    diagnostics: dict[str, object]
    candidate_results: list[ARIMACandidateResult]


def is_stationary(series: pd.Series, alpha: float = 0.05) -> bool:
    """ADF on the supplied development/fold-training Close observations only."""
    if not HAS_STATSMODELS:
        return False
    try:
        _, pvalue, *_ = adfuller(pd.Series(series, dtype=float).dropna())
        return bool(pvalue < alpha)
    except Exception:  # pragma: no cover
        log.warning("ADF failed; prioritizing d=1 without restricting the search.", exc_info=True)
        return False


def _candidate_orders(d_guess: int) -> list[tuple[int, int, int]]:
    """Bounded p<=3, d<=2, q<=3 candidates; ADF only prioritizes d."""
    d_order = sorted({d_guess, *range(MAX_D + 1)}, key=lambda d: (d != d_guess, d))
    return [
        (p, d, q)
        for d in d_order for p in range(MAX_P + 1) for q in range(MAX_Q + 1)
        if not (p == 0 and q == 0)
    ]


def _walk_forward_forecast(initial_fit, future_actuals: np.ndarray) -> np.ndarray:
    """Forecast then append revealed actuals with ``refit=False``.

    Parameters remain fixed across the sequence; only the state is updated
    after each one-step forecast has been issued.
    """
    predictions = np.empty(len(future_actuals), dtype=float)
    current = initial_fit
    for index, actual in enumerate(np.asarray(future_actuals, dtype=float)):
        forecast = current.forecast(steps=1)
        predictions[index] = float(np.asarray(forecast).reshape(-1)[0])
        if not np.isfinite(predictions[index]):
            raise ValueError("ARIMA produced a non-finite one-step forecast.")
        current = current.append([actual], refit=False)
    return predictions


def _fit_convergence_status(fit: object) -> bool | None:
    """Return optimizer convergence when statsmodels exposes it.

    ``None`` is deliberately distinct from ``False``: it means the fitted
    result did not expose a usable optimizer convergence flag.  Formal CV
    currently records this evidence without changing its approved
    finite-prediction fold-completion rule.
    """
    retvals = getattr(fit, "mle_retvals", None)
    if not hasattr(retvals, "get"):
        return None
    converged = retvals.get("converged")
    return None if converged is None else bool(converged)


def _all_folds_converged(statuses: tuple[bool | None, ...]) -> bool | None:
    """Summarize convergence without treating unavailable metadata as true."""
    if any(status is False for status in statuses):
        return False
    if statuses and all(status is True for status in statuses):
        return True
    return None


def _candidate_diagnostics(result: ARIMACandidateResult) -> dict[str, object]:
    """Return JSON-safe convergence and completeness evidence for one order."""
    return {
        "order": list(result.order),
        "required_fold_count": result.required_fold_count,
        "successful_fold_count": result.successful_fold_count,
        "valid_under_finite_prediction_rule": result.valid,
        "mean_validation_rmse": result.mean_validation_rmse,
        "failure_reasons": list(result.failure_reasons),
        "cv_fold_convergence": list(result.cv_fold_convergence),
        "all_cv_folds_converged": _all_folds_converged(result.cv_fold_convergence),
    }


def _evaluate_candidate(close: pd.Series, order: tuple[int, int, int]) -> ARIMACandidateResult:
    """Strict chronological CV: a failed fold invalidates its candidate."""
    splitter = expanding_window_splitter(len(close))
    required = splitter.get_n_splits()
    rmses: list[float] = []
    failures: list[str] = []
    convergence: list[bool | None] = []
    for fold_index, (train_idx, validation_idx) in enumerate(splitter.split(close), start=1):
        try:
            fit = ARIMA(close.iloc[train_idx], order=order).fit()
            converged = _fit_convergence_status(fit)
            convergence.append(converged)
            if converged is False:
                log.warning(
                    "Formal ARIMA order=%s fold=%d did not converge; recording it under the current finite-prediction rule.",
                    order,
                    fold_index,
                )
            actual = close.iloc[validation_idx].to_numpy(dtype=float)
            predicted = _walk_forward_forecast(fit, actual)
            if len(predicted) != len(actual) or not np.isfinite(predicted).all():
                raise ValueError("unusable validation prediction")
            rmses.append(float(np.sqrt(np.mean((actual - predicted) ** 2))))
        except Exception as exc:
            if len(convergence) < fold_index:
                convergence.append(None)
            failures.append(f"fold {fold_index}: {type(exc).__name__}: {exc}")
    valid = len(rmses) == required
    return ARIMACandidateResult(
        order=order, required_fold_count=required, successful_fold_count=len(rmses), valid=valid,
        mean_validation_rmse=float(np.mean(rmses)) if valid else None,
        failure_reasons=tuple(failures),
        cv_fold_convergence=tuple(convergence),
    )


def _select_formal_order(development_close: pd.Series) -> tuple[tuple[int, int, int], list[ARIMACandidateResult]]:
    """Select the lowest-RMSE valid bounded candidate or raise strictly."""
    if not HAS_STATSMODELS:
        raise ARIMAFormalSelectionError("statsmodels is unavailable; formal ARIMA cannot use a deployment fallback.")
    d_guess = 0 if is_stationary(development_close) else 1
    results = [_evaluate_candidate(development_close, order) for order in _candidate_orders(d_guess)]
    valid = [result for result in results if result.valid and result.mean_validation_rmse is not None]
    if not valid:
        raise ARIMAFormalSelectionError("No ARIMA candidate completed all required rolling-origin folds.")
    winner = min(valid, key=lambda result: (result.mean_validation_rmse, result.order))
    return winner.order, results


def _adf_metadata(series: pd.Series) -> dict:
    """Record the development-only ADF decision used to prioritize orders."""
    if not HAS_STATSMODELS:
        return {"computable": False, "reason": "statsmodels_unavailable"}
    try:
        statistic, p_value, *_ = adfuller(pd.Series(series, dtype=float).dropna())
        return {"computable": True, "statistic": float(statistic), "p_value": float(p_value), "stationary_at_alpha_0_05": bool(p_value < .05)}
    except Exception as exc:
        return {"computable": False, "reason": f"adf_failed:{type(exc).__name__}"}


def _holdout_ljung_box(errors: np.ndarray) -> dict:
    """Backward-compatible formal Ljung-Box entrypoint for hold-out errors."""
    return run_formal_residual_diagnostics(errors, include_ljung_box=True)["ljung_box"]


def train_formal_arima(df: pd.DataFrame, plan: FormalEvaluationPlan) -> FormalARIMAResult:
    """Fit/select on development only; issue exact planned OOS hold-out rows."""
    development = development_ohlcv_for_plan(df, plan)
    development_close = development["Close"].astype(float).reset_index(drop=True)
    order, candidates = _select_formal_order(development_close)
    formal_model = ARIMA(development_close, order=order).fit()
    final_fit_converged = _fit_convergence_status(formal_model)
    selected_candidate = next((candidate for candidate in candidates if candidate.order == order), None)

    all_dates = pd.to_datetime(df["Date"])
    lookup = pd.Series(df["Close"].to_numpy(dtype=float), index=all_dates)
    actual = lookup.reindex(plan.holdout_target_dates)
    if actual.isna().any() or len(actual) != plan.holdout_count:
        raise ValueError(f"{plan.symbol}: missing actual Close for a required ARIMA hold-out target date.")
    predicted = _walk_forward_forecast(formal_model, actual.to_numpy(dtype=float))
    forecasts = pd.DataFrame({
        "symbol": plan.symbol, "model": "arima",
        "origin_date": plan.holdout_origin_dates, "target_date": plan.holdout_target_dates,
        "actual_close": actual.to_numpy(dtype=float), "predicted_close": predicted,
    })
    forecasts["error"] = forecasts["actual_close"] - forecasts["predicted_close"]
    ljung_box = _holdout_ljung_box(forecasts["error"].to_numpy())
    diagnostics = {
        "selected_order": list(order),
        "cv_fold_convergence": list(selected_candidate.cv_fold_convergence) if selected_candidate else None,
        "all_cv_folds_converged": _all_folds_converged(selected_candidate.cv_fold_convergence) if selected_candidate else None,
        "final_fit_converged": final_fit_converged,
        "candidate_cv": [_candidate_diagnostics(candidate) for candidate in candidates],
        "adf": _adf_metadata(development_close),
        "ljung_box": ljung_box,
        **ljung_box,
        **run_formal_residual_diagnostics(forecasts["error"].to_numpy()),
    }
    metrics = compute_metrics(forecasts["actual_close"], forecasts["predicted_close"], y_train=development_close)
    metrics["ljung_box_pvalue"] = ljung_box["p_value"]
    return FormalARIMAResult(formal_model, order, metrics, forecasts, forecasts["predicted_close"].tolist(), diagnostics, candidates)


def train_deployment_arima(df: pd.DataFrame):
    """Fit the persisted operational model on all available approved Close data."""
    close = df["Close"].astype(float).reset_index(drop=True)
    if not HAS_STATSMODELS:
        return None, DEPLOYMENT_FALLBACK_ORDER
    try:
        order, _ = _select_formal_order(close)
    except ARIMAFormalSelectionError:
        # Operational resilience only; this bounded fallback is never used by
        # formal research evaluation or its metrics.
        log.warning("Deployment ARIMA CV failed; using bounded operational fallback %s.", DEPLOYMENT_FALLBACK_ORDER)
        order = DEPLOYMENT_FALLBACK_ORDER
    return ARIMA(close, order=order).fit(), order


def train(df: pd.DataFrame):
    """Backward-compatible deployment training tuple; use train_formal_arima for research."""
    model, order = train_deployment_arima(df)
    if model is None:
        return None, order, {}, float(df["Close"].iloc[-1]), [], [], []
    return model, order, {}, predict_next(model), [], [], []


def save(model, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if model is not None:
        joblib.dump(model, path)


def load(path):
    return joblib.load(path)


def predict_next(model) -> float:
    return float(np.asarray(model.forecast(steps=1)).reshape(-1)[0])
