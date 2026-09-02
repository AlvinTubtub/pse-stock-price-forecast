"""Strict formal ARIMA evaluation and separate deployment ARIMA fitting."""
from __future__ import annotations

import logging
from dataclasses import dataclass

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
DEPLOYMENT_FALLBACK_TREND = "n"
LJUNG_BOX_LAGS = 10
FORMAL_CV_FOLD_COUNT = 5
FORMAL_OPTIMIZER_ATTEMPTS = (
    {"method": "statespace", "maxiter": 500},
    {"method": "statespace", "maxiter": 2_000},
)


class ARIMAFormalSelectionError(RuntimeError):
    """Raised when no bounded candidate completes every required CV fold."""


@dataclass(frozen=True, order=True)
class ARIMAConfiguration:
    """A formal ARIMA candidate identified by both order and trend."""

    order: tuple[int, int, int]
    trend: str


@dataclass(frozen=True)
class ARIMAFitAttempt:
    """JSON-serializable evidence for one predeclared optimizer attempt."""

    attempt_number: int
    method: str
    maxiter: int
    fit_completed: bool
    converged: bool | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class ARIMAFoldResult:
    """Formal evidence for one expanding-window validation fold."""

    fold_number: int
    successful: bool
    converged: bool | None
    validation_rmse: float | None
    attempts: tuple[ARIMAFitAttempt, ...]
    failure_reason: str | None = None


@dataclass(frozen=True)
class ARIMACandidateResult:
    configuration: ARIMAConfiguration
    required_fold_count: int
    successful_fold_count: int
    valid: bool
    mean_validation_rmse: float | None
    failure_reasons: tuple[str, ...] = ()
    fold_results: tuple[ARIMAFoldResult, ...] = ()

    @property
    def order(self) -> tuple[int, int, int]:
        """Keep read-only order access for existing reporting code."""
        return self.configuration.order

    @property
    def trend(self) -> str:
        return self.configuration.trend

    @property
    def cv_fold_convergence(self) -> tuple[bool | None, ...]:
        return tuple(fold.converged for fold in self.fold_results)


@dataclass
class FormalARIMAResult:
    model: object
    order: tuple[int, int, int]
    trend: str
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


def _trends_for_d(d: int) -> tuple[str, ...]:
    """Return the predeclared statsmodels trend codes for differencing order."""
    return {0: ("n", "c"), 1: ("n", "t"), 2: ("n",)}[d]


def _candidate_configurations(d_guess: int) -> list[ARIMAConfiguration]:
    """Return the complete bounded order/trend grid; ADF only prioritizes d."""
    d_order = sorted({d_guess, *range(MAX_D + 1)}, key=lambda d: (d != d_guess, d))
    return [
        ARIMAConfiguration(order=(p, d, q), trend=trend)
        for d in d_order
        for p in range(MAX_P + 1)
        for q in range(MAX_Q + 1)
        for trend in _trends_for_d(d)
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
    treats both unavailable metadata and explicit non-convergence as failures.
    """
    retvals = getattr(fit, "mle_retvals", None)
    if not hasattr(retvals, "get"):
        return None
    converged = retvals.get("converged")
    return None if converged is None else bool(converged)


def _fit_with_formal_retries(
    values: pd.Series,
    configuration: ARIMAConfiguration,
    *,
    context: str,
) -> tuple[object | None, tuple[ARIMAFitAttempt, ...]]:
    """Fit using only the deterministic, predeclared increasing-maxiter policy."""
    attempts: list[ARIMAFitAttempt] = []
    for attempt_number, settings in enumerate(FORMAL_OPTIMIZER_ATTEMPTS, start=1):
        method = str(settings["method"])
        maxiter = int(settings["maxiter"])
        try:
            fit = ARIMA(
                values,
                order=configuration.order,
                trend=configuration.trend,
            ).fit(method=method, method_kwargs={"maxiter": maxiter})
            converged = _fit_convergence_status(fit)
            reason = (
                None if converged is True
                else "optimizer_not_converged" if converged is False
                else "convergence_status_unavailable"
            )
            attempts.append(ARIMAFitAttempt(
                attempt_number=attempt_number,
                method=method,
                maxiter=maxiter,
                fit_completed=True,
                converged=converged,
                failure_reason=reason,
            ))
            if converged is True:
                log.info(
                    "Formal ARIMA %s configuration=%s trend=%s converged on attempt=%d maxiter=%d.",
                    context,
                    configuration.order,
                    configuration.trend,
                    attempt_number,
                    maxiter,
                )
                return fit, tuple(attempts)
            log.warning(
                "Formal ARIMA %s configuration=%s trend=%s attempt=%d has convergence=%s.",
                context,
                configuration.order,
                configuration.trend,
                attempt_number,
                converged,
            )
        except Exception as exc:
            attempts.append(ARIMAFitAttempt(
                attempt_number=attempt_number,
                method=method,
                maxiter=maxiter,
                fit_completed=False,
                converged=None,
                failure_reason=f"{type(exc).__name__}: {exc}",
            ))
            log.warning(
                "Formal ARIMA %s configuration=%s trend=%s attempt=%d failed.",
                context,
                configuration.order,
                configuration.trend,
                attempt_number,
                exc_info=True,
            )
    return None, tuple(attempts)


def _attempt_diagnostics(attempt: ARIMAFitAttempt) -> dict[str, object]:
    return {
        "attempt_number": attempt.attempt_number,
        "method": attempt.method,
        "maxiter": attempt.maxiter,
        "fit_completed": attempt.fit_completed,
        "converged": attempt.converged,
        "failure_reason": attempt.failure_reason,
    }


def _fold_diagnostics(fold: ARIMAFoldResult) -> dict[str, object]:
    return {
        "fold_number": fold.fold_number,
        "successful": fold.successful,
        "converged": fold.converged,
        "validation_rmse": fold.validation_rmse,
        "failure_reason": fold.failure_reason,
        "optimizer_attempts": [_attempt_diagnostics(attempt) for attempt in fold.attempts],
    }


def _candidate_diagnostics(result: ARIMACandidateResult) -> dict[str, object]:
    """Return JSON-safe convergence and completeness evidence for one candidate."""
    return {
        "order": list(result.order),
        "trend": result.trend,
        "configuration": {"order": list(result.order), "trend": result.trend},
        "required_fold_count": result.required_fold_count,
        "successful_fold_count": result.successful_fold_count,
        "valid": result.valid,
        "mean_validation_rmse": result.mean_validation_rmse,
        "failure_reasons": list(result.failure_reasons),
        "folds": [_fold_diagnostics(fold) for fold in result.fold_results],
        "all_cv_folds_converged": result.valid and all(
            fold.converged is True for fold in result.fold_results
        ),
    }


def _evaluate_candidate(close: pd.Series, configuration: ARIMAConfiguration) -> ARIMACandidateResult:
    """Require five finite, confirmed-converged chronological folds."""
    splitter = expanding_window_splitter(len(close))
    available = splitter.get_n_splits()
    required = FORMAL_CV_FOLD_COUNT
    rmses: list[float] = []
    failures: list[str] = []
    fold_results: list[ARIMAFoldResult] = []
    if available != required:
        failures.append(f"expected {required} folds but splitter produced {available}")
    for fold_index, (train_idx, validation_idx) in enumerate(splitter.split(close), start=1):
        fit, attempts = _fit_with_formal_retries(
            close.iloc[train_idx],
            configuration,
            context=f"CV fold {fold_index}",
        )
        if fit is None:
            final_status = attempts[-1].converged if attempts else None
            reason = (
                "confirmed non-convergence after all retries"
                if attempts and all(attempt.converged is False for attempt in attempts)
                else "no optimizer attempt produced confirmed convergence"
            )
            failures.append(f"fold {fold_index}: {reason}")
            fold_results.append(ARIMAFoldResult(
                fold_number=fold_index,
                successful=False,
                converged=final_status,
                validation_rmse=None,
                attempts=attempts,
                failure_reason=reason,
            ))
            continue
        try:
            actual = close.iloc[validation_idx].to_numpy(dtype=float)
            predicted = _walk_forward_forecast(fit, actual)
            if len(predicted) != len(actual) or not np.isfinite(predicted).all():
                raise ValueError("unusable validation prediction")
            rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
            if not np.isfinite(rmse):
                raise ValueError("non-finite validation RMSE")
            rmses.append(rmse)
            fold_results.append(ARIMAFoldResult(
                fold_number=fold_index,
                successful=True,
                converged=True,
                validation_rmse=rmse,
                attempts=attempts,
            ))
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failures.append(f"fold {fold_index}: {reason}")
            fold_results.append(ARIMAFoldResult(
                fold_number=fold_index,
                successful=False,
                converged=True,
                validation_rmse=None,
                attempts=attempts,
                failure_reason=reason,
            ))
    valid = available == required and len(rmses) == required and len(fold_results) == required
    return ARIMACandidateResult(
        configuration=configuration,
        required_fold_count=required,
        successful_fold_count=len(rmses),
        valid=valid,
        mean_validation_rmse=float(np.mean(rmses)) if valid else None,
        failure_reasons=tuple(failures),
        fold_results=tuple(fold_results),
    )


def _select_formal_configuration(
    development_close: pd.Series,
) -> tuple[ARIMAConfiguration, list[ARIMACandidateResult]]:
    """Select the lowest-RMSE valid bounded candidate or raise strictly."""
    if not HAS_STATSMODELS:
        raise ARIMAFormalSelectionError("statsmodels is unavailable; formal ARIMA cannot use a deployment fallback.")
    d_guess = 0 if is_stationary(development_close) else 1
    results = [
        _evaluate_candidate(development_close, configuration)
        for configuration in _candidate_configurations(d_guess)
    ]
    valid = [
        result for result in results
        if result.valid
        and result.mean_validation_rmse is not None
        and np.isfinite(result.mean_validation_rmse)
    ]
    if not valid:
        raise ARIMAFormalSelectionError("No ARIMA candidate completed all required rolling-origin folds.")
    winner = min(
        valid,
        key=lambda result: (
            result.mean_validation_rmse,
            result.configuration.order,
            result.configuration.trend,
        ),
    )
    return winner.configuration, results


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
    configuration, candidates = _select_formal_configuration(development_close)
    formal_model, final_fit_attempts = _fit_with_formal_retries(
        development_close,
        configuration,
        context="final development fit",
    )
    final_fit_confirmed = (
        formal_model is not None
        and bool(final_fit_attempts)
        and final_fit_attempts[-1].converged is True
    )
    if not final_fit_confirmed:
        raise ARIMAFormalSelectionError(
            f"{plan.symbol}: selected ARIMA configuration {configuration.order} "
            f"trend={configuration.trend!r} did not achieve confirmed convergence "
            "on the final development fit."
        )
    selected_candidate = next(
        (candidate for candidate in candidates if candidate.configuration == configuration),
        None,
    )
    if selected_candidate is None or not selected_candidate.valid:
        raise ARIMAFormalSelectionError(
            f"{plan.symbol}: selected ARIMA configuration lacks complete converged CV evidence."
        )

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
        "selected_order": list(configuration.order),
        "selected_trend": configuration.trend,
        "selected_configuration": {
            "order": list(configuration.order),
            "trend": configuration.trend,
        },
        "selection_metric": "mean_full_precision_validation_rmse",
        "tie_breaking": "mean_validation_rmse_then_order_then_trend",
        "optimizer_retry_policy": [dict(settings) for settings in FORMAL_OPTIMIZER_ATTEMPTS],
        "cv_fold_results": [_fold_diagnostics(fold) for fold in selected_candidate.fold_results],
        "cv_fold_convergence": list(selected_candidate.cv_fold_convergence),
        "all_cv_folds_converged": all(
            fold.converged is True for fold in selected_candidate.fold_results
        ),
        "final_fit_converged": final_fit_confirmed,
        "final_fit_attempts": [_attempt_diagnostics(attempt) for attempt in final_fit_attempts],
        "candidate_cv": [_candidate_diagnostics(candidate) for candidate in candidates],
        "adf": _adf_metadata(development_close),
        "ljung_box": ljung_box,
        **ljung_box,
        **run_formal_residual_diagnostics(forecasts["error"].to_numpy()),
    }
    metrics = compute_metrics(forecasts["actual_close"], forecasts["predicted_close"], y_train=development_close)
    metrics["ljung_box_pvalue"] = ljung_box["p_value"]
    return FormalARIMAResult(
        formal_model,
        configuration.order,
        configuration.trend,
        metrics,
        forecasts,
        forecasts["predicted_close"].tolist(),
        diagnostics,
        candidates,
    )


def train_deployment_arima(df: pd.DataFrame):
    """Legacy/manual retuning API; scheduled refresh uses explicit configuration."""
    close = df["Close"].astype(float).reset_index(drop=True)
    if not HAS_STATSMODELS:
        return None, DEPLOYMENT_FALLBACK_ORDER
    try:
        configuration, _ = _select_formal_configuration(close)
    except ARIMAFormalSelectionError:
        # Operational resilience only; this bounded fallback is never used by
        # formal research evaluation or its metrics.
        log.warning("Deployment ARIMA CV failed; using bounded operational fallback %s.", DEPLOYMENT_FALLBACK_ORDER)
        configuration = ARIMAConfiguration(DEPLOYMENT_FALLBACK_ORDER, DEPLOYMENT_FALLBACK_TREND)
    model = ARIMA(
        close,
        order=configuration.order,
        trend=configuration.trend,
    ).fit()
    return model, configuration.order


def refit_deployment_arima(
    df: pd.DataFrame,
    configuration: ARIMAConfiguration,
):
    """Refit an approved operational configuration without selection or fallback."""
    p, d, q = configuration.order
    if not (0 <= p <= MAX_P and 0 <= d <= MAX_D and 0 <= q <= MAX_Q):
        raise ValueError(f"Unsupported approved ARIMA order: {configuration.order}.")
    if configuration.trend not in _trends_for_d(configuration.order[1]):
        raise ValueError(
            f"Approved ARIMA trend {configuration.trend!r} is invalid for order {configuration.order}."
        )
    close = df["Close"].astype(float).reset_index(drop=True)
    model = ARIMA(
        close,
        order=configuration.order,
        trend=configuration.trend,
    ).fit(method="statespace", method_kwargs={"maxiter": 2_000})
    convergence = _fit_convergence_status(model)
    if convergence is not True:
        raise RuntimeError(
            "Deployment refresh ARIMA fit lacks confirmed optimizer convergence "
            f"for order={configuration.order}, trend={configuration.trend!r}; status={convergence}."
        )
    log.info(
        "Deployment refresh ARIMA: order=%s, trend=%s, convergence confirmed.",
        configuration.order,
        configuration.trend,
    )
    return model


def retune_deployment_arima(df: pd.DataFrame) -> tuple[object, ARIMAConfiguration]:
    """Manually tune an operational challenger without formal-run output."""
    close = df["Close"].astype(float).reset_index(drop=True)
    configuration, _candidates = _select_formal_configuration(close)
    log.info(
        "Deployment retuning ARIMA challenger selected order=%s, trend=%s.",
        configuration.order,
        configuration.trend,
    )
    return refit_deployment_arima(df, configuration), configuration


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
