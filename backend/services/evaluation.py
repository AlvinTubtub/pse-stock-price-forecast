"""Shared model-evaluation utilities.

One metrics implementation used by every forecasting model (Lag-Informed
Regression, ARIMA, LSTM) and by the naive baseline, so RMSE/MAE/MASE/R²
are always computed the same way regardless of which model produced the
predictions. All metrics are computed on reconstructed peso prices, never
on a differenced/scaled internal target.

Also implements the capstone paper's cross-model statistical-significance
suite, run once per pipeline update across every ticker
(services/model_selector.py writes the results to statistical_tests.json):

  - Diebold-Mariano test, Newey-West HAC variance + Harvey-Leybourne-
    Newbold (HLN) small-sample correction — pairwise, *within* each
    company.
  - Holm-Bonferroni correction for the resulting multiple comparisons.
  - Friedman rank test — *across* companies, one observation per model
    per company (its test-set RMSE).
  - Holm-adjusted Wilcoxon signed-rank tests — pairwise post-hoc follow-up
    to a significant Friedman result.
  - Best-model consistency check — whether one model has the lowest RMSE
    on at least a majority (8 of 15, by default) of companies.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, shapiro, t as t_dist, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    HAS_STATSMODELS_DIAGNOSTICS = True
except Exception:  # pragma: no cover - environment-dependent optional diagnostics
    HAS_STATSMODELS_DIAGNOSTICS = False

log = logging.getLogger(__name__)

from services.time_series_cv import FormalEvaluationPlan, build_forecast_rows


def _naive_mae(y_reference: np.ndarray) -> float:
    """The mean absolute error of a naive one-step (yesterday's value)
    forecast, computed over ``y_reference`` — this is the MASE scaling
    denominator. Per Hyndman & Koehler, this should be the *in-sample*
    (training) series, not the held-out series being scored, so that the
    denominator reflects how hard the series is to forecast naively
    independent of the test window.
    """
    y_reference = np.asarray(y_reference, dtype=float)
    if len(y_reference) < 2:
        return 1e-8
    return float(np.mean(np.abs(np.diff(y_reference)))) or 1e-8


def common_mase_denominator(development_close) -> float:
    """Single company-level MASE denominator from development Close only.

    A zero denominator is non-computable rather than silently replaced by an
    epsilon, which would fabricate a MASE scale.
    """
    values = np.asarray(development_close, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Development Close must contain at least two finite observations for MASE.")
    denominator = float(np.mean(np.abs(np.diff(values))))
    if denominator == 0.0:
        raise ValueError("MASE is not computable: development naive MAE is zero.")
    return denominator


def compute_metrics(y_true, y_pred, y_train=None, mase_denominator: float | None = None) -> dict:
    """RMSE, MAE, MASE, R² as full-precision numeric values.

    ``y_train`` should be the in-sample (training) target series, used as
    the naive one-step-forecast baseline that scales MASE. When omitted
    (e.g. the naive baseline scoring itself), ``y_true`` is used instead.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    denominator = float(mase_denominator) if mase_denominator is not None else _naive_mae(y_true if y_train is None else y_train)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("MASE denominator must be a positive finite value.")
    mase = mae / denominator

    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "mase": float(mase),
        "r2": r2,
    }


def build_naive_formal_forecasts(df: pd.DataFrame, plan: FormalEvaluationPlan) -> pd.DataFrame:
    """Build one naive prediction per required formal hold-out target date.

    The naive forecast is exactly ``Close[t]`` for target ``t+1``.  This is
    intentionally date-keyed, so it is suitable for strict cross-model
    validation rather than positional array comparisons.
    """
    rows = build_forecast_rows(df, plan.symbol)
    holdout = rows.loc[rows["target_date"].isin(plan.holdout_target_dates)].copy()
    if len(holdout) != plan.holdout_count:
        raise ValueError(f"{plan.symbol}: naive forecast rows do not match the frozen hold-out plan.")
    holdout["model"] = "naive"
    holdout["predicted_close"] = pd.to_numeric(
        df.set_index(pd.to_datetime(df["Date"]))["Close"].reindex(holdout["origin_date"]).to_numpy(),
        errors="raise",
    )
    holdout["error"] = holdout["actual_close"] - holdout["predicted_close"]
    return holdout[["symbol", "model", "origin_date", "target_date", "actual_close", "predicted_close", "error"]]


def evaluate_naive(df: pd.DataFrame, plan: FormalEvaluationPlan | None = None) -> dict:
    """Baseline: walk-forward one-step naive forecast (predict tomorrow's
    close = today's close), evaluated on the frozen plan when supplied.

    The first test-set prediction is the last *training* observation;
    every prediction after that is the previous *actual* test-set value
    (not the model's own prior prediction) — a true one-step-ahead naive
    forecast, not a shifted copy of the full series. ``y_train`` is
    passed to ``compute_metrics`` so MASE is scaled by the in-sample
    (training-period) naive MAE, per Hyndman & Koehler, matching how the
    other models are scored.
    """
    if plan is not None:
        naive_rows = build_naive_formal_forecasts(df, plan)
        train_series = df.loc[pd.to_datetime(df["Date"]) <= plan.development_end_date, "Close"].to_numpy(dtype=float)
        test_series = naive_rows["actual_close"].to_numpy(dtype=float)
        y_pred = naive_rows["predicted_close"].to_numpy(dtype=float)
    else:
        close = df["Close"].values
        n_test = max(1, int(round(len(close) * 0.15)))
        train_series = close[: len(close) - n_test]
        test_series = close[len(close) - n_test :]
        y_pred = np.concatenate([[train_series[-1]], test_series[:-1]])

    log.info("Naive baseline: %d train rows, %d test rows", len(train_series), len(test_series))

    metrics = compute_metrics(test_series, y_pred, y_train=train_series)

    log.info("Naive baseline evaluation complete.")
    return metrics


def build_comparison_table(metrics_by_model: dict[str, dict], labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Turn {"lag_reg": {...}, "arima": {...}, "lstm": {...}, "naive": {...}}
    into a tidy comparison table — one row per model, ranked by RMSE.
    """
    labels = labels or {}
    rows = []
    for key, metrics in metrics_by_model.items():
        rows.append({
            "Model": labels.get(key, key),
            "RMSE": float(metrics["rmse"]),
            "MAE": float(metrics["mae"]),
            "MASE": float(metrics["mase"]),
            "R2": float(metrics["r2"]),
        })
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return table


def select_best_model(metrics_by_model: dict[str, dict], candidate_keys: list[str]) -> str:
    """Returns the key (from candidate_keys) with the lowest RMSE."""
    return min(candidate_keys, key=lambda k: float(metrics_by_model[k]["rmse"]))


# --------------------------------------------------------------------------
# Statistical significance suite
# --------------------------------------------------------------------------

def _newey_west_hac_variance(d: np.ndarray, max_lag: int) -> float:
    """Newey-West HAC (heteroskedasticity- and autocorrelation-consistent)
    long-run variance estimate of the loss-differential series ``d``,
    using a Bartlett kernel out to ``max_lag`` lags."""
    n = len(d)
    d_bar = d.mean()
    demeaned = d - d_bar
    variance = float(np.mean(demeaned**2))
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        autocov = float(np.mean(demeaned[lag:] * demeaned[:-lag]))
        variance += 2.0 * weight * autocov
    return max(variance, 1e-12)


def diebold_mariano_test(errors1, errors2, h: int = 1, power: int = 2) -> tuple[float, float]:
    """Diebold-Mariano test comparing two models' forecast errors on the
    *same* held-out window, with Newey-West HAC variance (automatic
    bandwidth) and the Harvey-Leybourne-Newbold (HLN) small-sample
    correction. ``errors1``/``errors2`` are actual-minus-predicted arrays,
    paired by forecast date. Returns (dm_statistic, p_value); the p-value
    uses a Student-t reference distribution (T-1 df) per HLN.
    """
    e1 = np.asarray(errors1, dtype=float)
    e2 = np.asarray(errors2, dtype=float)
    n = len(e1)
    if n < 2 or n != len(e2):
        return float("nan"), float("nan")

    loss_diff = np.abs(e1) ** power - np.abs(e2) ** power
    d_bar = float(loss_diff.mean())

    max_lag = max(int(np.floor(4 * (n / 100) ** (2 / 9))), h - 1, 0)
    var_d = _newey_west_hac_variance(loss_diff, max_lag)

    dm_stat = d_bar / np.sqrt(var_d / n)

    hln_correction = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n)
    dm_hln = dm_stat * hln_correction

    p_value = float(2 * (1 - t_dist.cdf(abs(dm_hln), df=max(n - 1, 1))))
    return float(dm_hln), p_value


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values in
    the same order as ``p_values`` (each clipped to [0, 1], monotone
    non-decreasing when sorted by the original p-value)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def dm_result(errors_a, errors_b, model_a: str, model_b: str, *, power: int = 2, alpha: float = .05) -> dict:
    """Detailed DM result. Negative mean differential means model_a has lower loss."""
    a, b = np.asarray(errors_a, float), np.asarray(errors_b, float)
    differential = np.abs(a) ** power - np.abs(b) ** power
    bandwidth = max(int(np.floor(4 * (len(a) / 100) ** (2 / 9))), 0)
    statistic, p_value = diebold_mariano_test(a, b, power=power)
    mean = float(differential.mean()) if len(differential) else float("nan")
    return {"model_a": model_a, "model_b": model_b, "loss": "squared_error" if power == 2 else "absolute_error", "mean_loss_differential": mean, "dm_statistic": statistic, "hln_statistic": statistic, "raw_p_value": p_value, "hac_bandwidth": bandwidth, "direction": "model_a_lower_loss" if mean < 0 else "model_b_lower_loss" if mean > 0 else "equal_loss", "n_observations": len(a), "alpha": alpha}


def moving_block_bootstrap(differential, *, replications: int = 5000, seed: int = 42, block_length: int | None = None) -> dict:
    """Centered moving-block bootstrap of a loss-differential mean.

    Uses ceil(n**(1/3)) contiguous blocks, a predeclared dependence-aware
    rule; tests may explicitly reduce ``replications``.
    """
    values = np.asarray(differential, float)
    n = len(values)
    if n < 2 or not np.isfinite(values).all():
        return {"method": "moving_block_bootstrap", "computable": False, "reason": "fewer_than_two_finite_observations", "bootstrap_replications": replications, "seed": seed}
    block = block_length or max(1, int(np.ceil(n ** (1 / 3))))
    centered = values - values.mean(); blocks = [centered[i:i + block] for i in range(n - block + 1)]
    rng = np.random.default_rng(seed); means = np.empty(replications)
    for index in range(replications):
        sample = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=int(np.ceil(n / block)))])[:n]
        means[index] = sample.mean()
    observed = float(values.mean())
    return {"method": "moving_block_bootstrap", "computable": True, "bootstrap_replications": replications, "seed": seed, "block_length": block, "observed_mean_loss_differential": observed, "bootstrap_p_value": float((1 + np.sum(np.abs(means) >= abs(observed))) / (replications + 1)), "confidence_interval": [float(np.percentile(means + observed, 2.5)), float(np.percentile(means + observed, 97.5))]}


def run_formal_residual_diagnostics(errors, *, include_ljung_box: bool = False) -> dict:
    """Diagnostics for the unmodified chronological formal hold-out errors.

    ARCH uses a deterministic ``min(5, n // 5)`` lag rule and requires at
    least ten errors.  Non-finite or too-short sequences are explicitly
    non-computable; this helper never drops or reorders observations.
    """
    values = np.asarray(errors, dtype=float)
    target = "holdout_forecast_errors"
    n = int(len(values))
    valid = bool(np.isfinite(values).all())
    reason = "nonfinite_errors" if not valid else "insufficient_errors"
    result: dict = {}
    if include_ljung_box:
        if not valid or n < 3 or not HAS_STATSMODELS_DIAGNOSTICS:
            result["ljung_box"] = {"diagnostic": "ljung_box", "diagnostic_target": target, "computable": False, "reason": reason if valid else reason, "lags": [], "n": n, "statistic": None, "p_value": None}
        else:
            lag = min(10, n - 1)
            table = acorr_ljungbox(values, lags=[lag], return_df=True)
            result["ljung_box"] = {"diagnostic": "ljung_box", "diagnostic_target": target, "computable": True, "lags": [lag], "n": n, "statistic": float(table["lb_stat"].iloc[0]), "p_value": float(table["lb_pvalue"].iloc[0])}
    if not valid or n < 3 or np.ptp(values) == 0:
        result["shapiro_wilk"] = {"diagnostic": "shapiro_wilk", "diagnostic_target": target, "computable": False, "reason": "zero_variance_errors" if valid and n >= 3 else reason, "n": n, "statistic": None, "p_value": None}
    else:
        statistic, p_value = shapiro(values)
        result["shapiro_wilk"] = {"diagnostic": "shapiro_wilk", "diagnostic_target": target, "computable": True, "n": n, "statistic": float(statistic), "p_value": float(p_value)}
    if not valid or n < 10 or not HAS_STATSMODELS_DIAGNOSTICS:
        result["arch_lm"] = {"diagnostic": "arch_lm", "diagnostic_target": target, "computable": False, "reason": reason if valid else reason, "n": n, "lags": [], "lm_statistic": None, "lm_p_value": None, "f_statistic": None, "f_p_value": None}
    else:
        lags = min(5, n // 5)
        lm_statistic, lm_p_value, f_statistic, f_p_value = het_arch(values, nlags=lags)
        result["arch_lm"] = {"diagnostic": "arch_lm", "diagnostic_target": target, "computable": True, "n": n, "lags": [lags], "lm_statistic": float(lm_statistic), "lm_p_value": float(lm_p_value), "f_statistic": float(f_statistic), "f_p_value": float(f_p_value)}
    return result


def _diagnostics(errors) -> dict:
    return run_formal_residual_diagnostics(errors)


def _stage(errors: dict[str, np.ndarray], metrics: dict[str, dict], power: int, alpha: float) -> dict:
    principals = ("lag_reg", "arima", "lstm")
    stage1 = [dm_result(errors[key], errors["naive"], key, "naive", power=power, alpha=alpha) for key in principals]
    adjusted = holm_correction([item["raw_p_value"] for item in stage1])
    eligible = []
    for item, holm in zip(stage1, adjusted):
        item["holm_adjusted_p_value"] = holm
        item["beats_naive_rmse"] = metrics[item["model_a"]]["rmse"] < metrics["naive"]["rmse"]
        item["significantly_beats_naive"] = item["mean_loss_differential"] < 0 and holm < alpha
        if item["significantly_beats_naive"]: eligible.append(item["model_a"])
    pairs = [(a, b) for i, a in enumerate(eligible) for b in eligible[i + 1:]]
    stage2 = [dm_result(errors[a], errors[b], a, b, power=power, alpha=alpha) for a, b in pairs]
    for item, holm in zip(stage2, holm_correction([item["raw_p_value"] for item in stage2])):
        item["holm_adjusted_p_value"] = holm; item["significant"] = holm < alpha
    return {"stage1_vs_naive": stage1, "eligible_principal_models": eligible, "stage2_executed": len(eligible) >= 2, "stage2_principal": stage2, "reason": None if len(eligible) >= 2 else "fewer_than_two_models_significantly_beat_naive"}


def friedman_test(rmse_by_model: dict[str, list[float]]) -> dict:
    """Friedman rank test across companies: each model contributes one
    RMSE observation per company (paired by company). Tests whether the
    average ranks of the models differ significantly.

    ``rmse_by_model``: {model_key: [rmse_company_1, rmse_company_2, ...]},
    all lists the same length and in the same company order.
    """
    groups = list(rmse_by_model.values())
    n_companies = len(groups[0]) if groups else 0
    if len(groups) < 3 or n_companies < 3:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_companies": n_companies}
    statistic, p_value = friedmanchisquare(*groups)
    return {"statistic": float(statistic), "p_value": float(p_value), "n_companies": n_companies}


def holm_wilcoxon_posthoc(rmse_by_model: dict[str, list[float]]) -> dict:
    """Pairwise Wilcoxon signed-rank tests (paired by company) between
    every pair of models, Holm-adjusted across all pairs. Post-hoc
    follow-up to a significant Friedman result.

    Returns {"model_a vs model_b": {"statistic":.., "p_value":..,
    "holm_p_value":..}, ...}.
    """
    keys = list(rmse_by_model.keys())
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1 :]]

    raw_results = []
    for a, b in pairs:
        try:
            statistic, p_value = wilcoxon(rmse_by_model[a], rmse_by_model[b])
            statistic, p_value = float(statistic), float(p_value)
        except ValueError:  # all differences zero, or too few samples
            statistic, p_value = float("nan"), 1.0
        raw_results.append((statistic, p_value))

    adjusted = holm_correction([p for _, p in raw_results])

    return {
        f"{a} vs {b}": {"statistic": stat, "p_value": p, "holm_p_value": holm_p}
        for (a, b), (stat, p), holm_p in zip(pairs, raw_results, adjusted)
    }


def best_model_consistency_check(rmse_by_model: dict[str, list[float]], min_companies: int = 8) -> dict:
    """Whether one model has the lowest RMSE on at least ``min_companies``
    (out of the total, paired by company) — a consistency check that the
    overall best model isn't a fluke of averaging."""
    keys = list(rmse_by_model.keys())
    n_companies = len(rmse_by_model[keys[0]]) if keys else 0

    counts = {k: 0 for k in keys}
    for company_idx in range(n_companies):
        rmses = {k: rmse_by_model[k][company_idx] for k in keys}
        winner = min(rmses, key=rmses.get)
        counts[winner] += 1

    dominant_count = max(counts.values(), default=0)
    tied_models = sorted(model for model, count in counts.items() if count == dominant_count)
    tie = len(tied_models) > 1
    dominant_model = None if tie else tied_models[0] if tied_models else None

    return {
        "counts": counts,
        "total_companies": n_companies,
        "min_required": min_companies,
        "dominant_model": dominant_model,
        "dominant_count": dominant_count,
        "tie": tie,
        "tied_models": tied_models if tie else [],
        "pass": dominant_count >= min_companies,
    }


def run_formal_statistical_tests(companies: dict[str, dict], *, alpha: float = .05, permutations: int = 10000, seed: int = 42) -> dict:
    """Phase-5 hierarchy over strictly date-aligned formal forecast frames.

    Each company entry contains ``forecasts`` keyed by the four model names
    and its development-only ``development_close`` series.
    """
    per_company, mase_rows, rmse_rows = {}, {}, {}
    keys = ("lag_reg", "arima", "lstm", "naive")
    for symbol, payload in sorted(companies.items()):
        denominator = common_mase_denominator(payload["development_close"])
        frames = payload["forecasts"]
        errors = {key: frames[key]["error"].to_numpy(float) for key in keys}
        metrics = {key: compute_metrics(frames[key]["actual_close"], frames[key]["predicted_close"], mase_denominator=denominator) for key in keys}
        per_company[symbol] = {"mase_denominator": denominator, "metrics": metrics, "dm_squared_error": _stage(errors, metrics, 2, alpha), "dm_absolute_error": _stage(errors, metrics, 1, alpha), "moving_block_bootstrap": {key: moving_block_bootstrap(np.abs(errors[key]) ** 2 - np.abs(errors["naive"]) ** 2) for key in ("lag_reg", "arima", "lstm")}, "diagnostics": {key: _diagnostics(errors[key]) for key in ("lag_reg", "arima", "lstm")}}
        mase_rows[symbol] = {key: metrics[key]["mase"] for key in keys}; rmse_rows[symbol] = {key: metrics[key]["rmse"] for key in keys if key != "naive"}
    ordered = sorted(mase_rows); matrix = {key: [mase_rows[s][key] for s in ordered] for key in keys}
    friedman = friedman_test(matrix); observed = friedman["statistic"]; rng = np.random.default_rng(seed); extreme = 0
    if len(ordered) >= 3:
        values = np.asarray([[mase_rows[s][key] for key in keys] for s in ordered])
        for _ in range(permutations):
            permuted = np.asarray([row[rng.permutation(len(keys))] for row in values])
            statistic = float(friedmanchisquare(*permuted.T).statistic)
            extreme += statistic >= observed
        friedman.update({"permutation_p_value": float((1 + extreme) / (1 + permutations)), "permutation_count": permutations, "seed": seed, "model_order": list(keys)})
    significant = bool(np.isfinite(friedman["permutation_p_value"]) and friedman["permutation_p_value"] < alpha) if "permutation_p_value" in friedman else False
    posthoc = holm_wilcoxon_posthoc(matrix) if significant else {}
    principal_matrix = {key: [rmse_rows[s][key] for s in ordered] for key in ("lag_reg", "arima", "lstm")}
    return {"per_company": per_company, "across_company": {"friedman_mase": friedman, "wilcoxon_posthoc": {"posthoc_executed": significant, "results": posthoc}, "rmse_consistency": best_model_consistency_check(principal_matrix)}}


def run_cross_model_statistical_tests(
    test_errors_by_symbol: dict[str, dict[str, np.ndarray]],
    rmse_by_symbol: dict[str, dict[str, float]],
    model_keys: tuple[str, ...] = ("lag_reg", "arima", "lstm"),
    min_companies: int = 8,
) -> dict:
    """Runs the full statistical-significance suite and returns a single
    JSON-serializable dict, written to disk by model_selector.py.

    ``test_errors_by_symbol[symbol][model_key]`` = 1-D array of
    (actual - predicted) reconstructed-price errors on that model's
    held-out test window for that symbol, already aligned so every model
    covers the *same* dates for that symbol (truncated to the shortest
    common test window, right-aligned on the most recent date).

    ``rmse_by_symbol[symbol][model_key]`` = that model's test-set RMSE for
    that symbol (reconstructed-price scale).
    """
    symbols = sorted(test_errors_by_symbol.keys())

    # --- Diebold-Mariano, within each company, Holm-corrected across its pairs ---
    dm_by_symbol = {}
    for symbol in symbols:
        errors = test_errors_by_symbol[symbol]
        pairs = [(a, b) for i, a in enumerate(model_keys) for b in model_keys[i + 1 :] if a in errors and b in errors]
        raw = []
        for a, b in pairs:
            dm_stat, p_value = diebold_mariano_test(errors[a], errors[b])
            raw.append((a, b, dm_stat, p_value))
        adjusted = holm_correction([p for *_, p in raw])
        dm_by_symbol[symbol] = {
            f"{a} vs {b}": {"dm_statistic": stat, "p_value": p, "holm_p_value": holm_p}
            for (a, b, stat, p), holm_p in zip(raw, adjusted)
        }

    # --- Friedman (across companies) + Wilcoxon-Holm post-hoc ---
    # Only symbols where every model has an RMSE keep the pairing valid.
    complete_symbols = [s for s in symbols if all(m in rmse_by_symbol[s] for m in model_keys)]
    rmse_by_model_paired = {
        model: [rmse_by_symbol[symbol][model] for symbol in complete_symbols] for model in model_keys
    }

    friedman_result = friedman_test(rmse_by_model_paired)
    wilcoxon_result = holm_wilcoxon_posthoc(rmse_by_model_paired) if complete_symbols else {}
    consistency = best_model_consistency_check(rmse_by_model_paired, min_companies=min_companies)

    return {
        "n_companies": len(complete_symbols),
        "companies": complete_symbols,
        "diebold_mariano": dm_by_symbol,
        "friedman": friedman_result,
        "wilcoxon_holm_posthoc": wilcoxon_result,
        "best_model_consistency": consistency,
    }
