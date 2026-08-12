# Final Unseen-Test Predictive-Accuracy Evaluation

A dedicated, reproducible evaluation suite that measures how well every
supported forecasting model — **Lag-Informed Regression/LASSO**, **ARIMA**,
**LSTM**, and a **naive baseline** (`tomorrow = today`) — generalizes to a
final block of dates none of them were trained, tuned, scaled, or selected
on, for every supported PSE ticker.

It is additive: nothing in `services/`, `run_pipeline.py`, or the dashboard
is modified. This suite only reads `data/raw/*.csv` and reuses the existing
data loading, feature engineering, and model-training code as a library.

## Methodology

### Temporal split

Every ticker's history is cut chronologically into exactly two pieces:

```
[ ---------------- TRAIN + VALIDATION ---------------- ] [ -- FINAL TEST -- ]
                                                          ^
                                            never seen by anything upstream
```

- **TRAIN + VALIDATION**: passed, unmodified, into the existing
  `services/forecasting/{lag_regression,arima_model,lstm_model}.train()`
  functions. Those functions already do their own internal
  training/validation work on whatever dataframe they're given — PACF lag
  selection and `LassoCV`'s expanding-window CV for the lag regression,
  order-search CV for ARIMA, and grid-search-with-holdout for the LSTM. By
  construction, none of that can ever see a final-test row, because the
  final-test rows are not present in the object being passed in.
- **FINAL TEST**: the most recent slice of each ticker's history, held out
  *before* any of the above runs. Every model is scored on the exact same
  final-test dates (naive baseline included).

The split point is configurable, not hard-coded (see [Configuration](#configuration)):
by default, the final 15% of each ticker's rows become the final-test
window; alternatively an explicit `[start, end]` date window can be set.

### Freezing

After `train()` returns, each model's artifact (scaler, fitted
`Lasso`/`LassoCV` model and its selected features, the fitted ARIMA/SARIMAX
result, or the LSTM's trained weights + `x_scaler`/`y_scaler`) is treated as
frozen. Nothing below that point calls `.fit()` again. Predicting a
final-test date still needs *inputs* — e.g. yesterday's realized close to
compute today's lag/technical features — and those inputs are allowed to
include realized values from within the final-test window itself, as long
as they are strictly earlier than the date being predicted. This is
standard walk-forward forecasting (it's how these models would actually be
deployed) and is **not** leakage — see the next section for the precise
distinction the suite enforces.

### Leakage controls

`leakage_checks.py` gives every check its own function, and
`run_evaluation.py` calls the relevant ones inline as the run progresses.
Every check raises `LeakageError` with a specific message and **aborts the
run** — nothing is caught and swallowed.

| Check | What it guarantees |
|---|---|
| `assert_no_date_overlap` | TRAIN+VALIDATION ends strictly before FINAL TEST begins; no shared dates. |
| `assert_fit_input_excludes_dates` | The dataframe handed to `train()` contains none of the final-test dates. |
| `assert_model_selection_before_final_test` | Every model's `selection_source` is `"trainval"` (or `"naive"`, which has nothing to select). |
| `assert_scaler_fit_row_count` | A fitted scaler's `n_samples_seen_` never exceeds the TRAIN+VALIDATION row count. |
| `assert_no_future_values_in_features` | Every date inside an LSTM input window is strictly before the date it's predicting. |
| `assert_naive_uses_prior_close_only` | The naive baseline is exactly `tomorrow = today` — no drift, no smoothing. |
| `assert_identical_test_dates` | Every model (after `align_runs_to_common_dates`) is scored on exactly the same final-test dates, in the same order. |

**Fit vs. input, precisely:** "leakage" here means a final-test
*observation* influenced a *fitted parameter* (a scaler's mean/std, a
LASSO coefficient, an ARIMA order or coefficient, an LSTM weight). It does
not mean a frozen model is forbidden from taking realized market data as
input at prediction time — a production forecaster deployed today
obviously gets to use today's actual closing price to predict tomorrow's.
ARIMA's walk-forward step (`.append(actual, refit=False)`) updates only the
model's *state* (a Kalman-filter-style update), never re-estimates its
parameters — matching the frozen-artifact contract above.

### Statistical tests

All computed strictly from the frozen final-test predictions, per ticker
and in aggregate:

- **Diebold-Mariano** (`services/evaluation.diebold_mariano_test`, HLN
  small-sample corrected) — every model pair per ticker, including each
  tuned model vs. the naive baseline, Holm-corrected across that ticker's
  pairs.
- **Friedman test** across tickers (`services/evaluation.friedman_test`) —
  once for the three tuned models, once including naive.
- **Pairwise Wilcoxon signed-rank with Holm correction**
  (`services/evaluation.holm_wilcoxon_posthoc`) across tickers.
- **Best-model consistency** check
  (`services/evaluation.best_model_consistency_check`) — does one model
  dominate (lowest RMSE) on a configurable minimum fraction of tickers?

All four re-use the exact functions `services/model_selector.py` already
relies on in production — there is exactly one statistical-testing
implementation in this repository, not two that could silently disagree.

### Metrics

Per ticker and in aggregate, for every model and the naive baseline:
RMSE, MAE, MASE, R² (via `services/evaluation.compute_metrics`, reused),
plus this suite's own `metrics.py`:

- **Directional accuracy** — fraction of days the predicted up/flat/down
  call (relative to the prior day's actual close) matched reality.
- **Mean directional error (MDE)** — mean of
  `(predicted_direction − actual_direction)`, each in `{-1, 0, +1}`; 0 = no
  systematic bullish/bearish bias.
- **Hit rate vs. naive** — fraction of final-test days a model's absolute
  error was ≤ the naive baseline's absolute error on that same date.
- **Prediction error statistics** — bias (mean signed error), error std,
  min/max, median absolute error.
- **95% confidence intervals** — percentile bootstrap over RMSE and MAE
  (2000 resamples by default). Omitted, not fabricated, when a ticker has
  too few final-test observations to bootstrap meaningfully (`< 8`, see
  `config.MIN_ROWS_FOR_CI`).

Exact definitions (important for interpreting output correctly) are in the
`metrics.py` module docstring.

## Data sources

`backend/data/raw/<TICKER>.csv` — the same OHLCV CSVs
`services/data_loader.py` / `run_pipeline.py` already use, loaded and
validated through the existing `services/data_validator.validate_ohlcv_csv`
(sorted, deduplicated, OHLC-consistency-checked). No separate data pipeline
was introduced.

## Models tested

| Model | Trained via | Notes in this environment |
|---|---|---|
| Lag-Informed Regression / LASSO | `services/forecasting/lag_regression.train` | Runs fully — only needs `scikit-learn`. |
| ARIMA | `services/forecasting/arima_model.train` + `._walk_forward_forecast` | Needs `statsmodels`. If unavailable, both the underlying module and this suite's runner deterministically fall back to a TRAIN+VALIDATION-derived drift model (flagged `fallback: true` in every output file — never silently substituted). |
| LSTM | `services/forecasting/lstm_model.train` | Needs `torch`. If unavailable (or there's insufficient TRAIN+VALIDATION history for even the smallest grid config), both the underlying module and this suite's runner fall back to the naive prediction (flagged `fallback: true`). |
| Naive baseline | `runners.run_naive` | `tomorrow = today`, no dependencies. |

## Running it

From `backend/`:

```bash
# Full 15-ticker canonical evaluation (uses whatever's installed;
# see requirements-pipeline.txt for the full-fidelity dependency set).
python -m tests.predictive_accuracy.run_evaluation

# Fast local smoke test — a couple of tickers only.
PA_TICKERS=BPI,ALI python -m tests.predictive_accuracy.run_evaluation

# Explicit final-test window instead of the default 15% tail.
PA_FINAL_TEST_START_DATE=2026-05-01 python -m tests.predictive_accuracy.run_evaluation

# Run the suite's own test suite.
python -m unittest discover -s tests/predictive_accuracy/tests -t .
```

### Configuration

Everything is overridable via environment variable — see `config.py` for
the authoritative list. The ones you're most likely to touch:

| Variable | Default | Meaning |
|---|---|---|
| `PA_FINAL_TEST_FRACTION` | `0.15` | Tail fraction of each ticker's rows reserved as FINAL TEST. |
| `PA_FINAL_TEST_START_DATE` / `PA_FINAL_TEST_END_DATE` | unset | Explicit `[start, end]` FINAL TEST window (overrides the fraction). |
| `PA_TICKERS` | every CSV in `data/raw/` | Comma-separated ticker subset. |
| `PA_SEED` | `42` | Seed for numpy/random/torch and the bootstrap CI. |
| `PA_MIN_CONSISTENCY_COMPANIES` | majority of tickers | Threshold for the best-model-consistency check. |
| `PA_BOOTSTRAP_ITERATIONS` | `2000` | Bootstrap resamples for the 95% CIs. |

## Expected output files

Written to `results/` (created if missing):

| File | Contents |
|---|---|
| `metrics.csv` | One row per (ticker, model): every metric, flattened, plus `fallback`. |
| `metrics.json` | The same data, nested by ticker → model, full precision. |
| `model_comparison.csv` | Aggregate ranking: mean RMSE, selection frequency, improvement vs. naive. |
| `statistical_tests.json` | Per-ticker Diebold-Mariano, rankings, Friedman, Wilcoxon-Holm, best-model consistency. |
| `evaluation_summary.md` | Human-readable roll-up of the above. |

`results/` is evaluation output, not source — safe to delete and
regenerate at any time by re-running `run_evaluation.py`.

## Environment / version requirements

- Python: 3.11–3.12 (matches `backend/.python-version` / CI).
- Required: `pandas`, `numpy`, `scikit-learn`, `scipy` (already required by
  the existing pipeline).
- Optional, for full-fidelity ARIMA/LSTM rather than their documented
  fallbacks: `statsmodels==0.14.6`, `torch==2.13.0` (versions pinned in
  `requirements-pipeline.txt`, the repository's existing canonical
  training environment — install that file for a full-fidelity run).
- No new dependencies were introduced by this suite.

## Reproducibility

- `PA_SEED` (default `42`) seeds `random`, `numpy`, and `torch` (when
  present) at the start of `run_evaluation.py`.
- The bootstrap CI takes its own explicit `seed` parameter
  (`metrics.bootstrap_ci`), independent of global state, and is verified
  deterministic given a fixed seed (`tests/test_metrics.py::TestBootstrapCI`).
- `LassoCV`, ARIMA's order search, and the LSTM's grid search are all
  deterministic given the same TRAIN+VALIDATION data (no shuffling; time
  series CV only) — verified end-to-end in
  `tests/test_integration.py::TestReproducibility`.
- The chronological split itself has no randomness (`tests/test_splits.py`).

## Tests

`tests/predictive_accuracy/tests/`, run via
`python -m unittest discover -s tests/predictive_accuracy/tests -t .` from
`backend/`:

| File | Covers |
|---|---|
| `test_splits.py` | Temporal split correctness — sizes, ordering, no lost/duplicated rows, explicit date windows, error cases. |
| `test_leakage_checks.py` | Every leakage-detection function, both the pass and fail path. |
| `test_metrics.py` | Directional accuracy / MDE / hit-rate / error-stats / bootstrap-CI calculations, including edge cases (flat direction, too-few observations). |
| `test_naive_baseline.py` | Naive baseline calculation — predictions equal the immediately preceding close, dates match final-test exactly. |
| `test_statistical_tests.py` | Ranking/selection-frequency helpers and Diebold-Mariano wiring, incl. minimal-input validation for the full suite. |
| `test_integration.py` | Feature alignment (predictions indexed to the correct date, inputs never reference the target date itself), identical final-test dates across all four models, a structural leakage tripwire (ARIMA's trainer is monkey-patched to raise if it ever receives a final-test row), and reproducibility. |

All 63 tests pass in this environment. `test_integration.py` exercises the
real `lag_regression`/`naive` runners against synthetic OHLCV data
end-to-end; ARIMA/LSTM are exercised through whichever code path
(full or fallback) the installed dependencies select, so the suite never
skips itself into a false sense of coverage.

## Known limitation in the evaluation environment used to generate the
## results currently checked into `results/`

`statsmodels` and `torch` were not installable in the sandbox this suite
was built and run in (no network access). The results in `results/`
therefore use the documented ARIMA/LSTM fallbacks for all 15 tickers
(every affected row is flagged `fallback: true` in `metrics.csv`/`.json`).
Re-running `python -m tests.predictive_accuracy.run_evaluation` after
`pip install -r requirements-pipeline.txt` will produce full-fidelity
ARIMA and LSTM results with no code changes required.
