# Manuscript Alignment Acceptance Record

## Deliverables

- `ForecastPH_Capstone_Aligned_Formal_Run_02.docx`
- `ForecastPH_Capstone_Aligned_Formal_Run_02.pdf`
- `Chapter_3_Proposed_Revisions.md`
- `Chapter_4_Results_Draft.md`
- `Manuscript_Alignment_Remaining_Steps.md`

## Authority and evidence identity

- Authoritative source: `/Users/alvintubtub/Downloads/MO-IT200D1 _ H3103 Group 4 Capstone 1 Paper (9).docx`
- Authoritative source SHA-256: `be6c351ef6f2bfa2e658ebc585f78eb73a891f986f86693b2d2e96da069d29f4`
- Formal run ID: `FORMAL_CORRECTED_20260828_02`
- Formal code commit: `bfb33b8c184c87cc8828af5529410da94addd71c`
- Source-data commit: `2e72058057f5ba2ef903147c8390c3f05f41ffe3`
- Downloaded evidence archive SHA-256: `2b2ed0ca6b88ea6cfef5ac14013440da1c7c55c1d9f9640e04a595fdafca5d24`
- Corporate-action registry SHA-256: `98199732d5372683c256d1e7c9b9ab1b9987b9f2a7240dbe0c43d00370d508eb`

## Acceptance checks

| Check | Result |
|---|---|
| Protected experiment artifacts | PASS — 155 of 155 SHA-256 values matched |
| Experiment status | PASS — complete, with 15 completed companies |
| Formal data scope | PASS — 1,624 rows per company; 24,360 company-date observations |
| Forecast-pair split | PASS — 1,380 development and 243 holdout targets per company |
| Holdout interval | PASS — September 2, 2025 through August 28, 2026 |
| LASSO procedure | PASS — 36 alpha values from 0.0001 through 1000; five folds |
| ARIMA procedure | PASS — 80 order-trend candidates; convergence required; fixed-parameter state updates |
| LSTM procedure | PASS — 48 configurations; five common-date folds; three tuning seeds; complete-development refit |
| Benchmark procedure | PASS — Naive is the mandatory formal benchmark; Seasonal Naive is outside the frozen comparison |
| Statistical procedure | PASS — benchmark-first DM/Holm gate; conditional principal comparison; permutation Friedman and Wilcoxon-Holm |
| Corporate-action treatment | PASS — primary observations retained; 26 registered target dates excluded only in sensitivity analysis |
| Reported significant improvements | PASS — Lag Regression for ICT and MBT only under the declared gate |
| Across-company conclusion | PASS — significant omnibus result; no Holm-significant post-hoc pair |
| Dominance claim | PASS — no principal model achieved the eight-of-fifteen threshold |
| Application architecture | PASS — Python analytics, versioned artifacts, Next.js/React, and Vercel; no active browser upload or Streamlit claim |
| DOCX package | PASS — valid ZIP package; no damaged member |
| PDF rendering | PASS — 359 A4 pages |
| Visual review | PASS — every rendered page reviewed; no clipping, overlap, or malformed inserted table/diagram found |
| Adviser approval | DEFERRED — not claimed as completed |

## Report-ready findings

- Lag Regression and ARIMA each had the lowest principal-model RMSE for seven companies; LSTM had the lowest for ICT.
- When Naive was included, Lag Regression and ARIMA each had five outright lowest-RMSE results, LSTM had one, Naive had two, and ARIMA tied Naive for GLO and SCC.
- Lag Regression significantly improved on Naive for ICT and MBT. ARIMA and LSTM did not pass the benchmark-first improvement gate for any company.
- No company had two eligible principal models, so the conditional second-stage principal-model comparisons were correctly skipped.
- The MASE Friedman permutation test was significant, but no Wilcoxon pair remained significant after Holm correction. The smallest adjusted post-hoc value was 0.061523 for Lag Regression versus LSTM.
- Removing the 26 registered corporate-action target dates did not change any company’s descriptive lowest-RMSE model or tie.
- Nine selected Lag Regression fits retained no nonzero feature coefficients; the manuscript therefore does not claim that technical indicators contributed to every final regression.

## Final artifact checksums

- DOCX SHA-256: `dc73caf36ca244f2c6aed67269571d2da86429e707c4f843c5c56cf85d5b6196`
- PDF SHA-256: `b38a29a4f2a80b5fcc2104a6323e49427ebabf68fefd8057d647fc70f503a26a`

## Decision

PASS for local manuscript alignment and Git review. The next action is to select the intended documentation files, inspect the staged diff, and create a clean documentation commit. Do not include the large `formal_evidence/` directory without an explicit repository storage decision.
