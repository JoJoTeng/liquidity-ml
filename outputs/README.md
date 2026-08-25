# outputs/ — Results Layout

Cleaned 2026-08-20: this tree holds only the current generation of results.
Superseded, bug-affected, or out-of-scope results were moved (with mirrored
paths) to `Junk/` at the repo root — see `Junk/README.md` for the manifest.
Nothing was deleted.

**Scope.** Models: `elastic_net` and `xgboost` (neural_network was dropped
from the analysis scope on 2026-06-18; its results are in `Junk/`). Weight
specs (`{spec}`): `dolvol`, `softmax_rank_lam2`, `softmax_rank_lam3`,
`tc_10m`, `tc_100m`, `tc_500m`, `tc_1000m`, `tc_rank_lam3_10m`,
`tc_rank_lam3_100m`, `tc_rank_lam3_500m`, `tc_rank_lam3_1000m` (plus the
unweighted `standard` benchmark in the experiment cache). The paper's hero is
`xgboost` / `tc_rank_lam3_500m`; the Section-4 exhibits co-report `tc_500m`
and `softmax_rank_lam2`. AUM deployment scenarios (`{aum}`): `PropTC`,
`100M`, `500M`, `1B`. Liquidity breakpoints (`{bp}`): `nyse`, `full_sample`.
Investable universes: `full`, `nyse`, `top60`.

---

## motivation/ — scripts 02–06 (motivation track, `--liquidity dvol`)

| Folder | Producer | Contents |
|---|---|---|
| `step1/dvol/` | 02 | Deployment-weight vs equal-weight divergence: `divergence_*` csv/parquet, `weight_regression_*`, `density_*`, `weight_distribution.png`, `tables/` (raw LaTeX). `density_panel_data.csv` is 383 MB. |
| `step2/dvol/` | 03 | Predictability heterogeneity: Fama–MacBeth quintile coefficients (`quintile_fm_*`), interaction regressions (`interaction_*`, `_full` variants), plots, `tables/`. |
| `step3/{model}/dvol/` | 04 | Standard-loss ML diagnostics: `predictions.parquet`, `feature_importance.csv`, `r2_by_quintile.*`, `importance_vs_liquid_r2.*`, `illiquidity_*`, `utility_weighted_r2.json`, `table3/table4` csv, `step3_meta.json`. Large: `predictions_with_quintile.csv` (~250 MB), `quintile_lookup.csv` (108 MB). |
| `step3_restriction/{model}/dvol/global/baseline/` | 05 | Progressive universe restriction: `predictions_MQ2+..MQ5+.parquet`, `restriction_comparison.csv`, `restriction_by_quintile.csv`, `restriction_curve.png`, `meta.json`. |
| `step3_quintile/{model}/dvol/global/baseline/` | 06 | Quintile-specific models: `predictions_q1..q5.parquet`, `r2_comparison.csv/png`, `meta.json`. |

Script 07 (regime analysis) writes `motivation/step1_regime/`, which has never
been generated locally. Generation note: the May-2026 xgboost runs of 05/06
remain valid — their pooled baselines match the regenerated 2026-07-10
step3/xgboost numbers digit-for-digit.

## formalanalysis/ — scripts 20 (training) and 21a–e, 22b (analysis)

| Folder | Producer | Contents |
|---|---|---|
| `experiment/{model}/{spec}/` | 20 (04 pre-populates `standard`) | Prediction cache per fit: `predictions.parquet`, `importance_shap.csv`, `importance_native.csv`, `tuned_params.csv`, `training_diagnostics.csv`, `meta.json` (`training_meta.json` for standard). Trained: xgboost 2026-05-30/31 (standard 05-11), elastic_net 2026-06-10/11. Never retrained since — the 2026-07-19 hysteresis fix was downstream of training. |
| `experiment/xgboost/tc_target/` | 20 `--include-tc-target` | TC-adjusted-target training track (nested standard + spec layout). Required by `22b`, which skips a model without it. The elastic_net twin is in `Junk/` (unused). |
| `analysis/{model}/{spec}/liquidity_breakpoints/{bp}/` | 21a, 21c, 21d | `r2_by_quintile.csv`, `table2_oos_r2_quintile.csv`, `utility_weighted_r2.csv`, `restriction_curve_comparison.*`, `liquid_squared_error_differential.*`, figures. |
| `analysis/{model}/{spec}/` (root files) | 21b | `importance_shift.csv`, `gamma_regression.json`, `group_shares.csv`, `importance_reallocation.png`. |
| `analysis/{model}/{spec}/{run}/stock_universe/{universe}/` | 21e | Conventional sorted-portfolio 2×2: `two_by_two_{aum}.csv`, `two_by_two_timeseries_{aum}.xlsx` (cell series), and the standalone quantile-ladder `prediction_quantile_timeseries_{aum}.csv/xlsx` (a different object). Runs: `prediction_quantile` (equal legs), `prediction_quantile_value_weight`, `prediction_quantile_signal_weight`. **Entire 21e grid regenerated 2026-08-20 with the post-hysteresis-fix builder** (commit 68842ec) for all specs × both models. |
| `tables/xgboost/full_sample/` | 22b | Two-sided-sort workbooks `table12/13/14_*_two_sided_halfleg_equal.xlsx` (documented run: `--portfolio-weighting equal --liquidity-screen-pct 0 --leg-capital half`), consumed by the paper's TwoSidedSort builder. |

## eval_realignment/ — scripts 41–47 (prime evaluation track; reads the experiment cache, no retraining)

| Folder / files | Producer | Contents |
|---|---|---|
| `analysis/{model}/{spec}/liquidity_breakpoints/{bp}/` | 41 | `deployment_weighted_r2.csv`, `deployment_weighted_error_diff*.csv/png`. |
| `analysis/{model}/{spec}/capacity_portfolio[_equal\|_value]_*_{aum}.*` | 42 | Signal-(/equal/value-)weighted capacity portfolio metrics, monthly series, net-cumret figures. |
| `analysis/{model}/{spec}/capacity_breakeven_*` | 43 | Breakeven-gated portfolio metrics/monthly + `capacity_breakeven_gate_diag.csv`. |
| `analysis/{model}/{spec}/two_by_two_{aum}.csv/xlsx` | 44 | Capacity 2×2 (training × execution) tables; per-model workbook in `tables/{model}/capacity_two_by_two_tables.xlsx`. |
| `analysis/{model}/{spec}/longonly_*` | 45 | Long-only Q5 capacity portfolio + hysteresis diagnostics; per-model workbook in `tables/{model}/longonly_two_by_two_tables.xlsx`. |
| `analysis/{model}/{spec}/inference_supplement.csv` | 46 | Seeded bootstrap inference (exists for xgboost dolvol/sm2/sm3/tc_500m of 2026-07-07 and both hero dirs; xgboost hero refreshed 2026-07-19). |
| `analysis/{model}/tc_rank_lam3_500m/gate_scale_diagnostics.csv` | 47 | Gate-scale / forecast-scale diagnostics (hero spec, both models). |
| `tables/capacity_*_training_effect.xlsx` | make_*_xlsx helpers | Cross-model/spec training-effect summaries (2026-06-16/17 vintage). |

**Known caveats (as of 2026-08-20):**

- `45` long-only outputs for every spec except `xgboost/tc_rank_lam3_500m`
  are 2026-06-18 vintage — before the top-60% screen rework (R1a, 07-06/07).
  The paper's LongOnlyTwoByTwo therefore mixes generations for `tc_500m` and
  `softmax_rank_lam2`. Rerun 45 for those specs before submission.
- `tables/xgboost/longonly_two_by_two_tables.xlsx` was clobbered on 07-07 by a
  single-spec rerun (the all-spec version is in `Junk/`); rerun 45 without
  `--weight-spec` to rebuild it.
- 42/43/44 capacity-track outputs are June vintage; per the trim log
  (`paper/review/trim_log_2026-07-18.md`) they were unaffected by the
  hysteresis fix (all A cells bit-for-bit unchanged).

## Loose files

- `audit_liquidity_zeros.csv` — one-off data-quality audit of the `liq_*`
  columns (script `audit_liquidity_zeros.py`, 2026-05-05); relevant to
  RERUN_PLAN decision D9.

## What the paper consumes

`scripts/build_paper_tables.py` and `scripts/build_paper_figures.py` are the
authoritative map from this tree to `paper/Tables` and `paper/Figures`
(fixed constants at the top of each script). In brief: motivation
`xgboost/dvol` (+ step1/step2), eval_realignment `xgboost` for the Section-4
trio + sweep/scale specs and `elastic_net/tc_rank_lam3_500m` (linear
benchmark), formal 21a/21e outputs for the Section-4 trio, the 22b two-sided
workbooks, and the experiment SHAP importances for `standard` vs hero. The
tree deliberately holds the **full result grid** beyond what the paper
selects.
