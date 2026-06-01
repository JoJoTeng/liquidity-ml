# Codex Project Guide

## Collaboration Preferences

- When reading or explaining code, go section by section from the actual file.
- Do not skip docstrings, imports, constants, utility lines, CLI arguments, or
  entrypoints.
- When a function calls a helper, shared API, or model method, open that helper
  and explain the actual implementation before moving on.
- Explain code using this project's real data flow and research logic, not
  generic programming summaries.
- Prefer concrete file paths, function names, column names, and output files.
- If a code section looks stale, fragile, or unnecessary, say so clearly and
  explain whether it affects the current pipeline.

## Project Goal

LiquidityML tests whether liquidity-aware machine-learning training improves
asset-pricing predictions and portfolio performance. The project has two main
branches:

- Motivation analyses in scripts `02` through `07`.
- Formal model training and analysis in scripts `20` and `21a` through `21e`,
  with `22` preparing paper-style Excel tables from `21e` outputs.

The main model input is `data/processed_panel.parquet`, produced by running
`scripts/00_fetch_data.py` and `scripts/01_process_data.py`.

## Current Script Order

```bash
python scripts/00_fetch_data.py
python scripts/01_process_data.py

python scripts/02_motivation_step1_divergence.py --liquidity dvol --vw
python scripts/03_motivation_step2_heterogeneity.py --liquidity dvol --full

python scripts/04_motivation_step3_ml_diagnostics.py --model xgboost --liquidity dvol
python scripts/04_motivation_step3_ml_diagnostics.py --model elastic_net --liquidity dvol
python scripts/04_motivation_step3_ml_diagnostics.py --model neural_network --liquidity dvol

python scripts/05_motivation_step3d_progressive_restriction.py --model xgboost --liquidity dvol --normalization global --use-baseline-params
python scripts/06_motivation_step3e_quintile_specific_models.py --model xgboost --liquidity dvol --normalization global --use-baseline-params
python scripts/07_motivation_regime_analysis.py --model xgboost --liquidity dvol

python scripts/20_formal_run_experiment.py --model xgboost --weights dolvol
python scripts/20_formal_run_experiment.py --model xgboost --weights softmax_rank --softmax-lambda 2
python scripts/20_formal_run_experiment.py --model xgboost --weights softmax_rank --softmax-lambda 3
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 10
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 100
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 500
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 1000
python scripts/20_formal_run_experiment.py --model xgboost --weights tc_rank --tc-rank-lambda 3 --aum 10
python scripts/20_formal_run_experiment.py --model xgboost --weights tc_rank --tc-rank-lambda 3 --aum 100
python scripts/20_formal_run_experiment.py --model xgboost --weights tc_rank --tc-rank-lambda 3 --aum 500
python scripts/20_formal_run_experiment.py --model xgboost --weights tc_rank --tc-rank-lambda 3 --aum 1000

python scripts/21a_formal_liquid_r2.py
python scripts/21b_formal_importance_reallocation.py
python scripts/21c_formal_restriction_curve.py
python scripts/21d_formal_error_differential.py
python scripts/21e_formal_portfolio_decomposition.py
python scripts/22_prepare_portfolio_excel_tables.py --table table11 --model all --weight-spec all --aum all --skip-missing
python scripts/22_prepare_portfolio_excel_tables.py --table table12 --model all --weight-spec all --aum all --skip-missing
python scripts/22_prepare_portfolio_excel_tables.py --table table13 --model all --weight-spec all --aum all --skip-missing
python scripts/22_prepare_portfolio_excel_tables.py --table table14 --model all --weight-spec all --aum all --skip-missing
```

Run the formal `20` grid for each active model (`elastic_net`, `xgboost`,
`neural_network`). The commands above show one model; the HPC generator emits
the full grid automatically.

For Apocrita, do not run compute-heavy Python scripts on the login node. Generate
and submit SLURM jobs instead:

```bash
bash scripts/generate_hpc_jobs.sh --from-processed --submit
```

Use `--from-processed` when `00` and `01` were run locally and the processed
data files were uploaded to the cluster. Add `--include-tc-target` only when
you also want target-adjusted formal models with target
`excess_ret - BidAskSpread/2`; these are written under
`outputs/formalanalysis/experiment/{model}/tc_target/`.

## Active Models

The active model registry is:

- `elastic_net`
- `xgboost`
- `neural_network`

Create models through:

```python
from src.models import create_model
model = create_model("xgboost")
```

All active models implement the `BaseReturnPredictor` API:

- `fit(X_train, y_train, X_val=None, y_val=None, sample_weight=None, sample_weight_val=None)`
- `predict(X)`
- `get_feature_importance(feature_names)`
- `get_shap_values(X_test, X_background=None, feature_names=None, config=None)`
- `tune_hyperparameters(X_train, y_train, X_val, y_val, sample_weight=None, sample_weight_val=None)`

## Data And Features

- `scripts/00_fetch_data.py` creates `data/signed_predictors_all_wide.csv` from
  WRDS CRSP data and the Chen-Zimmermann predictor zip.
- `scripts/01_process_data.py` creates:
  - `data/processed_panel.parquet`
  - `data/feature_list.json`
  - `config/feature_categories.json`
- Selected model features are rank-normalized to `[0, 1]` within each month.
- Missing feature values remain `NaN` in the processed panel and are filled with
  the neutral value `0.5` inside training and regression code when needed.
- Raw liquidity helpers are preserved as `liq_*` columns for weights, quintiles,
  market-cap robustness, and transaction-cost calculations.

## Weight Families

Formal weighted training supports four families:

- `dolvol`: `DolVol_it / mean_i(DolVol_it)`.
- `softmax_rank`: `exp(lambda * rank_it)` normalized to mean one, with formal
  lambdas `2` and `3`.
- `tc`: transaction-cost weights based on spread, daily-scaled volatility, ADV,
  and AUM.
- `tc_rank`: softmax weights on the within-month percentile rank of `-TC_it`,
  with formal lambda `3` and AUMs `10`, `100`, `500`, and `1000` million.

All training weights are normalized to mean one within each month.

## Formal Output Layout

Formal training writes to:

```text
outputs/formalanalysis/experiment/{model}/standard/
outputs/formalanalysis/experiment/{model}/dolvol/
outputs/formalanalysis/experiment/{model}/softmax_rank_lam2/
outputs/formalanalysis/experiment/{model}/softmax_rank_lam3/
outputs/formalanalysis/experiment/{model}/tc_10m/
outputs/formalanalysis/experiment/{model}/tc_100m/
outputs/formalanalysis/experiment/{model}/tc_500m/
outputs/formalanalysis/experiment/{model}/tc_1000m/
outputs/formalanalysis/experiment/{model}/tc_rank_lam3_10m/
outputs/formalanalysis/experiment/{model}/tc_rank_lam3_100m/
outputs/formalanalysis/experiment/{model}/tc_rank_lam3_500m/
outputs/formalanalysis/experiment/{model}/tc_rank_lam3_1000m/
```

If `--include-tc-target` is used, `20_formal_run_experiment.py` also writes
target-adjusted runs under:

```text
outputs/formalanalysis/experiment/{model}/tc_target/standard/
outputs/formalanalysis/experiment/{model}/tc_target/{weight_spec}/
```

Each fitted directory stores:

- `predictions.parquet`
- `importance_shap.csv`
- `importance_native.csv`
- `tuned_params.csv`
- `training_diagnostics.csv`
- `training_meta.json` for `standard`, or `meta.json` for weighted specs

Formal analysis scripts write to:

```text
outputs/formalanalysis/analysis/{model}/{weight_spec}/
```

For example:

```text
outputs/formalanalysis/analysis/xgboost/tc_500m/r2_by_quintile.csv
```

Portfolio decomposition outputs from `21e` are nested by portfolio run, for
example:

```text
outputs/formalanalysis/analysis/xgboost/tc_500m/prediction_quantile/two_by_three_500M.csv
```

By default `21e` uses equal-dollar weights within each prediction quantile.
Use `--portfolio-weighting value` to use `liq_me_raw` market-cap weights
inside each prediction quantile, or `--portfolio-weighting both` to write
equal-weight outputs under `prediction_quantile/` and value-weight outputs
under `prediction_quantile_value_weight/`.

Use `--stock-universe nyse` to construct portfolios only from NYSE stocks
without retraining models. NYSE-only outputs are written under
`{portfolio_run}/stock_universe/nyse/`. Use `--stock-universe both` to write
separate full-sample and NYSE folders without overwriting the legacy
full-sample outputs.

`21e` writes `two_by_three_{AUM}.csv` and
`two_by_three_timeseries_{AUM}.xlsx` when `tc_target` predictions are available.
It also writes `prediction_quantile_timeseries_{AUM}.csv/xlsx`. The 2x3
long-short cells are derived from Q5 minus Q1 after first computing standalone
Q1-Q5 prediction-quantile portfolios. Each prediction quantile uses
`AUM / portfolio.n_quantiles` for transaction-cost sizing.

Formatted Excel report tables from `22_prepare_portfolio_excel_tables.py` and
`22b_table12_two_sided.py` write to model- and stock-universe-specific
directories:

```text
outputs/formalanalysis/tables/{model}/{stock_universe}/
```

The active formatted portfolio tables are:

- `table11`: within-liquidity-quintile long-short performance.
- `table12`: 2x3 Q5-Q1 long-short decomposition.
- `table13`: standalone Q1-Q5 prediction-quantile portfolios.
- `table14`: Table-12-style 2x3 decompositions for each standalone prediction quantile.

## Normalization Contract

The shared rolling trainer in `src/training/rolling.py` expects the processed
panel from `load_processed_panel()`. It does not re-rank inputs. It only fills
selected feature `NaN` values with `0.5`.

The restricted-universe motivation scripts `05` and `06` support two modes:

- `--normalization global`: keep the processed full-cross-section ranks.
- `--normalization rerank`: re-rank after filtering the training universe.

## Important Pipeline Notes

- Do not add or use a quick formal mode; it was removed to avoid poisoning the
  full-run output cache.
- The only active model families are `elastic_net`, `xgboost`, and
  `neural_network`.
- Regime diagnostics are model-independent; the `--model` argument only
  namespaces their outputs.
- The primary formal R-squared tables use the zero benchmark. Extra benchmarks
  are available with `scripts/21a_formal_liquid_r2.py --extra-benchmarks`.
- Formal analysis scripts discover all complete experiment folders, including
  `tc_rank_lam3_*`. Use `--weights dolvol`, `--weights softmax_rank`,
  `--weights tc`, or `--weights tc_rank` to restrict analysis.
- Formal importance reallocation expects Step 2 full outputs, especially
  `interaction_regression_full.csv` and `quintile_fm_coefficients_full_raw.csv`.
- Formal restriction-curve comparison expects Step 3d baseline/global output:
  `outputs/motivation/step3_restriction/{model}/dvol/global/baseline/restriction_comparison.csv`.
- Formal portfolio decomposition builds prediction-quantile portfolios first.
  The 2x3 Q5-Q1 long-short series is derived from those quantile portfolios.
  Portfolio holding weights are controlled with
  `--portfolio-weighting equal|value|both`; `22_prepare_portfolio_excel_tables.py`
  selects the output folder with `--portfolio-run prediction_quantile|prediction_quantile_value_weight|both`.
  Stock-universe filtering is controlled with `--stock-universe full_sample|nyse|both`.
  It does not recompute portfolios; it reads the CSV outputs created by `21e`.

## Testing And Verification

Use focused checks after edits:

```bash
python -m compileall -q scripts src tests
python -m pytest tests -q
```

For script-only changes, at minimum run:

```bash
python -m compileall -q scripts src
python scripts/<changed_script>.py --help
```

Some SHAP and neural-network tests are skipped automatically if optional runtime
dependencies are unavailable.
