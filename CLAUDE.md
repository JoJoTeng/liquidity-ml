# LiquidityML Project Guide

## Project Goal

This project tests whether liquidity-aware machine-learning training improves
asset-pricing predictions and portfolio performance. The current pipeline has
three branches:

- Motivation analyses in scripts `02` through `07`.
- Formal model training and analysis in scripts `20` and `21`.
- Evaluation realignment in `scripts/eval_realignment/41`–`46`, which
  re-evaluates the formal predictions against the deployment-weighted training
  objective (see `docs/eval_realignment_pipeline.md`).

The main training data is `data/processed_panel.parquet`, produced by running
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
python scripts/21a_formal_liquid_r2.py
python scripts/21b_formal_importance_reallocation.py
python scripts/21c_formal_restriction_curve.py
python scripts/21d_formal_error_differential.py
python scripts/21e_formal_portfolio_decomposition.py
```

The eval_realignment track reads the formal prediction cache (no retraining)
and writes under `outputs/eval_realignment/`. Run it after `20` has produced
predictions for the model/spec; `44` reads the monthly series of `42` and `43`
and must run last:

```bash
python scripts/eval_realignment/41_deployment_weighted_prediction_metrics.py --model xgboost --weight-spec dolvol --liquidity-breakpoints both
python scripts/eval_realignment/42_signal_weighted_capacity_portfolio.py --model xgboost --weight-spec dolvol
python scripts/eval_realignment/43_breakeven_capacity_portfolio.py --model xgboost --weight-spec dolvol
python scripts/eval_realignment/44_capacity_two_by_two_tables.py --model xgboost --weight-spec dolvol
python scripts/eval_realignment/45_longonly_capacity_q5.py --model xgboost --weight-spec dolvol
python scripts/eval_realignment/46_inference_supplement.py --model xgboost --weight-spec tc_rank_lam3_500m
```

Script `46` computes the seeded bootstrap inference supplement from the cached
monthly cell series of `42`--`45`; run it after them.

Omit `--weight-spec` to cover every fitted spec of a model; `--aum all` is the
default grid. Full conventions, equations, and output layout are in
`docs/eval_realignment_pipeline.md`.

For Apocrita, generate SLURM jobs instead of running Python directly on the
login node:

```bash
bash scripts/generate_hpc_jobs.sh --from-processed --submit
```

Use `--from-processed` when `00` and `01` were run locally and the processed
data files were uploaded to the cluster.

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

`scripts/00_fetch_data.py` creates `data/signed_predictors_all_wide.csv` from
WRDS CRSP data and the Chen-Zimmermann predictor zip.

`scripts/01_process_data.py` creates:

- `data/processed_panel.parquet`
- `data/feature_list.json`
- `config/feature_categories.json`

The selected model features are rank-normalized to `[0, 1]` within each month.
Missing feature values remain `NaN` in the processed panel and are filled with
the neutral value `0.5` inside the training and regression code when needed.

Raw liquidity helpers are preserved as `liq_*` columns for weights, quintiles,
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

Each fitted directory stores:

- `predictions.parquet`
- `importance_shap.csv`
- `importance_native.csv`
- `tuned_params.csv`

The standard model cache is shared across all weight families for the same
model. `scripts/04_motivation_step3_ml_diagnostics.py` can pre-populate this
standard cache.

## Current Normalization Contract

The shared rolling trainer in `src/training/rolling.py` expects the processed
panel from `load_processed_panel()`. It does not re-rank inputs. It only fills
selected feature `NaN` values with `0.5`.

The restricted-universe motivation scripts `05` and `06` support two modes:

- `--normalization global`: keep the processed full-cross-section ranks.
- `--normalization rerank`: re-rank after filtering the training universe.

## Important Notes

- Do not use a quick formal mode; it was removed to avoid poisoning the full-run
  output cache.
- Regime diagnostics are model-independent; the `--model` argument only
  namespaces their outputs.
- The primary formal R-squared tables use the zero benchmark. Extra benchmarks
  are available in `scripts/21a_formal_liquid_r2.py --extra-benchmarks`.

## Testing

Run:

```bash
python -m pytest tests -q
python -m compileall -q scripts src tests
```

Some SHAP and neural-network tests are skipped automatically if optional runtime
dependencies are unavailable.
