# Run Order Cheat Sheet

This document gives the exact command order to reproduce the current project pipeline.

It is organized into prerequisites, the motivation pipeline, the formal
experiment pipeline, and optional robustness runs.

## 1. Before You Run Anything

Run everything from the repo root:

```bash
cd /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml
```

Make sure these input files are present:

- `data/temp/signed_predictors_dl_wide.zip`
- `data/FFResearch_Data_Factors.csv`
- `data/SignalDoc.csv`

If you use WRDS through the fetch script, set your username:

```bash
export WRDS_USERNAME="your_wrds_username"
```

## 2. Full Motivation Pipeline

These commands reproduce the current motivation branch for the main `dvol` specification.

### Step 0. Build the raw panel

```bash
python scripts/00_fetch_data.py
```

### Step 1. Build the processed motivation panel

```bash
python scripts/01_process_data.py
```

### Step 1 outputs. Distributional divergence

This version includes the market-cap overlay in the density plot.

```bash
python scripts/02_motivation_step1_divergence.py --liquidity dvol --vw
```

### Step 2 outputs. Heterogeneous predictability

Use `--full` because the current formal analysis script reuses the full-output files for Prediction 2.

```bash
python scripts/03_motivation_step2_heterogeneity.py --liquidity dvol --full
```

### Step 3 baseline. Model-compatible diagnostics

```bash
for MODEL in elastic_net xgboost neural_network; do
    python scripts/04_motivation_step3_ml_diagnostics.py --model ${MODEL} --liquidity dvol
done
```

### Step 3d. Progressive restriction curve

Use `--use-baseline-params` because the current formal Prediction 3 overlay reads the baseline-mode restriction output. The default normalization is `global`, which keeps the processed full-cross-section ranks so the main difference is the training universe. Add `--normalization rerank` for the robustness version.

```bash
for MODEL in elastic_net xgboost neural_network; do
    python scripts/05_motivation_step3d_progressive_restriction.py --model ${MODEL} --liquidity dvol --normalization global --use-baseline-params
done
```

### Step 3e. Quintile-specific models

Default mode is baseline tuned parameters and default normalization is `global`.
Add `--normalization rerank` to save the within-filter-reranked robustness version
in a separate folder.

```bash
for MODEL in elastic_net xgboost neural_network; do
    python scripts/06_motivation_step3e_quintile_specific_models.py --model ${MODEL} --liquidity dvol --normalization global --use-baseline-params
done
```

### Regime extension

Download regime data once:

```bash
python scripts/07_motivation_regime_analysis.py --download-regime-data
```

Then run the regime analysis:

```bash
python scripts/07_motivation_regime_analysis.py --model xgboost --liquidity dvol
```

Optional TC-compatible motivation run:

```bash
python scripts/02_motivation_step1_divergence.py --liquidity tc --aum 500 --vw
python scripts/03_motivation_step2_heterogeneity.py --liquidity tc --aum 500 --full
python scripts/04_motivation_step3_ml_diagnostics.py --liquidity tc --aum 500
```

TC motivation outputs use the liquidity key `tc_500m`, for example
`outputs/motivation/step1/tc_500m/`.

## 3. Full Formal Experiment Pipeline

The formal reporting script now reuses some motivation outputs:

- Step 2 full-mode outputs from:
  - `scripts/03_motivation_step2_heterogeneity.py --liquidity dvol --full`
- Step 3d baseline restriction output from:
  - `scripts/05_motivation_step3d_progressive_restriction.py --model {model} --liquidity dvol --normalization global --use-baseline-params`

So run those first if you want the full current Section 8 output set.
Step 04 should also be run for each model if you want `M_std` cached before the
formal weighted jobs start.

Weighting definitions used by these runs:
- `dolvol`: `DolVol_it / mean_i(DolVol_it)` within each month, AUM-independent.
- `softmax_rank`: `exp(lambda * percentile_rank(DolVol_it))` within each month, AUM-independent.
- `tc`: transaction-cost-based weights, AUM-dependent, run separately for $10M, $100M, $500M, and $1B.
- Current `tc` intensity: `alpha_t = 3.0 / median_i(TC_it)`.

### XGBoost experiments

```bash
python scripts/20_formal_run_experiment.py --model xgboost --weights dolvol
python scripts/20_formal_run_experiment.py --model xgboost --weights softmax_rank --softmax-lambda 2
python scripts/20_formal_run_experiment.py --model xgboost --weights softmax_rank --softmax-lambda 3
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 10
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 100
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 500
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 1000
```

### ElasticNet experiments

```bash
python scripts/20_formal_run_experiment.py --model elastic_net --weights dolvol
python scripts/20_formal_run_experiment.py --model elastic_net --weights softmax_rank --softmax-lambda 2
python scripts/20_formal_run_experiment.py --model elastic_net --weights softmax_rank --softmax-lambda 3
python scripts/20_formal_run_experiment.py --model elastic_net --weights tc --aum 10
python scripts/20_formal_run_experiment.py --model elastic_net --weights tc --aum 100
python scripts/20_formal_run_experiment.py --model elastic_net --weights tc --aum 500
python scripts/20_formal_run_experiment.py --model elastic_net --weights tc --aum 1000
```

### Neural network experiments

Run these only if your environment has the required TensorFlow and SHAP dependencies:

```bash
python scripts/20_formal_run_experiment.py --model neural_network --weights dolvol
python scripts/20_formal_run_experiment.py --model neural_network --weights softmax_rank --softmax-lambda 2
python scripts/20_formal_run_experiment.py --model neural_network --weights softmax_rank --softmax-lambda 3
python scripts/20_formal_run_experiment.py --model neural_network --weights tc --aum 10
python scripts/20_formal_run_experiment.py --model neural_network --weights tc --aum 100
python scripts/20_formal_run_experiment.py --model neural_network --weights tc --aum 500
python scripts/20_formal_run_experiment.py --model neural_network --weights tc --aum 1000
```

### Formal analysis tables and figures

After the experiment artifacts exist:

```bash
python scripts/21_formal_analyze_results.py
```

## 4. Minimal Local Smoke Run

This is a small local smoke run for XGBoost plus a subset of formal specs. It is
not the full 3-model formal grid. For the full current pipeline, use
`docs/apocrita_formal_experiment.md` and `scripts/generate_hpc_jobs.sh`.

```bash
python scripts/00_fetch_data.py
python scripts/01_process_data.py
python scripts/02_motivation_step1_divergence.py --liquidity dvol --vw
python scripts/03_motivation_step2_heterogeneity.py --liquidity dvol --full
python scripts/04_motivation_step3_ml_diagnostics.py --model xgboost --liquidity dvol
python scripts/05_motivation_step3d_progressive_restriction.py --model xgboost --liquidity dvol --normalization global --use-baseline-params
python scripts/06_motivation_step3e_quintile_specific_models.py --model xgboost --liquidity dvol --normalization global --use-baseline-params
python scripts/20_formal_run_experiment.py --model xgboost --weights dolvol
python scripts/20_formal_run_experiment.py --model xgboost --weights softmax_rank --softmax-lambda 2
python scripts/20_formal_run_experiment.py --model xgboost --weights softmax_rank --softmax-lambda 3
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 10
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 100
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 500
python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 1000
python scripts/21_formal_analyze_results.py
```

## 5. Useful Single-Spec Rebuilds

### Rebuild only one restriction level

```bash
python scripts/05_motivation_step3d_progressive_restriction.py --liquidity dvol --min-quintile 2 --use-baseline-params
python scripts/05_motivation_step3d_progressive_restriction.py --liquidity dvol --min-quintile 3 --use-baseline-params
python scripts/05_motivation_step3d_progressive_restriction.py --liquidity dvol --min-quintile 4 --use-baseline-params
python scripts/05_motivation_step3d_progressive_restriction.py --liquidity dvol --min-quintile 5 --use-baseline-params
```

### Rebuild only one quintile-specific model

```bash
python scripts/06_motivation_step3e_quintile_specific_models.py --model xgboost --liquidity dvol --quintile 1
python scripts/06_motivation_step3e_quintile_specific_models.py --model xgboost --liquidity dvol --quintile 2
python scripts/06_motivation_step3e_quintile_specific_models.py --model xgboost --liquidity dvol --quintile 3
python scripts/06_motivation_step3e_quintile_specific_models.py --model xgboost --liquidity dvol --quintile 4
python scripts/06_motivation_step3e_quintile_specific_models.py --model xgboost --liquidity dvol --quintile 5
```

### Re-analyze only one formal model family

```bash
python scripts/21_formal_analyze_results.py --model xgboost
python scripts/21_formal_analyze_results.py --model elastic_net
python scripts/21_formal_analyze_results.py --model neural_network
```

### Re-analyze only one weight family

```bash
python scripts/21_formal_analyze_results.py --weights dolvol
python scripts/21_formal_analyze_results.py --weights softmax_rank
python scripts/21_formal_analyze_results.py --weights tc
```

## 6. Output Locations To Check

After the runs finish, the main output folders to inspect are:

- Motivation:
  - `outputs/motivation/step1/dvol/`
  - `outputs/motivation/step2/dvol/`
  - `outputs/motivation/step3/{model}/dvol/`
  - `outputs/motivation/step3_restriction/{model}/dvol/global/baseline/`
  - `outputs/motivation/step3_quintile/{model}/dvol/global/baseline/`

- Formal:
  - `outputs/formalanalysis/experiment/`
  - `outputs/formalanalysis/analysis/tables/`
  - `outputs/formalanalysis/analysis/figures/`

## 7. Current Cross-Pipeline Dependency Reminder

For the current code, these dependencies matter:

1. `scripts/21_formal_analyze_results.py` Prediction 2 expects:
   - `outputs/motivation/step2/dvol/interaction_regression_full.csv`
   - `outputs/motivation/step2/dvol/quintile_fm_coefficients_full.csv`

2. `scripts/21_formal_analyze_results.py` Prediction 3 expects:
   - `outputs/motivation/step3_restriction/{model}/dvol/global/baseline/restriction_comparison.csv`

That is why the cheat sheet runs `03_motivation_step2_heterogeneity.py --full` and `05_motivation_step3d_progressive_restriction.py --use-baseline-params` before the formal analysis script.
