# LiquidityML Code Walkthrough

This guide is a code-reading map for the current repository.

It has two jobs:

1. Map the motivation section of the document to the actual code.
2. Give you a practical order for reading the project so you can understand how data, models, weighting, diagnostics, and portfolio outputs fit together.

The guide is descriptive of the current codebase. It is not a statement that every implementation choice matches the paper literally. It is meant to help you navigate the project efficiently.

Companion visual map:
- [project_data_flow.md](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/docs/project_data_flow.md)
- [motivation_module_flow.md](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/docs/motivation_module_flow.md)
- [run_order_cheat_sheet.md](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/docs/run_order_cheat_sheet.md)

## 1. Big Picture

The repository has two main analytical tracks:

1. The motivation pipeline.
   This is the "why does liquidity matter for ML training?" part.
   Main scripts:
   - `scripts/05_step1_divergence.py`
   - `scripts/06_step2_heterogeneity.py`
   - `scripts/07_step3_ml_diagnostics.py`
   - `scripts/10_quintile_specific_models.py`
   - `scripts/12_progressive_restriction.py`
   - `scripts/11_regime_analysis.py`

2. The formal 2x2 experiment pipeline.
   This is the "weighted training vs weighted portfolio construction" part.
   Main scripts:
   - `scripts/02_run_experiment.py`
   - `scripts/03_analyze_results.py`

The shared logic for the motivation pipeline lives mostly in:
- `src/analysis/motivation.py`

The shared logic for the formal pipeline lives mostly in:
- `src/weighting/schemes.py`
- `src/models/*`
- `src/portfolio/construction.py`
- `src/evaluation/statistics.py`

## 2. Directory Map

### Core directories

- `config/`
  Global configuration and feature-category metadata.

- `scripts/`
  Top-level entry points. If you want to know "what do I run?", start here.

- `src/`
  Reusable implementation modules.

- `data/`
  Raw and processed local data products.

- `outputs/`
  Generated empirical results.

- `paper/`
  LaTeX tables and figures for the paper.

### Most important files

- `config/config.yaml`
  The master configuration file.

- `src/config.py`
  Thin loader for config, data path, and output path.

- `src/data/loader.py`
  The main panel loader used across the project.

- `src/analysis/motivation.py`
  The central motivation-analysis library.

- `scripts/00_fetch_data.py`
  Builds the raw master panel.

- `scripts/01_process_data.py`
  Produces the processed panel and feature list for motivation work.

- `scripts/02_run_experiment.py`
  Runs the formal weighted-vs-standard training experiments.

- `scripts/03_analyze_results.py`
  Turns experiment outputs into formal tables and figures.

## 3. Which Files Are Primary vs Secondary

This matters a lot when reading the code.

### Primary current paths

- Motivation:
  - `scripts/05_step1_divergence.py`
  - `scripts/06_step2_heterogeneity.py`
  - `scripts/07_step3_ml_diagnostics.py`
  - `scripts/10_quintile_specific_models.py`
  - `scripts/12_progressive_restriction.py`
  - `src/analysis/motivation.py`

- Formal analysis:
  - `scripts/02_run_experiment.py`
  - `scripts/03_analyze_results.py`
  - `src/weighting/schemes.py`
  - `src/portfolio/construction.py`
  - `src/evaluation/statistics.py`

### Secondary or older convenience paths

- `scripts/04_motivation_analysis.py`
  Older combined motivation script. Useful for historical context, but the current motivation workflow is split across Steps 1, 2, and 3 scripts.

- `scripts/08_step3_elasticnet.py`
  ElasticNet counterpart to Step 3 XGBoost diagnostics. Useful as a robustness or comparison path.

- `src/evaluation/two_by_two.py`
  Older integrated 2x2 framework. Helpful for understanding the earlier project architecture, but the active formal pipeline is now `02_run_experiment.py` plus `03_analyze_results.py`.

## 4. Motivation Section Map

This section maps the motivation document to the code.

### Step 0. Build the raw panel

Purpose:
- Pull CRSP data.
- Merge with Chen-Zimmermann predictors.
- Create the raw panel used everywhere else.

Main script:
- `scripts/00_fetch_data.py`

What it does:
- Downloads CRSP monthly data.
- Downloads CRSP daily data.
- Merges the external signed predictors zip.
- Creates derived liquidity variables such as:
  - `dvol_21d`
  - `daily_sigma`
  - `liu_lm`
  - `me_raw`
  - `Size`
  - `Price`
  - `STreversal`
- `src/data/loader.py` later adds `excess_sigma_12m`, the lagged 12-month
  excess-return volatility used by the current square-root TC model.

Main output:
- `data/signed_predictors_all_wide.csv`

Why it matters:
- This is the root dataset for both motivation and formal analysis.

### Step 1. Process the panel for motivation work

Purpose:
- Create an analysis-ready panel for Steps 1 and 2 of the motivation section.

Main script:
- `scripts/01_process_data.py`

Shared functions used:
- `src/data/loader.py::load_panel`
- `src/analysis/motivation.py::load_signaldoc`
- `src/analysis/motivation.py::get_motivation_features`
- `src/analysis/motivation.py::build_feature_categories`
- `src/analysis/motivation.py::rank_transform_01`

What it does:
- Loads the raw panel.
- Loads `SignalDoc.csv`.
- Selects the motivation feature set.
- Drops features with more than 70% missingness.
- Saves raw copies of some robustness liquidity variables.
- Rank-transforms selected features to `[0, 1]`.
- Leaves remaining missing values as `NaN`.

Main outputs:
- `data/processed_panel.parquet`
- `data/feature_list.json`
- `config/feature_categories.json`

Why it matters:
- `processed_panel.parquet` is the main input to motivation Step 1 and Step 2.

### Step 1 in the motivation document. Distributional divergence

Purpose:
- Show that the training distribution and deployment distribution differ.

Main script:
- `scripts/05_step1_divergence.py`

Core implementation lives in:
- `src/analysis/motivation.py`

Key functions:
- `compute_implementability_weights`
- `assign_nyse_quintiles`
- `compute_marginal_divergence`
- `compute_divergence_stats`
- `summarize_divergence_by_category`
- `fama_macbeth_weight_regression`
- `plot_divergence_bar_chart`
- `plot_divergence_by_category`
- `plot_density_comparison`
- `plot_weight_distribution`

Inputs:
- `data/processed_panel.parquet`
- `data/feature_list.json`
- `config/feature_categories.json`

Outputs:
- `outputs/motivation/step1/{liquidity}/divergence_monthly.parquet`
- `outputs/motivation/step1/{liquidity}/divergence_stats.csv`
- `outputs/motivation/step1/{liquidity}/divergence_by_category.csv`
- `outputs/motivation/step1/{liquidity}/divergence_bar_chart.png`
- `outputs/motivation/step1/{liquidity}/divergence_bar_chart_appendix.png`
- `outputs/motivation/step1/{liquidity}/density_comparison.png`
- `outputs/motivation/step1/{liquidity}/weight_distribution.png`

Reading tip:
- If you want to understand the economics of Step 1, read `compute_implementability_weights`, then `compute_marginal_divergence`, then `fama_macbeth_weight_regression`.

### Step 2 in the motivation document. Heterogeneous predictability

Purpose:
- Show that the return-prediction function differs across liquidity groups.

Main script:
- `scripts/06_step2_heterogeneity.py`

Core implementation lives in:
- `src/analysis/motivation.py`

Key functions:
- `assign_nyse_quintiles`
- `quintile_fama_macbeth`
- `format_quintile_table`
- `interaction_fama_macbeth`
- `plot_quintile_coefficients`
- `plot_divergence_vs_heterogeneity`

Inputs:
- `data/processed_panel.parquet`
- `outputs/motivation/step1/{liquidity}/divergence_stats.csv` for the Step 2.4 scatter

Outputs:
- `outputs/motivation/step2/{liquidity}/quintile_fm_coefficients.csv`
- `outputs/motivation/step2/{liquidity}/quintile_fm_coefficients_raw.csv`
- `outputs/motivation/step2/{liquidity}/coefficient_plots.png`
- `outputs/motivation/step2/{liquidity}/interaction_regression.csv`
- `outputs/motivation/step2/{liquidity}/interaction_regression_dummy.csv`
- `outputs/motivation/step2/{liquidity}/divergence_vs_heterogeneity.png`
- `outputs/motivation/step2/{liquidity}/interaction_meta.json`

Important extension:
- When run with `--full`, this script also writes:
  - `interaction_regression_full.csv`
  - `interaction_regression_full_dummy.csv`
  - `interaction_by_category.csv`
  - `quintile_fm_coefficients_full.csv`
  - `quintile_fm_coefficients_full_raw.csv`

Why this matters:
- The current formal analysis script uses the full-mode Step 2 outputs to build Prediction 2's `Delta I_j ~ gamma_j` regression and grouped importance-share analysis.

Reading tip:
- Start with `quintile_fama_macbeth`.
- Then read `interaction_fama_macbeth`.
- Then read the plotting helpers that turn those estimates into visual diagnostics.

### Step 3 in the motivation document. Standard ML is affected

Purpose:
- Show that standard ML allocates capacity in a way that under-serves liquid stocks.

Main script:
- `scripts/07_step3_ml_diagnostics.py`

Core implementation lives in:
- `src/analysis/motivation.py`

Key functions:
- `rolling_xgboost_predict`
- `compute_illiquidity_relatedness`
- `compute_quintile_oos_r2`
- `compute_monthly_quintile_r2`
- `compute_utility_weighted_r2`
- `compute_monthly_utility_weighted_r2`
- `compute_univariate_liquid_r2`
- `plot_importance_vs_illiquidity`
- `plot_importance_vs_liquid_r2`
- `plot_r2_by_quintile`

Inputs:
- Raw panel from `src/data/loader.py::load_panel`
- `data/feature_list.json`

Outputs:
- `outputs/motivation/step3/{liquidity}/predictions.parquet`
- `outputs/motivation/step3/{liquidity}/feature_importance.csv`
- `outputs/motivation/step3/{liquidity}/tuned_params.csv`
- `outputs/motivation/step3/{liquidity}/r2_by_quintile.csv`
- `outputs/motivation/step3/{liquidity}/r2_monthly_by_quintile.csv`
- `outputs/motivation/step3/{liquidity}/utility_weighted_r2.json`
- `outputs/motivation/step3/{liquidity}/importance_vs_illiquidity.png`
- `outputs/motivation/step3/{liquidity}/importance_vs_liquid_r2.png`

Reading tip:
- This is the most important function chain in the motivation library:
  - `rolling_xgboost_predict`
  - `compute_quintile_oos_r2`
  - `compute_utility_weighted_r2`

### Step 3 robustness and extensions

#### Step 3 ElasticNet version

Purpose:
- Re-run the Step 3 diagnostic logic with a linear model.

Script:
- `scripts/08_step3_elasticnet.py`

Core function:
- `src/analysis/motivation.py::rolling_elasticnet_predict`

Use this when:
- You want a linear-model counterpart to the XGBoost baseline.

#### Step 3d. Progressive universe restriction

Purpose:
- Compare weighted training against hard training-universe restrictions.

Script:
- `scripts/12_progressive_restriction.py`

Core function:
- `src/analysis/motivation.py::rolling_xgboost_predict_restricted`

Outputs:
- `outputs/motivation/step3_restriction_rerank/{liquidity}/{mode}/restriction_curve.png`
- `outputs/motivation/step3_restriction_rerank/{liquidity}/{mode}/restriction_comparison.csv`
- `outputs/motivation/step3_restriction_rerank/{liquidity}/{mode}/restriction_by_quintile.csv`

#### Step 3e. Quintile-specific models

Purpose:
- Compare the pooled model against quintile-specific models.

Script:
- `scripts/10_quintile_specific_models.py`

Core function:
- `src/analysis/motivation.py::rolling_xgboost_predict_quintile`

Outputs:
- `outputs/motivation/step3_quintile_rerank/{liquidity}/{mode}/r2_comparison.csv`
- `outputs/motivation/step3_quintile_rerank/{liquidity}/{mode}/r2_comparison.png`

#### Regime analysis

Purpose:
- Revisit motivation-style distribution plots under recession, VIX, and NFCI regimes.

Script:
- `scripts/11_regime_analysis.py`

Reused Step 1 functions:
- `plot_density_comparison`
- `plot_weight_distribution`
- `compute_implementability_weights`

## 5. Formal Analysis Map

This section maps the later sections of the document to the code.

### Section 7. Importance-weighted training

Main script:
- `scripts/02_run_experiment.py`

Main supporting modules:
- `src/weighting/schemes.py`
- `src/models/__init__.py`
- `src/models/base.py`
- `src/models/xgboost_model.py`
- `src/models/elastic_net_model.py`
- `src/models/neural_network_model.py`

What happens here:
- The script loads the panel.
- It loads the feature list from `feature_list.json`.
- It computes standard or weighted sample weights.
- It runs rolling train/validation/test windows.
- It saves predictions, SHAP summaries, native importance, and tuned parameters.

Current weighting definitions:
- `dolvol` is AUM-independent and uses the within-month mean denominator: `w_it = DolVol_it / mean_i(DolVol_it)`.
- This keeps the average `dolvol` sample weight equal to 1 within each month.
- `softmax_rank` is AUM-independent and uses `exp(lambda * percentile_rank(DolVol_it))`, then normalizes to mean 1 within month. The formal robustness grid runs lambda 2 and lambda 3 separately.
- `tc` is AUM-dependent and uses the transaction-cost penalty defined in `src/weighting/schemes.py`.
- Current TC-weight intensity is `alpha_t = 3.0 / median_i(TC_it)`, so the median-cost stock has raw penalty `exp(-3)` before mean-normalization.

Implementation note:
- The current code intentionally uses the cross-sectional mean for `dolvol`, not the earlier median-denominator wording from the draft document.

Outputs:
- `outputs/formalanalysis/experiment/{model}/standard/*`
- `outputs/formalanalysis/experiment/{model}/dolvol/*`
- `outputs/formalanalysis/experiment/{model}/softmax_rank_lam2/*`
- `outputs/formalanalysis/experiment/{model}/softmax_rank_lam3/*`
- `outputs/formalanalysis/experiment/{model}/tc_{aum}m/*`

### Section 8. Formal prediction tests

Main script:
- `scripts/03_analyze_results.py`

What it computes:
- Prediction 1:
  OOS R2 by liquidity quintile and utility-weighted R2
- Prediction 2:
-  feature-importance shifts
-  regression of `Delta I_j` on Step 2 `gamma_j`
-  grouped importance shares using Step 2 full outputs
- Prediction 3:
  weighted-model overlay on the Step 3d restriction curve
- Prediction 4:
  cumulative squared-error differentials

Note:
- Prediction 2 depends on Step 2 full-mode files from `scripts/06_step2_heterogeneity.py --full`.
- Prediction 3 depends on the Step 3d baseline restriction output from `scripts/12_progressive_restriction.py --use-baseline-params`.

### Section 9. Portfolio construction and 2x2 decomposition

Main script:
- `scripts/03_analyze_results.py`

Main supporting modules:
- `src/portfolio/construction.py`
- `src/evaluation/statistics.py`

What happens here:
- The 2x2 cells are built.
- Gross and net returns are computed.
- Sharpe ratios are computed.
- Ledoit-Wolf tests are run.
- Net and gross decompositions are computed.
- Factor alphas are computed through the statistics module and exported in the Table 12 output.
- Table 11 and Table 12 style outputs are produced, including turnover.

## 6. Step-by-Step Reading Order

If I were onboarding to this project from scratch, I would read it in this order.

### Pass 1. Learn the project skeleton

1. `config/config.yaml`
   Read the global data range, model configs, training windows, liquidity settings, and transaction-cost settings.

2. `src/config.py`
   See how config and paths are loaded.

3. `src/data/loader.py`
   Understand how the master panel is built for downstream use.

This first pass gives you the common vocabulary used everywhere else.

### Pass 2. Understand the dataset construction

4. `scripts/00_fetch_data.py`
   Read this to understand where every important raw variable comes from.

5. `scripts/01_process_data.py`
   Read this to understand how the processed panel and feature list are created.

At this point you should understand:
- what the target is
- what a "feature" means in this project
- what the raw vs processed panel distinction is

### Pass 3. Understand the motivation argument

6. `src/analysis/motivation.py`
   Do not read it all at once. Read it in functional blocks:
   - constants and feature definitions
   - data prep helpers
   - Step 1 functions
   - Step 2 functions
   - Step 3 functions

7. `scripts/05_step1_divergence.py`
   This shows how Step 1 functions are orchestrated.

8. `scripts/06_step2_heterogeneity.py`
   This shows how Step 2 functions are orchestrated.

9. `scripts/07_step3_ml_diagnostics.py`
   This shows how Step 3 diagnostics are orchestrated.

10. `scripts/12_progressive_restriction.py`
    Read this after Step 3, because it extends the same baseline setup.

11. `scripts/10_quintile_specific_models.py`
    Read this next, since it is another Step 3 extension.

12. `scripts/11_regime_analysis.py`
    Read this last in the motivation family because it reuses Step 1 logic rather than defining new core methods.

### Pass 4. Understand the formal weighted-training framework

13. `src/weighting/schemes.py`
    Understand the formal weighting functions first.

14. `src/models/base.py`
    Read the model interface so the training flow makes sense.

15. `src/models/xgboost_model.py`
    This is the most important model implementation for the project.

16. `src/models/elastic_net_model.py`
    Read this next as the linear benchmark implementation.

17. `src/models/neural_network_model.py`
    Read this after the first two, because it is more engineering-heavy.

18. `scripts/02_run_experiment.py`
    This is the formal training orchestrator.

### Pass 5. Understand the portfolio and inference layer

19. `src/portfolio/construction.py`
    Understand:
    - how predictions turn into decile portfolios
    - how transaction costs are applied
    - how net returns are computed

20. `src/evaluation/statistics.py`
    Understand:
    - Sharpe ratio
    - Newey-West t-statistics
    - factor alpha
    - bootstrap Sharpe tests
    - 2x2 effect decomposition

21. `scripts/03_analyze_results.py`
    This is the top-level formal reporting script.

### Pass 6. Read the remaining helpers only if needed

22. `src/analysis/feature_importance.py`
    Useful when you want to focus specifically on SHAP aggregation and group-shift analysis in the formal pipeline.

23. `scripts/04_motivation_analysis.py`
    Read only for historical context or to compare with the split motivation workflow.

24. `src/evaluation/two_by_two.py`
    Read only if you need to understand the older integrated framework.

## 7. How Data Flows Through the Project

### Motivation flow

1. `scripts/00_fetch_data.py`
   creates `data/signed_predictors_all_wide.csv`

2. `scripts/01_process_data.py`
   creates:
   - `data/processed_panel.parquet`
   - `data/feature_list.json`
   - `config/feature_categories.json`

3. `scripts/05_step1_divergence.py`
   reads the processed panel and produces Step 1 motivation outputs

4. `scripts/06_step2_heterogeneity.py`
   reads the processed panel and Step 1 divergence stats and produces Step 2 outputs

5. `scripts/07_step3_ml_diagnostics.py`
   reads the raw panel plus feature list and produces Step 3 outputs

6. `scripts/12_progressive_restriction.py` and `scripts/10_quintile_specific_models.py`
   read the Step 3 setup and extend it

7. `scripts/03_analyze_results.py`
   reuses:
   - Step 2 full outputs for Prediction 2
   - Step 3d baseline restriction outputs for Prediction 3

### Formal analysis flow

1. `scripts/02_run_experiment.py`
   reads the raw panel and feature list

2. It writes prediction and importance artifacts under `outputs/formalanalysis/experiment/`

3. `scripts/03_analyze_results.py`
   reads those experiment artifacts and produces tables and figures under `outputs/formalanalysis/analysis/`

4. For the full current Section 8 output set, `scripts/03_analyze_results.py` also reads:
   - `outputs/motivation/step2/dvol/interaction_regression_full.csv`
   - `outputs/motivation/step2/dvol/quintile_fm_coefficients_full.csv`
   - `outputs/motivation/step3_restriction_rerank/dvol/baseline/restriction_comparison.csv`

## 8. How To Read `src/analysis/motivation.py`

This file is long, so it helps to read it by chunk rather than top to bottom in one sitting.

### Chunk A. Definitions and constants

Read:
- `FOCAL_CHARACTERISTICS`
- `DENSITY_PLOT_FEATURES`
- `CZ_TO_BROAD`

These tell you:
- which variables the motivation section emphasizes
- how the paper-style categories are represented in code

### Chunk B. Data-prep helpers

Read:
- `load_signaldoc`
- `get_motivation_features`
- `load_feature_categories`
- `build_feature_categories`
- `rank_transform_01`
- `compute_implementability_weights`
- `assign_nyse_quintiles`

These functions define the common setup used across motivation Steps 1 to 3.

### Chunk C. Step 1 functions

Read:
- `compute_marginal_divergence`
- `compute_divergence_stats`
- `summarize_divergence_by_category`
- `fama_macbeth_weight_regression`

Then read the Step 1 plotting helpers.

### Chunk D. Step 2 functions

Read:
- `quintile_fama_macbeth`
- `format_quintile_table`
- `interaction_fama_macbeth`
- `plot_quintile_coefficients`
- `plot_divergence_vs_heterogeneity`

### Chunk E. Step 3 model-training helpers

Read:
- `rolling_xgboost_predict`
- `_rolling_xgboost_core`
- `rolling_xgboost_predict_restricted`
- `rolling_xgboost_predict_quintile`
- `rolling_elasticnet_predict`

This block is the bridge between the motivation logic and the model layer.

### Chunk F. Step 3 diagnostics

Read:
- `compute_illiquidity_relatedness`
- `compute_quintile_oos_r2`
- `compute_monthly_quintile_r2`
- `compute_utility_weighted_r2`
- `compute_monthly_utility_weighted_r2`
- `compute_univariate_liquid_r2`
- `plot_importance_vs_illiquidity`
- `plot_importance_vs_liquid_r2`
- `plot_r2_by_quintile`

## 9. Mental Model of the Whole Project

One useful way to think about the repository is:

1. Data construction layer
   - `scripts/00_fetch_data.py`
   - `src/data/loader.py`
   - `scripts/01_process_data.py`

2. Motivation-evidence layer
   - `scripts/05_step1_divergence.py`
   - `scripts/06_step2_heterogeneity.py`
   - `scripts/07_step3_ml_diagnostics.py`
   - `src/analysis/motivation.py`

3. Formal weighted-learning layer
   - `src/weighting/schemes.py`
   - `src/models/*`
   - `scripts/02_run_experiment.py`

4. Portfolio and inference layer
   - `src/portfolio/construction.py`
   - `src/evaluation/statistics.py`
   - `scripts/03_analyze_results.py`

5. Paper-output layer
   - `paper/TablesNew/`
   - `paper/FiguresNew/`

## 10. Recommended First Hands-On Walkthrough

If you want to understand the project quickly, this is the shortest effective route.

### Day 1 walkthrough

1. Read `config/config.yaml`.
2. Read `src/data/loader.py`.
3. Read `scripts/01_process_data.py`.
4. Read only the Step 1 part of `src/analysis/motivation.py`.
5. Read `scripts/05_step1_divergence.py`.

Goal:
- understand the panel
- understand the feature set
- understand the weighting intuition

### Day 2 walkthrough

1. Read the Step 2 part of `src/analysis/motivation.py`.
2. Read `scripts/06_step2_heterogeneity.py`.
3. Read `rolling_xgboost_predict` in `src/analysis/motivation.py`.
4. Read `scripts/07_step3_ml_diagnostics.py`.

Goal:
- understand why the paper argues the shift is not benign
- understand how the baseline ML evidence is built

### Day 3 walkthrough

1. Read `src/weighting/schemes.py`.
2. Read `src/models/base.py`.
3. Read `src/models/xgboost_model.py`.
4. Read `scripts/02_run_experiment.py`.
5. Read `src/portfolio/construction.py`.
6. Read `src/evaluation/statistics.py`.
7. Read `scripts/03_analyze_results.py`.

Goal:
- understand the formal experiment from weighted training to final tables

## 11. Quick Answer to "Where Should I Start?"

If you only have 30 minutes:

1. `config/config.yaml`
2. `src/data/loader.py`
3. `scripts/01_process_data.py`
4. `scripts/05_step1_divergence.py`
5. `scripts/07_step3_ml_diagnostics.py`
6. `scripts/02_run_experiment.py`
7. `scripts/03_analyze_results.py`

If you want the single most important shared file:

- `src/analysis/motivation.py` for the motivation side
- `scripts/02_run_experiment.py` for the formal side

## 12. Suggested Companion Notes While Reading

As you walk through the code, it helps to keep a small note with four columns:

1. File
2. Input data
3. Output data
4. Economic question answered

If you do that for:
- `scripts/01_process_data.py`
- `scripts/05_step1_divergence.py`
- `scripts/06_step2_heterogeneity.py`
- `scripts/07_step3_ml_diagnostics.py`
- `scripts/02_run_experiment.py`
- `scripts/03_analyze_results.py`

you will have a compact mental model of almost the entire project.
