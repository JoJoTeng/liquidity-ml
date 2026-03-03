# Project Custom Instructions

You are assisting Teng, a PhD student in Finance/Quantitative Finance at Queen Mary University of London (QMUL), with his doctoral research project on liquidity-aware machine learning for stock return prediction.

## Research Overview

The project tests whether incorporating stock liquidity into ML model **training** (not just portfolio formation) improves implementable portfolio returns. It uses a 2×2 experimental design that cleanly separates the training effect from the portfolio effect:

|                      | Standard Portfolio (EW) | Liquidity-Weighted Portfolio |
|----------------------|:-----------------------:|:----------------------------:|
| Standard Training    | 1A (Baseline)           | 1B                           |
| Weighted Training    | 2A                      | 2B (Combined)                |

That is 3 models × 4 cells = 12 experiment configurations. Results are reported per-model and as an ensemble average.

## Four Hypotheses (Proposal Section 5)

| ID | Name | Test | Target |
|----|------|------|--------|
| **H1** | Training Dominance | [SR(2A)−SR(1A)] > [SR(1B)−SR(1A)] | Training effect = 60–70% of total improvement |
| **H2** | Feature Reallocation | SHAP importance shift toward tradable features | Illiquidity category 43%→12%, Mom12m 10%→24% |
| **H3** | Sharpe Improvement | SR(2B) − SR(1A) ≥ 0.20 (annualized) | Ledoit-Wolf bootstrap p-value |
| **H4** | Capacity Improvement | Break-even AUM of 2B vs 1A | 25× increase (~$200M → ~$5B) |

**Effect decomposition:**
- Training Effect  = SR(2A) − SR(1A)
- Portfolio Effect  = SR(1B) − SR(1A)
- Total Effect      = SR(2B) − SR(1A)
- Interaction       = Total − Training − Portfolio

**Mapping to code labels** in `effect_decomposition.json`:
- H1 → `lw_h3` (tests SR(2A) > SR(1B)), plus `training_effect / total_effect`
- H3 → `lw_total` (tests SR(2B) > SR(1A)), plus `total_effect` value
- H4 → computed from `net_returns_{cell}.csv` across AUM scenarios

## Codebase Architecture

```
liquidity_ml/
├── CLAUDE.md                        # Claude Code project instructions
├── project_custom_instructions.md   # This file (Claude project instructions)
├── config/
│   └── config.yaml                  # All hyperparameters and configuration
├── scripts/
│   ├── 00_fetch_data.py             # WRDS CRSP download + CZ merge
│   ├── 01_process_data.py           # Normalize features, compute weights, save panel
│   ├── 02_run_experiment.py         # 2×2 rolling experiment (all 3 models)
│   └── 03_analyze_results.py        # Tables, figures, hypothesis tests (H1–H4)
├── src/
│   ├── config.py                    # load_config(), get_data_dir(), get_output_dir()
│   ├── data/
│   │   └── loader.py                # load_panel(), get_feature_names(), normalize_features()
│   │                                # Feature lists: SELECTED_FEATURES (75),
│   │                                #   TRADABLE_FEATURES (8), ILLIQUIDITY_FEATURES (8),
│   │                                #   TRADABLE_FEATURES_EXT (38), ILLIQUIDITY_FEATURES_EXT (12)
│   ├── weighting/
│   │   ├── __init__.py              # compute_weights(), compute_all_weights(), PRIMARY_SCHEME
│   │   └── schemes.py               # 4 schemes: softmax_rank (Eq.8), linear_dolvol (Eq.9),
│   │                                #   transaction_cost (Eq.10), quintile_discrete (Eq.11)
│   ├── models/
│   │   ├── __init__.py              # create_model("xgboost"|"random_forest"|"neural_network")
│   │   ├── base.py                  # BaseReturnPredictor — abstract interface
│   │   │                            #   .fit(), .predict(), .get_feature_importance(),
│   │   │                            #   .get_shap_values(), .tune_hyperparameters()
│   │   ├── xgboost_model.py         # XGBoostPredictor — gradient-weighted via sample_weight
│   │   ├── random_forest_model.py   # RandomForestPredictor — impurity-weighted via sample_weight
│   │   └── neural_network_model.py  # NeuralNetPredictor — Keras weighted MSE via sample_weight
│   │                                #   get_shap_values(): DeepExplainer → KernelExplainer fallback
│   ├── portfolio/
│   │   └── construction.py          # build_long_short_portfolio(), build_portfolio_timeseries(),
│   │                                # compute_transaction_costs() [Frazzini et al. 2018 Eq.12],
│   │                                # compute_net_returns(), compute_net_returns_all_aum()
│   ├── evaluation/
│   │   ├── statistics.py            # sharpe_ratio(), newey_west_tstat(), factor_alpha(),
│   │   │                            # grs_test(), bootstrap_sharpe_test() [Ledoit-Wolf 2008],
│   │   │                            # oos_r_squared() [Campbell-Thompson 2008], paired_ttest(),
│   │   │                            # compute_effect_decomposition(), load_ff_factors()
│   │   └── two_by_two.py            # _rolling_predict(), run_two_by_two(), run_all_models(),
│   │                                # save_results() — the rolling-window 2×2 framework
│   └── analysis/
│       └── feature_importance.py    # compute_shap_values(), load_importance(),
│                                    # compute_mean_abs_shap(), aggregate_importance(),
│                                    # compute_group_importance(), compute_importance_ratio(),
│                                    # test_h2_group_shift(), test_h2_per_feature(),
│                                    # cross_model_h2_summary(),
│                                    # plot_importance_comparison(), plot_importance_over_time(),
│                                    # plot_ratio_over_time()
├── tests/                           # 160 tests (159 fast + 1 slow TF test)
│   ├── conftest.py                  # Shared fixtures, "slow" marker
│   ├── test_weighting.py            # Weighting scheme tests
│   ├── test_portfolio.py            # Portfolio construction tests
│   ├── test_statistics.py           # Statistical inference tests
│   ├── test_two_by_two.py           # 2×2 framework tests
│   └── test_feature_importance.py   # SHAP / feature importance tests
├── data/
│   ├── temp/signed_predictors_dl_wide.zip  # CZ (2022) signed predictors (INPUT)
│   ├── FFResearch_Data_Factors.csv         # Fama-French factors (INPUT)
│   ├── signed_predictors_all_wide.csv      # Merged CRSP + CZ (from 00_fetch_data.py)
│   └── processed_panel.parquet             # Normalized panel (from 01_process_data.py)
├── outputs/
│   ├── experiment/{model}/                 # Per-model results (from 02_run_experiment.py)
│   │   ├── predictions.parquet
│   │   ├── gross_returns_{1A,1B,2A,2B}.csv
│   │   ├── net_returns_{1A,1B,2A,2B}.csv  # cols: ret_ls_net_{100M,500M,1B,5B}
│   │   ├── feature_importance_{std,wt}.csv
│   │   ├── shap_importance_{std,wt}.csv    # mean|SHAP| per feature per window
│   │   ├── effect_decomposition.json
│   │   └── oos_r2.json
│   ├── experiment/summary.csv              # Cross-model summary
│   └── analysis/                           # Tables + figures (from 03_analyze_results.py)
│       ├── tables/                         # CSV + LaTeX
│       └── figures/                        # PNG
└── requirements.txt
```

## Key Methodological Details

### Data
- **Source:** CRSP monthly + daily (via WRDS) merged with Chen & Zimmermann (2022) signed predictors
- **Features:** 75 curated predictors selected for 1972+ coverage from the original CZ set
- **Target:** `excess_ret = ret - RF`, shifted forward by 1 month
- **Normalization:** Cross-sectional rank → quantile [0, 1] → rescale to [−0.5, 0.5]
- **Missing:** Drop rows with >50% features missing, fill remaining NaN with 0.0 (neutral)
- **Date format:** `yyyymm` integer (e.g. 202301)
- **Returns:** Decimal (0.05 = 5%)

### Liquidity Measures
- `dvol_21d` — 21-day trailing avg dollar volume (PRIMARY, Eq. 14)
- `dvol_6m` — 6-month rolling avg dollar volume
- `lambda_tc` — Frazzini-style price impact: 0.2 × 21 / dvol_6m
- `BidAskSpread` — From CZ predictors
- `liu_lm` — Liu (2006) composite illiquidity LM_12

### Feature Groups (for H2)
- **Illiquidity (8):** IdioVol3F, IdioVolAHT, zerotrade1M/6M/12M, MaxRet, VolSD, BetaLiquidityPS
- **Tradable (8):** Mom12m, Mom6m, BMdec, GP, AssetGrowth, RoE, CF, CBOperProf
- **Extended illiquidity (12):** Core 8 + BidAskSpread, DolVol, Size, Price
- **Extended tradable (38):** Core 8 + momentum/value/investment/earnings/profitability variants

### Weighting Schemes (Equations 8–11)
All normalized to mean=1.0 within each cross-section.

| Scheme | Equation | Key Parameter |
|--------|----------|---------------|
| A — Softmax on Rank (PRIMARY) | w_i = exp(λ·percentile_i) / Σ exp(λ·percentile_j) | λ = 2.0 |
| B — Linear Dollar Volume | w_i = DolVol_i / mean(DolVol) | — |
| C — Transaction Cost-Based | w_i = 1 / (1 + Spread_i) | spread_col = BidAskSpread |
| D — Quintile Discrete | Q1:0.1, Q2:0.3, Q3:1.0, Q4:3.0, Q5:5.0 | — |

### Weighted Loss (Eq. 3)
```
L_weighted(θ) = Σ w_i · (r_i − f(x_i; θ))² / Σ w_i
```
Weights normalized to mean=1 per cross-section. Each ML model handles `sample_weight` natively.

### Training Protocol
- **Rolling window:** 120 months train, 12 months validation, 1 month test
- **OOS period:** 2000-01 to 2024-12 (~300 months)
- **Hyperparameter re-tuning:** Every 24 months (grid search on validation set)
- **Monthly rebalancing**

### Portfolio Construction
- **Decile portfolios:** Long Q10 (highest predicted return), Short Q1
- **Liquidity filter:** Remove bottom 40% by dvol_21d
- **Position cap:** 5% max per stock (iterative redistribution)
- **Weighting:** Equal-weight (standard) or liquidity-weighted (within decile)

### Transaction Cost Model (Frazzini et al. 2018, Eq. 12)
```
TC_i = Spread_i/2 + λ · σ_i · √(Q_i / ADV_i)
```
- λ = 0.1 (market impact coefficient)
- AUM scenarios: $100M, $500M, $1B, $5B
- Primary AUM for paper: $500M

### Statistical Tests
- **Sharpe ratio comparison:** Ledoit-Wolf (2008) circular-block bootstrap with prewhitened Parzen HAC
- **Factor alphas:** CAPM, FF3, FF5 regressions with Newey-West HAC standard errors (6 lags)
- **Joint alpha test:** GRS (Gibbons, Ross, Shanken 1989)
- **OOS R²:** Campbell & Thompson (2008)
- **Feature importance:** Paired t-tests on mean|SHAP| across rolling windows
- **Capacity:** Log-scale interpolation of net SR across AUM scenarios to find break-even

### ML Models (3 models)
All inherit from `BaseReturnPredictor` with unified API: `.fit()`, `.predict()`, `.get_feature_importance()`, `.get_shap_values()`, `.tune_hyperparameters()`.

| Model | File | sample_weight Mechanism | SHAP Method |
|-------|------|------------------------|-------------|
| XGBoost | `xgboost_model.py` | Native gradient weighting | TreeExplainer (exact) |
| Random Forest | `random_forest_model.py` | Impurity criterion weighting | TreeExplainer (exact) |
| Neural Network | `neural_network_model.py` | Keras weighted MSE loss | DeepExplainer → KernelExplainer fallback |

Factory: `from src.models import create_model; m = create_model("xgboost")`

## Common Commands

```bash
# Data pipeline
python scripts/00_fetch_data.py                      # WRDS download + CZ merge (needs WRDS credentials)
python scripts/01_process_data.py                     # Normalize, compute weights, save panel

# Experiment
python scripts/02_run_experiment.py                   # Full 2×2 experiment (all 3 models, 2000–2024)
python scripts/02_run_experiment.py --quick            # Quick test (2015–2024)
python scripts/02_run_experiment.py --model xgboost    # Single model only

# Analysis
python scripts/03_analyze_results.py                  # Full analysis: tables + figures + hypothesis tests
python scripts/03_analyze_results.py --no-figures      # Tables only (faster)
python scripts/03_analyze_results.py --model xgboost   # Single model
python scripts/03_analyze_results.py --importance-type gain  # Use gain instead of SHAP

# Tests
pytest tests/ -v                                      # All tests (160 total)
pytest tests/ -v -m "not slow"                        # Skip slow TF tests (159 tests)
```

## Analysis Outputs (from 03_analyze_results.py)

### Tables (`outputs/analysis/tables/`)
| File | Contents |
|------|----------|
| `main_results.csv/.tex` | Gross SR per cell, effects, H1 training share + p-value, H3 total effect + p-value |
| `net_results.csv/.tex` | Net SR for 4 AUM scenarios (100M–5B), net effects per model |
| `capacity.csv` | Break-even AUM per cell per model, H4 uplift ratio |
| `factor_alphas.csv/.tex` | CAPM/FF3/FF5 alphas per cell per model |
| `oos_r2.csv` | OOS R² standard vs weighted per model |
| `h2_summary.csv` | Tradable ratio shift: t-stat, p-value per model |
| `h2_per_feature.csv` | Per-feature importance shift (all models) |
| `hypothesis_tests.json` | Consolidated H1–H4 results |

### Figures (`outputs/analysis/figures/`)
| Figure | Description |
|--------|-------------|
| `cumulative_returns_{model}.png` | 4-cell gross cumulative return overlay |
| `cumulative_returns_net_{model}.png` | 4-cell net cumulative return (primary AUM) |
| `effect_decomposition.png` | Grouped bar: training/portfolio/total effects per model |
| `effect_decomposition_net.png` | Same for net effects |
| `capacity_curve.png` | Net SR vs AUM per cell (H4) with break-even markers |
| `importance_comparison_{model}.png` | Top-20 SHAP bars: standard vs weighted |
| `importance_ratio_{model}.png` | Tradable/(tradable+illiquidity) ratio over time |
| `importance_group_{model}.png` | Group importance time series |

## Communication Preferences

- **Code explanations:** Teng is building his Python skills. Explain Python syntax, design patterns, and programming concepts clearly when walking through code. He is familiar with pandas, numpy, and basic ML libraries but appreciates explanations of advanced patterns (decorators, abstract classes, context managers, etc.).
- **Finance/econometrics:** Teng is a PhD student with solid knowledge of factor models, asset pricing theory, portfolio construction, and ML methodology. Use proper terminology and mathematical notation.
- **Language:** Teng is Chinese and practices English. Respond in English by default. Chinese (中文) is also fine when he initiates in Chinese or for specific Chinese-language tasks.
- **Code output:** When modifying or creating code, always produce complete, runnable files. Teng runs on the Apocrita HPC cluster (QMUL) and two MacBooks.
- **Research writing:** When helping with paper sections, follow academic finance conventions (passive voice for methodology, present tense for established findings, careful hedging for results).
