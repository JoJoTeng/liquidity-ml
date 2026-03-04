# Liquidity-Aware ML for Asset Pricing

## Project Goal
Test whether incorporating stock liquidity into ML training improves
return predictions — not just at the portfolio level but at the
*model training* level. Uses a 2×2 experimental design to decompose
the contribution of liquidity-weighted training vs. liquidity-weighted
portfolio construction.

## Only Two Input Files
- `data/temp/signed_predictors_dl_wide.zip` — Chen & Zimmermann (2022) signed predictors
- `data/FFResearch_Data_Factors.csv` — Fama-French factors (risk-free rate)

Everything else is derived from WRDS (CRSP monthly + CRSP daily).

## Architecture
```
liquidity_ml/
├── CLAUDE.md                  ← you are here
├── config/config.yaml         ← all hyperparameters
├── scripts/
│   ├── 00_fetch_data.py       ← WRDS download + merge CZ
│   ├── 01_process_data.py     ← normalize, compute weights, save panel
│   ├── 02_run_experiment.py   ← 2×2 rolling experiment (all 3 models)
│   └── 03_analyze_results.py  ← tables, figures, hypothesis tests (H1–H4)
├── src/
│   ├── config.py              ← load_config(), get_data_dir(), get_output_dir()
│   ├── data/
│   │   └── loader.py          ← load_panel(), get_feature_names(), normalize_features()
│   │                            Feature lists: SELECTED_FEATURES (75),
│   │                            TRADABLE_FEATURES (8), ILLIQUIDITY_FEATURES (8),
│   │                            TRADABLE_FEATURES_EXT (38), ILLIQUIDITY_FEATURES_EXT (12)
│   ├── weighting/
│   │   ├── __init__.py        ← compute_weights(), compute_all_weights(), PRIMARY_SCHEME
│   │   └── schemes.py         ← 4 weighting schemes (Eq. 8-11)
│   ├── models/
│   │   ├── __init__.py        ← model factory: create_model("xgboost")
│   │   ├── base.py            ← BaseReturnPredictor abstract interface
│   │   ├── xgboost_model.py
│   │   ├── random_forest_model.py
│   │   └── neural_network_model.py
│   ├── portfolio/
│   │   └── construction.py    ← long-short decile portfolios, transaction costs,
│   │                            compute_net_returns_all_aum()
│   ├── evaluation/
│   │   ├── statistics.py      ← sharpe_ratio(), bootstrap_sharpe_test() [Ledoit-Wolf],
│   │   │                        newey_west_tstat(), factor_alpha(), oos_r_squared(),
│   │   │                        oos_r_squared_monthly(),
│   │   │                        paired_ttest(), compute_effect_decomposition()
│   │   └── two_by_two.py      ← _rolling_predict(), run_two_by_two(), run_all_models(),
│   │                            save_results()
│   └── analysis/
│       └── feature_importance.py ← load_importance(), test_h2_group_shift(),
│                                   test_h2_per_feature(), cross_model_h2_summary(),
│                                   plot_importance_comparison(), plot_ratio_over_time()
├── tests/                     ← 160 tests (pytest tests/ -v -m "not slow")
│   ├── conftest.py
│   ├── test_weighting.py
│   ├── test_portfolio.py
│   ├── test_statistics.py
│   ├── test_two_by_two.py
│   └── test_feature_importance.py
├── data/                      ← raw + processed data
└── outputs/
    ├── experiment/{model}/    ← per-model results (from 02_run_experiment.py)
    └── analysis/              ← tables + figures (from 03_analyze_results.py)
```

## Build Order (ALL COMPLETE)
1. `scripts/00_fetch_data.py` — fetches CRSP + merges CZ
2. `src/models/*` — XGBoost, Random Forest, Neural Network
3. `src/data/loader.py` — load panel, define feature lists
4. `src/weighting/schemes.py` — 4 weighting schemes
5. `scripts/01_process_data.py` — orchestrate load + weights + save
6. `src/portfolio/construction.py` — long-short portfolios + transaction costs
7. `src/evaluation/statistics.py` — statistical tests
8. `src/evaluation/two_by_two.py` — 2×2 framework with rolling windows
9. `scripts/02_run_experiment.py` — run the experiment (all 3 models)
10. `src/analysis/feature_importance.py` — SHAP analysis + H2 testing
11. `scripts/03_analyze_results.py` — tables, figures, hypothesis tests (H1–H4)

## ML Models (3 models)
Each model runs independently through the 2×2 framework.
All support `sample_weight` in `.fit()` for liquidity-weighted training.

| Model | File | sample_weight mechanism | SHAP method |
|-------|------|------------------------|-------------|
| XGBoost | `xgboost_model.py` | Enters gradient computation | TreeExplainer (exact) |
| Random Forest | `random_forest_model.py` | Enters impurity criterion | TreeExplainer (exact) |
| Neural Network | `neural_network_model.py` | Keras weighted MSE loss | DeepExplainer → KernelExplainer |

Factory: `from src.models import create_model; m = create_model("xgboost")`

All models inherit from `BaseReturnPredictor` (src/models/base.py) with API:
  - `.fit(X_train, y_train, X_val, y_val, sample_weight, sample_weight_val)` → self
  - `.predict(X)` → np.ndarray
  - `.get_feature_importance(feature_names)` → pd.Series
  - `.get_shap_values(X_test, X_background, feature_names, config)` → pd.DataFrame
  - `.tune_hyperparameters(X_train, y_train, X_val, y_val, sample_weight, sample_weight_val)` → dict

## 2×2 Experimental Design (CORE)
Run INDEPENDENTLY for each of the 3 models (XGBoost, RF, NN):
```
                        Standard Portfolio    Liquidity-Weighted Portfolio
Standard Training       1A (Baseline)         1B
Weighted Training       2A                    2B (Combined)
```
That's 3 models × 4 cells = 12 experiment configurations.
Results are reported per-model and as an ensemble average.
Effect decomposition:
- Training Effect  = SR(2A) - SR(1A)
- Portfolio Effect = SR(1B) - SR(1A)
- Total Effect     = SR(2B) - SR(1A)
- Interaction      = Total - Training - Portfolio

## Key Hypotheses (Proposal Section 5)

| ID | Name | Test | Target |
|----|------|------|--------|
| H1 | Training Dominance | [SR(2A)−SR(1A)] > [SR(1B)−SR(1A)] | Training = 60–70% of total |
| H2 | Feature Reallocation | SHAP shift toward tradable features | Illiq 43%→12% |
| H3 | Sharpe Improvement | SR(2B) − SR(1A) ≥ 0.20 (annualized) | Ledoit-Wolf p-value |
| H4 | Capacity Improvement | Break-even AUM: 2B vs 1A | 25× increase |

Code label mapping in `effect_decomposition.json`:
- H1 → `lw_h3` (tests SR(2A) > SR(1B)), plus `training_effect / total_effect`
- H3 → `lw_total` (tests SR(2B) > SR(1A))
- H4 → computed from `net_returns_{cell}.csv` across AUM scenarios

H1 and H3 are tested on both gross and net returns:
- Gross: primary test (from `effect_decomposition.json`, computed during experiment)
- Net: robustness test per AUM level (computed during analysis by `compute_net_effect_tests()`
  in `03_analyze_results.py`). Results in `net_results.csv` (`h1_pval_{aum}`, `h3_pval_{aum}`)
  and `hypothesis_tests.json` (under `net_results` sub-dict in H1/H3 entries)

## Feature Groups (for H2)
- **Illiquidity (8):** IdioVol3F, IdioVolAHT, zerotrade1M/6M/12M, MaxRet, VolSD, BetaLiquidityPS
- **Tradable (8):** Mom12m, Mom6m, BMdec, GP, AssetGrowth, RoE, CF, CBOperProf
- Extended versions also defined: ILLIQUIDITY_FEATURES_EXT (12), TRADABLE_FEATURES_EXT (38)

## Weighting Schemes (Equations 8–11)
All normalized to mean=1.0 within each cross-section.
Liquidity percentile is based on dvol_21d (21-day trailing avg dollar volume, Eq. 14).

Scheme A — Softmax on Rank (Primary):
  w_i = exp(λ · percentile_i) / Σ_j exp(λ · percentile_j)
  λ = 2.0

Scheme B — Linear Dollar Volume:
  w_i = DolVol_i / mean(DolVol)

Scheme C — Transaction Cost-Based:
  w_i = 1 / (1 + Spread_i)
  where Spread_i = BidAskSpread (from CZ signed predictors)

Scheme D — Quintile Discrete:
  Q5 (most liquid): 5.0, Q4: 3.0, Q3: 1.0, Q2: 0.3, Q1 (most illiquid): 0.1

## Data Columns After Fetch (00_fetch_data.py output)
The signed_predictors_all_wide.csv contains:
- permno, yyyymm — identifiers (both int)
- ret — delisting-adjusted monthly return
- All CZ signed predictors (Mom12m, BMdec, GP, DolVol, BidAskSpread, etc.)
- STreversal, Price, Size — CRSP-derived signed predictors
- me_raw — raw market equity (for capacity analysis)
- dvol_monthly — monthly dollar volume (|prc| × vol_in_shares)
- dvol_21d — 21-day trailing avg dollar volume (PRIMARY liquidity measure, Eq. 14)
- dvol_6m — 6-month rolling avg dollar volume
- lambda_tc — price impact = 0.2 × 21 / dvol_6m (Frazzini-style)
- liu_lm — Liu (2006) composite illiquidity LM_12 (from daily CRSP)
- BidAskSpread — from CZ predictors (used for Scheme C weighting)

## Data Conventions
- Dates: yyyymm integer (e.g. 202301 for Jan 2023)
- Returns: decimal (0.05 = 5%)
- Target: excess_ret = ret - RF, shifted forward by 1 month
- Normalization: rank → quantile [0, 1] → rescale to [-0.5, 0.5]
- Missing: drop rows with >50% features missing, fill remaining NaN with 0.0

## Training Protocol
- Rolling window: 120 months train, 12 months validation, 1 month test
- OOS period: 2000-01 to 2024-12
- Hyperparameter re-tuning every 24 months
- Monthly rebalancing

## Portfolio Construction
- Decile portfolios: Long Q10 (highest predicted), Short Q1
- Liquidity filter: remove bottom 40% by dvol_21d
- Position cap: 5% max per stock (iterative redistribution)

## Transaction Costs (Frazzini et al. 2018, Eq. 12)
TC_i = Spread_i/2 + λ · σ_i · √(Q_i / ADV_i), where λ = 0.1
AUM scenarios: $100M, $500M, $1B, $5B
Net return columns in saved CSVs: `ret_ls_net_{100M,500M,1B,5B}`

## Experiment Outputs (02_run_experiment.py)
Per model in `outputs/experiment/{model_name}/`:
- `predictions.parquet` — permno, yyyymm, y_true, pred_std, pred_wt
- `gross_returns_{1A,1B,2A,2B}.csv` — yyyymm, ret_long_short
- `net_returns_{1A,1B,2A,2B}.csv` — yyyymm, ret_ls_net_{100M,500M,1B,5B}
- `feature_importance_{std,wt}.csv` — rows=yyyymm, cols=features (gain)
- `shap_importance_{std,wt}.csv` — rows=yyyymm, cols=features (mean|SHAP|)
- `effect_decomposition.json` — sharpe_ratios, effects, lw_* tests, factor_alphas
- `oos_r2.json` — {standard, weighted}
- `oos_r2_monthly.csv` — yyyymm, R2_std, R2_wt (monthly cross-sectional OOS R²)

Cross-model: `outputs/experiment/summary.csv`, `ensemble.json`

## Analysis Outputs (03_analyze_results.py)
Tables in `outputs/analysis/tables/`:
- `main_results.csv/.tex` — gross SR, effects, H1/H3 p-values
- `net_results.csv/.tex` — net SR per AUM, net effects, net H1/H3 p-values per AUM
- `capacity.csv` — break-even AUM per cell (H4)
- `factor_alphas.csv/.tex` — CAPM/FF3/FF5 alphas
- `oos_r2.csv` — OOS R² standard vs weighted (aggregate)
- `oos_r2_monthly_{model}.csv` — monthly OOS R² time series per model
- `h2_summary.csv` — tradable ratio shift per model
- `h2_per_feature.csv` — per-feature importance shift
- `hypothesis_tests.json` — consolidated H1–H4

Figures in `outputs/analysis/figures/`:
- `cumulative_returns_{model}.png` — 4-cell gross overlay
- `cumulative_returns_net_{model}.png` — 4-cell net (primary AUM)
- `effect_decomposition.png` / `_net.png` — grouped bar charts
- `capacity_curve.png` — net SR vs AUM (H4)
- `importance_comparison_{model}.png` — SHAP bars std vs wt
- `importance_ratio_{model}.png` — tradable ratio time series
- `importance_group_{model}.png` — group importance time series
- `oos_r2_monthly.png` — monthly OOS R² time series (std vs wt, all models)

## Common Commands
```bash
python scripts/00_fetch_data.py                      # WRDS download + merge (needs WRDS)
python scripts/01_process_data.py                    # normalize + compute weights
python scripts/02_run_experiment.py                  # run 2×2 experiment (full)
python scripts/02_run_experiment.py --quick          # quick test (2015–2024)
python scripts/02_run_experiment.py --model xgboost  # single model
python scripts/03_analyze_results.py                 # tables + figures + H1–H4
python scripts/03_analyze_results.py --no-figures    # tables only (faster)
python scripts/03_analyze_results.py --model xgboost # single model
pytest tests/ -v -m "not slow"                       # run tests (159 fast)
pytest tests/ -v                                     # all tests (160 total)
```

## Testing
Run: `pytest tests/ -v -m "not slow"` (159 tests, ~7s)
The 1 slow test (`test_nn_requires_background`) requires TensorFlow initialization.
Test files: test_weighting, test_portfolio, test_statistics, test_two_by_two, test_feature_importance.
