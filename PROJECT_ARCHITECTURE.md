# Liquidity-Aware ML for Stock Return Prediction
## Project Architecture & Technical Reference

---

## 1. Research Question

**Does incorporating stock liquidity into the training objective of machine learning models -- rather than only into portfolio construction -- improve the implementability and net performance of cross-sectional return predictions?**

Standard ML models for return prediction (Gu, Kelly, Xiu 2020) minimize unweighted MSE, treating all stocks equally. This means a microcap with $50K daily volume contributes the same gradient as Apple with $10B daily volume. The model may learn patterns in untradeable stocks that don't survive transaction costs.

This project proposes a liquidity-weighted training loss (Equation 3):

    L_weighted(theta) = Sum w_i * (r_i - f(x_i; theta))^2 / Sum w_i

where w_i reflects stock i's liquidity. This focuses the model on predicting returns for stocks that can actually be traded profitably.

---

## 2. Experimental Design (2x2 Matrix)

The design cleanly separates two channels through which liquidity can enter:

|                      | Standard Portfolio (EW) | Liquidity-Weighted Portfolio |
|----------------------|:-----------------------:|:----------------------------:|
| Standard Training    | 1A (Baseline)           | 1B                           |
| Weighted Training    | 2A                      | 2B (Combined)                |

3 models x 4 cells = 12 experiment configurations.
Results are reported per-model and as an ensemble average.

**Effect Decomposition:**
- Training Effect  = SR(2A) - SR(1A) -- different predictions, same portfolio weights
- Portfolio Effect = SR(1B) - SR(1A) -- same predictions, different portfolio weights
- Total Effect     = SR(2B) - SR(1A) -- both channels active
- Interaction      = Total - Training - Portfolio -- complementarity or redundancy

**Why this design matters:** Row 1 (1A vs 1B) only changes how predictions are deployed. Row 2 changes what the model learns. If the training effect exceeds the portfolio effect (H1), it means liquidity-aware learning produces fundamentally better predictions, not just better portfolio construction.

---

## 3. Hypotheses (Proposal Section 5)

| ID | Name | Test | Target |
|----|------|------|--------|
| **H1** | Training Dominance | [SR(2A)-SR(1A)] > [SR(1B)-SR(1A)] | Training effect = 60-70% of total improvement |
| **H2** | Feature Reallocation | SHAP importance shift toward tradable features | Illiquidity category 43%->12%, Mom12m 10%->24% |
| **H3** | Sharpe Improvement | SR(2B) - SR(1A) >= 0.20 (annualized) | Ledoit-Wolf bootstrap p-value |
| **H4** | Capacity Improvement | Break-even AUM of 2B vs 1A | 25x increase (~$200M -> ~$5B) |

**Mapping to code labels** in `effect_decomposition.json`:
- H1 -> `lw_h3` (tests SR(2A) > SR(1B)), plus `training_effect / total_effect` ratio
- H3 -> `lw_total` (tests SR(2B) > SR(1A)), plus `total_effect` value
- H4 -> computed from `net_returns_{cell}.csv` across AUM scenarios via log-scale interpolation

**Gross vs Net testing:**
- H1 and H3 are tested on both gross returns (primary, computed during experiment) and net returns (robustness, computed during analysis via `compute_net_effect_tests()` in `03_analyze_results.py`)
- Net tests use Ledoit-Wolf bootstrap on `ret_ls_net_{AUM}` columns at each AUM level
- Net p-values saved in `net_results.csv` (`h1_pval_{AUM}`, `h3_pval_{AUM}`) and `hypothesis_tests.json` (`net_results` sub-dict under H1/H3)

---

## 4. Data Pipeline

### 4.1 Data Sources
- **CRSP Monthly:** permno, date, ret, prc, shrout, vol, spread -- stock returns and characteristics
- **CRSP Daily:** Used to compute 21-day dollar volume (dvol_21d), Liu (2006) illiquidity (liu_lm)
- **Chen & Zimmermann (2022):** Open Source Asset Pricing database -- 75 signed predictors selected for 1972+ coverage
- **Fama-French Factors:** Mkt-RF, SMB, HML (FF3) + RMW, CMA (FF5) -- for alpha regressions
- **Risk-free rate:** From FF factors file, for computing excess returns

### 4.2 Liquidity Measures

| Measure | Column | Formula | Interpretation |
|---------|--------|---------|---------------|
| Dollar Volume (21d) | liq_dvol_21d | Mean of daily |prc| x volume over 21 days | PRIMARY. Higher = more liquid |
| Dollar Volume (6m) | liq_dvol_6m | 6-month rolling avg dollar volume | Used for lambda_tc |
| Bid-Ask Spread | liq_BidAskSpread | From CZ signed predictors | Higher = more illiquid |
| Liu (2006) LM | liq_liu_lm | Composite: zero-volume days + 1/turnover | Higher = more illiquid |
| Price Impact | liq_lambda_tc | 0.2 x 21 / dvol_6m (Frazzini-style) | Higher = more illiquid |
| Daily Volatility | daily_sigma | Std of daily returns over 21 days | Used in TC model |

### 4.3 Feature Groups (for H2 testing)

**Illiquidity features (8 core):** IdioVol3F, IdioVolAHT, zerotrade1M, zerotrade6M, zerotrade12M, MaxRet, VolSD, BetaLiquidityPS
**Extended (12):** Core 8 + BidAskSpread, DolVol, Size, Price

**Tradable features (8 core):** Mom12m, Mom6m, BMdec, GP, AssetGrowth, RoE, CF, CBOperProf
**Extended (38):** Core 8 + momentum variants, value, investment, earnings, profitability, accruals, issuance

### 4.4 Panel Structure
- ~5.4M stock-month rows covering ~38,870 unique permnos
- Date range: 1972-01 to 2024-12
- Columns: permno, yyyymm, ret, excess_ret, 75 feature columns, liq_* columns, weight_* columns
- Stored as Parquet for efficient I/O

### 4.5 Data Conventions
- Dates: yyyymm integer (e.g. 202301 for Jan 2023)
- Returns: decimal (0.05 = 5%)
- Target: excess_ret = ret - RF, shifted forward by 1 month
- Normalization: cross-sectional rank -> quantile [0, 1] -> rescale to [-0.5, 0.5]
- Missing: drop rows with >50% features missing, fill remaining NaN with 0.0 (neutral)

---

## 5. Weighting Schemes (src/weighting/schemes.py)

Four schemes, all normalize to mean=1 within each cross-section:

**Scheme A: Softmax on Rank (Primary)** -- w_i = exp(lambda * percentile_i) / mean(...), lambda=2.0. Smooth, differentiable. Most liquid gets ~7.4x weight of least liquid.

**Scheme B: Linear Dollar Volume** -- w_i = DolVol_i / mean(DolVol_i). Direct proportional weighting.

**Scheme C: Transaction Cost Inverse** -- w_i = 1/(1+Spread_i) / mean(...). Tighter spreads = higher weights.

**Scheme D: Quintile Discrete** -- Q1->0.1, Q2->0.3, Q3->1.0, Q4->3.0, Q5->5.0 then normalize. Most aggressive.

---

## 6. ML Models

### 6.1 Common Interface (src/models/base.py)

All three models implement BaseReturnPredictor(ABC) with:
- `fit(X_train, y_train, X_val, y_val, sample_weight, sample_weight_val)` -> self
- `predict(X)` -> np.ndarray
- `get_feature_importance(feature_names)` -> pd.Series
- `get_shap_values(X_test, X_background, feature_names, config)` -> pd.DataFrame
- `tune_hyperparameters(X_train, y_train, X_val, y_val, sample_weight, sample_weight_val)` -> dict

Factory: `from src.models import create_model; m = create_model("xgboost")`

### 6.2 How Weights Enter Each Model

| Model | Mechanism | Effect |
|-------|-----------|--------|
| XGBoost | sample_weight modifies gradient: g_i = -2*w_i*(r_i - y_hat_i) | Next tree focuses on high-weight stock errors |
| Random Forest | sample_weight modifies impurity calculation | Splits reduce weighted impurity; leaves are weighted averages |
| Neural Network | Keras multiplies loss: loss_i = w_i * (y_i - y_hat_i)^2 | Backprop gradients scaled by weights throughout network |

### 6.3 SHAP Integration

| Model | SHAP Method | Notes |
|-------|-------------|-------|
| XGBoost | `shap.TreeExplainer` (exact) | Fast, no background data needed |
| Random Forest | `shap.TreeExplainer` (exact) | Fast, no background data needed |
| Neural Network | `shap.DeepExplainer` -> `shap.KernelExplainer` fallback | Requires X_background (100 samples from training data) |

SHAP values are computed at each rolling window during the experiment (mean|SHAP| per feature per window), accumulated as time series, and saved as `shap_importance_{std,wt}.csv`.

### 6.4 Neural Network Architecture (Gu, Kelly, Xiu 2020)

Input(75) -> Dense(64)+BatchNorm+ReLU+Dropout(0.5) -> Dense(32)+BatchNorm+ReLU+Dropout(0.5) -> Dense(16)+BatchNorm+ReLU+Dropout(0.5) -> Dense(1, linear)

Callbacks: EarlyStopping(patience=10), ReduceLROnPlateau(patience=5, factor=0.5)

---

## 7. Portfolio Construction (src/portfolio/construction.py)

### 7.1 Long-Short Decile Portfolio (per OOS month)
1. Liquidity filter: Remove bottom 40% by dollar volume
2. Assign deciles by predicted return (1=lowest, 10=highest)
3. Long = decile 10, Short = decile 1
4. Within-decile weights: equal (cells 1A,2A) or liquidity-proportional (cells 1B,2B)
5. Position cap: 5% max per stock (iterative clip-and-redistribute)
6. Return: ret_long_short = weighted_return(long) - weighted_return(short)

### 7.2 Transaction Cost Model (Frazzini et al. 2018, Eq. 12)

    TC_i = Spread_i/2 + lambda * sigma_i * sqrt(Q_i / ADV_i)

- Half-spread: fixed cost of crossing bid-ask
- Market impact: lambda=0.1, sigma_i=daily vol, Q_i=trade size ($), ADV_i=avg daily volume
- AUM scenarios: $100M, $500M, $1B, $5B for capacity testing (H4)
- Primary AUM for the paper: $500M

Net return columns saved: `ret_ls_net_{100M,500M,1B,5B}`

---

## 8. Statistical Inference (src/evaluation/statistics.py)

- **sharpe_ratio():** annualized = (mean/std) * sqrt(12)
- **newey_west_tstat():** OLS on constant, HAC SEs (6 lags)
- **factor_alpha():** r_t = alpha + sum(beta_k * F_k,t) + epsilon_t with NW SEs (CAPM, FF3, FF5)
- **grs_test():** H0 = all portfolio alphas jointly zero (Gibbons, Ross, Shanken 1989)
- **bootstrap_sharpe_test():** Ledoit-Wolf (2008) studentized circular-block bootstrap for Sharpe ratio difference, with prewhitened VAR(1) residuals and Parzen kernel HAC
- **oos_r_squared():** Campbell-Thompson (2008): 1 - SS_pred/SS_hist (benchmark = expanding mean), aggregate scalar
- **oos_r_squared_monthly():** Cross-sectional OOS R² per month: R²_t = 1 - Σ_i(y_it - ŷ_it)² / Σ_i(y_it - ȳ_expanding_t)², returns time series DataFrame
- **paired_ttest():** for feature importance differences across rolling windows (H2)
- **compute_effect_decomposition():** full 2x2 with Sharpe ratios, effects, LW tests, factor alphas (gross returns)
- **compute_net_effect_tests():** (in `03_analyze_results.py`) Ledoit-Wolf bootstrap tests on net return series per AUM level for H1/H3 robustness

---

## 9. Feature Importance Analysis (src/analysis/feature_importance.py)

### 9.1 Loading & Aggregation
- `load_importance(model_name, importance_type="shap"|"gain")` -- loads from experiment output
- `compute_mean_abs_shap(df)` -- mean absolute SHAP per feature
- `aggregate_importance(df, method="mean"|"median")` -- aggregate across windows
- `compute_group_importance(df, group="tradable"|"illiquidity")` -- sum importance within feature group per window
- `compute_importance_ratio(df)` -- tradable/(tradable+illiquidity) ratio per window

### 9.2 H2 Hypothesis Testing
- `test_h2_group_shift(importance_std, importance_wt)` -- paired t-test on tradable ratio shift between standard and weighted training. Returns: ratio means, diff, t-stat, p-value
- `test_h2_per_feature(importance_std, importance_wt)` -- per-feature paired t-test. Returns DataFrame sorted by p-value with feature, mean_std, mean_wt, mean_diff, t_stat, p_value, significant, category
- `cross_model_h2_summary(model_names)` -- aggregates H2 results across all models

### 9.3 Visualization
- `plot_importance_comparison(std, wt, top_n=20)` -- top-N features side-by-side bars
- `plot_importance_over_time(std, wt, group)` -- group importance time series
- `plot_ratio_over_time(std, wt)` -- tradable/(tradable+illiquidity) ratio over time

---

## 10. Experiment Runner (scripts/02_run_experiment.py)

### Rolling Window Loop (_rolling_predict)
For each OOS month:
1. Split: Train (120 mo), Validation (12 mo), Test (1 mo)
2. Normalize: Train+Val together (prevent look-ahead), Test independently
3. Retune: Hyperparameters every 24 months (grid search on validation set)
4. Train: Standard model (equal weights) + weighted model (liquidity weights)
5. Predict: Generate predictions for test set
6. Collect: Feature importances (gain) + SHAP values (mean|SHAP| per feature)

### run_two_by_two(model_name)
Build 4 portfolio time series -> compute TC at all AUM -> align months -> effect decomposition -> OOS R-squared (aggregate + monthly) -> save

### run_all_models()
Runs all 3 models sharing one panel load. Produces per-model results + ensemble summary.

### CLI
```bash
python scripts/02_run_experiment.py                  # Full run (2000-2024)
python scripts/02_run_experiment.py --quick          # Quick test (2015-2024)
python scripts/02_run_experiment.py --model xgboost  # Single model
```

---

## 11. Analysis & Outputs (scripts/03_analyze_results.py)

### 11.0 Analysis Pipeline (main() execution order)
1. Load experiment results
2. Main results table (gross SR, effects, gross H1/H3 p-values)
3. Net return statistical tests (`compute_net_effect_tests()` — LW bootstrap on net returns per AUM)
4. Net results table (net SR, net effects, net H1/H3 p-values)
5. Capacity analysis (H4 break-even AUM)
6. Factor alpha table
7. OOS R² table
8. H2 analysis (feature importance)
9. Hypothesis test summary (consolidated H1–H4, gross + net)
10. Figures
11. Console summary

### 11.1 Tables (outputs/analysis/tables/)

| File | Contents |
|------|----------|
| main_results.csv/.tex | Gross SR per cell, effects, training share (H1), total effect (H3), LW p-values |
| net_results.csv/.tex | Net SR for 4 AUM scenarios, net effects, net H1/H3 p-values per AUM per model |
| capacity.csv | Break-even AUM per cell per model, H4 uplift ratio |
| factor_alphas.csv/.tex | CAPM/FF3/FF5 alphas per cell per model |
| oos_r2.csv | OOS R-squared standard vs weighted per model (aggregate) |
| oos_r2_monthly_{model}.csv | Monthly cross-sectional OOS R² time series per model |
| h2_summary.csv | Tradable ratio shift: t-stat, p-value per model |
| h2_per_feature.csv | Per-feature importance shift (all models) |
| hypothesis_tests.json | Consolidated H1-H4 results |

### 11.2 Figures (outputs/analysis/figures/)

| Figure | Description |
|--------|-------------|
| cumulative_returns_{model}.png | 4-cell gross cumulative return overlay |
| cumulative_returns_net_{model}.png | 4-cell net cumulative return at primary AUM ($500M) |
| effect_decomposition.png | Grouped bar: training/portfolio/total effects per model |
| effect_decomposition_net.png | Same for net effects |
| capacity_curve.png | Net SR vs AUM per cell with break-even markers (H4) |
| importance_comparison_{model}.png | Top-20 SHAP bars: standard vs weighted |
| importance_ratio_{model}.png | Tradable/(tradable+illiquidity) ratio over time |
| importance_group_{model}.png | Group importance time series |
| oos_r2_monthly.png | Monthly OOS R² time series: standard vs weighted (all models) |

### 11.3 H4 Capacity Analysis
For each model and each cell:
1. Compute net Sharpe ratio at each AUM level (100M, 500M, 1B, 5B)
2. Interpolate on log10(AUM) scale (scipy.interpolate.interp1d + brentq) to find break-even AUM where net SR = 0
3. Report uplift = breakeven_2B / breakeven_1A (target: 25x)

### CLI
```bash
python scripts/03_analyze_results.py                       # Full analysis
python scripts/03_analyze_results.py --no-figures          # Tables only
python scripts/03_analyze_results.py --model xgboost       # Single model
python scripts/03_analyze_results.py --importance-type gain # Use gain instead of SHAP
```

---

## 12. Key Config Parameters (config/config.yaml)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| train_window | 120 | 10 years training |
| validation_window | 12 | 1 year validation |
| test_window | 1 | 1 month OOS prediction |
| oos_start / oos_end | 200001 / 202412 | OOS test period |
| retune_frequency | 24 | Retune hyperparameters every 2 years |
| n_quantiles | 10 | Decile portfolios |
| position_cap | 0.05 | 5% max per stock |
| min_liquidity_pctile | 0.4 | Top 60% liquidity filter |
| lambda_market_impact | 0.1 | TC market impact coefficient |
| aum_scenarios | [100M, 500M, 1B, 5B] | Capacity testing (H4) |
| newey_west_lags | 6 | HAC lag count |
| bootstrap_samples | 5000 | LW bootstrap replications |
| primary weighting | softmax_rank | Default scheme (Scheme A) |
| compute_shap | true | Enable SHAP at each rolling window |
| background_samples | 100 | Background data for NN SHAP |
| seed | 42 | Reproducibility |

---

## 13. Codebase Map

```
liquidity_ml/
├── config/
│   └── config.yaml                  -- all hyperparameters
├── src/
│   ├── config.py                    -- load_config(), get_data_dir(), get_output_dir()
│   ├── data/
│   │   └── loader.py                -- load_panel(), get_feature_names(), normalize_features()
│   │                                   SELECTED_FEATURES (75), TRADABLE_FEATURES (8),
│   │                                   ILLIQUIDITY_FEATURES (8), *_EXT variants
│   ├── weighting/
│   │   ├── __init__.py              -- compute_weights(), compute_all_weights(), PRIMARY_SCHEME
│   │   └── schemes.py               -- 4 weighting schemes (Eq. 8-11)
│   ├── models/
│   │   ├── __init__.py              -- create_model() factory
│   │   ├── base.py                  -- BaseReturnPredictor(ABC)
│   │   ├── xgboost_model.py         -- XGBoostPredictor
│   │   ├── random_forest_model.py   -- RandomForestPredictor
│   │   └── neural_network_model.py  -- NeuralNetPredictor (TF/Keras)
│   ├── portfolio/
│   │   └── construction.py          -- portfolios, turnover, transaction costs, net returns
│   ├── evaluation/
│   │   ├── statistics.py            -- all statistical tests + effect decomposition
│   │   └── two_by_two.py            -- _rolling_predict(), run_two_by_two(),
│   │                                   run_all_models(), save_results()
│   └── analysis/
│       └── feature_importance.py    -- SHAP analysis, H2 testing, visualization
├── scripts/
│   ├── 00_fetch_data.py             -- WRDS CRSP download + CZ merge
│   ├── 01_process_data.py           -- normalize, compute weights, save panel
│   ├── 02_run_experiment.py         -- 2x2 rolling experiment (all 3 models)
│   └── 03_analyze_results.py        -- tables, figures, hypothesis tests (H1-H4)
├── tests/                           -- 160 tests total (159 fast + 1 slow TF test)
│   ├── conftest.py                  -- shared fixtures, "slow" marker
│   ├── test_weighting.py
│   ├── test_portfolio.py
│   ├── test_statistics.py
│   ├── test_two_by_two.py
│   └── test_feature_importance.py
├── data/
│   ├── temp/signed_predictors_dl_wide.zip  -- CZ (2022) signed predictors (INPUT)
│   ├── FFResearch_Data_Factors.csv         -- Fama-French factors (INPUT)
│   ├── signed_predictors_all_wide.csv      -- merged CRSP + CZ (from 00_fetch_data.py)
│   └── processed_panel.parquet             -- normalized panel (from 01_process_data.py)
├── outputs/
│   ├── experiment/{model}/                 -- per-model results (from 02_run_experiment.py)
│   │   ├── predictions.parquet
│   │   ├── gross_returns_{1A,1B,2A,2B}.csv
│   │   ├── net_returns_{1A,1B,2A,2B}.csv
│   │   ├── feature_importance_{std,wt}.csv
│   │   ├── shap_importance_{std,wt}.csv
│   │   ├── effect_decomposition.json
│   │   ├── oos_r2.json
│   │   └── oos_r2_monthly.csv
│   ├── experiment/summary.csv              -- cross-model summary
│   ├── experiment/ensemble.json            -- ensemble averages
│   └── analysis/                           -- from 03_analyze_results.py
│       ├── tables/                         -- CSV + LaTeX + JSON
│       └── figures/                        -- PNG
└── requirements.txt
```

---

## 14. Testing

Run: `pytest tests/ -v -m "not slow"` (159 tests, ~7s)

| Test File | Coverage |
|-----------|----------|
| test_weighting.py | Weighting scheme computation, normalization, all 4 schemes |
| test_portfolio.py | Portfolio construction, turnover, position capping, net returns |
| test_statistics.py | Sharpe, Newey-West, factor alpha, LW bootstrap, OOS R², GRS, paired t-test |
| test_two_by_two.py | Rolling windows, effect decomposition, save/load |
| test_feature_importance.py | SHAP computation, aggregation, H2 group/per-feature tests, load/save, cross-model |

The 1 slow test (`test_nn_requires_background`) requires TensorFlow initialization and is skipped by default.

---

## 15. Known Issues and Design Decisions

1. **Scheme C NaN handling:** Missing BidAskSpread set to 0.0 gives max weight. Acceptable since liquidity filter removes most problematic stocks.
2. **Validation weights:** sample_weight_val added to all models so early stopping and tuning optimize the same weighted objective as training.
3. **Lazy TF imports:** TensorFlow imported inside functions to avoid slow import when using tree models only. The `@pytest.mark.slow` marker skips TF-dependent tests by default.
4. **Position cap:** Iterative clip-and-redistribute (up to 20 iterations). Falls back to equal weights if infeasible.
5. **Block length calibration:** Disabled by default in LW bootstrap (too expensive). Uses grid [2,4,6,8,10] with b=6 as default.
6. **SHAP in rolling loop:** Mean|SHAP| per feature is accumulated at each window (not full SHAP matrices) to keep memory manageable. Only the aggregated importance is saved.
7. **H4 break-even interpolation:** Uses log10(AUM) scale with scipy.interpolate.interp1d + brentq. Reports ">5B" if net SR positive at all AUM levels, "<100M" if negative everywhere.
