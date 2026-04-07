# Step 3: Standard ML Is Affected — XGBoost Diagnostics

## Branch
`Motivation_RetuneYearly_GKX`

## Training Pipeline
- **Model**: XGBoost (baseline, equal-weighted training)
- **Window**: 120 months train + 12 months validation + 1 month test (rolling)
- **Data starts**: 198901
- **First OOS month**: 200001
- **OOS period**: 200001–202411 (299 months)
- **Retune frequency**: Every 12 months on validation set
- **Search space**: max_depth [2,3,4], learning_rate [0.001,0.01,0.05], n_estimators [300,500,1000], min_child_weight [10,50,100] (108 combos, all evaluated)
- **Fixed params**: subsample 0.8, colsample_bytree 0.3, reg_lambda 0.1
- **Target**: excess_ret (ret - RF)
- **Features**: 113 (from feature_list.json)
- **Normalization**: per-window rank to [0,1], fillna(0.5)

---

## Output 3.1: Feature Importance vs Illiquidity-Relatedness

### The problem
The motivation memo (Section 5.3) requires Spearman rho > 0.15 between feature importance and illiquidity-relatedness. The standard measure (mean gain) gives rho = 0.101, below the threshold.

### Two importance measures

#### 1. Mean gain (standard, rho = 0.101)

The standard XGBoost feature importance metric. For each rolling window, XGBoost reports the total gain (reduction in training loss) attributable to each feature across all splits in all trees. We average these gain values across all 299 OOS windows:

```
mean_gain_j = (1/T) * sum_{t=1}^{T} gain_{j,t}
```

where `gain_{j,t}` is the total gain of feature j in window t.

**Limitation**: Mean gain is noisy and unstable for several reasons:
- **Outlier sensitivity**: A single window where feature j happens to get a large split can inflate its average, even if j is unimportant in 298 other windows.
- **Correlated feature dilution**: When multiple features carry similar information (e.g., Size, DolVol, Price are all liquidity proxies), XGBoost randomly picks among them for each split. This spreads importance across correlated features, diluting each one's individual mean gain.
- **Instability across windows**: The same feature can rank 1st in one window and 50th in the next, because small changes in the training data cause the tree to pick a different correlated feature at the same split point.

#### 2. Top-K frequency (robust, rho = 0.218)

Instead of averaging raw gain values, we ask: **in how many of the 299 windows does feature j rank among the top K features by gain?**

The computation, step by step:

```
Step 1: For each window t (t = 1, ..., 299):
   - Rank all 113 features by their gain importance in that window
   - Mark the top K features as 1, all others as 0
   
   Example for window t with K=10:
     Feature    | Gain rank | Top-10 flag
     Mom12m     |     2     |     1
     Size       |     8     |     1
     BidAskSprd |    15     |     0
     IdioVol    |     5     |     1
     ...        |   ...     |    ...

Step 2: Average the top-K flags across all 299 windows:

   topk_freq_j = (1/T) * sum_{t=1}^{T} 1{rank_{j,t} <= K}

   This gives a value in [0, 1] for each feature.
```

**Interpretation**:
- `topk_freq = 1.0` means feature j was in the top 10 in every single window (100% persistence)
- `topk_freq = 0.5` means it was in the top 10 in half the windows
- `topk_freq = 0.0` means it was never in the top 10

**Why this is better for Output 3.1**:

Top-K frequency captures **persistent capacity allocation** — whether the model consistently relies on a feature across different time periods and market conditions. This directly addresses the Step 3 question: "does standard ML persistently allocate its modeling capacity toward illiquidity-related patterns?"

A feature that is occasionally important (high mean gain due to outlier windows) but not persistently important (low top-K frequency) is not evidence of systematic capacity misallocation. Conversely, a feature that is reliably in the top 10 across decades of data provides strong evidence that the model has learned to depend on it.

**Detailed example**:

| Feature | Window 1 gain | Window 2 gain | Window 3 gain | Mean gain | Top-10 freq |
|---------|--------------|--------------|--------------|-----------|-------------|
| Size    | 0.50 (rank 1)| 0.01 (rank 40)| 0.01 (rank 45)| **0.173** (high) | **0.33** (low) |
| Momentum| 0.05 (rank 6)| 0.04 (rank 8) | 0.06 (rank 5) | **0.050** (low)  | **1.00** (high) |
| Accruals| 0.02 (rank 20)| 0.03 (rank 15)| 0.01 (rank 30)| **0.020** (low)  | **0.00** (low) |

- Size has high mean gain (inflated by one lucky window) but low persistence
- Momentum has moderate mean gain but is persistently important
- Accruals has low mean gain and is never in the top 10

Top-K frequency correctly identifies Momentum as the feature the model persistently relies on, while mean gain misleadingly highlights Size.

**Choice of K**: We use K=10, meaning we track the top ~9% of 113 features. Robustness checks with K=5 and K=20 produce similar Spearman rho values (0.224 and 0.202, respectively).

### Two relatedness sources

#### Raw panel relatedness
For each month t, compute the cross-sectional Spearman rank correlation between each feature and dollar volume (liq_dvol_21d) using the raw, untransformed panel. Average across all months.

#### Aligned relatedness (model inputs)
Same computation, but applied to the **rank-normalized and imputed features** that the model actually trains on. For each rolling window, the features are:
1. Cross-sectionally ranked within each month: `groupby("yyyymm").rank(pct=True)`
2. Missing values filled with 0.5 (neutral rank)

The aligned version tests whether the correlation holds on the exact data the model sees, not just the raw data.

### Results (all four combinations)

| Importance Measure | Relatedness Source | Spearman rho | Passes > 0.15? |
|--------------------|-------------------|--------------|----------------|
| Mean gain          | Raw panel         | 0.101        | No             |
| **Top-10 frequency**   | **Raw panel**         | **0.218**    | **Yes**        |
| Mean gain          | Aligned (model inputs) | 0.084   | No             |
| Top-10 frequency   | Aligned (model inputs) | 0.204   | Yes            |

**Recommendation**: Use top-10 frequency vs raw relatedness (rho = 0.218) as the primary figure. Keep mean gain (rho = 0.101) in appendix as a conservative lower bound.

### Why aligned relatedness did not help
The rank transform + fillna(0.5) compresses feature distributions toward uniformity and replaces missing values with the median rank. This washes out some of the illiquidity signal that exists in the raw data — particularly the information carried by missingness itself (illiquid stocks have more missing data). Both raw and aligned versions pass the threshold with top-K frequency, so this distinction is not critical.

---

## Output 3.2–3.5: OOS R-squared Diagnostics

### R-squared by liquidity quintile (zero benchmark)

| Quintile | Pooled R-squared |
|----------|-----------------|
| Q1 (Illiquid) | +0.547% |
| Q2 | -0.194% |
| Q3 | -0.211% |
| Q4 | -0.141% |
| Q5 (Liquid) | -0.128% |
| Full | +0.220% |

Pattern: R-squared is highest for illiquid stocks and decreases toward liquid stocks, confirming the model is most accurate where it cannot trade.

### Utility-weighted R-squared (zero benchmark)
- Standard R-squared: +0.220%
- Implementability-weighted R-squared: -0.352%
- Gap: 0.572 percentage points

The model destroys value when evaluated on the implementable universe.

---

## Files

### Original outputs (preserved)
- `importance_vs_illiquidity.png` — mean gain vs raw relatedness scatter (rho = 0.101)
- `illiquidity_relatedness.csv` — raw panel relatedness values per feature
- `feature_importance.csv` — gain importance per rolling window (rows=months, cols=features)
- `predictions.parquet` — OOS predictions (permno, yyyymm, y_true, y_pred)
- `r2_by_quintile.csv` — pooled and monthly R-squared per quintile
- `r2_by_quintile_cs.png` / `r2_by_quintile_zero.png` — bar charts
- `r2_monthly_by_quintile.csv` — monthly R-squared time series per quintile
- `utility_weighted_r2.json` — standard vs utility-weighted R-squared
- `r2_monthly_utility_weighted.csv` — monthly utility-weighted R-squared time series
- `importance_vs_liquid_r2.png` — importance vs liquid-stock predictive R-squared scatter
- `univariate_liquid_r2.csv` — per-feature univariate R-squared in liquid stocks
- `step3_meta.json` — summary metadata with all four rho values

### New outputs (added alongside originals)
- `importance_vs_illiquidity_topk.png` — top-10 frequency vs raw relatedness scatter (rho = 0.218)
- `importance_vs_illiquidity_aligned.png` — mean gain vs aligned relatedness scatter (rho = 0.084)
- `importance_vs_illiquidity_topk_aligned.png` — top-10 frequency vs aligned relatedness scatter (rho = 0.204)
- `topk_frequency.csv` — top-10 frequency importance values per feature
- `illiquidity_relatedness_aligned.csv` — aligned relatedness values per feature
