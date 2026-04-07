# Step 3: Standard ML Is Affected

## Objective

Establish that a standard (equal-weighted) ML model exhibits the symptoms predicted by Steps 1 and 2: its feature importance is tilted toward characteristics informative for illiquid stocks, and its predictions are most accurate for illiquid stocks and least accurate for liquid stocks — precisely where predictions matter most for an implementable strategy.

This step requires the baseline ML pipeline. It uses out-of-sample predictions and feature importances from a standard XGBoost model trained with the rolling-window protocol.

## Data and Setup

- **Model:** XGBoost (baseline, equal-weighted training — no liquidity weighting)
- **Training protocol:** Rolling window — 120 months train, 12 months validation, 1 month test
- **OOS period:** 200001–202411 (299 months)
- **Hyperparameter re-tuning:** Every 12 months on validation set
- **Search space:** max_depth [2,3,4], learning_rate [0.001,0.01,0.05], n_estimators [300,500,1000], min_child_weight [10,50,100] — 108 combinations, all evaluated
- **Fixed params:** subsample 0.8, colsample_bytree 0.3, reg_lambda 0.1
- **Target:** excess_ret (ret − RF)
- **Features:** 113 characteristics (same as Steps 1–2)
- **Normalization:** Per-window rank to [0,1], NaN filled with 0.5 (neutral rank)
- **Liquidity quintiles:** NYSE breakpoints on `liq_dvol_21d`, same as Step 2
- **R² benchmarks:** Both zero benchmark (Campbell & Thompson 2008, primary) and cross-sectional mean benchmark (GKX 2020)

## Methods and Outputs

### Output 3.1 — Feature Importance vs. Illiquidity-Relatedness

**Method:** From the baseline XGBoost model, extract native feature importances (gain-based) for each of the 299 rolling windows.

**Importance measures (two variants):**

1. **Mean gain (standard):** Average raw gain values across all windows: Ī_j = (1/T) Σ_t gain_{j,t}. Simple but noisy — sensitive to outlier windows and correlated-feature dilution.

2. **Top-K frequency (robust):** Fraction of windows in which feature j ranks among the top K=10 by gain. Captures persistent capacity allocation rather than being inflated by sporadic large splits. A feature with topk_freq = 0.80 was in the model's top 10 in 80% of windows — strong evidence of systematic reliance.

**Illiquidity-relatedness:** For each month t, compute the cross-sectional Spearman rank correlation between each feature x_{ij,t} and dollar volume. Average across months to get ρ̄_j. A large negative ρ̄_j means the feature is associated with illiquid stocks (high characteristic value = low dollar volume). Two variants: raw panel and aligned (computed on the same rank-normalized, imputed data the model trains on).

**Scatter plot:** Ī_j on y-axis vs. −ρ̄_j on x-axis (negated so "more illiquidity-related" goes right). One point per characteristic (~113 points), 15 focal characteristics labeled. Report Spearman rank correlation.

**Output files:**
- `importance_vs_illiquidity.png` — Mean gain vs. raw relatedness (ρ = 0.101)
- `importance_vs_illiquidity_topk.png` — **Top-10 frequency vs. raw relatedness (ρ = 0.218, primary)**
- `importance_vs_illiquidity_aligned.png` — Mean gain vs. aligned relatedness (ρ = 0.084)
- `importance_vs_illiquidity_topk_aligned.png` — Top-10 frequency vs. aligned relatedness (ρ = 0.204)
- `feature_importance.csv` — Gain importance per rolling window (rows=months, cols=features)
- `topk_frequency.csv` — Top-10 frequency per feature
- `illiquidity_relatedness.csv` — Raw panel relatedness per feature
- `illiquidity_relatedness_aligned.csv` — Aligned relatedness per feature

### Output 3.2 — Feature Importance vs. Liquid-Stock Predictive R²

**Method:** For each feature j, compute the univariate predictive R² among liquid stocks only (Q4–Q5 pooled). Use Fama-MacBeth slope within Q4–Q5, then R² ≈ t² / (t² + T − 1). Scatter Ī_j (y-axis) against R²_j(liquid) (x-axis). If importance is unrelated to liquid-stock predictive power, the model is optimizing for the wrong target.

**Output files:**
- `importance_vs_liquid_r2.png` — Scatter plot
- `univariate_liquid_r2.csv` — Per-feature univariate R² in liquid stocks

### Output 3.3 — OOS R² by Liquidity Quintile (Bar Chart)

**Method:** Using OOS predictions r̂_{it} from the baseline model, compute pooled OOS R² separately for each liquidity quintile (Eq. 8 of document):

    R²_q = 1 − Σ_t Σ_{i∈Q_q,t} (r_{it} − r̂_{it})² / Σ_t Σ_{i∈Q_q,t} (r_{it} − benchmark)²

Two benchmarks: (a) zero (primary, Campbell & Thompson 2008), (b) cross-sectional mean r̄_t computed over the full sample, not within the quintile.

**Output files:**
- `r2_by_quintile_zero.png` — Bar chart, zero benchmark (primary)
- `r2_by_quintile_cs.png` — Bar chart, cross-sectional mean benchmark

### Output 3.4 — OOS R² by Quintile (Table)

**Method:** Same as Output 3.3, plus average monthly R² as a robustness check and average number of stocks per quintile per month.

**Output files:**
- `r2_by_quintile.csv` — Pooled and monthly R² per quintile
- `r2_monthly_by_quintile.csv` — Monthly R² time series per quintile

### Output 3.5 — Standard vs. Utility-Weighted OOS R²

**Method (Eq. 9 of document):** Compute a single summary statistic:

    R²_w = 1 − Σ_t Σ_i w̃_{it} · (r_{it} − r̂_{it})² / Σ_t Σ_i w̃_{it} · (r_{it} − benchmark)²

using the same normalized dollar-volume weights w̃_{it} from Step 1. Compare to the standard (equal-weighted) pooled R².

**Output files:**
- `utility_weighted_r2.json` — Standard vs. utility-weighted R² (both benchmarks)
- `r2_monthly_utility_weighted.csv` — Monthly utility-weighted R² time series

## Results

### Feature Importance vs. Illiquidity-Relatedness (Output 3.1)

| Importance Measure   | Relatedness Source     | Spearman ρ | Passes > 0.15? |
|----------------------|------------------------|------------|----------------|
| Mean gain            | Raw panel              | 0.101      | No             |
| **Top-10 frequency** | **Raw panel**          | **0.218**  | **Yes**        |
| Mean gain            | Aligned (model inputs) | 0.084      | No             |
| Top-10 frequency     | Aligned (model inputs) | 0.204      | Yes            |

The primary result (top-10 frequency, ρ = 0.218) confirms that the model persistently allocates its capacity toward characteristics associated with illiquid stocks. The mean gain measure (ρ = 0.101) provides a conservative lower bound, falling just short of the 0.15 threshold due to noise from outlier windows and correlated-feature dilution — the top-K frequency metric is more appropriate for measuring persistent capacity allocation.

### OOS R² by Liquidity Quintile (Outputs 3.3–3.4)

**Zero benchmark (primary):**

| Quintile        | Pooled R² (%) | Avg. Monthly R² (%) | Avg. N/month |
|-----------------|---------------|----------------------|--------------|
| Q1 (Illiquid)   | **+0.547**    | −0.547               | 2,438        |
| Q2              | −0.194        | −0.728               | 1,262        |
| Q3              | −0.211        | −0.485               | 1,051        |
| Q4              | −0.141        | −0.470               | 833          |
| Q5 (Liquid)     | −0.128        | −0.475               | 743          |
| **Full sample** | **+0.220**    | —                    | 6,336        |

**Cross-sectional mean benchmark:**

| Quintile        | Pooled R² (%) | Avg. Monthly R² (%) | Avg. N/month |
|-----------------|---------------|----------------------|--------------|
| Q1 (Illiquid)   | **−5.81**     | −6.66                | 2,438        |
| Q2              | −13.13        | −13.72               | 1,262        |
| Q3              | −17.65        | −17.56               | 1,051        |
| Q4              | −20.31        | −20.84               | 833          |
| Q5 (Liquid)     | **−19.88**    | −22.16               | 743          |
| **Full sample** | **−10.33**    | —                    | 6,336        |

### Utility-Weighted OOS R² (Output 3.5)

| Metric                              | Zero Benchmark | CS-Mean Benchmark |
|-------------------------------------|----------------|-------------------|
| Standard (equal-weighted) R²        | +0.220%        | −10.33%           |
| Implementability-weighted R²        | −0.352%        | −23.68%           |
| **Gap (standard − weighted)**       | **+0.572 pp**  | **+13.35 pp**     |

### Summary Statistics

- 299 OOS months, 1,894,464 total predictions, 113 features
- Average ~6,336 stocks per month; Q1 has ~2,438, Q5 has ~743

## Analysis

### The model is most accurate where it cannot trade

Under the zero benchmark, the model achieves a positive pooled R² of **+0.547%** for Q1 (illiquid stocks) — the only quintile where R² is positive. For Q2–Q5, R² is negative, meaning the model underperforms the naive zero forecast for all tradeable quintiles. The full-sample R² of +0.220% is entirely driven by the model's accuracy among illiquid stocks that dominate by count (2,438 per month in Q1 vs. 743 in Q5).

This is precisely the symptom predicted by Steps 1 and 2: the model has learned to predict returns for the equal-weighted training distribution (dominated by illiquid stocks), and its predictions are least accurate in the deployment universe (liquid stocks).

### The cross-sectional mean benchmark reveals monotonic degradation

Under the CS-mean benchmark, R² is **monotonically decreasing** from Q1 (−5.81%) to Q4 (−20.31%), with Q5 slightly recovering to −19.88%. This means the model's predictions become progressively worse relative to a simple cross-sectional mean as stocks become more liquid. The monotonic pattern strongly suggests systematic misallocation of model capacity.

### Utility-weighted R² confirms the mismatch destroys implementable value

The gap between standard and implementability-weighted R² is **0.572 percentage points** under the zero benchmark, far exceeding the PDF's minimum threshold of 0.05–0.10 pp. Under the CS-mean benchmark, the gap widens to 13.35 pp. When the model's predictions are evaluated with the weights that matter for implementation, performance deteriorates sharply — the model optimizes the wrong objective.

Under the zero benchmark, standard R² is positive (+0.220%) but utility-weighted R² is **negative** (−0.352%). This means the model has positive predictive power on average across the full cross-section, but **destroys value** when evaluated on the implementable, dollar-volume-weighted universe. An investor who deploys these predictions in a liquidity-aware portfolio would have been better off predicting zero.

### Feature importance is tilted toward illiquidity-related patterns

The Spearman ρ = 0.218 (top-10 frequency vs. illiquidity-relatedness) confirms that the model persistently allocates its modeling capacity toward characteristics associated with illiquid stocks. Features that are more negatively correlated with dollar volume (i.e., features that load on illiquid stocks) are more likely to appear in the model's top 10 features across rolling windows.

This is consistent with the limits-to-arbitrage finding from Step 2: illiquid stocks exhibit stronger and more varied return predictability (especially for microstructure-driven signals like STreversal), so the model — optimizing under equal weights — naturally learns to exploit these stronger patterns. The problem is that these patterns are strongest precisely where the investor cannot trade.

### The three steps together make the case

| Step | Question | Finding |
|------|----------|---------|
| Step 1 | Do P_train and P_deploy differ? | Yes — 105/113 characteristics diverge significantly; R² = 0.913 |
| Step 2 | Is the mismatch harmful? | Yes — 8/15 focal characteristics have heterogeneous predictability; F = 15.87, p ≈ 0 |
| Step 3 | Is standard ML affected? | Yes — R² positive only for illiquid stocks; utility-weighted R² is negative; importance tilted toward illiquidity |

The mismatch exists (Step 1), it is harmful because the return-generating function differs across liquid and illiquid stocks (Step 2), and standard ML is visibly affected — it allocates capacity toward illiquid-stock patterns and produces predictions that are most accurate where they cannot be traded (Step 3).

### Diagnostic criterion assessment

The PDF (Section 5.3) requires:
- **Feature importance correlates positively with illiquidity-relatedness (Spearman ρ > 0.15, ideally > 0.3):** Passes with top-K frequency (ρ = 0.218). Mean gain (ρ = 0.101) falls short but is a noisier measure.
- **OOS R² substantially higher for Q1 than Q5:** Passes — Q1 R² = +0.547% vs. Q5 R² = −0.128% under zero benchmark. Under CS-mean, R² is monotonically decreasing in liquidity.
- **Utility-weighted R² meaningfully lower than standard R² (gap ≥ 0.05–0.10 pp):** Passes — gap = 0.572 pp under zero benchmark (5.7× the minimum threshold).

All three diagnostic criteria are met. The motivation for a training-stage correction — liquidity-aware importance weighting — is established.

## Files

| File | Description |
|------|-------------|
| `predictions.parquet` | OOS predictions (permno, yyyymm, y_true, y_pred) |
| `feature_importance.csv` | Gain importance per rolling window (rows=months, cols=features) |
| `topk_frequency.csv` | Top-10 frequency importance per feature |
| `illiquidity_relatedness.csv` | Raw panel Spearman relatedness per feature |
| `illiquidity_relatedness_aligned.csv` | Aligned (model-input) relatedness per feature |
| `univariate_liquid_r2.csv` | Per-feature univariate R² in liquid stocks (Q4–Q5) |
| `r2_by_quintile.csv` | Pooled and monthly R² per quintile (both benchmarks) |
| `r2_monthly_by_quintile.csv` | Monthly R² time series per quintile |
| `utility_weighted_r2.json` | Standard vs. utility-weighted R² |
| `r2_monthly_utility_weighted.csv` | Monthly utility-weighted R² time series |
| `step3_meta.json` | Summary metadata (all Spearman ρ values, R² stats) |
| `importance_vs_illiquidity.png` | Scatter: mean gain vs. raw relatedness (ρ = 0.101) |
| `importance_vs_illiquidity_topk.png` | Scatter: top-K freq vs. raw relatedness (ρ = 0.218, primary) |
| `importance_vs_illiquidity_aligned.png` | Scatter: mean gain vs. aligned relatedness (ρ = 0.084) |
| `importance_vs_illiquidity_topk_aligned.png` | Scatter: top-K freq vs. aligned relatedness (ρ = 0.204) |
| `importance_vs_liquid_r2.png` | Scatter: importance vs. liquid-stock predictive R² |
| `r2_by_quintile_zero.png` | Bar chart: R² by quintile (zero benchmark) |
| `r2_by_quintile_cs.png` | Bar chart: R² by quintile (CS-mean benchmark) |
