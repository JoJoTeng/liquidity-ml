# Next Task: Implement Motivation Analysis (04_motivation_analysis.py)

## Goal

Create `scripts/04_motivation_analysis.py` that generates empirical evidence for Section 2 of the paper ("The Implementability Imbalance Problem"). This script produces three tables and two figures showing that standard ML training is misaligned with the implementable investment universe.

All models are trained with **standard (unweighted) training only** — this script documents the problem, not the solution.

---

## Outputs

All saved to `outputs/motivation/`:

| File | Description |
|------|-------------|
| `table1_distribution.csv` | Training sample vs implementable universe distribution by quintile × AUM |
| `table2_oos_r2_quintile.csv` | OOS R² by liquidity quintile × model |
| `table3_sr_quintile_{model}.csv` | Gross vs net Sharpe ratio by quintile, per model |
| `figure_r2_by_quintile.png` | Grouped bar chart of R² by quintile |
| `figure_sr_scissors.png` | Line chart: gross SR vs net SR by quintile (the "scissors" pattern) |
| `predictions_{model}.parquet` | Raw OOS predictions with quintile labels (for reuse) |

---

## Three Pieces of Evidence

### Evidence 1 — Table 1: Distribution Mismatch

**Question:** What fraction of implementable stocks come from each liquidity quintile?

**Method:**

1. Each month, assign stocks to quintiles Q1–Q5 by `liq_dvol_21d` (21-day trailing average daily dollar volume). Each quintile = 20% of stocks by construction.

2. For each AUM scenario ($100M, $500M, $5B), compute one-way TC for every stock using Frazzini et al. (2018):

```
TC_i = (spread_i * spread_scale) / 2 + lambda_mi * sigma_i * sqrt(Q_i / ADV_i)
```

where:
- `spread_i` = `liq_BidAskSpread` (Corwin-Schultz from CZ)
- `spread_scale` = 0.30 (from `config["transaction_costs"]["spread_scale"]`)
- `lambda_mi` = 0.1 (from `config["transaction_costs"]["lambda_market_impact"]`)
- `sigma_i` = `liq_daily_sigma` (21-day rolling daily return volatility)
- `ADV_i` = `liq_dvol_21d`
- `Q_i` = AUM / 500 (assuming 500 portfolio stocks)

3. A stock is "implementable" if `TC_i < 0.01` (1% one-way threshold).

4. For each quintile: `pct_implementable = count(implementable in Qq) / count(all implementable) * 100`

5. Average across all OOS months. Each AUM column sums to 100%.

**Table structure:**

| Quintile | % Training | Impl. ($100M) | Impl. ($500M) | Impl. ($5B) |
|----------|:---:|:---:|:---:|:---:|
| Q1 (Most Illiquid) | 20% | X% | X% | X% |
| Q2 | 20% | X% | X% | X% |
| Q3 | 20% | X% | X% | X% |
| Q4 | 20% | X% | X% | X% |
| Q5 (Most Liquid) | 20% | X% | X% | X% |

**Handle missing data:** Fill missing `liq_BidAskSpread` with cross-sectional median. Fill missing `liq_daily_sigma` with cross-sectional median. Skip stocks missing `liq_dvol_21d` (can't assign quintile).

---

### Evidence 2 — Table 2: OOS R² by Liquidity Quintile

**Question:** Does the standard model predict illiquid stocks better than liquid stocks?

**Method:**

1. Run rolling-window OOS predictions using standard (unweighted) training for each model: ElasticNet, XGBoost, Random Forest.

2. Rolling window protocol (from `config["training"]`):
   - Train window: 120 months
   - Validation window: 12 months
   - Test window: 1 month
   - OOS period: `oos_start` to `oos_end` (default 200001–202412)
   - Retune hyperparameters every 24 months (XGBoost, RF only)

3. For each test month, assign each stock to liquidity quintile Q1–Q5 based on `liq_dvol_21d` in that month.

4. Pool all (stock, month) OOS observations. Compute R² within each quintile using **zero benchmark** (GKX convention):

```python
R2_q = 1 - sum((y_true - y_pred)**2) / sum(y_true**2)   # for stocks in quintile q
```

Also compute pooled R² across all quintiles.

**Models:**

- **ElasticNet**: Use `sklearn.linear_model.ElasticNetCV` with `l1_ratio=[0.5, 0.7, 0.9, 0.95, 1.0]`, `cv=5`, `max_iter=5000`. No hyperparameter retuning schedule — ElasticNetCV handles alpha selection internally. No `sample_weight`.

- **XGBoost and Random Forest**: Use existing `src.models.create_model()`. Call `model.tune_hyperparameters()` every 24 months (same as the main experiment), then `model.fit()` monthly. Pass `sample_weight=None` and `sample_weight_val=None` for standard training.

**Normalization:** Follow the same protocol as `_prepare_xy()` in `two_by_two.py`:
- Train + Val normalised together, test normalised independently
- `normalize_features()` from `src.data.loader` (rank → [-1, 1])
- Fill remaining NaN in features with 0.0
- Drop rows with NaN in target column

**Table structure:**

| Quintile | ElasticNet | XGBoost | Random Forest |
|----------|:---:|:---:|:---:|
| Q1 (Most Illiquid) | X.XX% | X.XX% | X.XX% |
| ... | | | |
| Q5 (Most Liquid) | X.XX% | X.XX% | X.XX% |
| Pooled | X.XX% | X.XX% | X.XX% |

---

### Evidence 3 — Table 3: Gross vs. Net SR by Liquidity Quintile

**Question:** Is the gross alpha in illiquid stocks destroyed by transaction costs?

**Method:**

Using the **same OOS predictions** from Evidence 2 (no re-training needed):

1. For each test month and each quintile, form a **within-quintile** long-short portfolio:
   - Sort stocks in quintile $q$ by predicted return
   - Assign to deciles within quintile (use fewer bins if < 100 stocks: `n_bins = min(10, n_stocks // 10)`, minimum 5)
   - Long top decile, short bottom decile, equal-weight

2. **Gross return:** `r_LS = mean(r_long) - mean(r_short)`

3. **Transaction costs** at $500M AUM:
   - Track positions month-to-month to compute turnover
   - For each stock entering or exiting the portfolio, compute TC using Frazzini:
     ```
     TC_i = (liq_BidAskSpread * 0.30) / 2 + 0.1 * liq_daily_sigma * sqrt(Q_i / liq_dvol_21d)
     ```
     where `Q_i = 500_000_000 / n_portfolio_stocks`
   - Portfolio TC: `turnover * average(TC_i for traded stocks)`
   - First month: turnover = 2.0 (full establishment of both legs)

4. **Net return:** `r_LS_net = r_LS_gross - TC_portfolio`

5. **Annualised Sharpe ratio:** `SR = mean(r_LS) / std(r_LS) * sqrt(12)`

**Table structure (per model):**

| Quintile | Gross SR | Net SR ($500M) | Avg Ret (%) | Avg TC (%) | N months |
|----------|:---:|:---:|:---:|:---:|:---:|
| Q1 | X.XX | X.XX | X.XX | X.XX | XXX |
| ... | | | | | |
| Q5 | X.XX | X.XX | X.XX | X.XX | XXX |

**Figure: Scissors chart.** Two lines: gross SR (solid) declining from Q1→Q5, net SR (dashed) non-monotonic peaking at Q3–Q4. Use XGBoost as the primary model for the figure; report all models in the table.

---

## Codebase Context

### Key imports and functions

```python
from src.config import load_config, get_output_dir
from src.data.loader import load_panel, get_feature_names, normalize_features
from src.models import create_model  # factory for xgboost, random_forest
```

### Data loading

```python
config = load_config()
panel = load_panel(config)          # returns full panel with liq_* columns
features = get_feature_names(panel)  # returns list of 86 feature column names
```

The panel contains columns: `permno`, `yyyymm`, `ret`, `excess_ret`, 86 feature columns, `liq_dvol_21d`, `liq_BidAskSpread`, `liq_daily_sigma`, `liq_dvol_6m`, `liq_lambda_tc`, `liq_liu_lm`, `me_raw`, `weight_*` columns.

### Config values needed

```python
config["training"]["train_window"]         # 120
config["training"]["validation_window"]    # 12
config["training"]["oos_start"]            # 200001
config["training"]["oos_end"]              # 202412
config["training"]["retune_frequency"]     # 24
config["data"]["target_col"]               # "excess_ret"
config["data"]["selected_features"]        # list of 86 feature names
config["liquidity"]["primary"]             # "dvol_21d"
config["transaction_costs"]["spread_scale"]        # 0.30
config["transaction_costs"]["lambda_market_impact"] # 0.1
config["project"]["seed"]                  # 42
```

### Normalization protocol (from two_by_two.py)

```python
# Train + val normalised together, test independently
train_val = pd.concat([train_df, val_df])
train_val_norm = normalize_features(train_val, features)
test_norm = normalize_features(test_df, features)

# Split back
train_norm = train_val_norm.loc[train_df.index]
val_norm = train_val_norm.loc[val_df.index]

# Fill NaN with 0.0 (neutral rank)
for df in [train_norm, val_norm, test_norm]:
    df[features] = df[features].fillna(0.0)

# Drop rows with NaN target
train_norm = train_norm.dropna(subset=[target_col])
```

### Model usage pattern

```python
# XGBoost / Random Forest
model = create_model("xgboost", config=None, seed=42)
best_params = model.tune_hyperparameters(
    X_train, y_train, X_val, y_val,
    sample_weight=None, sample_weight_val=None,  # standard training
)
model = create_model("xgboost", config=best_params, seed=42)
model.fit(X_train, y_train, X_val, y_val,
          sample_weight=None, sample_weight_val=None)
preds = model.predict(X_test)
```

### Existing OOS months iteration pattern (from two_by_two.py)

```python
all_months = sorted(panel["yyyymm"].unique())
min_history = train_window + val_window

for test_month in all_months:
    if test_month < oos_start or test_month > oos_end:
        continue
    idx = all_months.index(test_month)
    if idx < min_history:
        continue

    months_before = all_months[:idx]
    val_months = set(months_before[-val_window:])
    train_months = set(months_before[-(train_window + val_window):-val_window])

    train_df = panel[panel["yyyymm"].isin(train_months)]
    val_df = panel[panel["yyyymm"].isin(val_months)]
    test_df = panel[panel["yyyymm"] == test_month]
```

---

## CLI Interface

```bash
python scripts/04_motivation_analysis.py                    # Full run, all 3 models
python scripts/04_motivation_analysis.py --quick            # OOS from 2015 only
python scripts/04_motivation_analysis.py --model xgboost    # Single model
python scripts/04_motivation_analysis.py --model elasticnet # Linear benchmark only
```

`--quick` sets `oos_start = 201501` for faster iteration (~120 vs ~300 months).

---

## Implementation Plan

### Step 1: Table 1 (no model training needed)

Load the panel, compute TC at each AUM scenario for each stock-month, assign quintiles, compute implementable shares. This runs in minutes — no ML training required.

### Step 2: Rolling prediction loop (shared across Evidence 2 & 3)

One loop per model. For each test month:
1. Split train/val/test
2. Normalise features
3. Train model (standard, no weights)
4. Predict test month
5. Store: permno, yyyymm, y_true, pred, ret, liq_quintile, liq_BidAskSpread, liq_daily_sigma, liq_dvol_21d

Save predictions to parquet after the loop completes.

### Step 3: Table 2 (post-prediction analysis)

Load predictions, group by quintile, compute R² within each. No additional model training.

### Step 4: Table 3 (post-prediction analysis)

Load predictions, for each quintile form within-quintile decile sorts month by month, track positions for turnover, compute gross and net SR. More involved than Table 2 but still no model training.

### Step 5: Figures

Bar chart for R² and scissors chart for SR. Use matplotlib with Agg backend.

---

## Important Design Decisions

1. **One model trained on full universe, evaluated on quintile subsets.** Do NOT train separate models per quintile. The point is that the single pooled model misallocates capacity.

2. **Standard training only.** No `sample_weight` anywhere. This script shows the problem, not the solution.

3. **ElasticNet is a new model class** not in `src/models/`. Implement as a simple wrapper class within the script itself (no need to add to the model registry). Use `sklearn.linear_model.ElasticNetCV`.

4. **Quintile assignment uses test-month data only.** Each test month's stocks are assigned to quintiles based on that month's `liq_dvol_21d`. No look-ahead.

5. **Within-quintile sorts for SR (Evidence 3).** NOT full-universe sorts decomposed by quintile. This isolates prediction quality within each liquidity level.

6. **Zero benchmark for OOS R² (Evidence 2).** Following GKX: `R² = 1 - SS_pred / SS_zero` where `SS_zero = sum(y_true²)`. Not `sum((y_true - mean)²)`.

7. **TC uses spread_scale = 0.30** (institutional calibration), not 1.0. The TC needs to be in realistic absolute units for both the implementability threshold (Table 1) and net return computation (Table 3).

8. **Implementability threshold τ = 1% one-way** for Table 1. Can add robustness with τ ∈ {0.5%, 1.0%, 2.0%} later.

9. **AUM scenarios for Table 1:** $100M, $500M, $5B. Use n_portfolio = 500 for all.

10. **Primary AUM for Table 3:** $500M (consistent with main experiment).
