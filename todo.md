# TODO: Daniele Feedback — All Before Thursday April 9, 11am

---

## Task 1: Expand Step 2 Interaction Regression to All ~113 Characteristics

**Est. time:** 2–3 hours  
**Daniele:** "Show the regression results in Step 2 for all characteristics (not just the selected 15)."

### Description

The current Step 2 interaction regression (Eq. 7) runs on 15 `FOCAL_CHARACTERISTICS` only. Expand it to all ~113 features. The core function `interaction_fama_macbeth()` already accepts any feature list — changes are only in the script and output formatting.

Also run the quintile-specific FM regressions (Output 2.1) on the full set. Note: Q5 averages ~743 stocks/month with ~113 regressors — multicollinearity may inflate standard errors. Report but caveat.

### Files to modify

**`scripts/06_step2_heterogeneity.py`**
- Add `--full` CLI flag
- When `--full`: load all ~113 features via `get_motivation_features(load_signaldoc(), panel)`
- Run `interaction_fama_macbeth(panel, all_features, "liq_rank")` → save as `interaction_regression_full.csv`
- Run `quintile_fama_macbeth(panel, all_features, "liq_quintile")` → save as `quintile_fm_coefficients_full_raw.csv`
- Produce a category summary table: # significant γ per broad category → save as `interaction_by_category.csv`
- Update `interaction_meta.json` with full-regression F-test stats

**`src/analysis/motivation.py`**
- No changes needed — functions already accept arbitrary feature lists
- Note: `build_feature_categories()` returns `{"broad": {feature: category, ...}}` for category mapping
- `load_signaldoc()` expects `data/SignalDoc.csv` — verify this file exists on local machine

### New outputs (in `outputs/motivation_raw/step2/dvol/`)

| File | Description |
|------|-------------|
| `interaction_regression_full.csv` | β̄, γ̄, t-stats for all ~113 features |
| `interaction_by_category.csv` | # significant γ per broad category |
| `quintile_fm_coefficients_full_raw.csv` | FM coefficients × 5 quintiles, all features |

### Run command

```bash
python scripts/06_step2_heterogeneity.py --full
```

---

## Task 2: Train 5 Quintile-Specific XGBoost Models

**Est. time:** 1 day coding + 10–20 hours runtime on local Mac  
**Daniele:** "Train five separate models, one per quintile, and compare their within-quintile R² to the pooled model's within-quintile R²."

### Description

For each liquidity quintile q ∈ {1,2,3,4,5}, train a separate XGBoost model using **only** stocks in that quintile. Same rolling-window protocol as the existing pooled model (120-month train, 12-month val, 1-month test, retune every 12 months). Compare within-quintile R² against the pooled model's within-quintile R² (already in `outputs/motivation/step3/dvol/r2_by_quintile.csv`).

If quintile-specific models outperform the pooled model within each quintile, it directly proves pooling across liquidity groups is suboptimal. This could replace the weak Step 3 importance-vs-illiquidity scatter (ρ = 0.10).

### Files to create

**`scripts/10_quintile_specific_models.py`** (new script)
- Load panel, assign NYSE quintiles via `assign_nyse_quintiles()`
- For each quintile (or `--quintile Q` for a single one):
  - Call `rolling_xgboost_predict_quintile()` to train and collect OOS predictions
  - Compute within-quintile pooled R² (zero benchmark)
- Load pooled model's R² from `outputs/motivation/step3/dvol/r2_by_quintile.csv`
- Produce comparison table and grouped bar chart
- Support `--recompute` flag to skip training and regenerate comparison from saved predictions

### Files to modify

**`src/analysis/motivation.py`**
- Add `rolling_xgboost_predict_quintile()` function
- Nearly identical to `rolling_xgboost_predict()` but with added filter per rolling window:
  ```python
  train_df = train_df[train_df[quintile_col] == quintile].copy()
  val_df = val_df[val_df[quintile_col] == quintile].copy()
  test_df = test_df[test_df[quintile_col] == quintile].copy()
  ```
- Quintile assignment (`liq_quintile` column) must already be in the panel before calling this function — computed per-month using NYSE breakpoints, no look-ahead

### New outputs (in `outputs/motivation/step3_quintile/dvol/`)

| File | Description |
|------|-------------|
| `predictions_q{1-5}.parquet` | OOS predictions per quintile model |
| `r2_comparison.csv` | Quintile-specific vs. pooled R², side by side |
| `r2_comparison.png` | Grouped bar chart (two bars per quintile) |
| `meta.json` | Summary statistics |

### Run command

```bash
# All 5 quintiles sequentially
python scripts/10_quintile_specific_models.py

# Single quintile (for parallel runs in separate terminals)
python scripts/10_quintile_specific_models.py --quintile 5
python scripts/10_quintile_specific_models.py --quintile 4
# etc.

# Regenerate comparison from saved predictions
python scripts/10_quintile_specific_models.py --recompute
```

### Runtime estimate

Each quintile model trains on fewer stocks than pooled (~740–2,400 vs ~6,300), so per-month fits are faster. Rough: ~2–4 hours per quintile on local Mac. Run quintiles in parallel across terminals to finish faster.

### Key considerations

- Q5 has ~743 stocks/month → 120 months × ~743 = ~89,000 train observations per window — fine for XGBoost
- Stocks move between quintiles over time — this is correct behavior
- Hyperparameter tuning within quintiles may select different params than pooled — expected and informative
- Use same config search space as pooled model (`config/config.yaml` → `models.xgboost.search_space`)

---

## Task 3: Add Value-Weight Line to `density_comparison.png`

**Est. time:** 1–2 hours  
**Daniele:** "Would it be possible to add value-weights to the density_comparison.png charts?"

### Description

Add a third density line (value-weighted via market cap `liq_me_raw`) to the characteristic-level density plots. Currently shows: (1) training = flat at 1.0, (2) deployment = dollar-volume-weighted KDE. Add: (3) value-weighted = market-cap-weighted KDE.

This lets Daniele see whether value-weighting already closes the gap, or whether dollar-volume weighting captures something beyond size.

### Files to modify

**`src/analysis/motivation.py` — `plot_density_comparison()`**
- Add parameter `vw_col: str | None = None`
- Inside `_plot_one()`: if `vw_col` is provided, compute market-cap-weighted KDE and plot as third line
  ```python
  # After the deployment (vol-wt) KDE plot:
  if vw_col_name is not None and vw_col_name in panel_data.columns:
      valid_vw = valid & panel_data[vw_col_name].notna()
      vals_vw = panel_data.loc[valid_vw, feat].values
      w_vw_raw = panel_data.loc[valid_vw, vw_col_name].values
      w_vw = w_vw_raw / w_vw_raw.mean()
      kde_vw = gaussian_kde(vals_vw, weights=w_vw)
      vw_y = kde_vw(x_grid)
      ax.plot(x_grid, vw_y, ":", color="seagreen", linewidth=2.0,
              label="Value-weighted")
  ```
- Pass `vw_col_name` through `_plot_one()` calls

**`scripts/05_step1_divergence.py`**
- Added `--vw` CLI flag (default off for backward compatibility)
- Around line 330, the call now reads:
  ```python
  plot_density_comparison(
      panel, density_features, "w_tilde", output_dir / "density_comparison.png",
      vw_col="liq_me_raw" if args.vw else None,
  )
  ```

### Output

With `--vw` flag: `density_comparison.png` has 3 lines per subplot:
- Solid blue: Training (equal-weight) — flat at 1.0
- Dashed orange: Deployment (dollar-volume-weighted)
- Dotted green: Value-weighted (market cap)

Without `--vw`: identical to previous output (2 lines only).

### Run command

```bash
# Old behavior (reproduce previous output exactly)
python scripts/05_step1_divergence.py

# New behavior with value-weight overlay (for Daniele)
python scripts/05_step1_divergence.py --vw
```

---

## Task 4: Regime-Conditional Plots

**Est. time:** 3–5 days (but start now, finish after meeting if needed)  
**Daniele:** "Would it be possible to see density_comparison.png and weight_distribution.png in three different regimes?"

### Description

Split `density_comparison.png` and `weight_distribution.png` by three regime indicators:

| Regime | Source | FRED Code | Split |
|--------|--------|-----------|-------|
| NBER recession/expansion | FRED | `USREC` | 1 = recession, 0 = expansion |
| VIX high/low | FRED | `VIXCLS` | Above/below full-sample median (month-end) |
| NFCI tight/loose | FRED | `NFCI` | Above/below zero |

### Data to obtain

Download from FRED as CSV (or use `pandas_datareader`/`fredapi`):
- `USREC`: monthly, binary
- `VIXCLS`: daily, take month-end close
- `NFCI`: weekly, take last observation per month

Merge into `data/regime_indicators.csv` with columns: `yyyymm, vix, nfci, recession`

### Files to create

**`data/regime_indicators.csv`** — monthly regime data

**`scripts/11_regime_analysis.py`** (new script)
- Load panel + regime indicators
- For each regime (3 regimes × 2 states = 6 subsets):
  - Filter panel to matching months
  - Recompute `w_tilde` (implementability weights) within the filtered panel
  - Call `plot_density_comparison()` and `plot_weight_distribution()` on filtered panel
- Optionally: side-by-side panels (e.g., recession left, expansion right)

### Files to modify

**`src/analysis/motivation.py`**
- Add `title_suffix` parameter to `plot_density_comparison()` and `plot_weight_distribution()` for regime labels in plot titles (e.g., " — Recession", " — High VIX")
- No other changes needed — both functions accept a `panel` DataFrame, so filtering before calling is sufficient

### New outputs (in `outputs/motivation/step1_regime/`)

```
outputs/motivation/step1_regime/
├── recession/
│   ├── density_comparison_recession.png
│   ├── density_comparison_expansion.png
│   ├── weight_distribution_recession.png
│   └── weight_distribution_expansion.png
├── vix/
│   ├── density_comparison_high_vix.png
│   ├── density_comparison_low_vix.png
│   ├── weight_distribution_high_vix.png
│   └── weight_distribution_low_vix.png
└── nfci/
    ├── density_comparison_tight.png
    ├── density_comparison_loose.png
    ├── weight_distribution_tight.png
    └── weight_distribution_loose.png
```

### Run command

```bash
python scripts/11_regime_analysis.py
```

---

## Execution Order

| # | Task | What | Est. Time |
|---|------|------|-----------|
| 1 | Task 3 | VW density overlay | 1–2 hrs |
| 2 | Task 1 | Full interaction regression | 2–3 hrs |
| 3 | Task 2 | Quintile-specific models (write script + start running) | 1 day code, 10–20 hrs run |
| 4 | Task 4 | Regime plots (start, may finish after meeting) | 3–5 days |

**Tip for Task 2:** Run quintiles in parallel across separate terminal sessions to cut wall-clock time from ~15 hrs to ~4 hrs.
