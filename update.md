# Project Update — March 2026

## Recent Changes (Completed)

### 1. Target Variable: `ret` to `excess_ret`
- **Config:** `target_col` changed from `"ret"` to `"excess_ret"`
- **Impact:** Models now predict forward 1-month excess returns (ret - RF)
- **Portfolio construction unchanged:** Rankings are identical since RF is constant across stocks each month. Raw `ret` is still used for computing portfolio returns (RF cancels in long-short).

### 2. OOS R-squared Benchmark: Zero (Campbell & Thompson 2008)
- Previously used expanding historical mean as benchmark
- Now uses zero benchmark, appropriate for excess returns (null = no predictability)
- **File changed:** `src/evaluation/two_by_two.py`

### 3. XGBoost min_child_weight: 100 to 10
- Old value (100) produced only ~12 unique predictions for ~5,777 stocks — the model was collapsing to a few buckets
- New value (10) allows finer-grained splits while maintaining regularization
- Search space: `[5, 10, 30, 50]` (was `[50, 100, 200]`)
- **File changed:** `config/config.yaml`

### 4. H1/H3 Primary Test: Net Returns at $500M
- H1 (Training Dominance) and H3 (Sharpe Improvement) now use **net returns at $500M AUM** as the primary test
- Gross returns kept as secondary
- `main_results.csv` includes both `h1_net_pval`/`h3_net_pval` and `h1_gross_pval`/`h3_gross_pval`
- `hypothesis_tests.json` top-level H1/H3 entries use net values, with `gross_results` sub-dict
- **File changed:** `scripts/03_analyze_results.py`

### 5. Neural Network Removed (Pending Fix)
- NN outputs deleted — model was not running correctly
- XGBoost and Random Forest are the two active models
- NN can be re-added once the BatchNorm/single-sample issue is fully resolved

### 6. Quick Mode Updated
- `--quick` flag now runs OOS from 2020-01 (last ~5 years) instead of 2015-01
- **File changed:** `scripts/02_run_experiment.py`

---

## Current State

**All experiment outputs have been cleared.** The pipeline needs to be re-run with the new settings:

```bash
python scripts/02_run_experiment.py --model xgboost --quick   # quick test first
python scripts/02_run_experiment.py --model random_forest --quick
python scripts/03_analyze_results.py                           # analyze both
```

For the full OOS period (2000-2024):
```bash
python scripts/02_run_experiment.py --model xgboost
python scripts/02_run_experiment.py --model random_forest
python scripts/03_analyze_results.py
```

---

## Next Phase: Two Planned Extensions

### Extension 1: Scheme E — TC-Based Training Weights (Replace Scheme A)

**Motivation:** Current Scheme A (softmax on dollar volume rank) is an ad-hoc proxy for implementability. Scheme E uses the full Frazzini et al. (2018) TC model, creating theoretical coherence between training and evaluation.

**Formula:**
```
TC_i_stock = Spread_i / 2 + lambda * sigma_i / sqrt(ADV_i)
w_i = 1 / (1 + kappa * TC_i_stock),  normalized to mean=1
```

**Key properties:**
- Captures all three TC dimensions: spread, volatility, market impact
- Uses data already in the pipeline (`BidAskSpread`, `daily_sigma`, `dvol_21d`)
- Will replace Scheme A as the primary weighting scheme
- Scheme A results become the robustness check

**Implementation:**
- Add `_tc_full()` to `src/weighting/schemes.py` (~20 lines)
- Add to `SCHEME_REGISTRY`
- Add config section under `weighting.tc_full`
- Change `weighting.primary` to `"tc_full"`

### Extension 2: TC-Penalized Portfolio Ranking

**Motivation:** Even with weighted training, illiquid stocks with high predicted returns can still enter the portfolio. TC-penalized ranking directly penalizes costly-to-trade stocks at the portfolio construction stage.

**Formula:**
```
Score_i = predicted_return_i - TC_i_stock
```

**Applied to cells 1B and 2B only** (liquidity-weighted portfolio cells). Cells 1A and 2A keep standard ranking for clean 2x2 contrast.

**Implementation:**
- Add `tc_penalize` parameter to `build_long_short_portfolio()` and `build_portfolio_timeseries()` in `src/portfolio/construction.py`
- Pass `tc_penalize=True` for cells 1B/2B in `src/evaluation/two_by_two.py`
- Add `portfolio.tc_penalize` and `portfolio.tc_penalize_scale` to config

### How These Extensions Relate

The two extensions target different stages of the pipeline:

| Stage | Current | After Extensions |
|-------|---------|-----------------|
| Training weights | Scheme A (dollar volume rank) | **Scheme E (full TC model)** |
| Portfolio ranking | Raw predicted return | **Predicted return minus TC** (cells 1B/2B) |
| Portfolio weighting | Equal or liquidity weights | Unchanged |

Together they create a fully TC-aware pipeline: the model learns from TC-weighted data and portfolios are constructed using TC-adjusted rankings.

---

## Mechanism Understanding (from Analysis)

The mechanism analysis document clarified that weighted training works through **capacity reallocation**, not stock exclusion:

1. ML models have finite learning capacity (tree splits, parameters)
2. Standard training allocates capacity to illiquidity patterns (predictable but untradable)
3. Weighted training redirects capacity toward tradable patterns (momentum, value, quality)
4. This improves ranking quality among liquid stocks, not prediction levels

This explains:
- **Modest effect sizes** — indirect mechanism, not exclusion
- **XGBoost helped, RF not** — XGBoost's sequential boosting concentrates capacity; RF's bagging diversifies naturally
- **H2 as strongest result** — the channel (feature reallocation) is stronger evidence than the outcome (Sharpe improvement)
- **Negative OOS R-squared** — models predict rankings, not levels; zero benchmark is appropriate
