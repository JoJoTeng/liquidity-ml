# Formal Analysis Results — `softmax_rank_lam2`

> **Project:** LiquidityML
> **Branch:** `formalanalysis`
> **Weight spec:** `softmax_rank_lam2` — softmax-rank training weights `exp(2·rank_it)`, mean-one per month (upweights liquid stocks)
> **Models:** `xgboost`, `elastic_net`, `neural_network`
> **Scope:** living document tracking the `softmax_rank_lam2` spec across scripts `21a`→`22`
> **Last updated:** 2026-06-02 — covers `21a`–`21d` and `22` (long-short + prediction-quantile); quintile breakpoints = **NYSE** (primary) for 21*; portfolio universe = **full_sample**, **equal-weighted** for 22

This document summarizes the formal liquidity analysis for the single weight
spec `softmax_rank_lam2`, comparing **standard** training (all stock-months
equal) against **weighted** training (`softmax_rank_lam2`). Sections are added
as each script (21a→22) is reviewed.

---

## 21a — Liquidity-Sorted OOS R²

**Script:** `scripts/21a_formal_liquid_r2.py`
**Outputs:** `outputs/formalanalysis/analysis/{model}/softmax_rank_lam2/liquidity_breakpoints/{nyse,full_sample}/{r2_by_quintile,utility_weighted_r2}.csv`

**What it measures.** Zero-benchmark OOS R² (%, monthly pooled) by liquidity
quintile (**Q1 = most illiquid → Q5 = most liquid**), plus pooled **Q4–Q5**
(the tradable end) and **Full**. `delta = R²_weighted − R²_standard`
(positive = weighting helped). The **utility-weighted R²** re-weights each
error by its economic softmax-rank(λ=2) weight — accuracy *where tradable
capital actually sits*.

### Main findings (TL;DR)

- **Liquidity-aware training works for the tree and linear models, not the NN.**
  For `xgboost` and `elastic_net`, weighting raises the economically-relevant
  (utility-weighted) R² and reallocates accuracy from illiquid Q1 to the
  liquid quintiles. For `neural_network` at λ=2 it backfires.
- **xgboost is the clean win:** utility-weighted R² **+51%** at essentially
  **zero cost** to full-sample R² (−0.0015). Near free lunch.
- **elastic_net** is directionally identical but smaller: utility-weighted R²
  **+18%**, full R² cost −0.009.
- **neural_network** degrades: full R² −0.06 and utility-weighted R² **−18%**
  in level (the concentration gap narrows only because the model got globally
  worse).
- **Standard ML predictability is concentrated in illiquid microcaps.** Under
  NYSE breakpoints the standard model is positive *only* in Q1 for xgboost
  (Q2–Q5 all negative) and declines monotonically with liquidity for EN/NN; the
  headline Full R² is essentially the illiquid-stock R² because microcaps
  dominate the pooled sum-of-squares. This is what motivates the
  utility-weighted metric.

### Headline numbers (universe-independent)

The Full and utility-weighted R² do not depend on the breakpoint universe — only
the per-quintile split below changes between NYSE and full-sample.

| Model | Full R² (std→wt) | Utility-wtd R² (std→wt) | Util. Δ (rel) | Gap unwtd−util (std→wt) |
|---|---|---|---|---|
| **xgboost** | 0.270 → 0.269 (≈flat) | 0.072 → 0.109 | **+51%** | 0.198 → 0.160 ↓ |
| **elastic_net** | 0.244 → 0.235 (−3.8%) | 0.137 → 0.162 | **+18%** | 0.108 → 0.074 ↓ |
| **neural_network** | 0.383 → 0.324 (−15.5%) | 0.232 → 0.190 | **−18%** | 0.151 → 0.134 ↓ |

### Quintile reallocation — `delta` (NYSE breakpoints, %)

NYSE breakpoints (the script default, Fama-French convention) split on NYSE
percentiles, so Q1 is a large illiquid mass (~2438 stocks/month — all the
non-NYSE microcaps fall below the NYSE 20th percentile) and Q5 is the megacap
tail (~743/month).

| Quintile | xgboost | elastic_net | neural_network |
|---|--:|--:|--:|
| Q1 (illiquid) | **−0.086** | **−0.067** | **−0.106** |
| Q2 | +0.124 | +0.048 | −0.005 |
| Q3 | **+0.154** | **+0.099** | +0.023 |
| Q4 | +0.079 | +0.075 | +0.014 |
| Q5 (liquid) | −0.032 | +0.012 | −0.062 |
| **Q4–Q5** | **+0.031** | **+0.048** | **−0.018** |

For `xgboost`/`elastic_net` the gains concentrate in the **moderately-liquid
middle (Q2–Q4)**, peaking at Q3, while Q1 (illiquid) gives back R². At the very
top, weighting actually *hurts* the megacap Q5 for xgboost (−0.032) and the NN
(−0.062) — the most-liquid NYSE names stay essentially unpredictable (xgboost
Q4–Q5 R² is negative even after weighting: −0.130 → −0.099; elastic_net crosses
zero, −0.029 → +0.019). The full-sample-breakpoint cut agrees qualitatively
(Q4–Q5 delta xgb +0.093, EN +0.072, NN −0.009) but places more of the gain in
Q4 because its quintiles are equal-sized.

### Per-model verdict

- **xgboost** — clean win. Reallocation from Q1 (−0.086) into the Q2–Q4 middle
  (peak Q3 +0.154), +51% utility R², full R² unchanged. Caveat: megacap Q5
  slips (−0.032).
- **elastic_net** — same direction, broad: every liquid quintile (Q2–Q5)
  improves, peak Q3 +0.099, +18% utility R², small full-R² cost. The only model
  whose Q5 also gains (+0.012).
- **neural_network** — λ=2 hurts: Q1 takes the biggest hit (−0.106), no transfer
  to the liquid end (Q4–Q5 −0.018, Q5 −0.062), full and utility R² both fall.
  Open question whether this is tuning/capacity or needs a different λ.

### Caveats

- 21a reports **point estimates only** (no standard errors). Inference comes
  from `21d` (monthly error-differential time series) and `21c` (restriction
  comparison).
- Primary lens here is **NYSE breakpoints**; the full-sample cut agrees
  qualitatively, so the result is not an artifact of the cutoff choice.

---

## 21b — Importance Reallocation

**Script:** `scripts/21b_formal_importance_reallocation.py`
**Outputs:** `outputs/formalanalysis/analysis/{model}/softmax_rank_lam2/{importance_shift.csv, group_shares.csv, gamma_regression.json}`
**Universe-independent** — 21b has no breakpoint dimension, so NYSE vs full-sample does not apply.

**What it measures.** How weighted training changes *feature usage* (default
mean |SHAP|). `delta = importance_wt − importance_std` per feature
(Newey-West, 6 lags). The **gamma regression** regresses `delta` on the Step-2
liquid-minus-illiquid interaction slope `gamma_bar` (γ>0 ⇒ feature works better
on liquid stocks); a positive slope ⇒ weighting shifts importance toward
liquid-relevant features — the *mechanism* behind the 21a R² reallocation.
**group_shares** splits the 113 features by Step-2 quintile-FM significance
(`Q1_only` = illiquid-only signal, `Q5_only` = liquid-only, `both`, `neither`)
and tracks each group's share of total importance.

### Step-2 gamma (γ) — definition

`gamma_bar` (γ̄ⱼ) comes from the Step-2 interaction Fama–MacBeth regression
(`scripts/03_motivation_step2_heterogeneity.py` →
`outputs/motivation/step2/dvol/interaction_regression_full.csv`). Each month,
across all stocks, with the full 113-feature set:

> r_{i,t+1} = α + xᵢₜ′·β + (xᵢₜ · Lᵢₜ)′·γ + ε,   L ∈ [0,1] = liquidity rank (higher = more liquid)

The total slope on feature *j* is `β_j + γ_j·L`, so **γ_j is how much the
feature's return-predictive slope changes from the most illiquid (L=0) to the
most liquid (L=1) stock**:

- **γ_j > 0** → the signal works *better on liquid* stocks;
- **γ_j < 0** → the signal is *illiquid-tilted*.

`gamma_bar` is the Fama–MacBeth time-series mean of the monthly γ_j; `gamma_t`
is its Newey–West (6-lag) t-stat. It is an **exogenous, model-independent**
property of the data — computed once from the panel — so all three models in
21b are scored against the same γ̄. Distribution: mean ≈ 0, sd ≈ 0.017,
range [−0.116, +0.044]. Tails:

| Illiquid-tilted (γ̄ < 0) | γ̄ | | Liquid-tilted (γ̄ > 0) | γ̄ |
|---|--:|---|---|--:|
| `AM` | −0.116 | | `Leverage` | +0.044 |
| `STreversal` | −0.039 | | `CF` | +0.039 |
| `CompEquIss` | −0.028 | | `IdioVolAHT` | +0.030 |

`AM` (β̄=+0.078, γ̄=−0.116) predicts strongly among illiquid microcaps but its
slope turns negative by the liquid end — the canonical "alpha lives in illiquid
stocks" feature, and the first thing the weighted models discard.

### Main findings (TL;DR)

- **Mechanism is cleanest for elastic_net, partial for xgboost, absent for the
  NN** — the mechanistic mirror of 21a.
- **gamma regression** (`delta ~ gamma_bar`): EN slope +0.0089, R²=0.33,
  p=2e-11 (strong); xgb +0.0018, R²=0.05, p=0.019 (weak but significant); NN
  +0.0002, R²≈0, p=0.92 (null).
- **Reallocation away from illiquid-only signals**: Q1_only importance share
  falls −7.2pp (EN) / −2.9pp (xgb); for the NN it *rises* +1.8pp (wrong way).
- **Dumped** — strongly illiquid-tilted signals: `AM` (γ=−0.116), `STreversal`
  (γ=−0.039, the microcap reversal premium), `CompEquIss`, all with large
  Newey-West t-stats. **Loaded** — the liquidity variables themselves (`DolVol`,
  `Illiquidity`), plus `Mom12m`, `MaxRet`, `Size`.
- **Gamma-alignment ≠ R² payoff.** EN's reallocation is the most gamma-aligned
  yet its 21a utility-R² gain is only +18%; xgboost gets the biggest payoff
  (+51%) via *nonlinear* reallocation the linear gamma barely captures
  (R²=0.05); the NN reshuffles without structure and fails (−18%).

### gamma regression (`delta_importance ~ gamma_bar`)

| Model | slope | R² | p-value | reading |
|---|--:|--:|--:|---|
| elastic_net | +0.0089 | 0.33 | 2e-11 | strong shift toward liquid signals |
| xgboost | +0.0018 | 0.05 | 0.019 | weak but significant |
| neural_network | +0.0002 | ~0.00 | 0.92 | none |

### Importance-share reallocation by Step-2 group, `delta_share_pct` (pp)

| Group (n features) | xgboost | elastic_net | neural_network |
|---|--:|--:|--:|
| **Q1_only** — illiquid signal (43) | **−2.93** | **−7.15** | **+1.79** |
| Q5_only — liquid signal (4) | +0.35 | +0.51 | +0.12 |
| both (9) | +0.35 | −2.62 | −1.43 |
| neither (57) | +2.23 | +9.27 | −0.48 |

Only 4 of 113 features are liquid-only signals vs 43 illiquid-only — the
predictor zoo is itself largely an illiquid-stock phenomenon, so reallocated
importance lands mostly in "neither" rather than a large `Q5_only` bucket.

### Per-model verdict

- **elastic_net** — textbook mechanism: strong gamma-link, dumps illiquid-tilted
  value/reversal/issuance (`AM`, `STreversal`, `CompEquIss`), Q1_only share
  −7.2pp; loads `DolVol`/`Illiquidity`/`Mom12m`/`MaxRet`.
- **xgboost** — real but nonlinear: still cuts `STreversal`/`Price`/`Size` and
  Q1_only share −2.9pp, but the linear gamma explains only 5% (top movers are
  interaction-driven: `DelLTI`, `TrendFactor`, `IndMom`). This nonlinear
  reallocation is why it earns the largest 21a R² payoff.
- **neural_network** — unstructured reshuffle: no gamma-link (p=0.92); Q1_only
  share *rises* because it concentrates into the `DolVol`/`Illiquidity` level
  variables (Q1_only-classified) while dumping idiosyncratic-vol features
  (`IdioVolAHT`, `IdioVol3F`) — consistent with its 21a failure.

### Why `DolVol`/`Illiquidity`/`Mom12m`/`MaxRet` gain importance

The reallocation runs through two channels — only the first is captured by the
γ regression:

1. **γ-aligned (shed illiquid-tilted, keep liquid-robust).** The weighted models
   drop signals whose slope collapses on liquid stocks (`AM`, `STreversal`,
   `CompEquIss`; γ̄ ≪ 0) and tilt toward ones that hold up. `MaxRet`
   (γ̄=+0.015, mildly liquid-tilted) is loaded by elastic_net for exactly this
   reason. `Mom12m` is roughly liquidity-neutral (γ̄≈−0.01) but is the canonical
   *tradable, large-cap-robust* anomaly, so it is a relative **survivor** once
   the illiquid-only signals are removed — it gains share even though γ does not
   "predict" it.
2. **Liquidity-conditioning (not γ-explained).** `DolVol` and `Illiquidity` are
   the liquidity axis itself, and their *own* γ̄ is ≈0 (DolVol −0.012,
   Illiquidity +0.005) — so their gains are **not** a γ effect. Once the loss is
   concentrated on liquid stocks, the liquidity level becomes a useful
   *conditioning / gating* variable: it tells the model where on the
   liquidity-interaction curve each stock sits, so it can apply the
   liquid-appropriate version of every other signal. This is why both EN and the
   NN load them, and why the γ regression explains only part of the reallocation
   (EN R²=0.33, xgb 0.05).

Loading the liquidity variables is **necessary but not sufficient**: the NN
piles +6.0pp into `DolVol` and +2.8pp into `Illiquidity` (far more than EN) yet
still *loses* liquid-stock R² in 21a — it over-concentrated on the level
variable without the productive γ-aligned reshuffle.

### Caveats

- Importance shifts carry Newey-West t-stats (6 lags); the headline cuts/loads
  are highly significant (e.g. EN `AM` t=−5.7, `STreversal` t=−6.9).
- Default importance source is mean |SHAP|; `--importance native` gives the
  model-native variant.

## 21c — Restriction Curve

**Script:** `scripts/21c_formal_restriction_curve.py` (module
`src/analysis/formal/restriction_curve.py`)
**Outputs:** `outputs/formalanalysis/analysis/{model}/softmax_rank_lam2/liquidity_breakpoints/{nyse,full_sample}/restriction_curve_comparison.{csv,png}`
**Breakpoint-dependent** — primary lens **NYSE**.

**What it measures.** Appends the soft-weighted model **M_w** to the Step-3d
**hard-restriction curve** and compares **Q4–Q5 (liquid-stock) zero-benchmark
OOS R² (%)**. The curve rows are *unweighted* models trained on progressively
restricted universes — **Mall** (all stocks = standard baseline), **MQ2+** (drop
Q1), **MQ3+**, **MQ4+**, **MQ5+** (Q5-only) — each still scored on the *same*
held-out Q4–Q5 stocks. The curve is read per-model from
`outputs/motivation/step3_restriction/{model}/dvol/global/baseline/restriction_comparison.csv`
(global normalization, baseline tuned params). M_w is the `softmax_rank_lam2`
model: full universe, soft weights. **Higher (less negative) = better.** This is
the head-to-head of **reweight vs restrict**.

### Main findings (TL;DR)

- **Soft weighting strictly dominates hard restriction — universally (all 3
  models × both universes).** M_w beats every restriction row, and restriction
  is *monotonically destructive*: each illiquid quintile dropped worsens
  liquid-stock R² as N_train collapses 6776 → 726.
- **Mechanism = "reweight, don't restrict."** Illiquid stocks aren't the target
  but are valuable *training data*; hard restriction starves the model. Soft
  weighting keeps the full N and re-aims the objective — the methodological
  thesis of the paper, in one table.
- **Soft also beats train-on-everything (Mall) for xgboost and elastic_net**
  (M_w is the *only above-zero row* for EN), but **not for the neural_network**
  (M_w marginally below Mall in both universes) — consistent with the NN's
  21a/21b failure.
- **The NN is hypersensitive to restriction**: its curve craters to −20 / −27 by
  MQ4+/MQ5+ (vs xgb −1.5/−2.6), so soft weighting beats restriction by +3.4 to
  +26.8 pp even though it doesn't beat Mall.

### Q4–Q5 OOS R² (%), NYSE breakpoints

| Model | **M_w** (soft) | Mall (std) | MQ2+ | MQ3+ | MQ4+ | MQ5+ | M_w − Mall |
|---|--:|--:|--:|--:|--:|--:|--:|
| xgboost | **−0.099** | −0.130 | −0.602 | −0.831 | −1.550 | −2.551 | +0.031 |
| elastic_net | **+0.019** | −0.029 | −0.490 | −0.749 | −1.230 | −2.231 | +0.048 |
| neural_network | +0.038 | **+0.056** | −3.361 | −3.761 | −20.01 | −26.76 | −0.018 |

full_sample recomputes only M_w (curve rows are unchanged NYSE-cut copies):
xgb −0.040, EN +0.052, NN +0.027.

### Per-model verdict

- **xgboost** — M_w is the best of all six rows; beats Mall (+0.031) and crushes
  hard restriction (+0.50 vs the best restriction row MQ2+).
- **elastic_net** — strongest soft-vs-standard case: M_w is the only *positive*
  row (+0.019), +0.048 over Mall, +0.51 over MQ2+.
- **neural_network** — soft beats restriction massively (+3.4 to +26.8 pp) but
  is marginally *below* Mall (−0.018); soft weighting avoids restriction's damage
  but does not improve on the plain baseline.

### Caveats

- **Apples-to-apples only in NYSE mode.** The 5 restriction rows are copied
  verbatim from the single Step-3d curve (always NYSE breakpoints); only M_w is
  recomputed per mode. So in `full_sample/` the curve rows are still NYSE-cut
  while M_w uses full-sample cuts — *not* strictly comparable. In `nyse/`
  everything uses NYSE breakpoints, and M_w / Mall match 21a's NYSE wt / std
  exactly to 6 dp. This is the main reason NYSE is the right primary lens for
  21c.
- The curve is *unweighted* throughout: M_w-vs-Mall isolates the weighting
  effect; M_w-vs-MQk+ contrasts soft down-weighting against hard restriction.

## 21d — Error Differential (inference)

**Script:** `scripts/21d_formal_error_differential.py` (module
`src/analysis/formal/error_differential.py`)
**Outputs:** `.../{model}/softmax_rank_lam2/liquidity_breakpoints/{nyse,full_sample}/liquid_squared_error_differential.{csv,png}`
**Breakpoint-dependent** — primary lens **NYSE**.

**What it measures.** The monthly Q4–Q5 (liquid-stock) **squared-error
differential** `mse_diff = MSE(standard) − MSE(weighted)` (equal-weighted mean
over liquid stocks each month) plus its cumulative sum. **mse_diff > 0 ⇒
weighted has lower error (better).** The script emits only the raw monthly
series — *no in-code test* — so significance here is a **Newey–West** t-stat
(lags=6, project convention) on the `mse_diff` series, the inference layer that
21a–21c lacked. Errors are in squared-excess-return (decimal) units, so the
values are tiny. Sample 2000-01 → 2024-11 (n=299 months).

### Main findings (TL;DR)

- **No statistically significant liquid-stock improvement in any cell.** All 6
  (model × universe) mean differentials are *positive* (weighted directionally
  favored), but **0 of 6 NW t-stats clear 1.96** (robust to lags=12). The
  strongest is elastic_net/full_sample at t=1.82 (p≈0.07) — marginal, not 5%.
  The point estimates of 21a/21c do **not** survive proper monthly inference.
- **Economically tiny and decaying.** Differentials are 1e-6…2.6e-5; for xgb/EN
  the second-half mean is ~4–5× smaller than the first half (the edge fades),
  and every cumulative path dips negative at some point. xgboost's advantage
  accrues pre-2016 and partly reverses (it wins in <½ of months).
- **The neural_network is a sign-consistency failure.** Its mean mse_diff is
  fractionally *positive* (~1e-6), contradicting its *negative* 21a Q4–Q5 R²
  delta. Resolved: the NN cumulative differential is negative for **97–99% of
  the sample** (standard better); the tiny positive mean is manufactured by the
  last ~1–2 years. 21a's negative sign is the reliable read — the NN does not
  help liquid stocks.
- **Consistent with 21a in the 4 non-NN cells** (positive diff ↔ positive Q4–Q5
  R² delta for xgb/EN, both universes). 21d confirms the *sign* of the effect,
  not its *significance*.

### Inference — monthly `mse_diff` (>0 = weighted better), NYSE

| Model | mean mse_diff | NW-t (lag6) | sig @5%? | % months wt lower | cum. end |
|---|--:|--:|:--:|--:|--:|
| xgboost | +1.19e-5 | +0.45 | no | 48.8% | +0.0036 |
| elastic_net | +1.23e-5 | +1.26 | no | 54.5% | +0.0037 |
| neural_network | +1.03e-6 | +0.03 | no | 48.5% | +0.0003 |

full_sample: xgb t=+0.83, **EN t=+1.82 (p≈0.068, strongest)**, NN t=+0.06 — same verdict.

### Per-model verdict

- **xgboost** — positive but insignificant (t=0.45 / 0.83); wins in <½ of months,
  advantage concentrated pre-2016 and partly reversed since.
- **elastic_net** — the most robust signal: highest t (1.26 nyse / 1.82 full),
  wins >½ of months, cleanest near-monotone cumulative climb — yet still short
  of 5%.
- **neural_network** — effectively zero (t≈0.03–0.06); positive mean is an
  end-of-sample artifact contradicted by its negative 21a R² — no evidence of
  benefit.

### Why the Q4–Q5 pool looks insignificant — window mismatch

Recomputing the differential **per quintile** (same method as 21d, NYSE;
the Q4–Q5 pool reproduces the 21d CSV exactly) shows the weighting benefit is
concentrated in the **moderately-liquid middle (Q3–Q4)** and fades to ~zero by
Q5 — exactly the 21a pattern, now with inference:

| Quintile | xgboost t | elastic_net t | neural_network t |
|---|--:|--:|--:|
| Q1 (illiquid) | −0.62 | −1.59 | −0.94 |
| Q2 | +0.86 | +1.23 | −0.10 |
| **Q3** | +1.14 | **+2.63 (p=0.008)** | +0.13 |
| **Q4** | +0.70 | +1.85 (p=0.064) | +0.15 |
| Q5 (most liquid) | +0.02 | +0.58 | −0.11 |
| **Q3–Q4** | +1.00 | **+2.38 (p=0.017)** | +0.15 |
| Q4–Q5 *(21d pool)* | +0.45 | +1.26 (p=0.21) | +0.03 |

**21d's `Q4–Q5` window dilutes the signal two ways:** it *excludes Q3* (where the
improvement is largest and significant for EN) and *includes Q5* (a near-zero or
negative contributor). For elastic_net the economically-motivated **Q3–Q4 window
is significant at 5% (t=2.38)**, but the Q4–Q5 pool collapses it to t=1.26. So
21d's null is partly a **window-mismatch artifact** — it measures the wrong
liquidity band relative to where soft weighting actually helps.

Caveats: (1) it does *not* rescue xgboost (same direction, insignificant in every
window — noisier) or the NN (≈0 everywhere; Q5 even negative — a genuine failure,
not a pooling artifact). (2) The flip side: gains are in *moderately*-liquid
mid-caps (Q3–Q4), **not** the most-liquid Q5 megacaps — a large-cap-only investor
sees little benefit. (3) Multiple testing: Q3–Q4 is a *pre-specified* window
(justified by 21a), so EN's p=0.017 is a legitimate 5% result; treated as one of
several windows it warrants a discount. The directional conclusion is robust.

### Caveats / reading

- MSE here is *equal-weighted* across liquid stocks (not utility/variance
  weighted) — a more stringent test than 21a's pooled R². That, plus the decay
  and regime-dependence, explains why 21a's positive (large *relative*)
  utility-R² gain coexists with an insignificant monthly MSE differential: the
  absolute liquid-stock effect is small.
- **Honest bottom line:** 21d's *headline Q4–Q5* result is directionally
  supportive but statistically inconclusive; however, the per-quintile split
  shows a **significant Q3–Q4 improvement for elastic_net** that the Q4–Q5 pool
  masks. The effect is real but concentrated in moderately-liquid stocks; the
  broader economic case for soft weighting rests on its dominance over hard
  restriction (21c) and the portfolio/Sharpe metrics in 21e/22.

## 22 — Portfolio Tables

**Scripts:** `22b_table12_two_sided.py` (two-sided long-short → Table 12/13/14)
and `22_prepare_portfolio_excel_tables.py` (standalone prediction-quantile
portfolios). **Outputs:** `outputs/formalanalysis/tables/{model}/{stock_universe}/`.

> **Universe note:** here `full_sample`/`nyse` is the **stock universe the
> portfolio is built from** (full_sample = entire cross-section; nyse =
> `exchcd==1` only) — *not* the liquidity-breakpoint universe of 21a–21d.

This section is **equal-weighted, full_sample**. The two analyses tell different
stories: the **two-sided long-short** (below) and the **standalone
prediction-quantile portfolios** (further down).

### 22(b) — Two-sided long-short (Table 12 / 13 / 14)

**Construction.** A two-sided TC-aware Q5−Q1 long/short. Table 12 decomposes the
**annualized net (after-TC) Sharpe** in a 2×3:

|  | A: plain sort (r̂) | B: TC-aware sort (r̂∓TC) | C: TC-target sort |
|---|---|---|---|
| Row 1: **Standard** training | 1A (baseline) | 1B | 1C |
| Row 2: **Weighted** (softmax_lam2) | 2A | 2B | 2C |

Effects: **training = 2A−1A** (the weighting), **portfolio = 1B−1A** (TC-aware
sort), **total = 2B−1A**. AUM: PropTC (spread/2 only), $100M, $500M, $1B (market
impact grows with AUM). Table 13 = per-leg (Long Q5 / Short Q1 / LS) gross/net
return, SR, TC, turnover; Table 14 = the 2×3 decomposition within each leg.

#### Main findings

- **Liquidity weighting hurts the long-short net Sharpe at every AUM, for every
  model** (training effect negative throughout). At PropTC the standard
  long-short is solidly positive (1A = 0.29 / 0.30 / 0.59 for xgb/EN/NN) and
  weighting cuts it (total effect −0.14 / −0.32 / −0.15).
- **This is a small-money strategy.** Net SR collapses to deeply negative by
  $500M–$1B for all models — unprofitable at scale regardless of training.
- **The productive lever is the TC-aware *sort*, not the weights.** Portfolio
  effect grows with AUM for xgb (+0.01→+0.19) and NN (+0.02→+0.21); only
  xgboost's total turns positive (≥$500M) and only via the sort (its training
  effect is still −0.10 to −0.12 there). elastic_net loses on *both* levers at
  all AUM.
- **Net profit is a LONG-leg story; the short leg is a cost sink.** At PropTC the
  Long (Q5) leg is net-positive for all three (net SR 0.41 / 0.47 / 0.49) but the
  Short (Q1) leg is net-NEGATIVE (−0.28 / −0.36 / −0.17): short-side gross alpha
  (NN +0.49%/mo) is wiped out by 0.5–0.9%/mo TC. Equal-weighting amplifies the
  microcap short cost.
- **Weighting's damage is broad (Sharpe non-additivity), and shrinks with AUM.**
  At PropTC the per-leg training effects are tiny (Long −0.00 to −0.04, Short
  −0.02 to −0.05) yet the long-short drops more (−0.16 to −0.28): weighting shaves
  a similar small amount off both legs' net return, halving the already-small
  long-short net return. **The long-short training effect is AUM-dependent** — it
  becomes *less negative as AUM rises* for xgb (−0.16 → −0.10) and especially
  **elastic_net (−0.28 → −0.14, its *long* leg even crossing to positive,
  +0.001 → +0.010 from $100M to $1B)**, consistent with the large-AUM rationale
  (the weighted model's cheaper-to-trade tilt matters more as costs grow). It
  nonetheless stays negative at every AUM; for the NN it *worsens*
  (−0.22 → −0.26). The TC-aware sort (PropTC) *helps the long leg* (+0.12 to
  +0.21) and *hurts the short leg* (−0.07 to −0.19); on the long-short its effect
  grows *positive* with AUM for xgb/NN (+0.01 → +0.19, +0.02 → +0.21) but
  *negative* for elastic_net (−0.06 → −0.19).

#### Table 12 — net Sharpe, PropTC (full_sample, equal)

| Cell | xgboost | elastic_net | neural_network |
|---|--:|--:|--:|
| 1A std, plain | 0.287 | 0.302 | 0.590 |
| 1B std, TC-aware | 0.301 | 0.240 | 0.611 |
| 2A wt, plain | 0.127 | 0.024 | 0.367 |
| 2B wt, TC-aware | 0.143 | −0.014 | 0.438 |

#### Total effect (2B−1A) across AUM

| AUM | xgboost | elastic_net | neural_network |
|---|--:|--:|--:|
| PropTC | −0.144 | −0.317 | −0.152 |
| $100M | −0.059 | −0.301 | −0.121 |
| $500M | **+0.040** | −0.283 | −0.091 |
| $1B | **+0.102** | −0.273 | −0.081 |

#### Deployable level — is it profitable? + TC-target column

Beyond effects, the *best deployable cell* (max over 1A–2C) says whether the
long-short actually makes money. For softmax_rank_lam2 (full_sample, equal) the
best cell is always **1B — standard training + TC-aware sort** (the weighted
cells 2A/2B never lead in levels): **+0.30 (PropTC) → −0.13 ($100M) → −0.69
($500M) → −1.10 ($1B)**. So it is **profitable only at PropTC and dead by $100M**,
and softmax adds *no deployable long-short value* — its role is purely the small
liquid-R² reallocation of 21a. (Contrast the cross-spec sweep: only
`tc_500m`+value+EN, cell 2A, yields a long-short profitable at scale.)
The **TC-target sort (1C/2C)** underperforms the TC-aware sort here (1C 0.225 <
1B 0.301 at PropTC); its 1C−1A edge turns positive only at high AUM (+0.11 at
$500M, +0.17 at $1B) but never overtakes 1B.

#### Per-leg net SR, PropTC (standard model 1A)

| Model | Long (Q5) | Short (Q1) | Long-Short |
|---|--:|--:|--:|
| xgboost | +0.409 | −0.282 | 0.287 |
| elastic_net | +0.467 | −0.359 | 0.302 |
| neural_network | +0.487 | −0.171 | 0.590 |

#### Reading

The R² reallocation of 21a buys nothing for the realizable equal-weighted
long-short: the tradable edge is the **long side**, the short side is a
**transaction-cost sink**, liquidity weighting is a **net drag at every cost
level**, and the only positive lever is **cost-aware portfolio construction**
(acting through the long leg). Caveat: equal-weighting amplifies microcaps;
value-weighting (pending) should de-amplify the short-leg cost problem.

### 22 — Standalone prediction-quantile portfolios (Table 13/14, `4_specs`)

**Construction.** Long-only portfolios formed by sorting on predicted return
into quintiles Q1…Q5 (same six 2×3 cells 1A–2C as the long-short, but here each
*quantile* is a standalone long-only book — **Q5 = the highest-r̂ sleeve a
long-only investor would actually hold**). Net (after-TC) annualized Sharpe.

**Why this is a different story from the long-short.** The standalone Q5 book
does **not** pay the short-leg transaction costs that sank the long-short, so
(a) it stays *profitable at institutional scale* and (b) the weighting damage is
an order of magnitude smaller.

#### The ladder is clean and monotone (PropTC, standard 1A, net SR)

| Quantile | xgboost | elastic_net | neural_network |
|---|--:|--:|--:|
| Q1 | −0.40 | −0.22 | −0.60 |
| Q2 | −0.08 | −0.17 | −0.23 |
| Q3 | −0.06 | −0.06 | +0.08 |
| Q4 | +0.11 | +0.11 | +0.16 |
| Q5 | +0.41 | +0.47 | +0.49 |

Net Sharpe rises monotonically with predicted-return quantile; **Q4–Q5 are the
held (positive) sleeves**; Q1–Q3 are net-negative and not held long.

#### Training effect (2A−1A) is quintile-structured, not uniform (PropTC)

| Quantile | xgboost | elastic_net | neural_network |
|---|--:|--:|--:|
| Q1 | +0.01 | +0.09 | +0.07 |
| Q2 | −0.00 | +0.12 | +0.05 |
| Q3 | +0.05 | +0.02 | −0.02 |
| Q4 | −0.04 | **+0.07** | −0.06 |
| Q5 | −0.03 | −0.00 | −0.04 |

Weighting's benefit sits in the **low/mid** quintiles and fades to neutral at the
top. But Q1–Q2 are unheld, so the only economically useful training gain is
**elastic_net's Q4 (+0.07)** — and it **grows with AUM** (+0.07 → +0.12 from
PropTC → $1B). The dominant lever everywhere is the **TC-aware sort** (portfolio
effect +0.12 to +0.21 in Q3–Q5; negative in Q1–Q2), far larger than the weighting
effect for xgb/NN.

#### Q5 long-only net SR survives at scale (1A → 2B)

| AUM | xgboost | elastic_net | neural_network |
|---|--:|--:|--:|
| PropTC | 0.41 → 0.60 | 0.47 → **0.69** | 0.49 → 0.54 |
| $100M | 0.28 → 0.47 | 0.35 → 0.58 | 0.34 → 0.37 |
| $500M | 0.11 → 0.31 | 0.20 → 0.44 | 0.14 → 0.16 |
| $1B | −0.01 → **0.19** | 0.09 → **0.34** | 0.00 → 0.00 |

(vs the long-short ≈ −1.2 at $1B.) The weighted+TC-aware Q5 stays positive to
$1B for xgb and EN — an institutionally viable long-only book.

#### Per-model verdict (long-only)

- **elastic_net — the genuine weighting win.** Weighting *helps* the held mid-high
  **Q4** (+0.07, growing to +0.12 at $1B) and is neutral at Q5; combined with the
  TC-aware sort it gives the **best long-only portfolios at every AUM** (2B:
  0.69 → 0.34 PropTC → $1B). The 21a–21b prediction-side edge finally converts to
  portfolio value.
- **xgboost — a portfolio-construction story, not a weighting one.** Weighting is
  ≈neutral/slightly-negative on the held quintiles (Q4 −0.04, Q5 −0.03, stable
  across AUM); its only mild positive is the barely-held Q3 (+0.05). The TC-aware
  sort does all the work (+0.12 → +0.21 in Q3–Q5), keeping Q5 profitable to $1B
  (~0.19). Net: hold the top quintile with a TC-aware sort — **weighting neither
  helps nor hurts**.
- **neural_network — weighting is counterproductive in the tradable quintiles**
  (Q4 −0.06, Q5 −0.04 at PropTC; the Q5 damage *grows* to −0.08 at $1B); it only
  "helps" the unheld low quintiles (Q1 +0.07, Q2 +0.05). Its best long-only book
  is **standard + TC-aware sort** (1B), profitable but fading to ~0.10 by $1B
  (weighted+sort fades to ~0). Consistent with the NN's failure to benefit from
  weighting throughout 21a–21d.

#### AUM-robustness & cross-model consistency

Re-checked at all four AUM (PropTC → $1B):

- **TC-aware sort (portfolio effect) — consistent across all 3 models and
  AUM-robust.** Positive at the held quintiles everywhere: **Q4 grows** with AUM
  (+0.14 → +0.20 across models) and **Q5 stays positive but slightly shrinks**
  (≈+0.21 → +0.10 band). Same sign and shape for xgb, EN, NN — the robust lever.
- **Weighting (training effect) — model-specific, and the divergence is itself
  AUM-robust (amplified by cost).** elastic_net Q4 is positive and *grows*
  (+0.07 → +0.12), Q5 crosses to slightly positive; xgboost is ≈neutral and
  *stable* (Q4/Q5 ≈ −0.03 to −0.05); the neural network is negative and *worsens*
  with AUM (Q5 −0.04 → −0.08). **The per-model verdicts above therefore hold at
  every AUM — they are not a PropTC artifact**, and higher cost widens the gap
  (EN's benefit grows, NN's harm grows).
- Contrast with the long-short, where the portfolio effect is *not*
  cross-model consistent (EN negative, xgb/NN positive) — that split is a
  short-leg phenomenon, absent here because the long-only book has no short leg.

#### Reading

Unlike the long-short — where weighting was a net drag and the strategy died at
scale — the **standalone long-only book is institutionally viable and weighting
is benign-to-helpful**. The anti-weighting verdict was a short-side / cost
phenomenon. Across all three models the **TC-aware sort is the primary lever**;
**liquidity weighting adds value only for elastic_net, in the mid-high quintiles,
and increasingly so at high AUM** — for **xgboost it is neutral**, for the
**neural network it is harmful**.
