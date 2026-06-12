# Evaluation Realignment — Audit: Portfolio Evaluation vs. Training Objective

> **Project:** LiquidityML
> **Branch:** `eval-realignment`
> **Created:** 2026-06-06
> **Status:** audit (no code changed)
> **Source note:** `LiquidityML_Portfolio_Issue.pdf` — "Aligning the Portfolio
> Evaluation with the Training Objective" (June 6, 2026), §1–§6.
> **Code anchor:** repo at commit `d07257e`; `file:line` references are as of that commit.

This document records the audit of the current evaluation pipeline against the
source note. It is the reference for the planned evaluation redesign. Section
numbers (§1–§6) refer to the note.

## 1. Section summaries

### §1 — The problem
The headline result scores models with a two-sided Q5−Q1 long–short sorted on
`r̂` over the *full* cross-section, equal/value legs, net of TC — and the training
effect (Δ net Sharpe, standard→weighted) is negative almost everywhere. That
metric does not measure what weighted training optimizes: importance weighting
minimizes the **DolVol-weighted prediction error on the deployment distribution**
(Eq. 1, Shimodaira 2000) — accuracy where tradable capital sits. The long–short
instead rewards the *extreme tails* of the `r̂` ranking in a dollar-neutral book
whose leg weights do not match `w̃`, over a universe still full of the illiquid
names the weighting discounts. There is no mapping from "lower deployment-weighted
error" to "higher Q5−Q1 Sharpe," and the tails are where the two objectives
diverge — so a genuine gain need not appear and can reverse.

### §2 — Evidence it is the metric, not the method
1. The training effect improves monotonically as the eval universe is restricted
   toward the liquid set, flipping positive on NYSE-only for cost weights
   (EW TC: −0.49 full → +0.20 NYSE proportional-cost → +0.39 at $1B).
2. Leg-level effects are tiny (long −0.01…−0.07, short −0.03…+0.16); the dramatic
   long–short figures are mostly Sharpe non-additivity from differencing two thin legs.
3. The long leg is essentially unaffected and survives at scale; both the large
   negative (raw dolvol) and positive (cost) effects concentrate in the **short
   leg** — the illiquid Q1 basket the paper itself calls untradeable.

The headline sign is thus largely a function of which universe it is scored on.

### §3 — Consistency principle
Use the *same* implementability measure at both stages: if `w̃ ∝ DolVol` defines
the importance weights, the same measure should define (i) the evaluation universe
and (ii) the portfolio weights. The current design violates this on all three
counts (trains on DolVol weights; evaluates an equal-/ME-weighted, full-universe,
tail-sorted long–short).

### §4 — Proposed evaluation variations (5)
Ordered closest-to-loss → most familiar:
1. **Deployment-weighted prediction metrics (primary)** — DolVol-weighted OOS R²
   (Eq. 2) + monthly error differential on the liquid universe under the same `w̃`.
2. **Signal-weighted capacity portfolio (preferred economic test)** —
   `θ_it ∝ DolVol_it · (r̂_it − r̄ᵂ_t)`, centered on the DolVol-weighted
   cross-sectional mean so the book is dollar-neutral *by construction*; no
   quantile cut, no microcap equal-weighting, no long–short non-additivity; report
   gross/net Sharpe + certainty-equivalent across the AUM grid.
3. **Capacity-weighted long-only book** from the liquid universe (drop the illiquid
   short leg).
4. **Liquid-universe sorts (for comparability)** — quantiles formed *within* the
   liquid/AUM-feasible universe, legs weighted by DolVol rather than equal/ME.
5. **Implementable-utility metric** — feed weighted predictions into a Jensen et al.
   (2024) mean–variance optimizer, report net certainty-equivalent.

Run (1)+(2) first; add (3)+(5) for the economic story and positioning.

### §5 — Code checks
> Note: §5 contains **three** numbered checks, not four. Check 1 bundles two
> sub-screens (the top-60% DV screen and the 5%-per-name cap).

1. **Traded-universe / implementability screen** — does the code restrict to the
   deployable set before sorting, or sort & short Q1 microcaps over the full
   cross-section? The plan specified a top-60%-DolVol screen and a 5%-per-name cap;
   the note suspects neither is implemented.
2. **TC-sort scaling** — in `r̂ ∓ c`, `c = ½·Spread` is a *per-trade* one-way cost
   while `r̂` is a *one-month* expected return; if `|c| ≈ |r̂|` the "cost-aware sort"
   is effectively a liquidity sort (symptom: it worsens the short leg's net Sharpe).
   Likely fix: scale `c` to a per-month drag (≈ `c × turnover`).
3. **Weight-drift timing** — the write-up drifts `t−1` weights by `r_{i,t-1}`, but
   with `r_it` the return over `[t−1,t]` it should drift by `r_it` (off-by-one if
   the code mirrors the write-up); drift should compound gross, not excess, returns.

### §6 — Inference & expectations
Sharpe/CE differences need proper inference (Ledoit–Wolf or block bootstrap);
monthly differentials need Newey–West t-statistics — not point estimates. The
liquid-stock monthly error differential is modest and not uniformly significant,
so a corrected evaluation should show a **real but moderate** effect, not a large
one. The aim is to stop the metric from converting a genuine prediction-level gain
into a portfolio-level loss by loading on the part of the universe the method discards.

## 2. Note → code mapping (§3–§5, verified)

| Note item | § | Repo location (file:function) | Found? | Note |
|---|---|---|---|---|
| Training weights `w̃ ∝ DolVol`, mean-one/month | §1,§3 | `src/weighting/schemes.py:135` `_dolvol_weights`; `:320` `compute_weights` | **yes** | `DolVol/mean(DolVol)`, mean-1 per month; entry `compute_weights(scheme='dolvol')` |
| Evaluation universe — quintile assignment | §3,§5.1 | `src/analysis/formal/common.py:162` `assign_liquidity_quintiles` | **yes** | NYSE (default) / full_sample breakpoints; Q1 illiquid…Q5 liquid |
| Evaluation universe — portfolio panel filter | §3,§5.1 | `scripts/21e_formal_portfolio_decomposition.py:164` `_filter_panel_for_stock_universe` | **partial** | only `full_sample` (default) or `nyse` (exchcd==1); **no liquidity/AUM screen** |
| Portfolio leg weights (equal / value) | §3 | `src/portfolio/construction.py:135` `_selected_leg_weights` | **yes** | equal=1/N; value=`liq_me_raw` |
| Consistency: DolVol-weighted portfolio legs | §3 | `src/portfolio/construction.py:58` `PORTFOLIO_WEIGHTINGS` | **no** | `{"equal","value"}` only — training is DolVol → core mismatch |
| Top-60%-dollar-volume screen | §5.1 | `src/portfolio/construction.py:162` `build_long_short_portfolio` | **no** | no pre-sort screen ("No explicit liquidity filter") |
| 5%-per-name position cap | §5.1 | `src/portfolio/construction.py:135` `_selected_leg_weights` | **no** | no cap; asserted in `tests/test_portfolio.py` |
| TC sort score `r̂ ∓ c`, `c=½Spread` | §5.2 | `src/portfolio/construction.py:223`; `src/weighting/schemes.py:411` `compute_tc_for_sorting` | **yes** | long `r̂−spread/2`, short `r̂+spread/2` |
| TC-sort scaling = per-month drag (`c×turnover`) | §5.2 | `src/weighting/schemes.py:411` `compute_tc_for_sorting` | **no** | uses raw per-trade `½Spread`, not `c×turnover` (currently by design; impact deferred to realized net returns) |
| Weight-drift timing (`r_it` vs `r_{i,t-1}`) | §5.3 | `src/portfolio/construction.py:806` `_drift_positions` | **yes (correct)** | drifts by **forward-shifted** `ret` (holding-period return) → off-by-one NOT present |
| Drift compounds gross vs excess returns | §5.3 | `src/portfolio/construction.py:806` `_drift_positions` | **yes (correct)** | compounds gross `ret` — matches note's preference |
| OOS R² — zero benchmark (pooled, unweighted) | §4.1 | `src/analysis/motivation.py:2001` `compute_quintile_oos_r2` | **yes** | `1−Σ(r−r̂)²/Σr²`, unweighted |
| DolVol/utility-weighted OOS R² (Eq. 2) | §4.1 | `src/analysis/motivation.py:2129` `compute_utility_weighted_r2` | **yes** | matches Eq. 2 exactly; wired via `src/analysis/formal/liquidity_sorted_r2.py` |
| §4.1 monthly error differential, liquid universe, DolVol-weighted | §4.1 | `src/analysis/eval_realignment/deployment_weighted_metrics.py` (script 41) | **yes (built)** | w̃-weighted per-quintile differential implemented in the eval_realignment track; the formalanalysis `error_differential.py:15` remains unweighted. See `docs/eval_realignment_pipeline.md` §2 |
| §4.2 signal-weighted capacity portfolio | §4.2 | `src/analysis/eval_realignment/capacity_portfolio.py` (script 42) | **yes (built)** | implemented in the eval_realignment track; see `docs/eval_realignment_pipeline.md` §3 |
| §5.2 cost-aware device, redesigned: per-stock breakeven no-trade gate | §5.2 | `src/analysis/eval_realignment/breakeven_capacity.py` (script 43) | **yes (built)** | gates the *trade* (not the signal rank) by \|α\|≥½Spread, in return units — cannot degenerate into a liquidity sort; see `docs/eval_realignment_pipeline.md` §4 |
| 2×2 decomposition deliverables (old `two_by_three` formats) | §3–§4 | `src/analysis/eval_realignment/two_by_two_tables.py` (script 44) | **yes (built)** | training × cost-device cells 1A/1B/2A/2B; long CSV + per-cell timeseries XLSX + Table-12-style workbook; see `docs/eval_realignment_pipeline.md` §4.4 |
| §4.3 capacity-weighted long-only book | §4.3 | `src/portfolio/construction.py:449` `build_prediction_quantile_timeseries` | **partial** | long-only quantile (Q5) book exists, but equal/ME weighted, not capacity-weighted, no liquid screen |
| §4.4 liquid-universe sorts, DolVol-weighted legs | §4.4 | `src/portfolio/construction.py:135` `_selected_leg_weights` | **no** | legs only equal/ME; no DolVol legs; no liquid-universe restriction |
| §4.5 implementable-utility / Jensen MV optimizer + CE | §4.5 | `src/analysis/eval_realignment/implementable_mv.py` (script 46) | **yes (built, Static tier)** | JKMP (2024) Static-ML closed form on the liquid top-1000 universe (Ledoit-Wolf Σ, Kyle-λ quadratic cost vs cost-blind Markowitz baseline), net CE(γ) at the book's own γ; Multiperiod/Portfolio-ML tiers deferred; see `docs/eval_realignment_pipeline.md` §6 |
| §6 inference (NW t-stats, LW bootstrap) | §6 | `src/evaluation/statistics.py` | **yes (infra)** | NW + Ledoit–Wolf bootstrap exist, not yet wired into portfolio tables (see `docs/formal_analysis_future_plan.md` tasks 1–3) |

## 3. Flags — note items not found / divergent

1. **Top-60% DolVol screen (§5.1) — absent.** Portfolio formation sorts the full
   input cross-section; no implementability screen before sorting.
2. **5%-per-name cap (§5.1) — absent.** No per-stock weight cap; intentional today.
3. **DolVol-weighted portfolio legs (§3, §4.4) — absent.** `PORTFOLIO_WEIGHTINGS`
   is `{"equal","value"}` only — the literal §3 consistency violation.
4. **Signal-weighted capacity portfolio (§4.2) — now implemented** in the eval_realignment track (script 42), extended by the breakeven-gated execution layer (script 43) and the old-format 2×2 deliverables (script 44); see `docs/eval_realignment_pipeline.md` §3–§4. (Was absent at audit time.)
5. **Implementable-utility / Jensen optimizer (§4.5) — now implemented at the Static tier** (script 46: closed-form mean-variance with quadratic trading costs vs a cost-blind Markowitz baseline, net CE(γ)); the Multiperiod and end-to-end Portfolio-ML tiers remain deferred.
6. **DolVol-weighted liquid-universe error differential (§4.1) — only unweighted exists.**
7. **§5.3 off-by-one — NOT borne out.** The note's allegation is *conditional*
   ("if the code mirrors the write-up"); the code does not — `_drift_positions`
   uses the forward-shifted holding-period return and compounds gross returns,
   which is correct. Subtle; worth independent confirmation before treated as settled.
8. **§5.2 TC-sort scaling — confirmed as the note describes** (raw `½Spread`, not
   `c×turnover`), but a deliberate current design (market impact added back at the
   realized-net-return stage) — "fix vs. keep" is a design decision, not a bug.

## 4. Open research-design choice (for the redesign)

The note writes `w̃ ∝ DolVol`, but the project trains **four** weight families
(`dolvol`, `softmax_rank`, `tc`, `tc_rank`). The §3 consistency principle requires
choosing *which* implementability measure defines (i) the evaluation universe and
(ii) the portfolio weights. This must be decided before implementing §4. **To be
resolved with the project owner — not chosen unilaterally.**

## 5. Planned construction — Signal-weighted capacity portfolio (§4.2)

> **Status:** IMPLEMENTED in the eval_realignment track —
> `scripts/eval_realignment/42_signal_weighted_capacity_portfolio.py` and
> `src/analysis/eval_realignment/capacity_portfolio.py`. The as-built methodology
> and outputs are documented in `docs/eval_realignment_pipeline.md` §3 (the
> unified signed-book, unit-gross `Σ|θ|=1` convention was adopted). The design
> notes below are retained for context.

**Idea.** Replace the quantile long–short with a continuous book that holds each
stock in proportion to its *capacity* and its *centered signal*:

```
θ_it  ∝  DolVol_it · ( r̂_it − r̄ᵂ_t ),     r̄ᵂ_t = ( Σ_i DolVol_it · r̂_it ) / ( Σ_i DolVol_it )
```

where `r̄ᵂ_t` is the **dollar-volume- (capacity-) weighted** cross-sectional mean
of the predictions.

**Why centered on `r̄ᵂ_t` (dollar-neutrality).** By construction,
`Σ_i θ_it ∝ Σ_i DolVol_it·r̂_it − r̄ᵂ_t·Σ_i DolVol_it = 0`, so the book is
dollar-neutral with no time-varying net long/short tilt. (Centering on the
*equal-weighted* mean would leave a drifting net tilt that contaminates the Sharpe.)

**Why it is the cleanest test (§1, §3).** The *same* capacity measure enters both
the training loss (Eq. 1) and the portfolio — satisfying the §3 consistency
principle. A stock is long when its predicted return exceeds the capacity-weighted
average and short when it falls below, with position size scaling in dollar volume.
There is **no quantile cut, no equal-weighting of microcaps, and no long–short
non-additivity** — removing exactly the artifacts §2 attributes the headline
negative training effect to.

**Reporting.** Gross and net Sharpe and certainty-equivalent across the AUM grid,
with proper inference per §6 (Ledoit–Wolf / block bootstrap for Sharpe,
Newey–West for monthly differentials).

**Gap vs. current code.** Absent today — all portfolio construction is
quantile-based (`build_long_short_portfolio` / `build_prediction_quantile_timeseries`,
leg weights `{equal, value}` only; see the §4.2 row in the mapping table). A new
construction path is required; it should reuse the existing realized-net-return /
turnover machinery (`_compute_net_returns_with_context`, `_drift_positions`) for
net returns.

**Open design choices (to resolve before building; not chosen here):**
- **Capacity measure.** The note uses `DolVol`. The project trains four weight
  families (`dolvol`, `softmax_rank`, `tc`, `tc_rank`); which defines `θ` (and the
  evaluation universe)? Ties into the open choice in Section 4; consistency argues
  for matching the training weight.
- **Gross exposure / leverage.** `θ` is defined up to a proportionality constant;
  needs a target gross leverage (e.g. `Σ|θ| = 1`, or scaled to a volatility/AUM
  target) so Sharpe/CE are comparable across specs and the AUM grid.
- **Universe.** Full cross-section vs liquid/AUM-feasible set for forming `θ` (the
  capacity weighting already down-weights illiquid names, but a screen may still be
  wanted — ties to §5.1).
- **Net-return / TC mapping.** How the continuous `θ` rebalances and how the
  realized, turnover-implied TC (full Frazzini) maps onto a continuous book vs. the
  current discrete-leg accounting.
- **Relation to the existing 2×2 / 2×3.** Standalone new exhibit, or an additional
  row/column in the decomposition?
