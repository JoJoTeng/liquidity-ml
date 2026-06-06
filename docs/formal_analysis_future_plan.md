# Formal Analysis — Future Plan (paper-completion tasks)

> **Project:** LiquidityML
> **Branch:** `formalanalysis`
> **Created:** 2026-06-03
> **Status:** planning (not started)
> **Source:** handwritten task note, turned into a concrete plan.
> **Companion docs:** [formal_analysis_softmax_rank_lam2.md](formal_analysis_softmax_rank_lam2.md)
> (single-spec worked example, 21a–22) and
> [formal_portfolio_spec_sweep.md](formal_portfolio_spec_sweep.md) (cross-spec sweep).

**Purpose.** The empirical analysis (21a–22) is done and documented, but it is
point-estimate only. These tasks turn it into **paper-ready results**: add
statistical inference, risk-adjusted alpha, a capacity metric, the gross-side
decomposition, and the methodology write-up.

## Decisions locked in

- **Alpha (#2): Fama–French 5-factor (FF5)** — MKT-RF, SMB, HML, RMW, CMA.
- **Breakeven (#6): breakeven *spread*** — the proportional bid-ask spread level
  at which the strategy's net return/Sharpe = 0 (not a breakeven-AUM).

## Task overview

| # | Task | Type | Status | Key dependency |
|---|---|---|---|---|
| 1 | t-statistics (inference) | new | not started | monthly return series + NW helper |
| 2 | Portfolio alpha (FF5) | new | not started | monthly returns + FF5 factor data |
| 3 | Statistics / Sharpe-ratio table | new | not started | #1 |
| 4 | Draft the data methodology | writing | not started | existing docs |
| 5 | Gross Sharpe decomposition | partial | not started | `gross_sr` already in Table 13 |
| 6 | Breakeven spread | new | not started | gross returns + TC model |

Suggested order: **1 → 2 → 3** together (the inference block), **5** (closest to
done), **6** (capacity), **4** (writing, any time).

---

## 1. t-statistics (inference)

**Goal.** Attach significance to the portfolio results — currently all Sharpe
differences / effects are point estimates with no standard errors.

**Method.** Newey–West t-stats (lags=6, project convention via
`src.evaluation.statistics.newey_west_tstat`) on the **monthly portfolio return
series** — net mean return, and the FF5 alpha (#2). Report for the long-short
(Q5−Q1) and the Q5 long-only book.

**Inputs.** 21e monthly series:
`outputs/formalanalysis/analysis/{model}/{spec}/prediction_quantile[/stock_universe/{u}]/prediction_quantile_timeseries_{aum}.csv`
and `two_by_three_timeseries_{aum}.xlsx`.

**Output.** t-stat columns added to the summary/alpha tables (#2, #3).

**Open.** Inference on the *Sharpe-difference effects* (training/portfolio
effects) is harder — needs a block bootstrap; default to t-stats on returns and
alpha first, treat effect-significance as a stretch goal.

## 2. Portfolio alpha — FF5

**Goal.** Risk-adjusted performance: does the strategy earn alpha beyond FF5?

**Method.** Regress each portfolio's **monthly net excess return** on FF5
(MKT-RF, SMB, HML, RMW, CMA) → report **α (annualized), t(α) [NW], factor
loadings, R²**. (Consider an FF5+MOM variant — these are anomaly-blend
portfolios, so momentum exposure is likely; decide at execution.)

**Inputs.** Monthly portfolio returns (as #1) + **FF5 factor series**
(Ken French data library or WRDS). **Prerequisite: confirm whether FF5 factors
already exist in `data/`; if not, fetch and store them.**

**Output.** Alpha table per (portfolio, spec, config). Scope to confirm: the
**deployable winners** (e.g. EN/`tc_500m`/value long-short cell 2A; Q5 long-only)
plus the standard baseline, rather than all 192 configs.

## 3. Statistics / Sharpe-ratio table

**Goal.** The paper performance table per portfolio.

**Method.** Summarize each monthly series: annualized mean, std, **gross & net
Sharpe**, t-stat (from #1), turnover, avg N, and optionally skew / max drawdown.

**Inputs.** Monthly series + #1.

**Output.** A clean stats table (one of the paper's main exhibits).

## 4. Draft the data methodology

**Goal.** The Data & Methodology section of the paper.

**Content.** Data sources (CRSP via `00_fetch_data.py`; Chen–Zimmermann
predictors); sample window; feature set + monthly rank-normalization to [0,1]
with 0.5 fill; liquidity measures (`liq_*`, primary = dvol); the four weight
families (dolvol / softmax_rank / tc / tc_rank) normalized to mean one; rolling
training; the two-sided TC-aware portfolio construction; the transaction-cost
model (spread, ADV, AUM).

**Inputs (reuse).** [weighting_schemes.md](weighting_schemes.md),
[portfolio_construction.md](portfolio_construction.md), CLAUDE.md, scripts
`00`/`01`/`20`. Mostly assembly + prose.

## 5. Gross Sharpe decomposition

**Goal.** Report the **gross (pre-TC)** 2×3 decomposition alongside the net one,
to separate the **pure-prediction effect from the transaction-cost effect**.

**Method.** Run the same 2×3 training/portfolio/total decomposition on the
**gross** long-short Sharpe (cells' gross SR) instead of net. The net version is
`_decomposition_dict` in `scripts/22b_table12_two_sided.py`; mirror it on the
gross series.

**Inputs.** `gross_sr` is already reported per leg in the Table 13
(`table13_legs_two_sided_*`) outputs; the gross long-short series is available in
21e. Largely a re-aggregation, not a new computation.

**Output.** Gross 2×3 tables paired with the existing net ones (a "before TC vs
after TC" view).

## 6. Breakeven spread

**Goal.** Capacity metric — the proportional bid-ask **spread** at which the
strategy's **net return (or net Sharpe) = 0**.

**Method.** Net = gross − TC(spread); TC scales with turnover × (spread/2) under
the PropTC model. Sweep / solve for the spread level where net crosses zero, per
portfolio. The TC machinery is `prepare_transaction_cost_context` /
`_compute_net_returns_with_context` in `src/portfolio/construction.py`; the
PropTC scenario is the natural base (spread/2 only, no market impact).

**Output.** Breakeven spread (bps) per (portfolio, spec, config) — interpretable
as "the strategy survives any real-world spread below X bps."

**Note.** Complements the AUM/TC sweep already done: AUM measures impact-cost
tolerance; breakeven spread measures spread-cost tolerance.

---

## Prerequisites / shared assets

- **Monthly portfolio return series** (foundation for 1, 2, 3, 5, 6) — already
  produced by 21e for the primary specs (`*_timeseries_*` files). Confirm
  coverage for the configs in scope.
- **FF5 factor data** — *verify availability in `data/`; fetch if missing* (the
  one external prerequisite, for #2).
- **NW helper** — `src.evaluation.statistics.newey_west_tstat(series, lags=6)`.

## Open sub-decisions (resolve at execution)

- Scope of #2/#3: which portfolios/specs/configs (suggest the deployable winners
  + standard baseline, not all 192).
- FF5 vs FF5+MOM for the alpha regression.
- Whether to push inference onto the Sharpe-difference effects (bootstrap) or
  keep it to returns/alpha.
