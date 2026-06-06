# Formal Portfolio Tables — Cross-Spec Sweep

> **Project:** LiquidityML
> **Branch:** `formalanalysis`
> **Scope:** the portfolio tables (scripts `22`/`22b`) across **all 4 primary weight specs**
> (`dolvol`, `softmax_rank_lam2`, `tc_500m`, `tc_rank_lam3_500m`) × **equal/value** weighting
> × **full_sample/NYSE** stock universe × 3 models × 4 AUM (PropTC, $100M, $500M, $1B).
> **Last updated:** 2026-06-02
> **Verification:** every headline number below was re-derived from the source `.xlsx`
> Panel-B floats by an independent adversarial pass (6/6 seed claims confirmed within ~0.001).

This complements the spec-specific [formal_analysis_softmax_rank_lam2.md](formal_analysis_softmax_rank_lam2.md).
It asks the cross-cutting question: **for which (spec, weighting, universe, model)
does liquidity-weighted training actually pay off in the portfolio?**

**Definitions** (annualized net-of-TC Sharpe, 2×3 cells): **training effect = 2A−1A**
(weighting, plain sort), **portfolio effect = 1B−1A** (TC-aware sort), **total
effect = 2B−1A** (both). **Universe** = the stock universe the portfolio is built
from (full_sample = all stocks; NYSE = `exchcd==1`). Higher = better.

---

## TL;DR

1. **Model–strategy dissociation (most robust):** **xgboost** is the *only* model
   that ever makes the long-short net-positive; **elastic_net** *owns* the
   long-only book. No model wins both; **no config helps all three models**.
2. **`tc_500m` (the TC-calibrated weight) is the cost-aware winner** — the only
   spec that turns xgboost's long-short total positive — but it is **AUM- and
   universe-gated**. `softmax_rank_lam2` is middling; **`dolvol` is the worst**.
3. **Weighting helps the *middle* of the book and hurts the *top decile*, across
   every spec** — mean training effect Q3 +0.07 vs Q5 −0.14.
4. **Two non-interchangeable levers:** **NYSE universe** helps the long-only side;
   **value-weighting** helps the long-short side. Both work by stripping microcaps
   at the portfolio level.
5. **AUM direction flips by universe:** more AUM *helps* weighting in full_sample
   (illiquid baseline) but *hurts* it in NYSE-value (already-liquid baseline).
6. **The neural network is harmed by weighting essentially everywhere.**
7. **Levels ≠ effects (important).** The effect-vs-standard view (cell 2B−1A)
   says xgboost owns the long-short — but in deployable net-SR *levels* the only
   long-short profitable at institutional scale is **elastic_net + `tc_500m` +
   value-weighting (cell 2A, +0.36 → +0.15 across PropTC→$1B)**; the standard
   long-short dies by $500M. The deployable cell is the *plain-sorted* weighted
   model (2A), not the TC-aware-sorted 2B the effect view uses. See
   "Net-SR levels" below.

---

## Net-SR levels — the deployable view (profitability, not just vs-standard)

The effect-vs-standard view answers *"does weighting beat the standard model?"* —
not *"is the result profitable?"* The level view (net SR of the best deployable
cell, max over the six 2×3 cells) reframes the long-short conclusion.

**At institutional AUM the long-short is dead for almost everything.** Best
deployable long-short net SR (equal/full_sample) is positive only at PropTC
(xgb/EN +0.30, NN +0.61 — all from the *standard* cells 1A/1B) and collapses to
≈ −0.5 to −0.7 by $500M for every spec and model. The standard-model long-short
is a small-money strategy.

**The one scalable exception — created by the weighting:** elastic_net +
`tc_500m` + value-weighting, **cell 2A (weighted, plain sort)**, is profitable at
*every* AUM:

| AUM | full_sample 2A | nyse 2A | standard 1A (full / nyse) |
|---|--:|--:|--:|
| PropTC | +0.351 | +0.360 | −0.018 / −0.215 |
| $100M | +0.287 | +0.295 | −0.054 / −0.245 |
| $500M | +0.207 | +0.214 | −0.098 / −0.283 |
| $1B | +0.147 | +0.154 | −0.131 / −0.311 |

The standard cell is negative throughout, so the profit is the weighting's doing.
**This is the single scalable, profitable long-short in the whole sweep, and the
effect view hid it** — because "LS total" uses cell 2B (the TC-aware sort), and
EN's TC-aware sort is counterproductive (2B = −0.23 to −0.52). The deployable cell
is the *plain-sorted* weighted model (2A). **Practical lesson: for elastic_net,
apply the `tc_500m` weight but NOT the TC-aware sort.**

**Long-only levels are healthier** (no short-leg cost sink): the Q5 book stays
positive at scale far more often than the long-short — e.g. softmax_rank_lam2's
best Q5 cell runs +0.62 (PropTC) → +0.19 ($1B), and EN under cost-aware spec +
value/NYSE pushes Q5 higher still.

**Short-leg: value-weighting does NOT fix the cost sink.** The standard short leg
(cell 1A, Q1) net SR *worsens* under value-weighting (full_sample PropTC:
xgb −0.28→−0.45, EN −0.36→−0.41, NN −0.17→−0.29) — the short-side alpha is a
microcap phenomenon that value-weighting strips out. So value-weighting helps the
long-short only through the *relative* effect and the *long* side, never by
rescuing the short leg.

**TC-target sort (C column):** rarely the deployable best — for the standout
EN/`tc_500m`/value config the best cell is 2A, not 2C. The TC-target sort is a
competitive-but-secondary third option, not a winner in levels.

---

## 1. Model × strategy dissociation

| | positive long-short *total* (2B−1A) | positive long-only Q5 *training* (2A−1A) | best cell |
|---|---|---|---|
| **xgboost** | **17/64 configs** (the only model that ever wins) | 6/64 | LS total +0.271 |
| **elastic_net** | 0/64 (max −0.107) | **37/64 configs** | Q5 train +0.160 |
| **neural_network** | 0/64 (max −0.081) | 4/64 | — |

→ **Trade long-short → use xgboost; trade long-only → use elastic_net.** The two
"best" configs sit in mutually exclusive corners:

- **Best long-short:** xgboost / `tc_500m` / equal / full_sample / $1B = **+0.271** (total).
- **Best long-only:** elastic_net / `tc_500m` / value / NYSE / $1B = **+0.160** (Q5 training).

No config makes the long-short total positive for ≥2 models; the closest to broad
benefit (cost-aware + NYSE on the long-only side) helps EN+NN but *hurts* xgboost.

> **Levels caveat:** "xgboost owns the long-short" is an *effect-view* (2B−1A)
> statement. In deployable net-SR *levels* (see "Net-SR levels" above), the only
> long-short profitable at scale is elastic_net + `tc_500m` + value (cell **2A**),
> because xgboost's positive *total* effect still nets to a negative Sharpe at
> high AUM while EN's plain-sorted weighted cell is genuinely positive. The
> dissociation holds for *who benefits from the full 2B treatment*; the
> *deployable* winner is EN.

## 2. Spec ranking — `tc_500m` wins, `dolvol` worst, `softmax` middling

Long-short **total effect (2B−1A)**, equal / full_sample (the baseline config):

| Spec | xgb PropTC | xgb $500M | EN PropTC | NN PropTC |
|---|--:|--:|--:|--:|
| `dolvol` | −1.084 | −0.597 | −1.009 | −1.520 |
| `softmax_rank_lam2` | −0.144 | +0.040 | −0.316 | −0.152 |
| `tc_500m` | −0.467 | +0.060 | −0.646 | −1.278 |
| `tc_rank_lam3_500m` | −0.266 | +0.014 | −0.415 | −0.731 |

Mean LS total by spec (all configs): **dolvol −0.83** (worst by far) vs
softmax −0.30, tc_500m −0.31, tc_rank_lam3_500m −0.31. `dolvol`'s aggressive
dollar-volume tilt sacrifices the most illiquid alpha.

**`tc_500m`'s win is gated.** For xgboost the long-short total is positive at
equal/nyse (+0.06 → +0.12), value/full (+0.11 → +0.12), best +0.271 at $1B
full_sample — but at **PropTC in full_sample it still hurts (−0.47)**; it only
turns positive at ≥$500M. So "tc_500m helps the long-short" is a *high-cost*
statement.

## 3. Weighting helps the mid-book, hurts the top decile (cross-spec)

Mean Q-level **training effect** across all configs/specs:

| Q1 | Q2 | Q3 | Q4 | Q5 |
|--:|--:|--:|--:|--:|
| +0.03 | +0.04 | **+0.07** | +0.03 | **−0.14** |

**Q5 is negative for all four specs.** Liquidity weighting consistently improves
the *middle* of the long-only book (peak Q3) and damages only the extreme top
decile (Q5). The headline "long-only harm" therefore understates the mid-book
benefit — this is the cross-spec generalization of the per-quintile pattern first
seen for `softmax_rank_lam2` (gains in Q3–Q4, neutral/negative at Q5).

## 4. Two distinct levers — NYSE (long-only) vs value (long-short)

| Lever | long-only Q5 training | long-short total |
|---|---|---|
| full_sample → **NYSE** | **+0.05 mean, helps 79%** | +0.004 (neutral) |
| equal → **value** | −0.01 (44%) | **+0.10 mean, helps 54%** |
| stack both | +0.09 | — |

**NYSE is the long-only lever; value-weighting is the long-short lever** — not
interchangeable. Both reduce the weighting "damage" because they already strip
microcaps at the portfolio level, so liquidity-weighted *training* becomes a
smaller sacrifice (verified e.g. dolvol/full_sample/xgb training −0.825 equal →
−0.052 value).

## 5. The elastic_net `tc_500m` tension (weighting helps, sort hurts)

For elastic_net, `tc_500m` weighting is *strongly* beneficial to the long-short
on its own — the **training effect reaches +0.575** (value/nyse PropTC: standard
1A = −0.215 → weighted plain-sort 2A = **+0.360**, a genuinely profitable weighted
long-short). **Yet EN's long-short *total* stays negative**, because its TC-aware
*sort* (portfolio effect) drags the long-short down hard via the short leg
(the same short-leg interaction documented for softmax). So for EN the **weighting
is the good lever and the cost-aware sort is the bad one** — they pull opposite
ways, and the deployable 2B nets out negative.

## 6. AUM direction flips by universe

Higher AUM **helps** weighting in **full_sample** (illiquid baseline → cost-aware
training gains more) but **hurts** it in **NYSE-value** (already-liquid baseline):

- xgb / `tc_500m` / equal / full_sample: −0.47 → −0.24 → +0.06 → **+0.27** (PropTC→$1B)
- xgb / `tc_500m` / value / nyse: +0.07 → +0.05 → +0.01 → **−0.01** (PropTC→$1B)

So "weighting helps more at scale" holds only when the baseline universe is
illiquid; in an already-liquid NYSE-value baseline, larger AUM erodes the edge.

## 7. The neural network

Harmed by weighting in **all 16 long-short PropTC cells** (4 specs × 2 weightings
× 2 universes), ranging −0.09 (least bad: tc_500m/value/nyse) to −1.74
(dolvol/full_sample/equal). Lone near-exception across the whole sweep:
`tc_500m` / value / NYSE Q5 long-only, marginally positive (+0.04 PropTC, +0.05
at $500M). Consistent with the NN's failure to benefit from weighting throughout
21a–21d.

---

## Caveats

- All figures are **Sharpe differences with no standard errors** — small values
  (±0.05) may be indistinguishable from zero.
- The portfolio tables cover only the **4 primary specs**; the "cost-aware helps"
  conclusion rests on `tc_500m` and its rank-λ3 variant (the `tc_10m/100m/1000m`
  and `softmax_rank_lam3` families are not in the portfolio tables).
- "Long-short total" (2B−1A) mixes the training and sort effects; for elastic_net
  the two pull in opposite directions (§5), so read the channels separately.
- These are point estimates; deployable conclusions also need the level (net-SR)
  view, not only the effect-vs-standard view.
