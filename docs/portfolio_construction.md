# Portfolio Construction

This appendix describes the portfolio construction underlying the empirical
results; it supersedes the earlier $2 \times 3$ design, whose
transaction-cost-target column has been removed (Section 5). Portfolios are
rebalanced monthly using one-month-ahead predictions formed at the end of
the prior month; reported performance is net of transaction costs computed
from realized turnover. Relative to a textbook quantile sort, the current
construction adds three design axes. First, the *stock universe* on which
quantiles are formed is itself a treatment: the full cross-section, the
NYSE subsample, or the top 60% of the dollar-volume distribution under a
config-driven screen. Second, *within-leg weights* are a treatment: equal,
value, dollar-volume, or — the default — the spec's own deployment weight
$\tilde w_{i,t}$, so that the book holds names in the same proportion the
training objective emphasizes them. Third, monthly membership carries
*hysteresis*: a held name is retained slightly past the fresh sort boundary
whenever the round-trip cost of replacing it exceeds the signal shortfall,
which converts the naive full-rebalance sort into an executable book. The
long–short is a true dollar-neutral quantile book, and the training
$\times$ execution comparison is a $2 \times 2$ factorial (Section 5).
Implementation and the output layout are collected in Section 10.

## 1. Notation

Let $\mathcal{U}_t$ denote the investable universe in month $t$ — the panel
cross-section after the universe filter and, where applicable, the liquidity
screen of Section 2. For each stock $i \in \mathcal{U}_t$:

- $r_{i,t}$ is the realized **excess** return earned between $t-1$ and $t$.
- $\hat{r}_{i,t}$ is a one-month-ahead return prediction formed at $t-1$;
  superscripts $\mathrm{base}$ and $w$ distinguish the uniformly weighted
  and liquidity-weighted training schemes of Section 5. (The
  eval-realignment appendices index the same objects one period later,
  writing $\hat r_{i,t+1}$ for the prediction formed at $t$ and $r_{i,t+1}$
  for the return it targets; the two conventions differ only by this
  one-period index offset.)
- $\tilde w_{i,t}$ is the spec's mean-one deployment weight — the **same
  weight that defines the training objective** (column `w_tilde`,
  recomputed by `compute_formal_utility_weights` in
  `src/analysis/formal/common.py`).
- $\mathrm{ME}_{i,t}$ is end-of-month market capitalization
  (`liq_me_raw`) and $\mathrm{DolVol}_{i,t}$ is 21-day average daily
  dollar volume (`liq_dvol_21d`).
- $\text{Spread}_{i,t}$ is the bid–ask spread (`liq_BidAskSpread`),
  $\sigma_{i,t}$ is daily-scaled realized volatility
  (`liq_excess_sigma_12m_daily`), and $\text{ADV}_{i,t}$ is average daily
  dollar volume (`liq_dvol_21d`).
- $A$ is total strategy capital (assets under management) under the
  transaction-cost scenario being reported, and $A_{\mathrm{leg}}$ is the
  capital assigned to a single leg (Section 6).

We define the one-way proportional cost, the *half-spread*, as

$$
\mathrm{hs}_{i,t} \;=\; \tfrac{1}{2}\,\text{Spread}_{i,t}.
$$

The half-spread enters the construction only through the membership
hysteresis bands of Section 4.3 — it is the price of replacing a held name
with a marginal entrant. The market-impact component of trading costs is
deliberately excluded at the membership stage; it is included when computing
realized net returns from turnover (Section 7). For the hysteresis bands,
half-spreads are taken from the spread-only, AUM-independent cost helper
(`compute_tc_for_sorting(panel, aum=0)` in `src/weighting/schemes.py`),
with missing spreads filled by the within-month median. Stocks with missing
predictions or returns in a month are excluded from that month's sort, not
imputed.

## 2. Investable universe: the stock-universe axis

Quantiles are formed on one of three parallel universes, selected by the
`--universe {full, nyse, top40, all}` flag of the driver script
(`scripts/21e_formal_portfolio_decomposition.py`); the config default
`portfolio.universe.default: "all"` runs all three. The `_universe_panel`
helper implements:

- **full** — the whole processed panel, unscreened.
- **nyse** — the panel pre-filtered to NYSE listings
  (`exchcd == 1`; the run errors if the exchange-code column is missing).
- **top40** — the full panel combined with a within-month liquidity
  screen applied *before* quantile assignment. With
  `portfolio.liquidity_screen_pct: 0.60`, a stock is investable in month
  $t$ if and only if

$$
\mathrm{pctrank}_t\!\bigl(\mathrm{DolVol}^{21d}_{i,t}\bigr) \;\ge\; 0.60,
$$

i.e., the portfolio trades only the **top 60%** of the month's dollar-volume
distribution (screen column `portfolio.dolvol_col = liq_dvol_21d`). The
screen is enforced identically in the long–short book and in the standalone
prediction-quantile books. A held name that falls out of the screen is
force-sold — the hysteresis bands of Section 4.3 never resurrect a name the
screen has removed. The same config knob drives the long-only capacity book
of the eval-realignment track
(`scripts/eval_realignment/45_longonly_capacity_q5.py`), which also derives
its top-quantile selection from `n_quantiles`, so the formal top-60%
universe and the eval-realignment top-quantile screen are aligned by
construction.

The universe axis is a treatment, not a robustness footnote: quantile
membership, hysteresis boundaries, and within-leg weights are all recomputed
inside each universe, so the three books answer three distinct deployment
questions (unconstrained paper portfolio, exchange-quality subsample, and a
liquidity-feasible universe).

## 3. Selected-stock weights: the leg-weighting axis

After stocks are selected into a leg or quantile — write $\mathcal{G}_t$
for the generic selected group — within-group weights are formed to sum to
one. Four schemes are supported
(`PORTFOLIO_WEIGHTINGS = {"equal", "value", "dolvol", "signal"}`; weight
formation in `_selected_leg_weights`, `src/portfolio/construction.py`):

$$
w_{i,t} \;=\; \frac{x_{i,t}}{\sum_{j \in \mathcal{G}_t} x_{j,t}},
\qquad
x_{i,t} \;=\;
\begin{cases}
1 & \text{equal},\\[2pt]
\mathrm{ME}_{i,t} & \text{value } (\texttt{liq\_me\_raw}),\\[2pt]
\mathrm{DolVol}_{i,t} & \text{dolvol } (\texttt{liq\_dvol\_21d}),\\[2pt]
\tilde w_{i,t} & \text{signal } (\texttt{w\_tilde}).
\end{cases}
$$

Non-finite or non-positive $x_{i,t}$ are dropped; if a leg's weight total
is non-positive the leg falls back to equal weights.

**Signal weighting is the default** (`portfolio.weighting: "signal"`,
read by the driver when `--portfolio-weighting` is omitted). Under signal
weighting, each leg holds names in proportion to the spec's own mean-one
deployment weight $\tilde w_{i,t}$ — the identical object that multiplies
the squared error in the weighted training loss — so the portfolio
evaluates the model on the capital allocation its training objective claims
to care about. This is the consistency principle of the eval-realignment
track (Section 9) applied to the sorted-portfolio object. The construction
layer is spec-agnostic: the driver recomputes $\tilde w_{i,t}$ per spec and
merges it onto the panel before building the book, and construction raises
a `KeyError` if the column is missing.

The equal- and value-weight schemes are retained as reportable variants.
The **gross** return series of the equal-weight, full-universe A cells
reproduces the conventional pre-rework sort; exact net-of-cost replication
additionally requires the legacy `--leg-capital full` sizing, and the old
cost-aware columns have no counterpart in the current grid (Section 9
details the replication conditions).

## 4. Portfolio sorting

### 4.1 Quantile assignment and prediction-quantile portfolios

Each month the screened universe is split into $Q = 5$ rank quintiles
(`n_quantiles: 5`, `long_quantile: 5`, `short_quantile: 1`, monthly
rebalancing; `portfolio` block of `config/config.yaml`). Assignment is a
rank-based `qcut` with ties broken by first occurrence:

$$
q_{i,t} \;=\; \mathrm{qcut}\bigl(\mathrm{rank}(\hat r_{i,t}),\, 5\bigr)
\;\in\; \{1, \dots, 5\},
\qquad
\mathcal{L}_t = \{i : q_{i,t} = 5\},
\quad
\mathcal{S}_t = \{i : q_{i,t} = 1\}.
$$

Months with fewer than $2Q$ stocks are skipped; a leg with fewer than two
names returns a missing month.

For each quantile $q$, the selected stocks constitute a standalone
long-only portfolio with within-group weights $w_{i,t}$ from Section 3:

$$
R_{q,t} \;=\; \sum_{i \in \mathcal{Q}_q} w_{i,t}\, r_{i,t}.
$$

These five quantile books are reported separately as a diagnostic of how
each training scheme performs across the entire prediction distribution
(`build_prediction_quantile_timeseries` in
`src/portfolio/construction.py`); their per-quantile effect decomposition
is described in Section 8.3.

### 4.2 The long–short book

The deployable long–short holds the top quintile long and the bottom
quintile short (`build_long_short_portfolio`). With within-leg weights
$w^{L}_{i,t}$, $w^{S}_{i,t}$ from Section 3, the gross return is

$$
R^{\mathrm{LS}}_{t} \;=\;
  \sum_{i \in \mathcal{L}_t} w^{L}_{i,t}\, r_{i,t}
  \;-\; \sum_{i \in \mathcal{S}_t} w^{S}_{i,t}\, r_{i,t}.
$$

Each cell of the $2 \times 2$ design (Section 5) is a **true
dollar-neutral book**: both legs are built, sized, and costed explicitly
(`_build_long_short_cell` in `src/analysis/formal/portfolio_tables.py`)
rather than reported as a difference of two long-only quantile series.

### 4.3 Membership hysteresis

The execution treatment of the design is a pair of parameter-free
hysteresis bands. Both are scaled by observable trading costs — the
stock's own half-spread $\mathrm{hs}_{i,t}$ plus a leg- or bin-level median
half-spread that proxies the cost of the replacement entrant — so the
band widens exactly where trading is expensive and vanishes as spreads go
to zero. Hysteresis is mutually exclusive with the legacy TC-aware sort.

**Symmetric long/short band** (`_apply_hysteresis_bands`; used by the B
cells of the long–short book). Let $b_{\mathrm{hi}}$ be the minimum signal
in the fresh long leg, $b_{\mathrm{lo}}$ the maximum signal in the fresh
short leg, and $\mathrm{cm}_{\mathrm{hi}}$, $\mathrm{cm}_{\mathrm{lo}}$ the
median half-spread of the fresh long and short legs (the "entrant toll").
A name held in the previous month is retained past the fresh boundary iff
its signal shortfall is within the round-trip cost of swapping it out:

$$
\text{retain held long } i \;\iff\;
\hat r_{i,t} \;\ge\; b_{\mathrm{hi}}
  - \bigl(\mathrm{hs}_{i,t} + \mathrm{cm}_{\mathrm{hi}}\bigr),
\qquad
\text{retain held short } i \;\iff\;
\hat r_{i,t} \;\le\; b_{\mathrm{lo}}
  + \bigl(\mathrm{hs}_{i,t} + \mathrm{cm}_{\mathrm{lo}}\bigr).
$$

A name is never held on both legs — short-side retention excludes
long-leg members. Retained names are *appended* to the fresh leg rather
than displacing entrants, so legs are not re-trued to a fixed count; the
leg is instead re-trued in weights, since `_selected_leg_weights`
renormalizes each leg to sum to one (retention dilutes fresh entrants).
Previous membership (the `prev_long`/`prev_short` permno sets) is threaded
across months in `build_portfolio_timeseries` and updated only when
hysteresis is active.

**Per-quantile sticky band** (`_apply_quantile_hysteresis`; used by the B
cells of the per-quantile diagnostic via
`build_prediction_quantile_timeseries(hysteresis=True)`). Each stock sits
in exactly one quantile. A held member of prior bin $q$ whose fresh bin
differs is put back into $q$ iff its score is within round-trip cost of
$q$'s edge toward the new bin. With $\mathrm{lo}_q$, $\mathrm{hi}_q$ the
fresh bin-$q$ score minimum and maximum and $\mathrm{cm}_q$ the median
half-spread of fresh bin $q$:

$$
\text{drifted below: }
\hat r_{i,t} \;\ge\; \mathrm{lo}_q - \bigl(\mathrm{hs}_{i,t} + \mathrm{cm}_q\bigr),
\qquad
\text{drifted above: }
\hat r_{i,t} \;\le\; \mathrm{hi}_q + \bigl(\mathrm{hs}_{i,t} + \mathrm{cm}_q\bigr).
$$

Reassignment moves the row out of its fresh bin, so no stock is
double-counted; the `prev_quantile` map $\{\text{permno} \to q\}$ is
carried month to month.

### 4.4 Legacy two-sided TC-aware sort

An earlier design implemented the execution treatment as a *sort
modification* — ranking the long leg on $\hat r_{i,t} - \mathrm{hs}_{i,t}$,
the short leg on $\hat r_{i,t} + \mathrm{hs}_{i,t}$, and a third column on
a TC-adjusted training target — with a fixed-stocks-per-leg refill on the
short side (`_select_quantile`, which holds the short leg at the fresh
quantile count after excluding long-leg overlaps). These code paths
(`tc_penalised`, `tc_target_score`) survive only for the legacy workbook
script `scripts/22b_table12_two_sided.py` and are not part of the current
formal design; the current execution treatment is the membership
hysteresis of Section 4.3.

## 5. Experimental design: the training $\times$ execution $2 \times 2$

Four portfolios are reported in a $2 \times 2$ factorial that crosses two
model-training schemes with two execution schemes
(`compute_two_by_two_decomposition` in
`src/analysis/formal/portfolio_tables.py`):

| | **A: plain full-rebalance sort** | **B: membership hysteresis** |
|:--|:--|:--|
| **1: standard training** ($\hat r^{\mathrm{base}}$) | 1A | 1B |
| **2: weighted training** ($\hat r^{w}$) | 2A | 2B |

Rows differ only in the prediction used for sorting — a baseline model
fitted with uniform observation weights versus a liquidity-weighted model
fitted under the spec's deployment weights (companion training appendix).
Columns differ only in execution: column A rebuilds the book from the
fresh sort every month, while column B applies the cost-scaled membership
hysteresis of Section 4.3, holding everything else fixed. Comparisons
across rows isolate the effect of liquidity-weighted training; comparisons
across columns isolate the effect of cost-aware execution; the interaction
measures whether weighted training and cost-aware execution are
substitutes or complements. The per-quantile diagnostic of Section 8.3 is
built in parallel with the same column semantics — A is the plain quantile
assignment, B the per-quantile sticky hysteresis.

**The tc-target column is removed.** The previous design carried a third
("C") column sorted on the prediction of a model trained to a
transaction-cost-adjusted target. That column was dropped when the
long–short was repointed to the true $2 \times 2$ book: the decomposition
function accepts the old `preds_tc_target_*` arguments but ignores them,
returns an empty `tc_target_effects` dictionary, and the driver no longer
loads tc-target predictions. For discoverability of the legacy naming:
`compute_two_by_three_decomposition` is retained as an alias of the
$2 \times 2$ function, and the current `two_by_two_{aum}.csv` deliverable
keeps the long `metric, value` format of the legacy
`two_by_three_{AUM}.csv` with metric names verbatim, minus the C-cell rows
— this is the format that the eval-realignment presentation script mirrors
"with the tc-target column out of scope"
(`docs/eval_realignment_pipeline.md`, §4.4).

Cells are aligned on common months before any statistic is computed.

## 6. Capital conventions

Sizing enters the construction only through the market-impact term of the
cost model (Section 7), via the per-leg capital $A_{\mathrm{leg}}$
(`_leg_aum` in `src/portfolio/construction.py`):

$$
A_{\mathrm{leg}} \;=\;
\begin{cases}
A/2 & \texttt{leg\_capital = "gross"}
      \quad (\$A \text{ gross book: } |L| + |S| = A;\ \text{default}),\\[2pt]
A   & \texttt{leg\_capital = "full"}
      \quad (\text{legacy } \$2A \text{ gross}).
\end{cases}
$$

Under the default `gross` convention the scenario capital $A$ is the
total gross exposure of the dollar-neutral book, so a \$500M scenario
deploys \$250M long and \$250M short; the legacy `full` convention sized
each leg at the full $A$. The driver exposes
`--leg-capital {gross, full}` with default `gross`, threaded through
`compute_net_returns` and its all-AUM wrapper.

**Exception.** Each standalone prediction-quantile book of Section 4.1 is
a long-only portfolio and is sized with the **full** scenario AUM. This is
a deliberate convention difference: the quantile diagnostic asks what a
\$$A$ long-only book in that bin would cost, while the long–short book
splits \$$A$ of gross capital across two legs.

## 7. Transaction costs and net returns

Net returns are computed from realized turnover at the per-leg capital
$A_{\mathrm{leg}}$ of Section 6 (`_compute_net_returns_with_context` in
`src/portfolio/construction.py`). Between rebalances, position weights
drift with realized returns. Let $r^{\mathrm{raw}}_{i,t}$ denote the raw
(total) return over month $t$ (the panel's `ret` column) — position drift
uses the raw return, while portfolio performance is stated in the excess
return $r_{i,t}$ of Section 1, since dividends and the risk-free leg move
position sizes even when performance is measured in excess terms. Writing
$w^{\mathrm{tgt}}_{i,t}$ for the post-trade target weight,

$$
w^{\mathrm{drift}}_{i,t} \;=\;
  \frac{w^{\mathrm{tgt}}_{i,t-1}\,\bigl(1 + r^{\mathrm{raw}}_{i,t-1}\bigr)}
       {\sum_{j} w^{\mathrm{tgt}}_{j,t-1}\,\bigl(1 + r^{\mathrm{raw}}_{j,t-1}\bigr)}.
$$

Measuring trades against the drifted book rather than the previous target
avoids charging for market moves the portfolio did not trade. The monthly
trade size in stock $i$, as a weight change and as a dollar amount, is

$$
\Delta w_{i,t} \;=\;
  \bigl|\, w^{\mathrm{tgt}}_{i,t} - w^{\mathrm{drift}}_{i,t} \,\bigr|,
\qquad
\Delta Q_{i,t} \;=\; \Delta w_{i,t}\, A_{\mathrm{leg}}.
$$

The stock-level cost rate $\tau_{i,t}$ follows the standard square-root
price-impact specification with an additive half-spread term (Grinold and
Kahn 2000; Almgren, Thum, Hauptmann, and Li 2005; Kyle and Obizhaeva 2016),
with the impact coefficient disciplined by the live-trade estimates of
Frazzini, Israel, and Moskowitz (2018) — the same $\tau_{i,t}$ notation used
in the eval-realignment appendices. The per-dollar cost *rate* is concave in
trade size (square root); the *total dollar cost* $\Delta Q_{i,t}\,\tau_{i,t}$
is convex, growing as $\Delta Q_{i,t}^{3/2}$:

$$
\tau_{i,t}(A_{\mathrm{leg}}) \;=\;
  \mathrm{hs}_{i,t}
  \;+\; \lambda\,\sigma_{i,t}\,
        \sqrt{\frac{\Delta Q_{i,t}}{\text{ADV}_{i,t}}},
$$

and portfolio costs aggregate across both legs:

$$
\mathrm{TC}_t \;=\;
\sum_{\mathrm{legs}} \sum_i \Delta w_{i,t}\,
  \tau_{i,t}(A_{\mathrm{leg}}),
\qquad
r^{\mathrm{net}}_t \;=\; r^{\mathrm{gross}}_t - \mathrm{TC}_t,
$$

with calibration constant $\lambda = 0.1$
(`transaction_costs.lambda_market_impact`, `config/config.yaml`, resolved
via `resolve_market_impact_lambda`). Cost inputs are `liq_BidAskSpread`,
`liq_excess_sigma_12m_daily`, and `liq_dvol_21d`; degenerate values fall
back to a spread of $0.01$, volatility of $0.02$, and ADV of $10^6$; a
stock missing from the cost lookup is charged a flat 50 basis points;
eliminated holdings are costed with their previous-month inputs.

**Turnover convention.** The raw trade sum is
$T^{\mathrm{raw}}_t = \sum_{\mathrm{legs}} \sum_i \Delta w_{i,t}$;
reported turnover follows the one-way convention
$T_t = \tfrac{1}{2}\, T^{\mathrm{raw}}_t$, which counts a buy and the
matching sell as one unit of trading. Costs, in contrast, are charged on
the full raw trade sum — one-way costs are paid on both the buy and the
sell. The first month's turnover row (the initial position build) is
excluded from turnover means.

**Reporting scenarios.** Realized net performance is reported under four
transaction-cost scenarios (`transaction_costs.aum_scenarios`,
`config/config.yaml`):

| Scenario | Strategy capital $A$ | Cost model |
|:---------|:---------------------|:-----------|
| $\mathrm{PropTC}$ | — | Half-spread only (impact zeroed) |
| $\$100\mathrm{M}$ | $A = \$100\mathrm{M}$ | Half-spread $+$ market impact |
| $\$500\mathrm{M}$ | $A = \$500\mathrm{M}$ | Half-spread $+$ market impact (primary) |
| $\$1\mathrm{B}$ | $A = \$1\mathrm{B}$ | Half-spread $+$ market impact |

In the $\mathrm{PropTC}$ scenario the impact term is suppressed, so the
cost rate collapses to the half-spread and net returns are
size-independent; the three dollar scenarios trace out the cost–capacity
profile across realistic deployment sizes, with $\$500\mathrm{M}$ as the
primary case. These reporting scenarios are distinct from the
training-time scenario capital that indexes the cost-aware sample weights
in the companion training appendix; only the latter enters the
model-fitting loss.

## 8. Reported statistics and decompositions

### 8.1 Portfolio time-series statistics

For each cell $c \in \{1A, 1B, 2A, 2B\}$ — long–short book, single leg,
or prediction quantile — and each scenario
$A \in \{\mathrm{PropTC}, \$100\mathrm{M}, \$500\mathrm{M}, \$1\mathrm{B}\}$,
statistics are built from the monthly series of Sections 4 and 7: the
gross return $R^{\mathrm{LS}}_{c,t}$ and the leg-level series

$$
R^{L}_{c,t} \;=\; \sum_{i \in \mathcal{L}_t} w^{L}_{i,t}\, r_{i,t},
\qquad
R^{S}_{c,t} \;=\; \sum_{i \in \mathcal{S}_t} w^{S}_{i,t}\, r_{i,t},
\qquad
R^{\mathrm{LS}}_{c,t} \;=\; R^{L}_{c,t} - R^{S}_{c,t},
$$

together with the one-way turnover $T_{c,t}$, the cost series
$\mathrm{TC}_{c,t}(A)$, and the net return
$R^{\mathrm{LS,net}}_{c,t}(A) = R^{\mathrm{LS}}_{c,t} - \mathrm{TC}_{c,t}(A)$.
The exported short-leg series (`ret_short` in the
`two_by_two_timeseries_{aum}` workbook) is the **raw long-only return of
the bottom-quantile basket** $R^{S}_{c,t}$; the P&L contribution of the
short *position* is its negative, and the book return is the difference
above. Transaction costs are computed for the book as a whole — both legs'
trades enter the single series $\mathrm{TC}_{c,t}(A)$; the true-book cells
carry no leg-level cost attribution (Section 8.4). Prediction-quantile
tables remain standalone long-only books. The four scenarios produce four
net-return series per cell; the gross series is invariant to $A$.

**Aggregation conventions.** For a monthly series $\{X_t\}_{t=1}^{T}$,
$\overline{X}$ and $\hat\sigma_X$ denote the sample mean and standard
deviation. Tabulated returns, costs, and turnover in the summary CSVs are
means of the monthly **decimal** series; percentage-point columns
(suffixed `_pct`) appear only in the timeseries workbooks. Only Sharpe
ratios and factor-model alphas are annualized:

$$
\mathrm{SR}^{\mathrm{gross}}_{c} \;=\;
  \sqrt{12}\,\frac{\overline{R^{\mathrm{LS}}_{c}}}{\hat\sigma_{R^{\mathrm{LS}}_{c}}},
\qquad
\mathrm{SR}^{\mathrm{net}}_{c}(A) \;=\;
  \sqrt{12}\,\frac{\overline{R^{\mathrm{LS,net}}_{c}(A)}}
                  {\hat\sigma_{R^{\mathrm{LS,net}}_{c}(A)}}.
$$

For each $(c, A)$ pair we tabulate mean monthly gross and net returns,
annualized gross and net Sharpe ratios, mean monthly turnover and
transaction cost, and annualized factor-model alphas (CAPM, Fama–French
three- and five-factor, five-factor plus momentum) estimated by ordinary
least squares on the monthly net-return series. Monthly leg counts are not
part of the summary tables; they are available as the `n_long`/`n_short`
columns of the timeseries workbook sheets.

### 8.2 Aggregate effect decomposition

The training-vs-execution decomposition on the $2 \times 2$ grid, in
annualized net-Sharpe units (the decomposition is computed on both the net
and gross series, with net as the headline), is

$$
\underbrace{\mathrm{SR}(2B) - \mathrm{SR}(1A)}_{\Delta_{\text{tot}}}
\;=\;
\underbrace{\mathrm{SR}(2A) - \mathrm{SR}(1A)}_{\Delta_{\text{train}}}
\;+\;
\underbrace{\mathrm{SR}(1B) - \mathrm{SR}(1A)}_{\Delta_{\text{port}}}
\;+\;
\underbrace{\Delta_{\text{tot}} - \Delta_{\text{train}} - \Delta_{\text{port}}}_{\Delta_{\text{int}}}.
$$

$\Delta_{\text{train}}$ is the pure training effect (weighted vs. standard
predictions under plain execution), $\Delta_{\text{port}}$ the pure
execution effect (hysteresis vs. full rebalance under standard training),
and $\Delta_{\text{int}}$ the interaction residual.

**Inference** (`compute_effect_decomposition` in
`src/evaluation/statistics.py`). Each Sharpe difference is tested with the
Ledoit–Wolf (2008) studentized circular-block bootstrap (Politis and
Romano, 1992) with prewhitened Parzen-kernel HAC standard errors using the
Andrews (1991) bandwidth (`bootstrap_sharpe_test`; 5,000 draws, block-size
grid $\{2, 4, 6, 8, 10\}$, sidedness set in the `inference` block of
`config/config.yaml`). Four tests are reported: `lw_training`
($2A$ vs. $1A$), `lw_portfolio` ($1B$ vs. $1A$), `lw_total`
($2B$ vs. $1A$), and the one-sided hypothesis `lw_h3` that
$\mathrm{SR}(2A) > \mathrm{SR}(1B)$ — liquidity-aware *training* beats the
cost-aware *execution* fix. Factor alphas are estimated per cell. The
shared workbook renderer (`src/analysis/common/two_by_two_block.py`)
presents Panel A (net and gross Sharpe per cell), Panel B (the
$2 \times 2$ decomposition with `*` marking Ledoit–Wolf $p < 0.05$), and
Panel C diagnostics.

### 8.3 Per-prediction-quantile decomposition

To assess whether the training and execution effects concentrate in
particular regions of the prediction distribution, the standalone
quantile books of Section 4.1 are each treated as their own long-only
portfolio, with A/B columns given by the plain quantile assignment and
the per-quantile sticky hysteresis of Section 4.3. Net returns follow
Section 7 (full-AUM sizing per Section 6), and the $2 \times 2$
decomposition of Section 8.2 is evaluated within each quantile:

$$
\mathrm{SR}^{\mathrm{net}}_{2B,q} - \mathrm{SR}^{\mathrm{net}}_{1A,q}
\;=\;
\bigl[\mathrm{SR}^{\mathrm{net}}_{2A,q} - \mathrm{SR}^{\mathrm{net}}_{1A,q}\bigr]
+
\bigl[\mathrm{SR}^{\mathrm{net}}_{1B,q} - \mathrm{SR}^{\mathrm{net}}_{1A,q}\bigr]
+ \text{Interaction}_q .
$$

*Status.* The driver currently exports the per-cell quantile time series
(`prediction_quantile_timeseries_{aum}`) for all four cells; the tabulated
per-quantile decomposition workbook is produced by script `22`, which has
not yet been realigned to the $2 \times 2$ layout (Section 10). The
decomposition above is therefore the design of record, pending that
realignment.

This diagnostic is essential, not redundant with the aggregate long–short
summary. The long–short differences only the top and bottom quintiles, so
any effect that concentrates in the middle of the distribution
($q = 2, 3, 4$) is mechanically invisible to it: a training improvement
that sharpens the ordering of mid-conviction names contributes nothing to
a $\mathcal{Q}_5 - \mathcal{Q}_1$ spread. The per-quantile decomposition
surfaces such localized effects directly — a training effect that is
large at $q = 5$ indicates sharper ranking of the most attractive long
candidates, while concentration in middle quantiles indicates improved
ordering of mid-conviction stocks without changing the extreme tails.

### 8.4 Per-leg decomposition

For the long–short books, a complementary decomposition attributes each
headline effect across the two sides of the trade. The long side is the
contribution $R^{L}_{c,t}$ of the selected long book; the short side
contributes with the opposite sign of the raw bottom-quantile basket
return, $-R^{S}_{c,t}$, consistent with the book identity
$R^{\mathrm{LS}}_{c,t} = R^{L}_{c,t} - R^{S}_{c,t}$ of Section 8.1. In the
current true-book cells transaction costs are computed for the whole book,
so no leg-level net return is defined there; leg-level cost attribution
($\mathrm{TC}^{L}_t$, $\mathrm{TC}^{S}_t$) exists only in the legacy
quantile-derived path used by the two-sided workbook script (`22b`), which
is not part of the current design. The $2 \times 2$ effect decomposition
of Section 8.2, computed within each side and for the combined book, is
therefore also pending the realignment of script `22` (Section 10).

Interpreted at the design level, concentration of the training effect on
the long side indicates that liquidity-weighted training improves
selection primarily among expected winners; concentration on the short
side indicates gains from better identification of expected losers.
Sections 8.3 and 8.4 are complementary: the former spans the entire
prediction distribution at fixed five-quantile granularity, the latter
restricts attention to the extremes actually held in the deployable
strategy.

## 9. Relation to the eval-realignment track

The sorted book of this appendix is the eval-realignment consistency
principle — evaluate the model under the same deployment weighting that
defines its training objective — applied to the literature's standard
portfolio object, the quantile sort. The signal-weighted legs of Section 3
hold each selected name in proportion to $\tilde w_{i,t}$, exactly as the
capacity portfolios of the eval-realignment track weight positions by the
deployment weight itself.

Three connections anchor the two tracks. First, the **equal $\times$ full
grid cell** (equal-weighted legs on the unscreened universe) connects to
the conventional pre-rework scoreboard: its A-column **gross** return
series reproduces the pre-rework sort exactly. Net-of-cost replication
additionally requires the legacy `--leg-capital full` sizing, because the
pre-rework long–short was derived from standalone quantile books each
sized at the full scenario AUM, whereas the current default splits the
gross capital across the two legs (Section 6); and the pre-rework B
(TC-aware sort) and C (tc-target) columns are different treatments with no
counterpart in the current grid. The historical record lives in
`docs/formal_analysis_softmax_rank_lam2.md` and
`docs/formal_portfolio_spec_sweep.md`, both marked superseded. Second, the
eval-realignment presentation layer
(`scripts/eval_realignment/44_capacity_two_by_two_tables.py`) deliberately
mirrors the deliverable formats of this appendix's $2 \times 2$ —
"analogous to the formalanalysis decomposition, with the tc-target column
out of scope" (`docs/eval_realignment_pipeline.md`, §4.4) — including the
long `metric, value` layout inherited from the legacy
`two_by_three_{AUM}.csv`. Third, the **dose–response contrast** between
this sorted book and the capacity book of scripts 42–45 — the same
predictions evaluated under progressively more deployment-consistent
constructions — is the paper's Section 6.1 exhibit.

## 10. Implementation and output layout

**Machinery.** Core construction:
`src/portfolio/construction.py` (`build_long_short_portfolio`,
`build_portfolio_timeseries`, `build_prediction_quantile_timeseries` at
`construction.py:641`, hysteresis helpers `_apply_hysteresis_bands` and
`_apply_quantile_hysteresis`, cost engine
`_compute_net_returns_with_context`). Decomposition layer:
`src/analysis/formal/portfolio_tables.py`
(`compute_two_by_two_decomposition`, cell builder
`_build_long_short_cell`). Inference:
`src/evaluation/statistics.py` (`compute_effect_decomposition`,
`bootstrap_sharpe_test`). Workbook rendering:
`src/analysis/common/two_by_two_block.py`.

**Configuration** (`config/config.yaml`): quantile block
(`n_quantiles`, `long_quantile`, `short_quantile`; lines 264–267),
`portfolio.weighting: "signal"` (line 273),
`portfolio.liquidity_screen_pct: 0.60` (line 277), the universe block
(lines 278–281), and `transaction_costs.aum_scenarios` (lines 300–304).

**Driver.** `scripts/21e_formal_portfolio_decomposition.py` builds all
cells from the formal prediction cache (no retraining). CLI: `--model`,
`--weight-spec`,
`--portfolio-weighting {equal, value, dolvol, signal, both, all}`
(`both` = equal $+$ value, `all` = equal $+$ value $+$ signal; default
from config: `signal`), `--universe {full, nyse, top40, all}` (default
from config: `all`), `--leg-capital {gross, full}` (default `gross`),
`--aum` (default: all scenarios).

**Output directories** (run naming `_portfolio_run_name`:
`equal → prediction_quantile`,
`value → prediction_quantile_value_weight`,
`dolvol → prediction_quantile_dolvol_weight`,
`signal → prediction_quantile_signal_weight`):

```text
outputs/formalanalysis/analysis/{model}/{weight_spec}/
    {prediction_quantile|prediction_quantile_value_weight|
     prediction_quantile_dolvol_weight|prediction_quantile_signal_weight}/
        stock_universe/{full|nyse|top40}/
            two_by_two_{PropTC|100M|500M|1B}.csv     # four cells, no C column
            two_by_two_timeseries_{aum}.xlsx         # one sheet per cell
            prediction_quantile_timeseries_{aum}.csv / .xlsx
```

The earlier `prediction_quantile_dolvol_weight_liq60` namespace is
superseded for 21e outputs: the liquidity screen moved onto the universe
axis, and the current screen keeps the top 60% (liquidity_screen_pct: 0.40 = keep dvol percentile >= 0.40)
(`liquidity_screen_pct: 0.60`), superseding the top-60% screen of the
earlier iteration. Script `22` still defaults to the liq60 run name (see
below).

**Workbooks.** Script `scripts/22_prepare_portfolio_excel_tables.py` is
the historical writer of the per-quantile and leg-decomposition workbooks
under
`outputs/formalanalysis/tables/{model}/{stock_universe}/{table13|table14}_{specs}_{portfolio_run}_{aums}.xlsx`.
**It has not yet been realigned to the $2 \times 2$ rework** and cannot
consume the 21e outputs described above: its per-quantile loader requires
all six cells of the old $2 \times 3$ layout, including the removed
1C/2C cells; its run choices omit `prediction_quantile_signal_weight` and
its `--portfolio-run` default remains the pre-rework
`prediction_quantile_dolvol_weight_liq60`; and its `--stock-universe` axis
is the legacy `{full_sample, nyse}` namespace, which does not resolve the
`{full, nyse, top40}` folders that 21e now writes. Until script 22 is
repointed, no code path produces the Section 8.3/8.4 decomposition
workbooks from the rebuilt pipeline; the 21e per-cell timeseries exports
are the current per-quantile deliverable. Script
`scripts/22b_table12_two_sided.py` writes the legacy two-sided workbooks
`tables/{model}/{stock_universe}/table12_two_sided_[halfleg_]{weighting}[_liqNN].xlsx`
plus `table13_legs_two_sided_*` and `table14_legdecomp_two_sided_*`; note
that 22b implements the pre-rework $2 \times 3$ tc-target design
(Section 4.4) and retains the legacy per-leg-capital convention, so its
outputs are historical companions rather than cells of the current design.

## References

Andrews, D. W. K. (1991). "Heteroskedasticity and Autocorrelation
Consistent Covariance Matrix Estimation." *Econometrica* 59(3), 817–858.

Frazzini, A., R. Israel, and T. J. Moskowitz (2018). "Trading Costs."
Working Paper, AQR Capital Management.

Ledoit, O., and M. Wolf (2008). "Robust Performance Hypothesis Testing
with the Sharpe Ratio." *Journal of Empirical Finance* 15(5), 850–859.

Politis, D. N., and J. P. Romano (1992). "A Circular Block-Resampling
Procedure for Stationary Data." In R. LePage and L. Billard (eds.),
*Exploring the Limits of Bootstrap*, Wiley, 263–270.
