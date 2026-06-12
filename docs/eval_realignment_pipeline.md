# Realigned Portfolio Evaluation

This appendix documents the *evaluation-realignment* track, which re-evaluates the
existing liquidity-weighted prediction models against the quantity their training
objective actually optimizes. The companion note
(`LiquidityML_Portfolio_Issue.pdf`) argues that the headline $Q_5 - Q_1$
long–short metric does not measure the deployment-weighted prediction error that
importance-weighted training minimizes, and proposes two complementary evaluations:
(i) *deployment-weighted prediction metrics* (note §4.1) and (ii) a
*signal-weighted capacity portfolio* (note §4.2). Both are implemented as an
isolated track that **reads the existing formal predictions** (no retraining) and
writes under `outputs/eval_realignment/`; the formalanalysis pipeline (scripts
`20`–`22b`) is left untouched.

## 1. Notation

Let $\mathcal{U}_t$ be the cross-section in month $t$. For stock
$i \in \mathcal{U}_t$:

- $r_{i,t}$ is the realized excess return earned over the holding period
  (the model target, `excess_ret`).
- $\hat{r}_{i,t}$ is the one-month-ahead return prediction. We write
  $\hat{r}^{\,\mathrm{std}}_{i,t}$ for the baseline (uniform-weight) model and
  $\hat{r}^{\,w}_{i,t}$ for the liquidity-weighted model.
- $\tilde{w}_{i,t}$ is the **same implementability weight that defined training**,
  normalized to monthly mean one ($\mathcal{N}_t^{-1}\sum_i \tilde{w}_{i,t} = 1$).
  Each formal specification carries its own family — dollar-volume, softmax-rank,
  transaction-cost, or transaction-cost-rank — as defined in the
  sample-weighting appendix. For the dollar-volume specification
  $\tilde{w}_{i,t} \propto \mathrm{DolVol}_{i,t}$.

The consistency principle of the note (§3) is that the *same* $\tilde{w}$ enters
both the training loss and the evaluation; the two sections below apply it to a
prediction metric and to a portfolio object, respectively.

## 2. Deployment-weighted prediction metrics

These are the literal empirical counterpart of the importance-weighted training
loss. They compare the standard and weighted predictions under the *same*
$\tilde{w}$, i.e. they isolate the training effect at the prediction level.

### 2.1 Deployment-weighted out-of-sample $R^2$

The deployment-weighted (zero-benchmark) out-of-sample $R^2$ of a prediction set is

$$
R^2_{\tilde{w}}
  \;=\; 1 \;-\;
  \frac{\sum_t \sum_{i} \tilde{w}_{i,t}\,\bigl(r_{i,t} - \hat{r}_{i,t}\bigr)^2}
       {\sum_t \sum_{i} \tilde{w}_{i,t}\,r_{i,t}^2}.
\tag{1}
$$

This is the empirical counterpart of minimizing $\sum_i \tilde{w}_{i,t}(r_{i,t+1} -
\hat{r}_{i,t})^2$ (note Eq. 1) and is reported for both $\hat{r}^{\,\mathrm{std}}$
and $\hat{r}^{\,w}$; the **training effect** is
$\Delta R^2_{\tilde{w}} = R^2_{\tilde{w}}(\hat{r}^{\,w}) -
R^2_{\tilde{w}}(\hat{r}^{\,\mathrm{std}})$. The zero benchmark matches the project's
primary $R^2$ convention; since the metric is a ratio, it is invariant to a global
rescaling of $\tilde{w}$, so subsets are directly comparable.

### 2.2 Deployment-weighted squared-error differential

To track the effect through time we report the monthly $\tilde{w}$-weighted mean
squared error of each model,

$$
\mathrm{wMSE}_t(\hat{r})
  \;=\;
  \frac{\sum_{i} \tilde{w}_{i,t}\,\bigl(r_{i,t} - \hat{r}_{i,t}\bigr)^2}
       {\sum_{i} \tilde{w}_{i,t}},
\tag{2}
$$

and the differential
$D_t = \mathrm{wMSE}_t(\hat{r}^{\,\mathrm{std}}) -
\mathrm{wMSE}_t(\hat{r}^{\,w})$, so $D_t > 0$ means weighted training lowered the
deployment-weighted error in month $t$. Inference on the mean differential uses
Newey–West HAC standard errors (6 lags); the cumulative series $\sum_{s \le t} D_s$
is reported and plotted.

### 2.3 Evaluation universes

Both metrics are reported per liquidity quintile $Q_1$–$Q_5$ (NYSE or full-sample
breakpoints on $\tilde{w}$'s liquidity measure), for the pooled liquid set
$Q_4\!-\!Q_5$, and for the full cross-section, mirroring the formalanalysis
`r2_by_quintile` convention. Quintile membership only partitions the sum in
$(1)$–$(2)$; the weight $\tilde{w}$ itself is the global mean-one weight.

### 2.4 Implementation

Script `scripts/eval_realignment/41_deployment_weighted_prediction_metrics.py`
reuses `compute_utility_weighted_r2` for $(1)$ and a $\tilde{w}$-weighted analogue
of the liquid-error-differential for $(2)$
(`src/analysis/eval_realignment/deployment_weighted_metrics.py`). CLI:
`--model`, `--weights`, `--weight-spec`, `--liquidity-breakpoints {nyse,full_sample,both}`,
`--no-figures`. Per spec and breakpoint mode it writes, under
`outputs/eval_realignment/analysis/{model}/{weight_spec}/liquidity_breakpoints/{mode}/`:
`deployment_weighted_r2.csv` (one row per universe), `deployment_weighted_error_diff.csv`
(stacked monthly series), `deployment_weighted_error_diff_stats.csv` (Newey–West per
universe), and `deployment_weighted_error_diff_quintiles.png` (the $Q_1$–$Q_5$
cumulative-differential overlay).

## 3. Signal-weighted capacity portfolio

The capacity portfolio is the note's preferred *economic* test. It replaces the
quantile long–short with a continuous, dollar-neutral book whose position size
scales with capacity and whose sign is the centered signal, so the *same*
$\tilde{w}$ enters both loss and portfolio.

### 3.1 Construction

For each month, define the unnormalized holdings

$$
\tilde{\theta}_{i,t}
  \;=\; \tilde{w}_{i,t}\,\bigl(\hat{r}_{i,t} - \bar{r}^{\,w}_t\bigr),
\qquad
\bar{r}^{\,w}_t
  \;=\; \frac{\sum_{i} \tilde{w}_{i,t}\,\hat{r}_{i,t}}{\sum_{i} \tilde{w}_{i,t}},
\tag{3}
$$

where $\bar{r}^{\,w}_t$ is the $\tilde{w}$-weighted cross-sectional mean of the
predictions. For the dollar-volume specification $(3)$ is exactly the note's Eq. 3.
A stock is held long when its predicted return exceeds the capacity-weighted
average and short when it falls below, with size scaling in $\tilde{w}$. There is no
quantile cut, no equal-weighting of microcaps, and no portfolio-stage cost sort.

### 3.2 Dollar-neutrality and leverage normalization

Centering on $\bar{r}^{\,w}_t$ makes the book dollar-neutral by construction:

$$
\sum_{i} \tilde{\theta}_{i,t}
  \;=\; \sum_{i} \tilde{w}_{i,t}\,\hat{r}_{i,t}
       \;-\; \bar{r}^{\,w}_t \sum_{i} \tilde{w}_{i,t}
  \;=\; 0 .
\tag{4}
$$

(Centering on the equal-weighted mean would leave a time-varying net tilt that
contaminates the Sharpe ratio.) Because $(3)$ defines the book only up to scale, we
fix unit **gross** exposure,

$$
\theta_{i,t} \;=\; \frac{\tilde{\theta}_{i,t}}{\sum_{j} |\tilde{\theta}_{j,t}|},
\qquad \sum_i |\theta_{i,t}| = 1 ,
\tag{5}
$$

so that $A$ (assets under management) is the gross capital deployed. Fixing the
leverage is necessary because the certainty-equivalent and the net-of-cost drag are
not scale-invariant (the Sharpe ratio is).

### 3.3 Gross and net returns

The gross monthly return is $R_t = \sum_i \theta_{i,t}\, r_{i,t}$. Between
rebalances the carried book drifts with realized returns and is renormalized to
unit gross,

$$
\theta^{\mathrm{drift}}_{i,t}
  \;=\; \frac{\theta_{i,t-1}\,(1 + r_{i,t})}{\sum_j |\theta_{j,t-1}\,(1 + r_{j,t})|},
$$

and the realized net return charges a one-way transaction cost on the full traded
notional $\Delta\theta_{i,t} = |\theta_{i,t} - \theta^{\mathrm{drift}}_{i,t}|$:

$$
R^{\mathrm{net}}_t
  \;=\; R_t \;-\; \sum_i \Delta\theta_{i,t}\,\tau_{i,t},
\qquad
\tau_{i,t}
  \;=\; \tfrac{1}{2}\,\text{Spread}_{i,t}
       \;+\; \lambda\,\sigma_{i,t}\sqrt{\frac{\Delta\theta_{i,t}\,A}{\text{ADV}_{i,t}}},
\tag{6}
$$

following the transaction-cost model of the portfolio-construction appendix
(Frazzini et al., 2018). The market-impact term is omitted under the
proportional-cost scenario ($A = 0$). Reported turnover is the standard diagnostic
$T_t = \tfrac{1}{2}\sum_i \Delta\theta_{i,t}$; note that the cost in $(6)$ uses the
*full* $\sum_i \Delta\theta_{i,t}$ (a one-way cost is paid on every dollar traded),
not the halved turnover figure.

### 3.4 Performance metrics

For each model we report the annualized gross and net Sharpe ratio
($\sqrt{12}\,\mathbb{E}[R]/\mathrm{sd}[R]$), the annualized mean return, turnover,
and the mean-variance certainty-equivalent

$$
\mathrm{CE}(\gamma) \;=\; \mathbb{E}[R] - \tfrac{1}{2}\,\gamma\,\mathrm{Var}[R],
\qquad \gamma \in \{1, 5, 10\},
\tag{7}
$$

evaluated on monthly net returns and annualized by a factor of 12. Metrics are
reported across the AUM grid $A \in \{\text{PropTC},\, \$100\mathrm{M},\,
\$500\mathrm{M},\, \$1\mathrm{B}\}$.

### 3.5 Training comparison and inference

The capacity book is built from both $\hat{r}^{\,\mathrm{std}}$ and $\hat{r}^{\,w}$
using the spec's own $\tilde{w}$; the **training effect** is the difference
(weighted $-$ standard) in net Sharpe and certainty-equivalent. The Sharpe-ratio
difference is tested with the Ledoit–Wolf (2008) studentized circular-block
bootstrap, and the mean net-return difference with Newey–West $t$-statistics.

### 3.6 Implementation

Script `scripts/eval_realignment/42_signal_weighted_capacity_portfolio.py` builds
the book over the **full cross-section** (no quintile/breakpoint split) and reuses
the realized-cost primitives (`prepare_transaction_cost_context` and the Frazzini
formula); only the unified signed-book net-return loop and the
certainty-equivalent are new (`src/analysis/eval_realignment/capacity_portfolio.py`).
CLI: `--model`, `--weights`, `--weight-spec`, `--aum`, `--no-figures`. Per spec and
AUM it writes, under `outputs/eval_realignment/analysis/{model}/{weight_spec}/`:
`capacity_portfolio_metrics_{aum}.csv` (rows: standard, weighted, difference),
`capacity_portfolio_monthly_{aum}.csv` (stacked monthly gross/net series), and
`capacity_portfolio_net_cumret_{aum}.png`.

## 4. Breakeven-gated capacity portfolio

The capacity book of §3 rebalances fully to its target every month, so it trades on
transient signal noise and, at institutional AUM, the cost drag of $(6)$ can absorb
the entire gross premium. This section adds a *parameter-free execution layer* that
gates each name's **trade** — not its signal — by the stock's own breakeven.

### 4.1 The breakeven gate

A trade adds value only if the edge it captures exceeds the cost of capturing it.
Both sides are observable in the same units (monthly returns): the deployment alpha
$\alpha_{i,t} = \hat{r}_{i,t} - \bar{r}^{\,w}_t$ — the centered signal the
dollar-neutral book harvests, from $(3)$ — and the one-way proportional cost
$\tfrac{1}{2}\text{Spread}_{i,t}$, the cost that $(6)$ charges per unit traded. At
each rebalance the executed position is

$$
\theta^{\mathrm{exec}}_{i,t}
  \;=\;
  \begin{cases}
    \theta_{i,t}, & |\alpha_{i,t}| \;\ge\; \tfrac{1}{2}\,\text{Spread}_{i,t},\\[2pt]
    \theta^{\mathrm{drift}}_{i,t}, & \text{otherwise},
  \end{cases}
\tag{8}
$$

i.e. a name trades *fully* to its §3 target when its alpha clears its own cost and
otherwise holds the drifted position (a name with no existing position is simply
not opened). Under proportional costs the one-period trade-or-not decision is
bang-bang — trading toward the target is optimal precisely when
$|\alpha| \ge \tfrac{1}{2}\text{Spread}$ — so $(8)$ is the closed-form myopic
optimum rather than a heuristic, and it contains no tuning parameter: the threshold
*is* the cost. The gate is independent of $A$; market impact remains in the realized
net-return accounting of $(6)$.

Names that leave the prediction universe are closed (the closing trade is charged
in $(6)$), and the gated book is rescaled to half-unit long and short legs.
Unlike the full-rebalance book — whose raw holdings sum to zero by construction,
so the single normalization factor of $(5)$ automatically yields $\pm 0.5$ legs —
the *gated* raw book is generally not dollar-neutral (held drifts and forced
closes break the sum), so the two legs are rescaled **independently** to $+0.5$
and $-0.5$. This preserves within-leg relative weights and re-imposes both
$\sum_i \theta_i = 0$ and $\sum_i |\theta_i| = 1$, leaving the two books on the
same scale; in the all-pass limit the per-leg rescale coincides with $(5)$.
Degenerate months carry the previous executed book into the next month's drift.
By construction the gate nests §3: as $\text{Spread} \to 0$ every name passes and
the gated book coincides with the full-rebalance book.

### 4.2 Diagnostics and caveats

Per month we report the count pass fraction and the gross-weighted pass fraction
$\sum_i |\theta_{i,t}|\,\mathbf{1}\{|\alpha_{i,t}| \ge \tfrac{1}{2}\text{Spread}_{i,t}\}$
(the share of target gross exposure that clears its breakeven), together with the
median $|\alpha|$ and median half-spread. Empirically the threshold bites in the
interior of the alpha distribution and the gross-weighted pass fraction exceeds the
count fraction — the gate trims the low-conviction tail while keeping the names that
carry the book.

Three caveats are accepted by design. First, the test is *myopic*: the cost is paid
once while the alpha accrues over the holding period, so a name with
$|\alpha| < \tfrac{1}{2}\text{Spread}$ could still break even over
$H^* = \tfrac{1}{2}\text{Spread}/|\alpha|$ months; amortizing would require a
signal-persistence parameter, which the design deliberately avoids — the gate is
therefore conservative. Second, the rule is bang-bang, so names near the threshold
alternate between full trades and inaction. Third, the gate binds asymmetrically
across the training comparison: flatter predictions clear the cost threshold less
often, so part of the gated standard-versus-weighted difference reflects *whose
alpha clears costs* — itself an economically meaningful margin, made transparent by
the per-book diagnostics.

### 4.3 Implementation

Script `scripts/eval_realignment/43_breakeven_capacity_portfolio.py` builds the
full-rebalance (§3) and breakeven-gated books from both prediction sets
(`src/analysis/eval_realignment/breakeven_capacity.py`; the §3 primitives are reused
unchanged). CLI: `--model`, `--weights`, `--weight-spec`, `--aum`, `--no-figures`.
Per spec it writes, under `outputs/eval_realignment/analysis/{model}/{weight_spec}/`:
`capacity_breakeven_metrics_{aum}.csv` (book $\times$ standard/weighted/difference,
same metric schema as §3.4–3.5), `capacity_breakeven_monthly_{aum}.csv` (gated-book
monthly series), `capacity_breakeven_gate_diag.csv` (per-month pass fractions,
AUM-independent), and `capacity_breakeven_net_cumret_{aum}.png`.

### 4.4 Two-by-two deliverables (script 44)

The full-rebalance book of §3 and the breakeven-gated book of §4 form a
$2\times2$ design analogous to the formalanalysis decomposition, with the
tc-target column out of scope:

|  | $A$ = full rebalance (§3) | $B$ = breakeven gate (§4) |
|---|---|---|
| **1 = standard training** | 1A | 1B |
| **2 = weighted training** | 2A | 2B |

Script `scripts/eval_realignment/44_capacity_two_by_two_tables.py` is a
presentation layer that reads the monthly series written by scripts 42 and 43
(no retraining, no book reconstruction), aligns the four cells on their common
months, and reproduces the formalanalysis deliverable formats
(`src/analysis/eval_realignment/two_by_two_tables.py`):

- `two_by_two_{aum}.csv` — the long `metric, value` format of
  `two_by_three_{AUM}.csv`, with per-cell return/Sharpe/cost/turnover rows and
  the effect decomposition on net annualized Sharpe ratios: the *training
  effect* $\mathrm{SR}(2A)-\mathrm{SR}(1A)$, the *portfolio effect*
  $\mathrm{SR}(1B)-\mathrm{SR}(1A)$ (here the breakeven-gate effect), the
  *total effect* $\mathrm{SR}(2B)-\mathrm{SR}(1A)$, and the *interaction*
  (total $-$ training $-$ portfolio), each with Ledoit–Wolf bootstrap
  $p$-values; factor alphas (CAPM, FF3, FF5, FF5+Mom) are reported per cell.
  Metric names are kept verbatim from the formalanalysis format.
- `two_by_two_timeseries_{aum}.xlsx` — one sheet per cell (1A/1B/2A/2B) with
  the book-level monthly series (gross/net return, transaction cost, turnover,
  leg counts).
- `outputs/eval_realignment/tables/{model}/capacity_two_by_two_tables.xlsx` —
  the Table-12-style formatted workbook (one sheet per weight specification,
  AUM blocks stacked): Panel A (net and gross Sharpe ratios per cell), Panel B
  (the $2\times2$ decomposition with Ledoit–Wolf $p$-values), and Panel C
  (breakeven-gate diagnostics, replacing the old tc-target panel).

## 5. Capacity-weighted long-only book

The note's §4.3 calls a long-only capacity-weighted portfolio from the liquid
universe "the most defensible implementable object": it drops the illiquid short
leg — the least realistic component of any long–short backtest, since borrow
costs and locate constraints are not modeled — so every remaining position is
one a real fund could hold at the stated AUM.

### 5.1 Construction

Each month the **liquid universe** $L_t$ is the top 60% of the cross-section by
dollar volume (the note §5.1's screen), applied before any selection. With
$b_t$ the 80th percentile of $\hat{r}$ within $L_t$, the **Q5 target** is
$\{i \in L_t : \hat{r}_{i,t} \ge b_t\}$, held at the specification's own
capacity weights,

$$
\theta_{i,t} \;=\; \frac{\tilde{w}_{i,t}}{\sum_{j \in \mathrm{members}_t} \tilde{w}_{j,t}},
\qquad \sum_i \theta_{i,t} = 1,\;\; \theta_{i,t} \ge 0,
\tag{9}
$$

re-trued monthly. Weighting the leg by $\tilde{w}$ (rather than equally or by
market equity) restores the note's §3 consistency principle inside the leg.
Gross returns, drift, and realized net-of-cost returns reuse the §3 machinery
unchanged (the cost model $(6)$ charges the re-truing trades automatically);
degenerate months follow the §4 skip convention.

### 5.2 Cost-aware execution: membership hysteresis

A quantile book's turnover is **membership churn at the selection boundary** —
names whose predictions remain large but cross $b_t$ by small amounts — so the
§4 breakeven gate does not transfer (an $|\alpha| \ge \frac{1}{2}\text{Spread}$
test passes exactly the high-alpha boundary trades that need damping). The
correct parameter-free device is a **cost-scaled hysteresis band**: replacing a
held name $i$ with the marginal entrant earns $b_t - \hat{r}_{i,t}$ per month
and costs $\tfrac{1}{2}\text{Spread}_i + \tfrac{1}{2}\text{Spread}_{\text{entrant}}$,
so a held name that has left the Q5 target is retained while

$$
\hat{r}_{i,t} \;\ge\; b_t - \bigl(\tfrac{1}{2}\text{Spread}_{i,t} + \bar{c}_{m,t}\bigr),
\tag{10}
$$

with $\bar{c}_{m,t}$ the median half-spread over the month's Q5 target (the
entrant's toll). Entries are ungated; leaving the screen or the prediction
universe forces the sale; members are re-trued to $(9)$. As spreads $\to 0$
the buffer vanishes and the hysteresis book coincides with the plain book.
The threshold is again the actual trading cost in return units — no tuning
parameter.

### 5.3 Implementation

Script `scripts/eval_realignment/45_longonly_capacity_q5.py`
(`src/analysis/eval_realignment/longonly_capacity.py`) builds the four cells
1A/1B/2A/2B = training (standard/weighted) $\times$ book (plain/hysteresis) and
writes, per spec: `longonly_q5_metrics_{aum}.csv` (the §3.4–3.5 metric schema),
`longonly_q5_monthly_{aum}.csv` (stacked monthly series, all four cells),
`longonly_two_by_two_{aum}.csv` and `longonly_two_by_two_timeseries_{aum}.xlsx`
(the §4.4 old-format deliverables — factor alphas per cell provide the
beta-adjusted view a long-only Sharpe requires), `longonly_hysteresis_diag.csv`
(per-month membership/buffer diagnostics, AUM-independent), and
`longonly_net_cumret_{aum}.png`; plus the per-model formatted workbook
`outputs/eval_realignment/tables/{model}/longonly_two_by_two_tables.xlsx`
(Panel C = hysteresis diagnostics). In the long-only $2\times2$,
"portfolio effect" denotes the hysteresis effect $\mathrm{SR}(1B)-\mathrm{SR}(1A)$.

## 6. Implementable mean–variance frontier

The note's §4.5 proposes feeding the predictions into an implementable
mean–variance optimizer (Jensen, Kelly, Malamud, and Pedersen, 2024) and
reporting the net certainty-equivalent — a utility rather than an ad hoc
Sharpe. Of their three tiers, the one matched to our inputs (one-month-ahead
predictions only) is the *static* allocator: a one-period mean–variance
problem with a quadratic trading-cost penalty and the book inherited from the
previous month,

$$
\max_{\theta}\;\; \theta'\hat{\mu}_t
  \;-\; \frac{\gamma}{2}\,\theta'\Sigma_t\,\theta
  \;-\; \frac{1}{2}\,(\theta - \theta^{\mathrm{drift}}_{t-1})'\,\Lambda\,
        (\theta - \theta^{\mathrm{drift}}_{t-1}),
\tag{11}
$$

with the closed-form solution

$$
\theta_t \;=\; (\gamma\Sigma_t + \Lambda)^{-1}
               \bigl(\hat{\mu}_t + \Lambda\,\theta^{\mathrm{drift}}_{t-1}\bigr).
\tag{12}
$$

Here $\theta$ is in *wealth fractions* (JKMP's $\omega$): its scale is a choice
variable, so the book is not gross-normalized — net and gross exposure are
outputs, reported as leverage diagnostics — and the drift does not renormalize:
$\theta^{\mathrm{drift}} = \theta \odot (1+r) / (1 + \theta'r)$.

### 6.1 Inputs

- **Universe**: the within-month **top-60% of the cross-section by dollar
  volume** — identical to the §5 long-only screen and the note §5.1 — so the
  optimizer and the long-only book are evaluated on the same deployable set
  (JKMP analogously restrict to the top half of the market by capitalization).
  All four cells of the comparison share the same universe each month. A
  Ledoit–Wolf alternative (`--cov lw`, capped at the largest 1,000 names with
  36 months of history, since the unstructured estimate degrades with $N$)
  is retained as a robustness mode.
- **Covariance** $\Sigma_t$ — a monthly adaptation of JKMP's Barra-style
  characteristic-factor model,

$$
\Sigma_t \;=\; X_t\,\Omega^F_t\,X_t' \;+\; D_t ,
\tag{13}
$$

  where $X_t$ ($N\times K$, $K=9$) holds an intercept and eight broad
  characteristic-theme exposures (Value, Momentum, Profitability, Investment,
  Quality, Risk, Liquidity, Other — the feature-category clusters of
  `config/feature_categories.json`, the in-house analogue of JKMP's thirteen
  themes): a stock's theme exposure is the mean of its available
  rank-normalized features in the theme, z-scored cross-sectionally each
  month. Factor returns $f_s$ are monthly cross-sectional OLS coefficients of
  the (forward-shifted) excess return on $X_s$; they realize at $s{+}1$, so a
  rebalance at $t$ uses only $\{f_s : s \le t-1\}$. $\Omega^F_t$ is an EWMA of
  the factor returns with the monthly translation of JKMP's half-lives —
  variances 6 months, correlations 18 months (their 126/378 trading days) —
  recombined as $\mathrm{diag}(\sigma)\,C\,\mathrm{diag}(\sigma)$; $D_t$ is a
  per-name EWMA of squared residuals (half-life 6 months, floored), with
  names under 12 residual months assigned the cross-sectional median
  idiosyncratic variance (the analogue of JKMP's 12-month validity rule).
  $\Sigma_t$ is never formed densely: solves of $(12)$ use the **Woodbury
  identity** with a $K$-dimensional inner system, $O(NK^2)$ per month, which
  is what makes the top-60% universe tractable. Fidelity gaps versus the
  paper, by design: no industry factors (the panel carries no industry codes)
  and monthly rather than daily estimation.
- **Cost matrix** $\Lambda$ (diagonal): the quadratic (Kyle-lambda) form
  implied by linear price impact at the Frazzini calibration,
  $\Lambda_{ii} = 2\lambda\,\sigma_{i}\,A/\mathrm{ADV}_{i}$. The quadratic
  shape is what makes the first-order condition linear — the closed form of
  $(12)$; realized net returns are always charged with the *true* half-spread
  plus square-root-impact accounting of $(6)$, so the approximation is policed
  by the evaluation. The proportional-cost scenario is not run ($A=0$ makes
  $\Lambda$ vanish and the two columns coincide).

### 6.2 The comparison and reporting

Columns: $A$ = cost-blind Markowitz ($\Lambda = 0$, memoryless — the failing
baseline that answers whether the cost penalty is needed) versus $B$ =
cost-aware static allocator; rows = standard versus weighted predictions
through the identical allocator, $\Sigma$, and costs. The headline is the
**net certainty-equivalent** $\mathrm{CE}(\gamma)$ of $(7)$ with each book
built *and* evaluated at its own $\gamma \in \{10, 100, 500\}$ (reaching past
JKMP's base case $\gamma=10$ because raw, unshrunk $\hat{\mu}$ through an
unconstrained optimizer produces extreme leverage at low $\gamma$ — in
mean–variance the book scales as $1/\gamma$, so the grid traces the frontier
from aggressive to deployable sizes). Every book is reported in two **views**:
`raw` (the literal optimizer output — the JKMP frontier object, with gross
leverage reported as a diagnostic) and `unit_gross` (positions rescaled to
$\sum_i|\theta_i| = 1$ before the net-of-cost evaluation — the track-comparable
scale of §3–§5; Sharpe is scale-invariant, the certainty-equivalent is not).
The old-format $2\times2$ deliverables use the unit-gross view at the base
case $\gamma=10$. The implementable-frontier figure plots net mean against net
standard deviation (annualized) across $\gamma$ from the raw view, per book
and training row.

### 6.3 Implementation

Script `scripts/eval_realignment/46_implementable_mv_frontier.py`
(`src/analysis/eval_realignment/implementable_mv.py` and
`src/analysis/eval_realignment/factor_covariance.py`) runs month-outer so each
month's risk model and factorizations are shared across all books; the factor
state ($\Omega^F$, $D$) is updated with month $t$'s factor return only *after*
the month-$t$ solve, preserving the no-lookahead convention. Per spec it
writes `implementable_mv_metrics_{aum}.csv` (book $\times$ row $\times$
$\gamma$ $\times$ view with difference rows), `implementable_mv_monthly_{aum}.csv`
(stacked series incl. leverage), the old-format `mv_two_by_two_{aum}.csv` and
`mv_two_by_two_timeseries_{aum}.xlsx` (unit-gross view, $\gamma=10$),
`implementable_mv_frontier_{aum}.png`, `implementable_mv_diag.csv`, and the
per-model workbook `tables/{model}/implementable_mv_tables.xlsx` (Panel C =
optimizer diagnostics). CLI: `--aum` (impact-bearing scenarios, $M),
`--gamma`, `--cov {factor,lw}`, `--screen-pct` (factor mode), `--top-n`
(LW mode), `--no-figures`. The Ledoit–Wolf robustness mode writes
`*_lw`-suffixed files so it never overwrites the canonical factor outputs.
In factor mode the universe matches §5's screen,
so the §5/§6 exhibits share the deployable set; the full cross-section
remains covered by script 41, and level comparisons against scripts 42–43
still mix the device and the universe.

## 7. Output layout

```text
outputs/eval_realignment/analysis/{model}/{weight_spec}/
├── capacity_portfolio_metrics_{PropTC,100M,500M,1B}.csv      # script 42
├── capacity_portfolio_monthly_{PropTC,100M,500M,1B}.csv
├── capacity_portfolio_net_cumret_{PropTC,100M,500M,1B}.png
├── capacity_breakeven_metrics_{PropTC,100M,500M,1B}.csv      # script 43
├── capacity_breakeven_monthly_{PropTC,100M,500M,1B}.csv
├── capacity_breakeven_gate_diag.csv
├── capacity_breakeven_net_cumret_{PropTC,100M,500M,1B}.png
├── two_by_two_{PropTC,100M,500M,1B}.csv                      # script 44
├── two_by_two_timeseries_{PropTC,100M,500M,1B}.xlsx
├── longonly_q5_metrics_{PropTC,100M,500M,1B}.csv             # script 45
├── longonly_q5_monthly_{PropTC,100M,500M,1B}.csv
├── longonly_two_by_two_{PropTC,100M,500M,1B}.csv
├── longonly_two_by_two_timeseries_{PropTC,100M,500M,1B}.xlsx
├── longonly_hysteresis_diag.csv
├── longonly_net_cumret_{PropTC,100M,500M,1B}.png
├── implementable_mv_metrics_{100M,500M,1B}.csv               # script 46
├── implementable_mv_monthly_{100M,500M,1B}.csv
├── mv_two_by_two_{100M,500M,1B}.csv
├── mv_two_by_two_timeseries_{100M,500M,1B}.xlsx
├── implementable_mv_frontier_{100M,500M,1B}.png
├── implementable_mv_diag.csv
└── liquidity_breakpoints/{nyse,full_sample}/                 # script 41
    ├── deployment_weighted_r2.csv
    ├── deployment_weighted_error_diff.csv
    ├── deployment_weighted_error_diff_stats.csv
    └── deployment_weighted_error_diff_quintiles.png

outputs/eval_realignment/tables/{model}/
├── capacity_two_by_two_tables.xlsx                           # script 44
├── longonly_two_by_two_tables.xlsx                           # script 45
└── implementable_mv_tables.xlsx                              # script 46
```

Scripts `41`–`46` write disjoint paths and never overwrite each other's output;
script `44` reads the monthly series of `42` and `43` and must run after them
(scripts `45` and `46` are self-contained).

## 8. Scope and relation to the formalanalysis pipeline

This track is restricted to the standard-versus-weighted comparison (the $2\times2$
training axis); the transaction-cost-target column (cell C of the
portfolio-construction appendix) is out of scope, and there is no portfolio-stage
cost sort — the device the note (§5.2) moves away from. Cost-awareness enters
through $\tilde{w}$ (for the transaction-cost families), the realized net-of-cost
returns of $(6)$, and the execution-stage breakeven gate of §4 — which gates the
*trade* rather than the signal rank, and therefore cannot degenerate into a
liquidity sort. Predictions are read from
`outputs/formalanalysis/experiment/{model}/{weight_spec}/predictions.parquet`; the
track adds no training and modifies no formalanalysis code, configuration, or
output. The capacity-weighted long-only book (note §4.3) is implemented in §5
(script `45`), with the note §5.1 top-60% dollar-volume screen and
$\tilde{w}$-weighted legs. The remaining note variations — liquid-universe
dollar-volume-weighted *long–short* sorts (§4.4) and the full
implementable-utility / mean–variance optimizer (§4.5, Jensen et al., 2024) —
are deferred; factor-model alphas on the capacity and long-only books are
reported per cell in the §4.4/§5.3 deliverables.

## References

- Frazzini, A., R. Israel, and T. J. Moskowitz (2018). "Trading Costs." Working Paper.
- Gârleanu, N., and L. H. Pedersen (2013). "Dynamic Trading with Predictable Returns and Transaction Costs." *Journal of Finance* 68(6).
- Jensen, T. I., B. T. Kelly, S. Malamud, and L. H. Pedersen (2024). "Machine Learning and the Implementable Efficient Frontier." *Review of Financial Studies*.
- Ledoit, O., and M. Wolf (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis* 88(2).
- Ledoit, O., and M. Wolf (2008). "Robust performance hypothesis testing with the Sharpe ratio." *Journal of Empirical Finance* 15(5).
- Shimodaira, H. (2000). "Improving predictive inference under covariate shift by weighting the log-likelihood function." *Journal of Statistical Planning and Inference* 90(2).
