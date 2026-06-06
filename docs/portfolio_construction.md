# Portfolio Construction

This appendix describes the portfolio construction underlying the empirical
results. Portfolios are rebalanced monthly using one-month-ahead predictions
formed at the end of the prior month; reported performance is net of
transaction costs computed from realized turnover.

## 1. Notation

Let $\mathcal{U}_t$ denote the investable universe in month $t$. For each
stock $i \in \mathcal{U}_t$:

- $r_{i,t}$ is the realized excess return earned between $t-1$ and $t$.
- $\hat{r}_{i,t}$ is a one-month-ahead return prediction formed at $t-1$.
- $s_{i,t}$ is a prediction from a model whose target is a
  transaction-cost-adjusted return rather than the raw excess return.
- $\text{Spread}_{i,t}$ is the bid–ask spread, $\sigma_{i,t}$ is realized
  return volatility, and $\text{ADV}_{i,t}$ is average daily dollar volume.
- $A$ is total strategy capital (assets under management) under the
  transaction-cost scenario being reported.

We define the one-way proportional cost penalty used for portfolio sorting as

$$
c_{i,t} \;=\; \tfrac{1}{2}\,\text{Spread}_{i,t}.
$$

The market-impact component is deliberately excluded from $c_{i,t}$ at the
sorting stage; it is included separately when computing realized net returns
from turnover (Section 5).

## 2. Experimental design

Six portfolios are reported in a $2 \times 3$ factorial design that crosses
two model-training schemes with three ranking signals. The two training
schemes are a *baseline* model fitted with uniform observation weights and a
*liquidity-weighted* model fitted under a weight scheme indexed by parameter
$\theta$ (detailed in the companion training appendix). The three ranking
signals are: the raw model prediction, the prediction reduced by the one-way
cost penalty, and the prediction of a separate model trained to a
transaction-cost-adjusted target.

For each combination, the table below defines the score $z_{i,t}$ used to
rank stocks each month:

| Cell | Training scheme       | Ranking score $z_{i,t}$                  |
|:-----|:----------------------|:-----------------------------------------|
| 1A   | Baseline              | $\hat{r}^{\mathrm{base}}_{i,t}$          |
| 1B   | Baseline              | $\hat{r}^{\mathrm{base}}_{i,t} - c_{i,t}$|
| 1C   | Baseline (TC-target)  | $s^{\mathrm{base}}_{i,t}$                |
| 2A   | Liquidity-weighted    | $\hat{r}^{w}_{i,t}$                      |
| 2B   | Liquidity-weighted    | $\hat{r}^{w}_{i,t} - c_{i,t}$            |
| 2C   | Liquidity-weighted (TC-target) | $s^{w}_{i,t}$                   |

Comparisons across columns isolate the effect of incorporating transaction
costs into either the ranking signal (B vs. A) or the model's training target
(C vs. A); comparisons across rows isolate the effect of liquidity-weighted
training (row 2 vs. row 1).

## 3. Selected-stock weights

After stocks are selected into a leg or quantile $\mathcal{S}_t$, within-group
weights $w_{i,t}$ are formed to sum to one. We report two schemes:

$$
w_{i,t}^{\mathrm{EW}} \;=\; \frac{1}{|\mathcal{S}_t|}, \qquad
w_{i,t}^{\mathrm{VW}} \;=\;
  \frac{\mathrm{ME}_{i,t}}{\sum_{j \in \mathcal{S}_t} \mathrm{ME}_{j,t}},
$$

where $\mathrm{ME}_{i,t}$ is end-of-month market capitalization. When the
value-weight denominator is non-positive the equal-weight scheme is used as a
fallback.

## 4. Portfolio sorting

The universe is split into $Q = 5$ rank quintiles each month using the cell
score $z_{i,t}$. We employ two distinct constructions on top of this
quintile assignment, each serving a different analytical purpose: the
prediction-quintile portfolios of Section 4.1 are reported as a *diagnostic*
of how each $(\text{training} \times \text{sort})$ combination performs
across the entire prediction distribution, while the *deployable*
long–short portfolio is constructed by the two-sided sort of Section 4.2.

### 4.1 Prediction-quintile portfolios

For each cell $c$ and each prediction quintile $q \in \{1, \ldots, 5\}$,
the stocks selected into $\mathcal{Q}_q(z_c)$ constitute a standalone
long-only portfolio with within-group weights $w_{i,t}$ from Section 3:

$$
R_{c, q, t} \;=\; \sum_{i \in \mathcal{Q}_q(z_c)} w_{i, t}\, r_{i, t}.
$$

These five quintile portfolios are reported as five separate books rather
than combined into a single long–short series; their per-quintile effect
decomposition is described in Section 6.3. The reason a naive long–short
of the form $R_{c, 5, t} - R_{c, 1, t}$ is *not* used as the deployable
long–short — even though it could be constructed mechanically — is that
this construction is economically asymmetric for cells B and C and would
distort the cost-aware comparison; we develop the deployable long–short
separately in Section 4.2.

### 4.2 Two-sided TC-aware long–short construction

A naive long–short formed as $R_{c, 5, t} - R_{c, 1, t}$ from the same
score $z_{i,t}$ is asymmetric for cells B and C. The long side profits
from low cost, so subtracting the cost penalty $c_{i,t}$ from $\hat{r}$
correctly tilts the long ranking toward high expected, low cost names.
The short side, however, is selected as the *bottom* of the same score,
i.e., the *smallest* values of $\hat{r} - c$, which mechanically favors
stocks with both low predicted return *and* high cost. Economically, a
short position with predicted return $-\hat{r}_{i,t}$ pays the same
one-way cost penalty and should therefore be ranked (for sorting purposes)
by

$$
\tilde z^{\mathrm{short}}_{i,t} \;=\; \hat{r}_{i,t} + c_{i,t},
$$

not by $\hat{r}_{i,t} - c_{i,t}$. The deployable long–short construction
restores this symmetry by giving the short side its own ranking signal.
The long leg selects the top quintile of one score; the short leg selects
the bottom quintile of another:

| Cell        | Long score                | Short score                    |
|:------------|:--------------------------|:-------------------------------|
| 1A, 2A      | $\hat{r}_{i,t}$           | $\hat{r}_{i,t}$                |
| 1B, 2B      | $\hat{r}_{i,t} - c_{i,t}$ | $\hat{r}_{i,t} + c_{i,t}$      |
| 1C, 2C      | $s_{i,t}$                 | $s_{i,t} + 2\,c_{i,t}$         |

For the C cells, the TC-target score satisfies
$s_{i,t} \approx \hat{r}_{i,t} - c_{i,t}$, so $s_{i,t} + 2 c_{i,t}$
recovers the symmetric $\hat{r}_{i,t} + c_{i,t}$. Long and short sets are
kept disjoint; ties resolve to the long leg.

Letting $\mathcal{L}_t$ and $\mathcal{S}_t$ denote the selected long and
short legs and $w_{i,t}^{L}$, $w_{i,t}^{S}$ the within-leg weights from
Section 3, the gross long–short return is

$$
R^{\mathrm{LS}}_{c,t} \;=\;
  \sum_{i \in \mathcal{L}_t} w_{i,t}^{L}\, r_{i,t}
  - \sum_{i \in \mathcal{S}_t} w_{i,t}^{S}\, r_{i,t}.
$$

All long–short results reported in the main text use this two-sided
construction.

## 5. Transaction costs and net returns

Net returns are computed from realized turnover at a specified strategy size
$A$. Each leg of the long–short portfolio trades the full scenario AUM, so
a sized position deploys $A$ in long exposure and a separate $A$ in short
exposure, and per-stock dollar trade sizes scale with $A$ on each side.

Between rebalances, position weights drift with realized returns:

$$
\tilde w_{i,t} \;=\;
  \frac{w_{i,t-1}\,(1 + r_{i,t-1})}
       {\sum_{j} w_{j,t-1}\,(1 + r_{j,t-1})}.
$$

The monthly trade size in stock $i$, expressed both as a weight change and as
a dollar amount, is

$$
\Delta w_{i,t} \;=\; \bigl|\,w_{i,t}^{\mathrm{target}} - \tilde w_{i,t}\,\bigr|,
\qquad
\Delta Q_{i,t} \;=\; \Delta w_{i,t}\, A.
$$

The stock-level transaction-cost rate follows a square-root market-impact
specification with a half-spread floor, as in Frazzini, Israel, and Moskowitz
(2018):

$$
\tau_{i,t} \;=\;
  \tfrac{1}{2}\,\text{Spread}_{i,t}
  \;+\; \lambda\,\sigma_{i,t}\,\sqrt{\Delta Q_{i,t}\,/\,\text{ADV}_{i,t}},
$$

where $\lambda$ is a calibration constant. Aggregating over traded stocks
yields the portfolio-level monthly transaction cost,

$$
\mathrm{TC}_t \;=\; \sum_i \Delta w_{i,t}\, \tau_{i,t},
$$

and the realized net long–short return,

$$
R^{\mathrm{LS,net}}_{c,t}
  \;=\; R^{\mathrm{LS}}_{c,t}
        - \mathrm{TC}^{L}_t - \mathrm{TC}^{S}_t.
$$

Realized net long–short performance is reported under four transaction-cost
scenarios, capturing strategies of progressively larger size and a
size-independent benchmark:

| Scenario        | Strategy capital $A$ | Cost model                                      |
|:----------------|:---------------------|:------------------------------------------------|
| $\mathrm{PropTC}$ | —                  | Half-spread only ($\lambda = 0$)                 |
| $\$100\mathrm{M}$ | $A = \$100\mathrm{M}$ | Half-spread $+$ market impact at $A = \$100\mathrm{M}$ |
| $\$500\mathrm{M}$ | $A = \$500\mathrm{M}$ | Half-spread $+$ market impact at $A = \$500\mathrm{M}$ |
| $\$1\mathrm{B}$   | $A = \$1\mathrm{B}$   | Half-spread $+$ market impact at $A = \$1\mathrm{B}$   |

In the $\mathrm{PropTC}$ scenario the market-impact term is suppressed
($\lambda = 0$), so $\tau_{i,t}$ collapses to $\tfrac{1}{2}\,\text{Spread}_{i,t}$
and net returns reflect only the proportional cost. This case serves both as
a benchmark against models that ignore impact and as a robustness check
verifying that qualitative conclusions do not hinge on a particular impact
calibration. The three dollar scenarios trace out the cost–capacity profile
of each strategy across realistic deployment sizes; the $\$500\mathrm{M}$
scenario is reported as the primary case in the main text, with
$\$100\mathrm{M}$ and $\$1\mathrm{B}$ providing the size-sensitivity range.

These reporting scenarios are distinct from the training-time scenario
capital indexing the cost-aware sample weights in the companion training
appendix; only the latter enters the model-fitting loss, while the four
scenarios above govern post-fit performance reporting.

## 6. Reported statistics and decompositions

### 6.1 Portfolio time-series statistics

For each portfolio cell $c$ — be it a long–short portfolio, a single leg,
or a prediction quintile (Section 6.3) — and each strategy-size scenario
$A \in \{\mathrm{PropTC},\,\$100\mathrm{M},\,\$500\mathrm{M},\,\$1\mathrm{B}\}$ from Section 5, the reported statistics
are built from the following monthly time series.

**Monthly gross return.** From the selection of Section 4, the deployable
long-short portfolio has gross return

$$
R^{\mathrm{LS}}_{c,t} \;=\;
  \sum_{i \in \mathcal{L}_t} w^{L}_{i,t}\, r_{i,t}
  \;-\; \sum_{i \in \mathcal{S}_t} w^{S}_{i,t}\, r_{i,t},
$$

with leg-level contribution variants

$$
R^{L}_{c,t} \;=\; \sum_{i \in \mathcal{L}_t} w^{L}_{i,t}\,r_{i,t},
\qquad
R^{S,\mathrm{short}}_{c,t}
\;=\; -\sum_{i \in \mathcal{S}_t} w^{S}_{i,t}\,r_{i,t}.
$$

Thus the short-leg rows in the two-sided leg tables are the P&L
contribution of a short position, not the raw long-only return of the Q1
stock basket. Prediction-quintile tables in Section 6.3 remain standalone
long-only quantile portfolios.

**Monthly turnover.** The total absolute change in weights between the
drifted holding $\tilde w_{i,t}$ from Section 5 and the new target weight
$w_{i,t}$ is

$$
T^{\mathrm{raw}}_{c,t} \;=\;
  \sum_i \bigl|\, w_{i,t} - \tilde w_{i,t}\,\bigr|
  \;=\; \sum_i \Delta w_{i,t}.
$$

Reported turnover follows the standard one-way convention, which counts a
buy and the matching sell as one unit of trading:

$$
T_{c,t} \;=\; \tfrac{1}{2}\,T^{\mathrm{raw}}_{c,t}.
$$

**Stock-level transaction-cost rate** (recapped from Section 5):

$$
\tau_{i,t}(A) \;=\;
  \tfrac{1}{2}\,\text{Spread}_{i,t}
  \;+\; \lambda\,\sigma_{i,t}\,\sqrt{\Delta Q_{i,t}\,/\,\text{ADV}_{i,t}},
$$

where $\Delta Q_{i,t} = \Delta w_{i,t}\, A$ is the realized
dollar trade size at scenario $A$. The $\mathrm{PropTC}$ scenario imposes
$\lambda = 0$, retaining only the half-spread floor.

**Monthly portfolio transaction cost.** Aggregating per-stock costs by
trade size, then summing across the two legs,

$$
\mathrm{TC}_{c,t}(A) \;=\; \sum_i \Delta w_{i,t}\, \tau_{i,t}(A)
                          \;=\; \mathrm{TC}^{L}_{c,t}(A) + \mathrm{TC}^{S}_{c,t}(A).
$$

**Monthly net return.**

$$
R^{\mathrm{LS,net}}_{c,t}(A) \;=\;
  R^{\mathrm{LS}}_{c,t} \;-\; \mathrm{TC}_{c,t}(A).
$$

The four scenarios of Section 5 therefore produce four distinct net-return
series per cell; the gross-return series is invariant to $A$.

**Aggregation conventions.** For any monthly series $\{X_t\}_{t=1}^{T}$ we
denote its sample mean and sample standard deviation by $\overline{X}$ and
$\hat\sigma_X$. Tabulated returns, transaction costs, and turnover are
reported as monthly means (returns and costs expressed in percentage
points); only the Sharpe ratio and the factor-model alphas are annualized.
Within the Sharpe-ratio formula below, the monthly mean is scaled by
$\sqrt{12}$ to produce the annualized statistic, exploiting the standard
identity $\sqrt{12}\,\overline{X}/\hat\sigma_X = (12\,\overline{X})/(\sqrt{12}\,\hat\sigma_X)$.

**Annualized Sharpe ratios.** The annualized gross and net Sharpe ratios
for cell $c$ at scenario $A$ are

$$
\mathrm{SR}^{\mathrm{gross}}_{c} \;=\;
  \sqrt{12}\,\frac{\overline{R^{\mathrm{LS}}_{c}}}{\hat\sigma_{R^{\mathrm{LS}}_{c}}},
\qquad
\mathrm{SR}^{\mathrm{net}}_{c}(A) \;=\;
  \sqrt{12}\,\frac{\overline{R^{\mathrm{LS,net}}_{c}(A)}}
                  {\hat\sigma_{R^{\mathrm{LS,net}}_{c}(A)}}.
$$

**Reported aggregates per $(c, A)$ pair.** For each cell and each scenario
we tabulate: the mean monthly gross and net returns
($\overline{R^{\mathrm{LS}}_{c}}$ and
$\overline{R^{\mathrm{LS,net}}_{c}(A)}$, reported in percentage points);
the annualized gross and net Sharpe ratios above; the mean monthly turnover
$\overline{T_{c}}$ and the mean monthly transaction cost
$\overline{\mathrm{TC}_{c}(A)}$ (also in percentage points); the average
number of stocks per leg; and annualized factor-model alphas (CAPM,
Fama–French three- and five-factor, with momentum) estimated by ordinary
least squares on the monthly net-return series. Subsequent subsections
build the headline effect decompositions of Section 6.2 from the
annualized net Sharpe ratios.

### 6.2 Aggregate effect decomposition

The training-vs-portfolio decomposition on the baseline $2 \times 2$ subgrid,
expressed in net-Sharpe units, is

$$
\underbrace{\mathrm{SR}^{\mathrm{net}}_{2B} - \mathrm{SR}^{\mathrm{net}}_{1A}}_{\text{Total}}
\;=\;
\underbrace{\mathrm{SR}^{\mathrm{net}}_{2A} - \mathrm{SR}^{\mathrm{net}}_{1A}}_{\text{Training}}
\;+\;
\underbrace{\mathrm{SR}^{\mathrm{net}}_{1B} - \mathrm{SR}^{\mathrm{net}}_{1A}}_{\text{Portfolio}}
\;+\;
\underbrace{\text{Interaction}}_{\text{residual}}.
$$

Extending to the full $2 \times 3$ grid, the TC-target effects are reported
as differences relative to the A and B cells of the same row
($\mathrm{SR}^{\mathrm{net}}_{1C}-\mathrm{SR}^{\mathrm{net}}_{1A}$ and
$\mathrm{SR}^{\mathrm{net}}_{1C}-\mathrm{SR}^{\mathrm{net}}_{1B}$, with
symmetric statistics for the weighted row), together with the overall
TC-target total effect
$\mathrm{SR}^{\mathrm{net}}_{2C}-\mathrm{SR}^{\mathrm{net}}_{1A}$ and the
incremental training effect within the TC-target column,
$\mathrm{SR}^{\mathrm{net}}_{2C}-\mathrm{SR}^{\mathrm{net}}_{1C}$.

The remainder of Section 6 reports two complementary refinements of this
aggregate decomposition. The first refines it along the **prediction
distribution** (Section 6.3); the second isolates the **long-side and
short-side contributions** of the two-sided long–short portfolio
(Section 6.4). Each provides a different lens on where the headline effects
originate.

### 6.3 Per-prediction-quantile decomposition

To assess whether the training, portfolio, and TC-target effects concentrate
in particular regions of the model's prediction distribution, we exploit the
prediction-quintile portfolios of Section 4.1. For each cell $c$ and each
prediction quintile $q \in \{1, \ldots, 5\}$, the standalone long-only
portfolio

$$
R_{c, q, t} \;=\; \sum_{i \in \mathcal{Q}_q(z_c)} w_{i, t}\, r_{i, t}
$$

is treated as its own book, with realized net returns
$R^{\mathrm{net}}_{c, q, t}$ obtained from realized turnover as in
Section 5. Annualized net Sharpe ratios $\mathrm{SR}^{\mathrm{net}}_{c, q}$
are computed for each $(c, q)$ pair, and the $2 \times 2$ decomposition of
Section 6.2 is then evaluated *within each quintile*:

$$
\mathrm{SR}^{\mathrm{net}}_{2B, q} - \mathrm{SR}^{\mathrm{net}}_{1A, q}
\;=\;
\bigl[\mathrm{SR}^{\mathrm{net}}_{2A, q} - \mathrm{SR}^{\mathrm{net}}_{1A, q}\bigr]
\;+\;
\bigl[\mathrm{SR}^{\mathrm{net}}_{1B, q} - \mathrm{SR}^{\mathrm{net}}_{1A, q}\bigr]
\;+\;\text{Interaction}_q,
$$

with TC-target effects defined analogously within each quintile using cells
$1C, q$ and $2C, q$. The collection
$\{(\text{Training}_q,\, \text{Portfolio}_q,\, \text{Total}_q)\}_{q=1}^{5}$
shows how each headline effect varies across the prediction distribution.

This diagnostic is essential, not redundant with the aggregate long–short
summary. By construction the long–short takes a difference between only
the top and bottom quintile portfolios, so any effect that concentrates in
the middle quintiles ($q = 2, 3, 4$) is invisible to it: a training
improvement that sharpens the model's ordering of mid-conviction names
contributes nothing to a $\mathcal{Q}_5 - \mathcal{Q}_1$ spread and is
therefore mechanically absent from both the single-score and the two-sided
long–short Sharpe comparisons. The per-quintile decomposition surfaces
such localized effects directly: a training effect that is large at $q = 5$
indicates that liquidity-weighted training sharpens the ranking of the
most attractive long candidates; a training effect that concentrates in
middle quintiles indicates that it improves the model's ordering of
mid-conviction stocks without changing the extreme tails — a pattern that
is consistent with our empirical findings and that motivates reporting
this decomposition as a primary, rather than supplementary, diagnostic.

### 6.4 Per-leg decomposition

For the two-sided long–short portfolios of Section 4.2 we report a
complementary decomposition across the two sides of the trade. The long side
is reported as the contribution of the selected long book. The short side is
reported as the contribution of the actual short position, so it enters with
the opposite sign of the underlying Q1 stock-basket return. The $2 \times 3$
effect decomposition of Section 6.2 is then computed within each side:

$$
R^{\mathrm{short,net}}_{c, t}
\;=\; -\sum_{i \in \mathcal{S}_t} w^{S}_{i, t}\, r_{i, t}
\;-\; \mathrm{TC}^{S}_{t}.
$$

The triple
$\{\text{Training}_{\mathcal{L}},\, \text{Training}_{\mathcal{S}},\, \text{Training}_{\mathcal{L S}}\}$,
together with its portfolio, TC-target, total, and interaction analogues,
attributes each headline effect to the long side, the short side, and the
combined long–short. A pattern in which the training effect concentrates on
the long side indicates that liquidity-weighted training improves stock
selection primarily among expected winners; concentration on the short side
indicates the gains come from better identification of expected losers.

The two decompositions in Sections 6.3 and 6.4 are complementary rather than
redundant. Section 6.3 spans the *entire* prediction distribution at fixed
five-quantile granularity and is diagnostic of where in that distribution
training or sorting choices add value; Section 6.4 restricts attention to
the *extremes* actually held in the deployable strategy (the top long
candidates and the bottom short candidates under the two-sided sort) and
attributes the operational long–short performance to each side.

## Reference

Frazzini, A., R. Israel, and T. J. Moskowitz (2018). "Trading Costs."
Working Paper, AQR Capital Management.
