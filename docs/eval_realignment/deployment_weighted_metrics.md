# Deployment-Weighted Prediction Metrics

This appendix describes the deployment-weighted prediction metrics used to assess
whether importance-weighted training improves return predictions where deployable
capital is concentrated. The headline long–short sorted on the raw prediction does
not measure the quantity that importance-weighted training minimizes — the
deployment-weighted prediction error under covariate shift (Shimodaira, 2000). The
metrics below are the direct empirical counterpart of that training objective.
Because they are defined on predictions rather than on a traded portfolio, they
carry neither transaction costs nor a notion of assets under management.

## 1. Notation

Let $\mathcal{U}_t$ denote the priced cross-section in month $t$. For each stock
$i \in \mathcal{U}_t$:

- $r_{i,t}$ is the realized excess return earned over the holding period.
- $\hat{r}^{\,\mathrm{std}}_{i,t}$ and $\hat{r}^{\,w}_{i,t}$ are one-month-ahead
  return predictions from, respectively, the baseline model fitted with uniform
  observation weights and the model fitted with importance weights.
- $\tilde{w}_{i,t} \ge 0$ is the **same importance weight that defines the
  training objective**, normalized to unit cross-sectional mean each month,
  $\mathcal{N}_t^{-1}\sum_{i \in \mathcal{U}_t} \tilde{w}_{i,t} = 1$. The identical
  weight scores both prediction sets, so the comparison reflects the training
  scheme alone.
- $q_{i,t} \in \{1,\dots,5\}$ is the stock's liquidity quintile, with $Q_5$ the
  most liquid. Quintile membership is formed from a single liquidity variable
  under a stated breakpoint convention (NYSE or full-sample).

An observation enters a metric only if its return, prediction, and weight are all
present; incomplete observations are excluded rather than imputed.

## 2. Evaluation universes

Each metric is reported on a nested family of universes $\mathcal{S}$:

- each single liquidity quintile, $\mathcal{S} = Q_k$ for $k = 1,\dots,5$;
- the pooled liquid (deployable) subset, $\mathcal{S} = Q_4 \cup Q_5$;
- the full cross-section.

The single-quintile and pooled universes restrict to stocks carrying the
corresponding quintile label. The full universe is the entire priced
cross-section, including stocks that cannot be assigned a quintile in a given
month (for example when too few names are available to form breakpoints); it is
therefore not in general the union of the five quintiles.

Two weight conventions are used throughout. First, the monthly mean-one weight
$\tilde{w}$ is held fixed on every universe and is **not** renormalized within a
subset; quintile membership only restricts the index set of the sums, it does not
redefine the weight. Second, every metric below is a *ratio* of $\tilde{w}$-weighted
sums and is therefore invariant to a global rescaling of $\tilde{w}$; the monthly
normalization fixes a harmless scale and leaves the universes directly comparable.

## 3. Deployment-weighted out-of-sample $R^2$

For a prediction set $\hat{r}$ on universe $\mathcal{S}$, the zero-benchmark,
$\tilde{w}$-weighted out-of-sample $R^2$ pools all stock-months,

$$
R^2_{\tilde{w}}(\hat{r}; \mathcal{S})
  \;=\; 1 \;-\;
  \frac{\displaystyle\sum_{t}\sum_{i \in \mathcal{S}_t}
        \tilde{w}_{i,t}\,(r_{i,t} - \hat{r}_{i,t})^2}
       {\displaystyle\sum_{t}\sum_{i \in \mathcal{S}_t}
        \tilde{w}_{i,t}\,r_{i,t}^2}.
$$

The benchmark is a forecast of zero, matching the project's primary $R^2$
convention: a positive value means the model beats a null forecast on the
deployment-weighted distribution. The aggregation is *pooled* — a single ratio
over the entire panel of $\mathcal{S}$ — rather than an average of monthly $R^2$
values.

Both prediction sets are scored under the same $\tilde{w}$ and the same universe,
and the training effect is reported in percentage points,

$$
\Delta R^2_{\tilde{w}}(\mathcal{S})
  \;=\; 100\,\Bigl[\,R^2_{\tilde{w}}(\hat{r}^{\,w}; \mathcal{S})
        - R^2_{\tilde{w}}(\hat{r}^{\,\mathrm{std}}; \mathcal{S})\,\Bigr],
$$

so $\Delta R^2_{\tilde{w}} > 0$ indicates that importance-weighted training lowered
the deployment-weighted prediction error on $\mathcal{S}$.

## 4. Monthly weighted squared-error differential

To follow the effect through time, the pooled level is replaced by a per-month
$\tilde{w}$-weighted mean squared error,

$$
\mathrm{wMSE}_t(\hat{r}; \mathcal{S})
  \;=\;
  \frac{\displaystyle\sum_{i \in \mathcal{S}_t}
        \tilde{w}_{i,t}\,(r_{i,t} - \hat{r}_{i,t})^2}
       {\displaystyle\sum_{i \in \mathcal{S}_t}\tilde{w}_{i,t}},
$$

defined for months with positive weight mass. The differential between the two
models and its running total are

$$
D_t(\mathcal{S}) \;=\;
  \mathrm{wMSE}_t(\hat{r}^{\,\mathrm{std}}; \mathcal{S})
  - \mathrm{wMSE}_t(\hat{r}^{\,w}; \mathcal{S}),
\qquad
C_t(\mathcal{S}) \;=\; \sum_{s \le t} D_s(\mathcal{S}),
$$

evaluated on months for which both models have a defined weighted error. The sign
convention is that $D_t > 0$ when the baseline model carries the larger weighted
error in month $t$, i.e. when importance-weighted training helped that month; the
cumulative series $C_t$ is reported by quintile.

The two metrics aggregate differently by design. The out-of-sample $R^2$ pools
every stock-month into one ratio and summarizes the *level* of the effect, whereas
the differential is a monthly quantity and so forms a *time series*, which is what
admits formal time-series inference and a cumulative description of when the effect
accrues.

## 5. Inference

For each universe the monthly differential $\{D_t(\mathcal{S})\}_{t=1}^{T}$ is
summarized by a test of a zero mean using Newey–West heteroskedasticity- and
autocorrelation-consistent standard errors (Bartlett kernel, six lags),

$$
\bar{D} \;=\; \frac{1}{T}\sum_{t=1}^{T} D_t,
\qquad
t_{\mathrm{NW}} \;=\;
  \frac{\bar{D}}{\widehat{\mathrm{se}}_{\mathrm{NW}}(\bar{D})}.
$$

The note's prior is that the genuine effect is real but moderate, so cells are
expected to be directionally positive without uniformly attaining conventional
significance; a non-significant cell is a statement about power, not evidence that
the effect is absent.

## 6. Conventions

The same importance weight scores both prediction sets, so each comparison
isolates the training scheme and nothing else. Only the quintile assignment
depends on the breakpoint convention; under both the NYSE and full-sample
conventions the metrics are otherwise identical in definition. A universe with no
usable observations, or a month with zero weight mass, yields a missing value
rather than a defined statistic.

For reference, the sign conventions are:

| Quantity | Positive value means |
|:---|:---|
| $\Delta R^2_{\tilde{w}}$ | weighted training has the higher deployment-weighted $R^2$ |
| $D_t,\;\bar{D}$ | weighted training has the lower deployment-weighted error (that month / on average) |

## 7. Results

Across three model families and eleven importance-weight specifications, the
realigned metrics deliver a consistent verdict. The table reports the training
effect $\Delta R^2_{\tilde{w}}$ on the full cross-section under the NYSE breakpoint
convention, in percentage points, with the Newey–West $t$-statistic of the monthly
differential in parentheses; the out-of-sample window is 2000–2024 ($T = 299$
months).

| Model | dolvol | softmax-rank ($\lambda{=}2$) | TC ($\$10$M) | TC-rank ($\lambda{=}3$, $\$100$M) |
|:--|--:|--:|--:|--:|
| ElasticNet     | $-0.96\;(-2.13)$ | $\;\;0.02\;(0.86)$  | $0.46\;(4.64)$ | $0.18\;(4.13)$ |
| XGBoost        | $-0.45\;(-1.03)$ | $\;\;0.04\;(0.46)$  | $0.46\;(2.36)$ | $0.10\;(0.64)$ |
| Neural network | $-0.41\;(-0.94)$ | $-0.04\;(-0.23)$    | $0.65\;(2.22)$ | $0.19\;(0.92)$ |

Three regularities hold across all three models. First, the raw dollar-volume
weight (dolvol) — the specification underlying the original portfolio results —
*lowers* deployment-weighted accuracy, significantly so for ElasticNet: an
unbounded capacity weight over-concentrates on the largest names and degrades the
very objective it is meant to serve. Second, the bounded rank transform of dollar
volume (softmax-rank) is approximately neutral, whereas the cost-aware weights
(transaction-cost and transaction-cost-rank) are uniformly positive and are the
only specifications to attain significance; the low-capital transaction-cost weight
(TC, $\$10$M) is the single specification significant in all three models. Third,
the effect is larger in the liquid (deployable) subset than in the full
cross-section — for ElasticNet's TC ($\$10$M) weight, $\Delta R^2_{\tilde{w}}$ rises
from $0.46$ on the full cross-section to $0.64$ on $Q_4 \cup Q_5$ — confirming that
importance-weighted training improves accuracy precisely where tradable capital is
concentrated.

The strength of the evidence is graded by model. For ElasticNet the cost-aware
gains are large and strongly significant ($t \approx 4$ on the full cross-section,
$t \approx 3$ on $Q_4 \cup Q_5$). For XGBoost and the neural network the same
specifications are positive but, with the exception of the low-capital
transaction-cost weight, fall short of conventional significance — and for the
neural network the most aggressive softmax weight is mildly harmful. This grading
matches the prior of a real but moderate effect: the realigned prediction metric
recovers a genuine benefit to cost-aware importance weighting that the headline
long–short obscures, decisively for the linear model and directionally for the two
non-linear models.

## References

Newey, W. K., and K. D. West (1987). "A Simple, Positive Semi-Definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
*Econometrica* 55(3), 703–708.

Shimodaira, H. (2000). "Improving Predictive Inference under Covariate Shift by
Weighting the Log-Likelihood Function." *Journal of Statistical Planning and
Inference* 90(2), 227–244.
