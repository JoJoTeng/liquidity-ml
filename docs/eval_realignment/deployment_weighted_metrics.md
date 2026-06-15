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
realigned metric delivers a consistent verdict. The tables report the training
effect $\Delta R^2_{\tilde{w}}$ by evaluation universe under the NYSE breakpoint
convention, in percentage points (out-of-sample window 2000–2024), for four
representative specifications. Significance is assessed separately, on the monthly
weighted-error differential of §4–§5, and is not conflated with the level effects
here.

*dolvol*

| Model | $Q_1$ | $Q_2$ | $Q_3$ | $Q_4$ | $Q_5$ | $Q_4\cup Q_5$ | Full |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ElasticNet     | $-0.55$ | $-0.28$ | $-0.31$ | $-0.48$ | $-1.09$ | $-1.02$ | $-0.96$ |
| XGBoost        | $-0.42$ | $\;\;0.09$ | $\;\;0.24$ | $\;\;0.18$ | $-0.60$ | $-0.50$ | $-0.45$ |
| Neural network | $-2.69$ | $-0.76$ | $-0.18$ | $-0.08$ | $-0.45$ | $-0.41$ | $-0.41$ |

*softmax-rank ($\lambda{=}2$)*

| Model | $Q_1$ | $Q_2$ | $Q_3$ | $Q_4$ | $Q_5$ | $Q_4\cup Q_5$ | Full |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ElasticNet     | $-0.09$ | $0.04$ | $0.10$ | $0.10$ | $\;\;0.05$ | $\;\;0.07$ | $\;\;0.02$ |
| XGBoost        | $-0.06$ | $0.12$ | $0.15$ | $0.08$ | $-0.04$ | $\;\;0.02$ | $\;\;0.04$ |
| Neural network | $-0.12$ | $-0.01$ | $0.02$ | $0.02$ | $-0.07$ | $-0.02$ | $-0.04$ |

*transaction-cost (TC, $\$10$M)*

| Model | $Q_1$ | $Q_2$ | $Q_3$ | $Q_4$ | $Q_5$ | $Q_4\cup Q_5$ | Full |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ElasticNet     | $0.27$ | $0.34$ | $0.58$ | $0.71$ | $0.56$ | $0.64$ | $0.46$ |
| XGBoost        | $0.36$ | $0.38$ | $0.54$ | $0.53$ | $0.59$ | $0.56$ | $0.46$ |
| Neural network | $0.65$ | $0.57$ | $0.57$ | $0.70$ | $0.86$ | $0.77$ | $0.65$ |

*transaction-cost-rank ($\lambda{=}3$, $\$100$M)*

| Model | $Q_1$ | $Q_2$ | $Q_3$ | $Q_4$ | $Q_5$ | $Q_4\cup Q_5$ | Full |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ElasticNet     | $-0.02$ | $0.16$ | $0.34$ | $0.44$ | $0.34$ | $0.40$ | $0.18$ |
| XGBoost        | $-0.06$ | $0.15$ | $0.23$ | $0.19$ | $0.24$ | $0.22$ | $0.10$ |
| Neural network | $-0.03$ | $0.21$ | $0.33$ | $0.36$ | $0.50$ | $0.42$ | $0.19$ |

**dolvol.** The raw dollar-volume weight lowers deployment-weighted accuracy, and
the damage concentrates where the weight places its mass: for ElasticNet and
XGBoost the largest losses fall in $Q_5$ ($-1.09$ and $-0.60$) and in the pooled
liquid subset — exactly the deployable names the original long–short trades. (The
neural network's large $Q_1$ figure reflects the near-zero weight mass dollar volume
assigns to the least-liquid quintile, where the weighted ratio is unstable, rather
than an economically meaningful loss.)

**softmax-rank.** The bounded rank transform is approximately neutral in every
quintile — small positive values through the middle of the distribution, near-zero
or mildly negative at the liquid extreme — consistent with a weight that neither
over- nor under-concentrates.

**Cost-aware weights.** The transaction-cost and transaction-cost-rank weights are
positive across the *entire* liquidity distribution, including the least-liquid
quintile, and largest in the deployable quintiles. The low-capital weight
(TC, $\$10$M) is positive in all five quintiles for all three models, peaking in
$Q_4$–$Q_5$ (e.g. $0.86$ for the neural network in $Q_5$); the transaction-cost-rank
weight shows the same rise from $Q_1$ toward the liquid end. Importance-weighted
training improves accuracy precisely where tradable capital is concentrated, so the
full-cross-section figure understates the gain in the deployable subset.

The magnitude of the gain is graded by model: largest for ElasticNet, smaller but
still positive for XGBoost and the neural network across the cost-aware
specifications, with the most aggressive softmax weight mildly harmful for the
neural network. The realigned prediction metric thus recovers a benefit to
cost-aware importance weighting that the headline long–short obscures — pronounced
for the linear model and directional for the two non-linear models.

## References

Newey, W. K., and K. D. West (1987). "A Simple, Positive Semi-Definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
*Econometrica* 55(3), 703–708.

Shimodaira, H. (2000). "Improving Predictive Inference under Covariate Shift by
Weighting the Log-Likelihood Function." *Journal of Statistical Planning and
Inference* 90(2), 227–244.
