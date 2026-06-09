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

## 4. Output layout

```text
outputs/eval_realignment/analysis/{model}/{weight_spec}/
├── capacity_portfolio_metrics_{PropTC,100M,500M,1B}.csv      # script 42
├── capacity_portfolio_monthly_{PropTC,100M,500M,1B}.csv
├── capacity_portfolio_net_cumret_{PropTC,100M,500M,1B}.png
└── liquidity_breakpoints/{nyse,full_sample}/                 # script 41
    ├── deployment_weighted_r2.csv
    ├── deployment_weighted_error_diff.csv
    ├── deployment_weighted_error_diff_stats.csv
    └── deployment_weighted_error_diff_quintiles.png
```

Scripts `41` and `42` write disjoint paths and never overwrite each other's output.

## 5. Scope and relation to the formalanalysis pipeline

This track is restricted to the standard-versus-weighted comparison (the $2\times2$
training axis); the transaction-cost-target column (cell C of the
portfolio-construction appendix) is out of scope, and there is no portfolio-stage
cost sort — the device the note (§5.2) moves away from. Cost-awareness enters only
through $\tilde{w}$ (for the transaction-cost families) and the realized net-of-cost
returns of $(6)$. Predictions are read from
`outputs/formalanalysis/experiment/{model}/{weight_spec}/predictions.parquet`; the
track adds no training and modifies no formalanalysis code, configuration, or
output. The remaining note variations — the capacity-weighted long-only book
(§4.3), liquid-universe dollar-volume-weighted sorts (§4.4), and the
implementable-utility / mean–variance metric (§4.5, Jensen et al., 2024) — together
with factor-model alphas on the capacity book, are deferred.

## References

- Frazzini, A., R. Israel, and T. J. Moskowitz (2018). "Trading Costs." Working Paper.
- Jensen, T. I., B. T. Kelly, S. Malamud, and L. H. Pedersen (2024). "Machine Learning and the Implementable Efficient Frontier." *Review of Financial Studies*.
- Ledoit, O., and M. Wolf (2008). "Robust performance hypothesis testing with the Sharpe ratio." *Journal of Empirical Finance* 15(5).
- Shimodaira, H. (2000). "Improving predictive inference under covariate shift by weighting the log-likelihood function." *Journal of Statistical Planning and Inference* 90(2).
