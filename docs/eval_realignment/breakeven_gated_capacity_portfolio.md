# Breakeven-Gated Capacity Portfolio

This appendix adds a cost-aware execution layer to the signal-weighted capacity portfolio of the preceding section. That book rebalances fully to its target each month, so it churns on the month-to-month noise in the predictions and pays a transaction-cost bill that grows with assets under management. The construction below leaves the target unchanged but governs *trading* with a parameter-free, per-stock no-trade gate: a name is traded all the way to its target only when the edge it offers clears the cost of capturing it, and is otherwise left to drift. As before, the gated book is built identically from the baseline and the importance-weighted predictions and is compared, both with its own full-rebalance counterpart and across the two training schemes, so the comparison isolates the effect of cost-aware execution and of the training scheme.

## 1. Notation

The notation of the capacity-portfolio appendix carries over. In month $t$, for each stock $i$ in the priced cross-section $\mathcal{U}_t$:

- $r_{i,t}$ is the realized excess return over month $t$ (between $t-1$ and $t$) and $r^{\mathrm{raw}}_{i,t}$ the corresponding raw return; $\hat{r}_{i,t}$ is a forecast of $r_{i,t}$, and the decision-time forecast of next month's return, available at the end of month $t$, is $\hat{r}_{i,t+1}$.
- $\tilde{w}_{i,t} \ge 0$ is the deployment weight, mean-one each month, and $\bar{r}^{\,W}_t = \sum_i \tilde{w}_{i,t}\hat{r}_{i,t+1} / \sum_i \tilde{w}_{i,t}$ the deployment-weighted mean forecast.
- $\theta^{*}_{i,t}$ is the **target** position — the dollar-neutral, unit-gross capacity book of the preceding section, $\theta^{*}_{i,t} \propto \tilde{w}_{i,t}\,(\hat{r}_{i,t+1} - \bar{r}^{\,W}_t)$.

Two new quantities define the gate, both expressed as monthly returns:

- the **centered signal**, the per-dollar edge the book seeks to harvest, $$\alpha_{i,t} \;=\; \hat{r}_{i,t+1} - \bar{r}^{\,W}_t;$$
- the **one-way proportional cost** of trading the name, $$c_{i,t} \;=\; \tfrac{1}{2}\,s_{i,t},$$ where $s_{i,t}$ is the quoted bid–ask spread.

Finally, $\theta^{e}_{i,t}$ denotes the **executed** position — the weight actually held after gating, as opposed to the target $\theta^{*}_{i,t}$.

## 2. The breakeven gate

Under proportional transaction costs the value of a position is linear in its size and the cost of acquiring it is linear in the trade, so the one-period decision to establish or keep a position reduces to a single per-dollar comparison: the edge it earns, $|\alpha_{i,t}|$, against the one-way cost of trading it, $c_{i,t}$. The book therefore trades each name to its target if and only if the edge clears the cost, and holds the inherited (drifted) position otherwise,

$$
\theta^{e}_{i,t} \;=\;
\begin{cases}
\theta^{*}_{i,t} & \text{if } |\alpha_{i,t}| \ge c_{i,t},\\[2pt]
\tilde{\theta}^{e}_{i,t} & \text{if } |\alpha_{i,t}| < c_{i,t},
\end{cases}
$$

where $\tilde{\theta}^{e}_{i,t}$ is the previous executed position carried forward (§3). This is the bang-bang solution of the one-period trade-or-not problem — because costs are linear, the optimal action is to move fully to the target or not to trade at all, never part way — so the gate is the closed-form myopic optimum rather than a heuristic filter. It is the discrete, single-period analogue of the no-trade region that proportional costs induce in the dynamic problem (Constantinides, 1986).

Two properties follow. First, the rule has **no tuning parameter**: the trading threshold *is* the cost $c_{i,t}$, in contrast to a no-trade band whose width must be chosen. Second, the gate is **independent of assets under management**: it compares two monthly-return quantities, neither of which scales with capital. Market impact enters only later, through the realized net-of-cost accounting, so a single gated book serves the entire range of capital levels.

## 3. The executed book

The executed book is path-dependent and is built sequentially. Entering month $t$ with the previous executed book $\theta^{e}_{i,t-1}$, three steps produce the current holdings.

**Drift.** The inherited book is aged by the realized raw return over the month and renormalized to unit gross,

$$
\tilde{\theta}^{e}_{i,t} \;=\;
\frac{\theta^{e}_{i,t-1}\,(1 + r^{\mathrm{raw}}_{i,t})}
     {\sum_j \bigl|\theta^{e}_{j,t-1}\,(1 + r^{\mathrm{raw}}_{j,t})\bigr|},
$$

so that a name failing the gate is held at the position the market has carried it to, not at a stale target.

**Gate.** Each name in the target is set to $\theta^{*}_{i,t}$ or to $\tilde{\theta}^{e}_{i,t}$ according to §2. A name that has left the prediction universe is absent from the target and is **closed**; a name that fails the gate and has no inherited position simply stays flat.

**Renormalize.** Gating mixes freshly traded targets with drifted holdings, so the long and short notionals no longer balance. Each leg is therefore rescaled separately to restore a dollar-neutral, unit-gross book, the long leg to $+\tfrac12$ and the short leg to $-\tfrac12$,

$$
\sum_{i}\theta^{e}_{i,t} = 0,
\qquad
\sum_{i}\bigl|\theta^{e}_{i,t}\bigr| = 1.
$$

Per-leg rescaling is required precisely because gating breaks the algebraic neutrality that centering guarantees for the target: a single gross rescaling would preserve unit gross but leave a net directional tilt.

A month in which a leg holds fewer than two names after gating is **skipped**, and the last executed book is carried into the next month's drift, so the book coasts through degenerate months. The realized gross return of the executed book is

$$
r^{g}_{t} \;=\; \sum_{i}\theta^{e}_{i,t}\,r_{i,t+1}.
$$

Because the executed positions differ from the target, this gross return is a distinct object from that of the full-rebalance book and is recomputed from realized returns rather than inherited.

## 4. Net returns and performance

Net returns and performance statistics follow the capacity-portfolio appendix without change. Transaction costs are charged on the realized trades of the executed book under the same square-root market-impact model, so the gate's reduction in turnover passes directly into a lower cost and a smaller gap between gross and net performance. Gross and net annualized Sharpe ratios and mean–variance certainty-equivalents (at risk aversion $\gamma \in \{1,5,10\}$) are reported for both books, with the first month excluded from the average turnover and cost as before.

## 5. The training effect under gated execution

For each weight specification the book is built four ways — full-rebalance and breakeven-gated, each from the baseline and from the importance-weighted predictions — and summarized over the same out-of-sample window. The standard-versus-weighted comparison uses the inference of the capacity-portfolio appendix: the difference in net Sharpe ratios with the studentized bootstrap test of Ledoit and Wolf (2008), the Newey–West $t$-statistic of the monthly net-return differential, and the certainty-equivalent differential, all in the weighted-minus-standard direction.

This design supports two complementary readings. Holding the book fixed, the **training effect** is the weighted-minus-standard comparison within each book — the question of the preceding section, re-asked under cost-aware execution. Holding the training scheme fixed, the **gate effect** is the comparison of the gated book with its full-rebalance counterpart. The object of interest is their interaction: whether the gate, by suppressing low-edge trading, widens the net advantage of importance-weighted training at scale — turning a cost-channel benefit that is modest under full rebalancing into a larger and more robust one once trading is governed by its own breakeven.

## 6. Conventions and caveats

The gate is deliberately myopic, and two consequences are accepted by design. The cost of trading is paid once, whereas the edge $\alpha_{i,t}$ accrues every month the position is held; a name with $|\alpha_{i,t}| < c_{i,t}$ may still be worth holding over a horizon $H^{*}_{i,t} = c_{i,t}/|\alpha_{i,t}|$ months, which the one-period test ignores. And a held name whose signal weakens or reverses but still fails the gate retains its existing direction until the edge again clears the cost.

The remaining conventions mirror the capacity-portfolio appendix: the same deployment weight sizes both books; names leaving the prediction universe are closed, with the closing trade still charged in the net-return accounting; missing liquidity inputs are replaced by conservative defaults rather than allowed to zero out a cost; and a month with no admissible book is omitted from every series.

For reference, the sign conventions are:

| Quantity | Positive value means |
|:---|:---|
| net Sharpe difference | weighted training has the higher net Sharpe ratio |
| Newey–West $t$ of the net differential | weighted training has the higher mean net return |
| certainty-equivalent differential | weighted training has the higher net certainty-equivalent |

## References

Constantinides, G. M. (1986). "Capital Market Equilibrium with Transaction Costs." *Journal of Political Economy* 94(4), 842–862.

Frazzini, A., R. Israel, and T. J. Moskowitz (2018). "Trading Costs." Working paper, AQR Capital Management.

Ledoit, O., and M. Wolf (2008). "Robust Performance Hypothesis Testing with the Sharpe Ratio." *Journal of Empirical Finance* 15(5), 850–859.

Newey, W. K., and K. D. West (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55(3), 703–708.
