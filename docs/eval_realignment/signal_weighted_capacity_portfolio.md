# Signal-Weighted Capacity Portfolio

This appendix describes the signal-weighted capacity portfolio used to test whether the prediction-level improvement from importance-weighted training survives at the portfolio level, net of trading costs. The headline long–short sorted on the raw prediction equal-weights the names inside each quantile, over-rewards illiquid microcaps that cannot absorb capital, and is non-additive across the liquidity dimension; it does not reward holding more where capital can actually be deployed. The construction below replaces it with a continuous, dollar-neutral book that holds each stock in proportion to its deployable capacity and its centered signal. As in the prediction metrics, the book is built identically from the baseline and the importance-weighted predictions and scored with the same deployment weight, so the comparison isolates the training scheme. Unlike the prediction metrics, it is a traded portfolio and therefore carries transaction costs and a notion of assets under management.

## 1. Notation

Let $\mathcal{U}_t$ denote the priced cross-section in month $t$. For each stock $i \in \mathcal{U}_t$:

- $r_{i,t}$ is the realized **excess** return over month $t$ (between $t-1$ and $t$), and $r^{\mathrm{raw}}_{i,t} = r_{i,t} + r^{f}_t$ is the corresponding **raw** return (the total return, including the risk-free component), differing from the excess return only by the risk-free rate $r^{f}_t$.
- $\hat{r}^{\,\mathrm{std}}_{i,t}$ and $\hat{r}^{\,w}_{i,t}$ are forecasts of the month-$t$ return $r_{i,t}$, formed at the end of month $t-1$, from, respectively, the baseline model fitted with uniform observation weights and the model fitted with importance weights. A statement that holds for either set is written $\hat{r}_{i,t}$; the forecast of next month's return $r_{i,t+1}$, available at the end of month $t$, is accordingly $\hat{r}_{i,t+1}$.
- $\tilde{w}_{i,t} \ge 0$ is the **same deployment weight that defines the training objective**, normalized to unit cross-sectional mean each month, $\mathcal{N}_t^{-1}\sum_{i \in \mathcal{U}_t}\tilde{w}_{i,t} = 1$. The identical weight sizes both books, so the comparison reflects the training scheme alone.
- $s_{i,t}$, $\sigma_{i,t}$, and $\mathrm{ADV}_{i,t}$ are the bid–ask spread, the return volatility, and the average daily dollar volume — the liquidity inputs to the trading-cost model.
- $A$ is the assets under management, interpreted as the gross dollar capital at which the book is run. It is a fixed scenario parameter, swept over a grid; it enters only the trading-cost term.
- $\lambda$ is the market-impact coefficient of the trading-cost model.

A stock-month enters the book only when its return, prediction, and deployment weight are all present; incomplete observations are excluded rather than imputed.

## 2. The capacity book

Let $\bar{r}^{\,W}_t$ be the deployment-weighted cross-sectional mean of the forecasts the month-$t$ book acts on,

$$
\bar{r}^{\,W}_t
  \;=\;
  \frac{\sum_{i \in \mathcal{U}_t}\tilde{w}_{i,t}\,\hat{r}_{i,t+1}}
       {\sum_{i \in \mathcal{U}_t}\tilde{w}_{i,t}}.
$$

The unnormalized position in stock $i$ is its capacity weight times its centered signal,

$$
\theta^{\mathrm{raw}}_{i,t}
  \;=\;
  \tilde{w}_{i,t}\,\bigl(\hat{r}_{i,t+1} - \bar{r}^{\,W}_t\bigr).
$$

Centering the prediction on $\bar{r}^{\,W}_t$ makes the book **dollar-neutral by construction**, since

$$
\sum_{i \in \mathcal{U}_t}\theta^{\mathrm{raw}}_{i,t}
  \;=\;
  \sum_{i}\tilde{w}_{i,t}\hat{r}_{i,t+1}
  - \bar{r}^{\,W}_t\sum_{i}\tilde{w}_{i,t}
  \;=\; 0,
$$

so the long and short notionals are equal in every month without any after-the-fact demeaning. The book is then sized to **unit gross exposure**,

$$
\theta_{i,t}
  \;=\;
  \frac{\theta^{\mathrm{raw}}_{i,t}}
       {\sum_{j \in \mathcal{U}_t}\bigl|\theta^{\mathrm{raw}}_{j,t}\bigr|},
\qquad
\sum_{i \in \mathcal{U}_t}\bigl|\theta_{i,t}\bigr| = 1,
$$

so that the book trades exactly one dollar of gross capital and the assets under management $A$ can be interpreted as the gross dollar size of the position. The book chosen at $t$ earns, over the following month, the gross return

$$
r^{g}_{t} \;=\; \sum_{i \in \mathcal{U}_t}\theta_{i,t}\,r_{i,t+1}.
$$

The construction holds a stock long when its prediction is above the deployment-weighted cross-sectional mean and short when it is below, with a position size scaled by its capacity $\tilde{w}_{i,t}$: more weight is placed where more capital can be absorbed. This is the economic content that the equal-weighted quantile long–short discards.

A month is admitted only when the total weight mass is positive, the signal is not degenerate ($\theta^{\mathrm{raw}}_t \not\equiv 0$), and each side holds at least two names. A month failing any of these is omitted from the series rather than recorded as a zero-return month, so that a degenerate cross-section does not enter the performance statistics as spurious low-variance data.

**Capacity-weighting benchmarks.** The deployment weight $\tilde{w}_{i,t}$ in $\theta^{\mathrm{raw}}_{i,t}=\tilde{w}_{i,t}\bigl(\hat{r}_{i,t+1}-\bar{r}^{\,W}_t\bigr)$ is the object under test — the *signal* capacity book. Two benchmarks replace it with a generic capacity weight $w_{i,t}$ that ignores $\tilde{w}$, holding the forecasts and the rest of the construction fixed:

$$
w_{i,t}=
\begin{cases}
\tilde{w}_{i,t} & \text{signal: the deployment weight,}\\
1 & \text{equal: the naive book, } \theta_{i,t}\propto \hat{r}_{i,t+1}-\bar{r}_t,\\
\mathrm{ME}_{i,t} & \text{value: market capitalization,}
\end{cases}
$$

where the centering mean $\bar{r}^{\,W}_t$, the dollar-neutralization, and the unit-gross normalization are all formed from the *same* $w$. Because the unit-gross step fixes $\sum_{i}|\theta_{i,t}|=1$, the book is **invariant to any positive rescaling of $w$**: replacing $w_{i,t}$ by $c\,w_{i,t}$ for $c>0$ leaves $\bar{r}^{\,W}_t$, every $\theta_{i,t}$, and every return unchanged, so no mean-one normalization of the capacity weight is applied — unlike the importance weights, whose per-month scale enters the training loss. The *equal* book (the continuous, dollar-neutral analogue of $1/N$ with $w=1$ — not the equal-weighted quintile long–short of the introduction) trusts the signal everywhere and so loads precisely the illiquid, high-forecast names that the capacity weight down-sizes; the *value* book weights by market capitalization, a coarser liquidity proxy than dollar volume.

Both benchmarks are run through the identical pipeline at each training row. Comparing across capacity weights — *signal* against *equal* and *value* — isolates the **deployment-stage** value of weighting by tradeable capacity, while comparing across training rows — baseline against importance-weighted — isolates the **training-stage** effect of $\tilde{w}$, so the two liquidity levers are separated. The benchmark books are written under the same spec directory with an `_equal` / `_value` filename infix; the *signal* book keeps the unsuffixed names.

## 3. Holding drift and turnover

Between rebalances the positions drift with realized returns: the book set at the previous rebalance, $\theta_{i,t-1}$, is not held fixed but grows over month $t$ with the raw return $r^{\mathrm{raw}}_{i,t}$ its holdings earn. The book carried into the rebalance at $t$ is therefore

$$
\tilde{\theta}_{i,t}
  \;=\;
  \frac{\theta_{i,t-1}\,(1 + r^{\mathrm{raw}}_{i,t})}
       {\sum_{j}\bigl|\theta_{j,t-1}\,(1 + r^{\mathrm{raw}}_{j,t})\bigr|},
$$

renormalized to unit gross so that the drifted book and the new target are on the same footing. The **raw** return $r^{\mathrm{raw}}$ is used here, rather than the excess return, because a holding's market value compounds at the raw return; the gross profit and loss of §2 nonetheless uses the excess return, and the two are interchangeable there because the book is dollar-neutral — $\sum_i\theta_{i,t}=0$ annihilates the common risk-free term. The trade required at the rebalance is the per-name change

$$
\Delta\theta_{i,t} \;=\; \bigl|\theta_{i,t} - \tilde{\theta}_{i,t}\bigr|,
$$

with $\tilde{\theta}_{i,1} \equiv 0$ at the first rebalance, where the entire book is established from cash. Measuring trades against the **drifted** book rather than against the previous target avoids charging for the position changes that realized returns produce on their own. The reported portfolio turnover is the conventional one-sided measure

$$
\mathrm{turnover}_t \;=\; \tfrac{1}{2}\sum_{i}\Delta\theta_{i,t}.
$$

## 4. Transaction costs

Each rebalance is priced with a square-root market-impact model in the style of Frazzini, Israel, and Moskowitz (2018). The per-name cost rate combines a proportional half-spread and a concave impact term sized on the dollar value of the trade,

$$
\tau_{i,t}
  \;=\;
  \underbrace{\tfrac{1}{2}\,s_{i,t}}_{\text{proportional}}
  \;+\;
  \underbrace{\lambda\,\sigma_{i,t}\,
    \sqrt{\dfrac{\Delta\theta_{i,t}\,A}{\mathrm{ADV}_{i,t}}}}_{\text{market impact}},
$$

and the monthly cost is the trade-weighted sum

$$
c_t \;=\; \sum_{i \in \mathcal{U}_t}\Delta\theta_{i,t}\,\tau_{i,t}.
$$

The half-spread is the size-independent cost of crossing the quoted spread; the impact term grows with the fraction of average daily volume consumed, $\Delta\theta_{i,t}A/\mathrm{ADV}_{i,t}$, but with diminishing marginal cost. The assets under management $A$ enter the book **only** through this term: gross returns are invariant to $A$, and a proportional-cost benchmark is obtained by setting the impact term to zero, leaving a pure half-spread cost. The net return is

$$
r^{n}_{t} \;=\; r^{g}_{t} - c_t.
$$

## 5. Performance metrics

The book is summarized over its live months by gross and net statistics. With monthly mean $\hat{\mu}$ and sample standard deviation $\hat{s}$ (computed with the $T-1$ denominator), the annualized Sharpe ratio is

$$
\mathrm{SR} \;=\; \sqrt{12}\,\frac{\hat{\mu}}{\hat{s}},
$$

and the annualized mean is $12\,\hat{\mu}$. The mean–variance certainty-equivalent at risk aversion $\gamma$ is reported on a grid $\gamma \in \{1,5,10\}$,

$$
\mathrm{CE}(\gamma) \;=\; 12\,\Bigl(\hat{\mu} - \tfrac{1}{2}\gamma\,\hat{V}\Bigr),
$$

with $\hat{V}$ the sample variance; under the same linear-annualization convention as the Sharpe ratio (annual variance equal to twelve times the monthly variance), this equals the certainty-equivalent of the annual return. Each statistic is reported for both the gross and the net series. The gross figures are independent of $A$ and isolate the **training effect**; the gap between the gross and net figures is the **cost drag**, which widens as $A$ grows.

The first month is excluded from the reported average turnover and average cost, because it reflects the one-time cost of establishing the book rather than the steady-state trading rate; the first month's cost is nonetheless retained in the net-return series and therefore in the Sharpe and certainty-equivalent figures.

## 6. The training effect

The standard and weighted books are compared on the months in which both are live, so the difference is a matched, paired sample. Three statistics summarize the training effect, all computed on the net returns and all in the weighted-minus-standard direction, so that a positive value favors importance weighting:

- the **difference in Sharpe ratios**, $\mathrm{SR}(\hat{r}^{w}) - \mathrm{SR}(\hat{r}^{\mathrm{std}})$, with a $p$-value from the studentized bootstrap test of Ledoit and Wolf (2008), which is robust to the heavy tails and serial correlation of return series and to the dependence between the two books;
- the **Newey–West $t$-statistic** of the monthly net-return differential $r^{n,w}_t - r^{n,\mathrm{std}}_t$, using a heteroskedasticity- and autocorrelation-consistent standard error (Bartlett kernel, six lags), testing the mean rather than the risk-adjusted improvement;
- the **certainty-equivalent differential** at each $\gamma$, the difference of the annualized net certainty-equivalents of the two books.

The two test statistics answer complementary questions — a risk-adjusted improvement (the Sharpe difference) and a mean-return improvement (the Newey–West $t$) — and each is computed on the quantity it describes rather than borrowed from a companion series. The certainty-equivalent differential measures the utility gain a mean–variance investor of risk aversion $\gamma$ books from the weighted training.

## 7. Conventions

The same deployment weight $\tilde{w}$ sizes both books, so each comparison isolates the training scheme and nothing else. The assets under management $A$ is a fixed scenario parameter: the book is run at a constant target gross size each month, and its profit and loss is not compounded back into $A$, so the cost-versus-capital gradient is read off cleanly across the grid. Transaction costs are charged on the full traded notional $\sum_i\Delta\theta_{i,t}$, whereas the reported turnover is the one-sided $\tfrac{1}{2}\sum_i\Delta\theta_{i,t}$ diagnostic and does not itself enter net returns. Missing liquidity inputs are replaced by conservative defaults rather than allowed to zero out a cost, and a stock-month with no usable cost data is charged a flat proportional rate. A month with no admissible book is omitted from every series.

For reference, the sign conventions are:

| Quantity | Positive value means |
|:---|:---|
| $\mathrm{SR}(\hat{r}^{w}) - \mathrm{SR}(\hat{r}^{\mathrm{std}})$ | weighted training has the higher net Sharpe ratio |
| Newey–West $t$ of $r^{n,w}_t - r^{n,\mathrm{std}}_t$ | weighted training has the higher mean net return |
| certainty-equivalent differential | weighted training has the higher net certainty-equivalent |

## References

Frazzini, A., R. Israel, and T. J. Moskowitz (2018). "Trading Costs." Working paper, AQR Capital Management.

Ledoit, O., and M. Wolf (2008). "Robust Performance Hypothesis Testing with the Sharpe Ratio." *Journal of Empirical Finance* 15(5), 850–859.

Newey, W. K., and K. D. West (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55(3), 703–708.
