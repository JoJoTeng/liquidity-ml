# Long-Only Capacity-Weighted Quintile Portfolio

This appendix describes the long-only capacity-weighted quintile portfolio, the most defensible implementable object in the evaluation track. The capacity books of the preceding sections are dollar-neutral long–short, and the short leg is the least realistic part of any backtest: borrow costs and locate availability are not modeled at all. This construction drops the short leg entirely and holds only a long, fully-invested, capacity-weighted top-quintile book drawn from the liquid universe, so that every position is one a real fund could take at the stated assets under management. As before, the book is built identically from the baseline and the importance-weighted predictions; its cost-aware variant is governed by a parameter-free membership hysteresis; and it is compared both across training schemes and across the execution device, so the comparison isolates each margin.

## 1. Notation

The notation of the capacity-portfolio appendix carries over. In month $t$, for each stock $i$ in the priced cross-section:

- $r_{i,t}$ is the realized excess return over month $t$ (between $t-1$ and $t$); $\hat{r}_{i,t}$ is a forecast of $r_{i,t}$, and the decision-time forecast of next month's return, available at the end of month $t$, is $\hat{r}_{i,t+1}$.
- $\tilde{w}_{i,t} \ge 0$ is the deployment weight, mean-one each month.
- $s_{i,t}$ is the quoted bid–ask spread; $A$ is the assets under management and $\lambda$ the market-impact coefficient of the trading-cost model.
- $D_{i,t}$ is the stock's dollar volume, the liquidity variable that defines the investable screen.

Two quantities define the book each month:

- the **liquid universe** $\mathcal{L}_t$, the most liquid $60\%$ of the cross-section by dollar volume;
- the **quintile cutoff** $b_t$, the $80$th percentile of the decision-time forecast $\hat{r}_{i,t+1}$ within $\mathcal{L}_t$.

## 2. The liquid universe and the quintile book

The investable universe is screened for liquidity before any selection. In month $t$,

$$
\mathcal{L}_t \;=\; \bigl\{\, i : D_{i,t} \ge \text{the } 40\text{th percentile of } D_{\cdot,t} \,\bigr\},
$$

the top $60\%$ of the cross-section by dollar volume, formed within month. Selection then takes the top quintile of the *liquid* names by forecast: the book targets

$$
\mathcal{Q}_t \;=\; \bigl\{\, i \in \mathcal{L}_t : \hat{r}_{i,t+1} \ge b_t \,\bigr\},
\qquad
b_t \;=\; \text{the } 80\text{th percentile of } \hat{r}_{i,t+1} \text{ over } \mathcal{L}_t.
$$

Because the screen precedes the quintile cut, $\mathcal{Q}_t$ is the best-predicted fifth of the *liquid* universe — never a microcap with an extreme forecast.

The members are held **long-only and fully invested**, weighted by the spec's own deployment weight,

$$
\theta_{i,t} \;=\; \frac{\tilde{w}_{i,t}}{\sum_{j \in \mathcal{Q}_t}\tilde{w}_{j,t}},
\qquad
\theta_{i,t} \ge 0,
\qquad
\sum_{i}\theta_{i,t} = 1.
$$

Weighting the leg by $\tilde{w}$ — rather than equal- or value-weighting it, as a conventional quantile sort would — aligns the deployment with the training objective: the same weight that defined the loss now sizes the positions. The weights are re-trued to $\tilde{w}$ over the current members every month. The book chosen at $t$ earns, over the following month, the gross return

$$
r^{g}_{t} \;=\; \sum_{i \in \mathcal{Q}_t}\theta_{i,t}\,r_{i,t+1}.
$$

A month with fewer than five admissible members is omitted from the series; the last recorded membership is carried into the next month's hysteresis test.

## 3. Membership hysteresis

The cost-aware variant replaces the plain quintile rule with a parameter-free **membership hysteresis** — a no-trade band around the quintile cutoff. The economic logic is a one-period breakeven on *replacement*: selling a held name $i$ that has slipped below the cutoff to buy the marginal entrant earns $(b_t - \hat{r}_{i,t+1})$ per month but costs a round trip — half the spread to sell $i$ plus the entrant's toll. A held name is therefore retained while

$$
\hat{r}_{i,t+1} \;\ge\; b_t - \bigl(\tfrac{1}{2}\,s_{i,t} + \bar{c}_{t}\bigr),
$$

where $\bar{c}_{t}$ is the median half-spread over the month's quintile target, the typical replacement's toll. The retained names are added to the fresh quintile, and the leg is re-weighted to $\tilde{w}$ as before.

The device has three deliberate features. First, it is **asymmetric**: a name enters the moment its forecast clears $b_t$, but once held it is sold only after its forecast falls a full toll-width *below* the cutoff — entry at $b_t$, exit at $b_t-(\tfrac12 s_{i,t}+\bar{c}_{t})$. Second, **wider-spread names are stickier**: the buffer uses the held name's own half-spread, so a more expensive name is retained over a wider band, which is economically correct since selling it costs more. Third, **leaving the screen forces the sale**: a held name that drops out of the liquid universe is sold regardless of its forecast — the buffer protects only names still tradable. As spreads vanish the band collapses and the hysteresis book coincides with the plain book.

A per-name alpha gate of the kind used for the continuous capacity book does **not** transfer to a selection book. A quantile book's turnover is membership churn at the boundary — names whose forecast is still large, sitting just on either side of $b_t$ — so a rule that traded only when the forecast edge exceeded its cost would pass exactly those boundary trades, leaving the churn undamped. Hysteresis on membership, a band rather than a threshold, is the device that suppresses it.

## 4. Net returns and performance

Net returns and performance statistics follow the capacity-portfolio appendix without change. Transaction costs are charged on the realized trades under the same square-root market-impact model; the trades comprise both membership changes and the monthly re-truing of the surviving names to $\tilde{w}$, so the hysteresis device's reduction in membership churn passes directly into lower turnover, a lower cost, and a smaller gap between gross and net performance. Gross and net annualized Sharpe ratios and mean–variance certainty-equivalents (at risk aversion $\gamma \in \{1,5,10\}$) are reported for both books, with the first month excluded from the average turnover and cost.

Unlike the dollar-neutral books of the preceding sections, this book is fully invested and long-only, so it carries net market exposure. Its Sharpe ratio therefore reflects the market together with the selection alpha, and the absolute level is not comparable to a market-neutral spread. The comparisons that the appendix reports — standard versus weighted, and plain versus hysteresis — difference out the common market exposure and isolate the selection-alpha and cost-device contributions, so it is those differences, not the absolute Sharpe levels, that carry the economic content.

## 5. The training and device effects

For each weight specification the book is built four ways — plain and hysteresis, each from the baseline and from the importance-weighted predictions — and summarized over the same out-of-sample window:

$$
1A = \text{standard} \times \text{plain},\quad
1B = \text{standard} \times \text{hysteresis},\quad
2A = \text{weighted} \times \text{plain},\quad
2B = \text{weighted} \times \text{hysteresis}.
$$

Relative to the baseline cell $1A$, the decomposition reports the **training effect** $2A-1A$ (the effect of importance-weighted training, holding the execution device fixed), the **device effect** $1B-1A$ (the effect of the hysteresis, holding the training fixed), the **total effect** $2B-1A$, and their interaction. The standard-versus-weighted comparison within each book uses the inference of the capacity-portfolio appendix: the difference in net Sharpe ratios with the studentized bootstrap test of Ledoit and Wolf (2008), the Newey–West $t$-statistic of the monthly net-return differential, and the certainty-equivalent differential, all in the weighted-minus-standard direction.

As in the preceding sections the baseline cell $1A$ is not a single object across specifications: it is the standard predictions deployed under each spec's own deployment weight $\tilde{w}$, so it differs by specification. This is by design — holding $\tilde{w}$ fixed within a specification makes the training effect a clean comparison of predictions alone — and it means the decomposition is read within each specification, not across them.

## 6. Conventions and caveats

The book is long-only and fully invested, a deliberate retreat to the implementable: the short leg, whose borrow and locate costs the backtest cannot model, is dropped rather than assumed frictionless. The weights are re-trued to $\tilde{w}$ over the surviving members every month, and those re-truing trades are charged transaction costs alongside the membership changes. A held name that leaves the liquid screen is sold regardless of the hysteresis band. Missing liquidity inputs are replaced by conservative defaults rather than allowed to zero out a cost, and a month with fewer than five admissible members is omitted from the series, with the last membership carried into the next month's test. The liquidity screen and the quintile cutoff are formed within month and recomputed each period.

For reference, the sign conventions are:

| Quantity | Positive value means |
|:---|:---|
| net Sharpe difference | weighted training has the higher net Sharpe ratio |
| Newey–West $t$ of the net differential | weighted training has the higher mean net return |
| certainty-equivalent differential | weighted training has the higher net certainty-equivalent |
| device effect $1B-1A$ | the hysteresis raises the standard book's net Sharpe |

## References

Constantinides, G. M. (1986). "Capital Market Equilibrium with Transaction Costs." *Journal of Political Economy* 94(4), 842–862.

Frazzini, A., R. Israel, and T. J. Moskowitz (2018). "Trading Costs." Working paper, AQR Capital Management.

Ledoit, O., and M. Wolf (2008). "Robust Performance Hypothesis Testing with the Sharpe Ratio." *Journal of Empirical Finance* 15(5), 850–859.

Newey, W. K., and K. D. West (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55(3), 703–708.
