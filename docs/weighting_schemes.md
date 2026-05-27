# Sample-Weighted Training

This appendix specifies the sample-weighting schemes used to fit the
liquidity-weighted prediction-model variants compared against the baseline
uniform-weight specification in the main text. The notation extends that of
the portfolio-construction appendix.

## 1. Motivation

A return-prediction model fit with uniform observation weights treats each
stock-month as equally informative for the training loss. Two consequences
are inconsistent with an implementable trading strategy. First, predictive
errors in small or illiquid stocks contribute to the fit in proportion to
their universe count rather than their realized capacity in a deployable
portfolio. Second, when predictions are subsequently used to trade, the
realized payoff depends on execution cost, which the loss does not see.

We therefore train alternative model variants in which each training
observation receives a sample weight $w_{i,t}^{(*)}$, normalized to monthly
mean one, that tilts the fit toward stocks that are either representative of
the investable universe or inexpensive to trade. Throughout, $w_{i,t}^{(*)}$
enters only the model fit; it does not modify the predictions used
downstream, the portfolio construction, or the realized-turnover net-return
calculation.

## 2. Notation

In addition to the notation of the portfolio-construction appendix, let
$\mathrm{DVOL}_{i,t}$ denote the average daily dollar volume of stock $i$
over a 21-day trailing window, and $\mathcal{N}_t$ the cross-sectional
cardinality of the training panel in month $t$. Under a training-time
scenario capital $A$, the implied flat per-stock dollar allocation is

$$
Q_t \;=\; A\,/\,\mathcal{N}_t,
$$

and the corresponding ex-ante one-way stock-level transaction cost is

$$
\tau_{i,t}(A) \;=\;
  \tfrac{1}{2}\,\text{Spread}_{i,t}
  \;+\; \lambda\,\sigma_{i,t}\,\sqrt{Q_t\,/\,\text{ADV}_{i,t}},
$$

with calibration constant $\lambda$ as in the portfolio-construction
appendix. We write $\rho_{i,t}(x)$ for the within-month percentile rank of a
quantity $x$ at stock $i$, with $\rho = 1$ for the largest value.

## 3. Weighting families

Four families of sample weights are considered. Each is normalized to
monthly mean one,

$$
\mathcal{N}_t^{-1} \sum_i w_{i,t}^{(*)} \;=\; 1,
$$

so that total training loss remains comparable across specifications and
uniform weighting is recovered as a limiting case.

### 3.1 Dollar-volume weights ($\mathrm{dvol}$)

Capacity-aware tilt based directly on traded liquidity:

$$
w^{\mathrm{dvol}}_{i,t} \;=\;
  \frac{\mathrm{DVOL}_{i,t}}{\overline{\mathrm{DVOL}}_t},
$$

where $\overline{\mathrm{DVOL}}_t = \mathcal{N}_t^{-1}\sum_j \mathrm{DVOL}_{j,t}$. The scheme is independent of $A$.

### 3.2 Softmax-rank weights ($\mathrm{sr}$)

Capacity-aware tilt smoothed through within-month ranks, with a single
concentration parameter $\theta > 0$:

$$
w^{\mathrm{sr}}_{i,t}(\theta) \;=\;
  \frac{\exp\!\bigl(\theta\,\rho_{i,t}(\mathrm{DVOL})\bigr)}
       {\mathcal{N}_t^{-1}\sum_j \exp\!\bigl(\theta\,\rho_{j,t}(\mathrm{DVOL})\bigr)}.
$$

The limit $\theta \to 0$ recovers uniform weighting; larger $\theta$
concentrates weight on the most liquid stocks. The scheme is independent
of $A$.

### 3.3 Transaction-cost weights ($\mathrm{tc}$)

Direct cost-aware tilt by exponential down-weighting of $\tau_{i,t}(A)$.
The temperature is set within each month so that a median-cost stock
receives a fixed raw weight $e^{-\kappa}$ before normalization:

$$
\alpha_t(A) \;=\;
  \kappa\,\big/\,\mathrm{median}_i\!\bigl(\tau_{i,t}(A)\bigr),
\qquad
w^{\mathrm{tc}}_{i,t}(A) \;=\;
  \frac{\exp\!\bigl(-\alpha_t(A)\,\tau_{i,t}(A)\bigr)}
       {\mathcal{N}_t^{-1}\sum_j \exp\!\bigl(-\alpha_t(A)\,\tau_{j,t}(A)\bigr)}.
$$

We set $\kappa = 3$. The scheme depends on $A$ through $Q_t$ in
$\tau_{i,t}$ and is therefore indexed by the training-time scenario
capital.

### 3.4 Transaction-cost rank weights ($\mathrm{tcr}$)

Rank-smoothed analogue of (3.3) that retains the cost ordering but not the
cost-level magnitudes. For $\theta > 0$:

$$
w^{\mathrm{tcr}}_{i,t}(A;\, \theta) \;=\;
  \frac{\exp\!\bigl(\theta\,\rho_{i,t}\!\bigl(-\tau(A)\bigr)\bigr)}
       {\mathcal{N}_t^{-1}\sum_j \exp\!\bigl(\theta\,\rho_{j,t}\!\bigl(-\tau(A)\bigr)\bigr)}.
$$

The lowest-cost stock has $\rho(-\tau) = 1$ and receives the largest
weight. The scheme depends on $A$ through $\tau(A)$.

## 4. Primary specifications

The four families are reported at one principal parameterization each,
selected to span the implementable-tilt spectrum:

| Family                    | Symbol                                          | Parameter setting                            | AUM-dependent |
|:--------------------------|:------------------------------------------------|:---------------------------------------------|:--------------|
| Dollar-volume             | $\mathrm{dvol}$                                 | —                                            | No            |
| Softmax-rank              | $\mathrm{sr}(\theta)$                           | $\theta = 2$                                 | No            |
| Transaction-cost          | $\mathrm{tc}(A)$                                | $A = \$500\mathrm{M}$                        | Yes           |
| Transaction-cost rank     | $\mathrm{tcr}(A;\, \theta)$                     | $A = \$500\mathrm{M},\; \theta = 3$          | Yes           |

These are the four *primary* weighted-training specifications underlying
the headline empirical results in the main text. The remaining points on
the training-AUM grid ($A \in \{\$10\mathrm{M},\, \$100\mathrm{M},\, \$1\mathrm{B}\}$) and the secondary softmax-rank value ($\theta = 3$) are
retained for robustness checks reported alongside the main tables.

A conclusion that holds across all four primary specifications is therefore
robust to (i) a raw capacity tilt, (ii) a smooth capacity-rank tilt, (iii)
an explicit cost-level tilt, and (iv) a smooth cost-rank tilt.

## 5. TC-adjusted training target

Alongside the four sample-weight families, we also fit a model variant in
which the *training target* itself is shifted to deduct the one-way
proportional cost. Letting $r_{i,t}$ denote the realized excess return and
$c_{i,t} = \tfrac{1}{2}\,\mathrm{Spread}_{i,t}$ as in the
portfolio-construction appendix, the variant predicts

$$
y^{\mathrm{tc}}_{i,t} \;=\; r_{i,t} - c_{i,t}.
$$

The resulting prediction $s_{i,t}$, used in cells 1C and 2C of the
$2 \times 3$ portfolio design (portfolio-construction appendix, Section 2),
is therefore a stock-level cost-adjusted forecast. Unlike (3.1)–(3.4),
this is a re-specification of the loss target rather than a sample weight;
it can be combined with any of the four sample-weight families when
fitting the cost-adjusted target.

## 6. Relation to portfolio-level cost adjustments

The sample weights of (3.1)–(3.4) operate exclusively at the *training*
stage. They are distinct from two cost adjustments that operate at the
*use* stage and are described in the portfolio-construction appendix:

- the **portfolio-sorting cost penalty** $c_{i,t} = \tfrac{1}{2}\,
  \mathrm{Spread}_{i,t}$ that enters the TC-aware ranking score
  (portfolio-construction appendix, Section 4); and
- the **realized monthly transaction cost**
  $\sum_i \Delta w_{i,t}\,\tau_{i,t}(\Delta Q_{i,t})$ that enters the
  realized net long–short return (portfolio-construction appendix,
  Section 5), in which $\tau_{i,t}$ is evaluated at the realized,
  turnover-implied trade size $\Delta Q_{i,t}$ rather than at the flat
  ex-ante allocation $Q_t$ used here.

Sample-weighted training thus modifies the *fitted predictor*, while the
portfolio-construction adjustments modify the *use of a given predictor*.
The two channels are jointly identified within the $2 \times 3$
experimental design of the portfolio-construction appendix: comparing rows
isolates the training-stage channel, while comparing columns isolates the
use-stage channel.
