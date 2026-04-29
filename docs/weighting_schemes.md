# Weighting Schemes

This note documents the weighting schemes currently implemented in the formal experiment code. It is meant to be the project reference for how training sample weights are constructed, how they relate to NYSE dollar-volume quintiles, and how the TC scale parameter changes the weighting profile.

## Current Code Locations

- Config: [`config/config.yaml`](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/config/config.yaml)
- Weight construction: [`src/weighting/schemes.py`](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/src/weighting/schemes.py)
- Formal training script: [`scripts/20_formal_run_experiment.py`](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/scripts/20_formal_run_experiment.py)
- Formal analysis script: [`scripts/21_formal_analyze_results.py`](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/scripts/21_formal_analyze_results.py)
- HPC job generator: [`scripts/generate_hpc_jobs.sh`](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/scripts/generate_hpc_jobs.sh)

## Implemented Schemes

All schemes are computed separately within each `yyyymm` cross-section and are normalized so that the average sample weight equals 1 within the month. This keeps weighted training comparable to unweighted training in terms of average loss scale.

### 1. Linear DolVol

Formula:

```text
w_it = DolVol_it / mean_i(DolVol_it)
```

Implementation:

```text
scheme = "dolvol"
```

Interpretation:

- AUM-independent.
- Directly proportional to trading capacity.
- Very aggressive because dollar volume is extremely right-skewed.
- High-volume stocks receive very large weights; low-volume stocks can receive weights near zero.

### 2. Softmax Rank

Formula in sample-weight form:

```text
rank_it = percentile_rank_i(DolVol_it) within month t
w_it = exp(lambda * rank_it) / mean_i(exp(lambda * rank_it))
```

Config setting:

```yaml
weighting:
  softmax_rank_lambda: 2.0
  softmax_rank_lambdas: [2.0, 3.0]
```

The scalar `softmax_rank_lambda` is the low-level default used by
`compute_weights()` when no override is supplied. The formal training script
requires an explicit `--softmax-lambda` and checks that value against
`softmax_rank_lambdas`, which is the formal robustness grid.

Implementation:

```text
scheme = "softmax_rank"
python scripts/20_formal_run_experiment.py --weights softmax_rank --softmax-lambda 2
python scripts/20_formal_run_experiment.py --weights softmax_rank --softmax-lambda 3
```

Interpretation:

- AUM-independent.
- Preserves the monotone liquidity ordering from dollar volume.
- Much smoother than raw dollar-volume weighting because it uses ranks rather than raw levels.
- Useful when we want Q1 and Q2 downweighted and Q4/Q5 upweighted without allowing mega-volume stocks to dominate the loss.

### 3. Transaction-Cost Weights

Per-stock transaction cost:

```text
TC_it = Spread_it / 2 + lambda * sigma_it * sqrt(Q_it / ADV_it)
```

Current inputs:

```yaml
transaction_costs:
  spread_col: "BidAskSpread"
  lambda_market_impact: 0.1
  sigma_col: "excess_sigma_12m_daily"
  adv_col: "dvol_21d"
  aum_scenarios:
    - 10_000_000
    - 100_000_000
    - 500_000_000
    - 1_000_000_000
```

Training-weight transformation:

```text
alpha_t = scale / median_i(TC_it)
w_it = exp(-alpha_t * TC_it)
```

Current setting:

```yaml
transaction_costs:
  weight_alpha:
    mode: "inverse_median"
    scale: 3.0
```

Interpretation:

- AUM-dependent.
- Uses spreads, volatility, and ADV rather than dollar volume alone.
- The median TC denominator is intentional. It makes `scale` interpretable: a median-cost stock receives raw penalty `exp(-scale)` before mean-normalization.
- With `scale = 3`, the median-cost stock has raw penalty `exp(-3)` before mean-normalization.

## NYSE-Breakpoint Quintile Comparison

The table below reports the average training sample weight by NYSE dollar-volume breakpoint quintile. Quintile assignment follows the project code: each month, breakpoints are computed using NYSE stocks only (`exchcd == 1`) and then applied to all stocks.

Q1 is least liquid; Q5 is most liquid.

| NYSE Quintile   | Linear DolVol   | Softmax Rank lambda=2   | TC $10M   | TC $100M   | TC $500M   | TC $1B   | Stock-Months   |
|:----------------|:----------------|:---------------|:----------|:-----------|:-----------|:---------|:---------------|
| Q1              | 0.013           | 0.500          | 0.722     | 0.648      | 0.562      | 0.517    | 1,227,792      |
| Q2              | 0.072           | 0.880          | 1.056     | 1.066      | 1.060      | 1.048    | 552,676        |
| Q3              | 0.258           | 1.240          | 1.081     | 1.127      | 1.177      | 1.201    | 455,752        |
| Q4              | 0.897           | 1.642          | 1.325     | 1.405      | 1.508      | 1.567    | 367,017        |
| Q5              | 7.714           | 2.076          | 1.498     | 1.609      | 1.763      | 1.857    | 312,897        |

Key takeaways:

- Linear DolVol is the most aggressive liquidity tilt. Q5 receives `7.714` times average weight, while Q1 receives only `0.013`.
- Softmax Rank gives the clean monotone rank shape: Q1 and Q2 are below 1, while Q3-Q5 are above 1.
- TC weights are economically different from rank weights. They penalize high estimated trading cost, so Q2 can remain slightly above 1 because it is much cheaper than Q1 and receives some redistributed weight after Q1 is downweighted.

## How TC Scale Changes Quintile Weights

For TC weights, `scale` controls how aggressively transaction costs enter the exponential penalty:

```text
alpha_t = scale / median_i(TC_it)
w_it = exp(-alpha_t * TC_it)
```

Higher `scale` means stronger downweighting of high-cost stocks and stronger upweighting of low-cost stocks after mean-normalization.

For the primary `$500M` AUM case:

| scale   | Q1    | Q2    | Q3    | Q4    | Q5    | Q5_Q1   |
|:--------|:------|:------|:------|:------|:------|:--------|
| 2       | 0.584 | 1.063 | 1.203 | 1.474 | 1.672 | 2.864   |
| 3       | 0.562 | 1.060 | 1.177 | 1.508 | 1.763 | 3.136   |
| 4       | 0.558 | 1.064 | 1.149 | 1.516 | 1.807 | 3.238   |
| 6       | 0.560 | 1.082 | 1.109 | 1.510 | 1.832 | 3.269   |
| 10      | 0.554 | 1.106 | 1.087 | 1.518 | 1.839 | 3.320   |
| 20      | 0.454 | 1.070 | 1.148 | 1.718 | 1.972 | 4.346   |

Interpretation:

- Increasing `scale` from 1 to 3 makes the TC tilt much stronger.
- Increasing beyond 3 mostly makes individual weights more extreme rather than producing the desired Softmax-Rank-style Q2-below-1 pattern.
- Q2 remains above 1 even at high scales because TC weighting is cost-based, not rank-quintile-based. Once high-cost Q1 stocks are downweighted, the lost weight is redistributed to all cheaper stocks, including Q2.

## Why We Keep Median TC Instead Of Mean TC

We tested `alpha_t = scale / mean_i(TC_it)` against the current median denominator. Mean TC is pulled up by high-cost outliers, so `scale / mean(TC)` gives a smaller alpha and a weaker penalty.

At `$500M` and `scale = 3`:

| center   | Q1    | Q2    | Q3    | Q4    | Q5    | Q5_Q1   |
|:---------|:------|:------|:------|:------|:------|:--------|
| median   | 0.562 | 1.060 | 1.177 | 1.508 | 1.763 | 3.136   |
| mean     | 0.611 | 1.071 | 1.188 | 1.436 | 1.619 | 2.650   |

Conclusion:

- Mean TC does not solve the Q2-above-1 issue.
- Median TC gives a stronger and more interpretable TC penalty.
- The current primary TC setting is therefore `mode = inverse_median`, `scale = 3.0`.

## Formal Experiment Grid

The formal experiment now has the following weighted specifications per model:

```text
dolvol
softmax_rank_lam2
softmax_rank_lam3
tc_10m
tc_100m
tc_500m
tc_1000m
```

With 3 models, this gives:

```text
3 models x 7 weighted specifications = 21 weighted jobs
```

The standard model (`M_std`) is shared within each model class, so the full set of fitted rolling-window artifacts is:

```text
3 standard fits + 21 weighted fits = 24 model-output directories
```

## Recommended Interpretation In The Paper

Use the schemes as complementary design choices rather than forcing them to behave identically:

- Linear DolVol: aggressive capacity proxy.
- Softmax Rank: smooth monotone liquidity-rank proxy.
- TC: explicit cost-sensitive proxy based on spread, volatility, ADV, and AUM.

If a result holds under all three, it is stronger because it survives raw capacity weighting, smooth rank weighting, and explicit transaction-cost weighting.

## Softmax Lambda Reference Table

The table below shows how the softmax-rank parameter `lambda` changes the average training weight by NYSE dollar-volume breakpoint quintile. We currently keep `lambda = 2.0` as the main reference case and add `lambda = 3.0` as the stronger-liquidity-tilt robustness case.

Formula:

```text
w_it = exp(lambda * percentile_rank_i(DolVol_it)) / mean_i(exp(lambda * percentile_rank_i(DolVol_it)))
```

| Lambda   | Q1    | Q2    | Q3    | Q4    | Q5    | Q5_Q1   |
|:---------|:------|:------|:------|:------|:------|:--------|
| 0        | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000   |
| 0.5      | 0.861 | 0.996 | 1.086 | 1.165 | 1.236 | 1.437   |
| 1        | 0.729 | 0.973 | 1.156 | 1.331 | 1.498 | 2.055   |
| 1.5      | 0.608 | 0.933 | 1.208 | 1.492 | 1.779 | 2.928   |
| 2        | 0.500 | 0.880 | 1.240 | 1.642 | 2.076 | 4.155   |
| 2.5      | 0.405 | 0.818 | 1.252 | 1.776 | 2.381 | 5.871   |
| 3        | 0.325 | 0.749 | 1.246 | 1.894 | 2.689 | 8.261   |
| 4        | 0.204 | 0.606 | 1.188 | 2.069 | 3.297 | 16.149  |
| 5        | 0.125 | 0.472 | 1.089 | 2.169 | 3.874 | 31.055  |
| 6        | 0.075 | 0.358 | 0.969 | 2.203 | 4.407 | 58.765  |

Reading the table:

- `lambda = 0` gives equal weights.
- Higher `lambda` shifts weight monotonically from Q1/Q2 toward Q4/Q5.
- `lambda = 2` is the main reference setting and keeps Q1/Q2 below 1 while keeping Q3 above 1.
- `lambda = 3` is the stronger robustness case: Q1/Q2 are downweighted more heavily and Q5 receives about `2.689` times average weight.
- `lambda >= 4` is substantially more aggressive; by `lambda = 6`, Q3 falls below 1.
