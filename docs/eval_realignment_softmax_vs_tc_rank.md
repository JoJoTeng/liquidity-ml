# Evaluation Realignment — Softmax-Rank vs TC-Rank Training Weights

> **Project:** LiquidityML
> **Created:** 2026-06-08
> **Scope:** Deployment-weighted prediction metrics from script `41`
> **Models:** `elastic_net`, `xgboost`, `neural_network`
>
> ⚠ **Staleness note (added 2026-06-11):** all `elastic_net` results in this
> note were produced under the pre-`e901e31` configuration
> (`l1_ratio = 0.5`, alpha grid up to `0.1`), in which the model collapses to
> flat predictions in many months. After the ridge-dominant retrain
> (`l1_ratio = 0.01`, capped grid) the `elastic_net` tables, rankings, and
> best-spec recommendations below must be regenerated and re-read before being
> relied upon. The `xgboost` and `neural_network` results are unaffected.
> **Specs compared:** `softmax_rank_lam2`, `softmax_rank_lam3`, and
> `tc_rank_lam3_{10m,100m,500m,1000m}`
> **Primary breakpoint universe:** `nyse`; `full_sample` checked as robustness.

This note documents why the rank-based transaction-cost specification behaves
better than the dollar-volume softmax-rank specification in the realigned
prediction metrics.

The source outputs are:

```text
outputs/eval_realignment/analysis/{model}/{weight_spec}/liquidity_breakpoints/{nyse,full_sample}/deployment_weighted_r2.csv
outputs/eval_realignment/analysis/{model}/{weight_spec}/liquidity_breakpoints/{nyse,full_sample}/deployment_weighted_error_diff_stats.csv
```

Positive `delta` means the weighted model has higher deployment-weighted
zero-benchmark R2 than the standard model. Positive `mean_wmse_diff` means
`wMSE(standard) - wMSE(weighted) > 0`, so weighted training lowered the
deployment-weighted prediction error.

## 1. Weight definitions

### 1.1 Softmax-rank

Code anchor: `src/weighting/schemes.py::_softmax_rank_weights`.

```python
rank = liq.where(liq > 0).rank(pct=True, method="average").fillna(0.5)
raw = np.exp(float(lam) * rank)
return _normalize_to_mean_one(raw)
```

Here `liq` is `liq_dvol_21d`, the primary liquidity variable. Thus:

```text
rank_it = percentile_rank(DolVol_it)
w_it = exp(lambda * rank_it) / monthly_mean(exp(lambda * rank_t))
```

The only ranked variable is dollar volume. High-dollar-volume stocks receive
larger training weights. The formal grid uses `lambda = 2` and `lambda = 3`.

### 1.2 TC-rank

Code anchor: `src/weighting/schemes.py::_tc_rank_weights`.

```python
tc = _compute_tc_per_stock(...)
rank = (-tc).rank(pct=True, method="average").fillna(0.5)
raw = np.exp(float(rank_lam) * rank)
return _normalize_to_mean_one(raw)
```

The ranked variable is negative transaction cost, not dollar volume. The
stock-level transaction cost is:

```text
TC_it = BidAskSpread_it / 2
        + lambda_market_impact * sigma_it * sqrt(Q_t / ADV_it)
```

where `Q_t = AUM / N_t`. Low-cost stocks receive higher ranks and higher
weights. The formal grid uses `lambda = 3` and AUMs `10m`, `100m`, `500m`,
and `1000m`.

The practical distinction:

```text
softmax_rank: train more on high-dollar-volume stocks.
tc_rank:      train more on low-transaction-cost stocks.
```

These are related but not equivalent. `tc_rank` uses spread, volatility, ADV,
and AUM, so it is closer to the implementability/deployment objective.

## 2. Weight distribution intuition

Both softmax-rank and TC-rank are rank-based and therefore avoid the extreme
raw dollar-volume weights. But their economic content differs.

For softmax-rank, the liquidity-quintile mean weights are:

| Spec | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| `softmax_rank_lam2` | 0.500 | 0.880 | 1.240 | 1.642 | 2.076 |
| `softmax_rank_lam3` | 0.325 | 0.749 | 1.246 | 1.894 | 2.689 |

`lambda = 3` is a much stronger dollar-volume tilt than `lambda = 2`.

For TC-rank, all AUMs share the same overall marginal weight distribution
because the final step ranks costs and applies the same `lambda = 3`. AUM
changes which stocks are ranked as low cost.

Mean TC-rank weights by liquidity quintile:

| Spec | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| `tc_rank_lam3_10m` | 0.774 | 1.040 | 1.089 | 1.262 | 1.384 |
| `tc_rank_lam3_100m` | 0.729 | 1.048 | 1.118 | 1.309 | 1.448 |
| `tc_rank_lam3_500m` | 0.674 | 1.049 | 1.150 | 1.371 | 1.540 |
| `tc_rank_lam3_1000m` | 0.645 | 1.044 | 1.167 | 1.407 | 1.596 |

Higher AUM increases the Q4-Q5 tilt, but not through an unbounded raw weight.
This bounded, cost-aware tilt is especially important for the neural network,
which is sensitive to extreme or poorly aligned sample weights.

## 3. Primary NYSE results

The table reports deployment-weighted R2 deltas from script `41`, with
Newey-West t-statistics for the monthly weighted-MSE differential in
parentheses. Positive numbers favor weighted training.

### 3.1 Full cross-section and pooled Q4-Q5

| Model | Spec | Full delta | Full t | Q4-Q5 delta | Q4-Q5 t |
|---|---|---:|---:|---:|---:|
| ElasticNet | `softmax_rank_lam2` | 0.025 | 0.85 | 0.044 | 1.12 |
| ElasticNet | `softmax_rank_lam3` | 0.044 | 0.95 | 0.073 | 1.13 |
| ElasticNet | `tc_rank_lam3_10m` | 0.157 | 3.16 | 0.324 | 2.44 |
| ElasticNet | `tc_rank_lam3_100m` | 0.180 | 3.68 | 0.365 | 2.80 |
| ElasticNet | `tc_rank_lam3_500m` | 0.185 | 3.65 | 0.361 | 2.76 |
| ElasticNet | `tc_rank_lam3_1000m` | 0.182 | 3.54 | 0.357 | 2.73 |
| XGBoost | `softmax_rank_lam2` | 0.037 | 0.46 | 0.024 | 0.39 |
| XGBoost | `softmax_rank_lam3` | -0.073 | -0.29 | -0.045 | -0.08 |
| XGBoost | `tc_rank_lam3_10m` | 0.031 | 0.47 | 0.081 | 0.61 |
| XGBoost | `tc_rank_lam3_100m` | 0.101 | 0.64 | 0.215 | 0.82 |
| XGBoost | `tc_rank_lam3_500m` | 0.079 | 0.59 | 0.139 | 0.61 |
| XGBoost | `tc_rank_lam3_1000m` | 0.068 | 0.60 | 0.017 | 0.24 |
| Neural Network | `softmax_rank_lam2` | -0.042 | -0.23 | -0.022 | 0.01 |
| Neural Network | `softmax_rank_lam3` | -0.416 | -1.78 | -0.492 | -1.60 |
| Neural Network | `tc_rank_lam3_10m` | 0.008 | 0.18 | 0.153 | 0.61 |
| Neural Network | `tc_rank_lam3_100m` | 0.192 | 0.92 | 0.423 | 1.50 |
| Neural Network | `tc_rank_lam3_500m` | 0.032 | 0.35 | 0.156 | 0.88 |
| Neural Network | `tc_rank_lam3_1000m` | 0.136 | 0.98 | 0.238 | 1.29 |

### 3.2 Q1 and Q5 deltas

| Model | Spec | Q1 delta | Q1 t | Q5 delta | Q5 t |
|---|---|---:|---:|---:|---:|
| ElasticNet | `softmax_rank_lam2` | -0.053 | -1.26 | 0.011 | 0.54 |
| ElasticNet | `softmax_rank_lam3` | -0.106 | -1.40 | 0.024 | 0.61 |
| ElasticNet | `tc_rank_lam3_10m` | -0.005 | -0.24 | 0.273 | 2.06 |
| ElasticNet | `tc_rank_lam3_100m` | -0.007 | -0.29 | 0.314 | 2.40 |
| ElasticNet | `tc_rank_lam3_500m` | -0.016 | -0.41 | 0.307 | 2.36 |
| ElasticNet | `tc_rank_lam3_1000m` | -0.029 | -0.55 | 0.300 | 2.31 |
| XGBoost | `softmax_rank_lam2` | -0.064 | -0.35 | -0.035 | -0.01 |
| XGBoost | `softmax_rank_lam3` | -0.231 | -1.10 | -0.112 | -0.46 |
| XGBoost | `tc_rank_lam3_10m` | -0.068 | -0.50 | 0.111 | 0.56 |
| XGBoost | `tc_rank_lam3_100m` | -0.056 | -0.29 | 0.243 | 0.77 |
| XGBoost | `tc_rank_lam3_500m` | -0.059 | -0.36 | 0.130 | 0.50 |
| XGBoost | `tc_rank_lam3_1000m` | 0.005 | 0.15 | -0.065 | -0.04 |
| Neural Network | `softmax_rank_lam2` | -0.119 | -1.04 | -0.069 | -0.14 |
| Neural Network | `softmax_rank_lam3` | -0.492 | -2.60 | -0.543 | -1.58 |
| Neural Network | `tc_rank_lam3_10m` | -0.140 | -0.61 | 0.209 | 0.65 |
| Neural Network | `tc_rank_lam3_100m` | -0.026 | 0.15 | 0.502 | 1.49 |
| Neural Network | `tc_rank_lam3_500m` | -0.099 | -0.67 | 0.218 | 0.94 |
| Neural Network | `tc_rank_lam3_1000m` | 0.036 | 0.12 | 0.279 | 1.23 |

## 4. Interpretation by model

### ElasticNet

ElasticNet benefits from all rank-based weighting specs, but TC-rank is much
stronger than softmax-rank. The softmax specs are directionally positive in
full and Q4-Q5, but weak. TC-rank produces consistent and statistically
meaningful improvements, especially in Q5 and Q4-Q5.

Best ElasticNet specs:

```text
Full:    tc_rank_lam3_500m / tc_rank_lam3_1000m / tc_rank_lam3_100m are similar.
Q4-Q5:   tc_rank_lam3_100m is slightly strongest.
Q5:      tc_rank_lam3_100m is strongest.
```

### XGBoost

XGBoost tolerates `softmax_rank_lam2`, but `softmax_rank_lam3` is negative.
TC-rank improves the deployment metrics, but the monthly error-differential
t-statistics are weak. The best XGBoost TC-rank AUM is `100m`.

Best XGBoost spec:

```text
tc_rank_lam3_100m
```

It is the strongest in both full and Q4-Q5 deployment-weighted R2.

### Neural Network

The neural network is the most sensitive to softmax-rank. `lambda = 2` is
slightly negative, while `lambda = 3` is clearly harmful. This is consistent
with over-tilting the NN toward high-dollar-volume names without using the
broader transaction-cost information.

Within TC-rank, `tc_rank_lam3_100m` is the strongest NN spec:

```text
Full delta:   0.192
Q4-Q5 delta:  0.423
Q5 delta:     0.502
```

The error-differential t-stats are positive but not decisive, so this should be
described as economically promising rather than statistically conclusive.

## 5. Main conclusion

The realigned prediction metrics support the following hierarchy:

```text
tc_rank_lam3_*  >>  softmax_rank_lam2  >  softmax_rank_lam3
```

`softmax_rank` is a smooth rank transform of dollar volume. It captures
capacity in a narrow sense, but it does not distinguish between high dollar
volume and low trading cost. The aggressive `lambda = 3` version can hurt,
especially for the neural network.

`tc_rank` is a smooth rank transform of estimated transaction cost. It is still
bounded and NN-friendly, but it ranks stocks by the implementability object that
matters for deployment: spread plus market impact. That makes it much better
aligned with script `41`'s deployment-weighted R2 and weighted-error
differential.

For reporting, the most defensible rank-based comparison is:

```text
softmax_rank_lam2 as the mild dollar-volume-rank benchmark
tc_rank_lam3_100m as the primary cost-rank benchmark
```

`tc_rank_lam3_100m` is the most consistent choice across XGBoost and neural
network, and it is also one of the strongest ElasticNet specifications.
