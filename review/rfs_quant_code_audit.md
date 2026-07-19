# Independent quantitative-code and RFS-style referee audit

**Repository:** `liquidity_ml`  
**Audit date:** 19 July 2026  
**Mode:** read-only audit of code, cached outputs, generated tables, and paper source

I did not retrain a model, change any source/data/output/paper file, or run a long job. The only repository addition is this report. I also observed the requested independence rule: I did **not** open or list `paper/review/` or `.claude/`.

**Overall verdict:** the specified headline values are faithfully transcribed from the cached outputs, but the current paper is not ready for RFS. Three implementation/identification failures are central: XGBoost's two training arms did not receive the same realized hyperparameter menu; the conventional hysteresis portfolio often places the same stock in both legs; and conventional long--short returns and costs use a one-leg denominator despite being described per dollar of total gross capital. In addition, the main-text breakeven equation is not the final executed rule. These are not cosmetic. They require code corrections, controlled reruns of affected downstream analyses, and a narrower statement of what has been identified.

## 1. SUMMARY

### 1.1 Claim structure

The paper makes a coherent four-link argument.

1. Equal-stock training emphasizes the illiquid portion of the panel, where measured predictability is strongest but institutional capital is hardest to deploy. The motivation track documents distributional divergence, liquidity-varying predictive slopes, and conventional out-of-sample fit concentrated in the least liquid quintile.
2. A mean-one implementability weight, especially the within-month softmax rank of negative estimated trading cost, is used to shift training toward deployable names. Hard screening is presented as an inferior substitute because it discards transferable signal.
3. Forecasts must be evaluated under the deployment measure. The paper therefore introduces deployment-weighted prediction R-squared and a signal-weighted, dollar-neutral capacity portfolio.
4. In the primary two-by-two design, cost-aware execution supplies nearly all of the economically and statistically strong improvement; weighted training is explicitly presented as a conditional, mostly insignificant lever. A sorted-portfolio dose-response and feature-importance reallocation are intended to show that the evaluation object, rather than a new source of gross alpha, explains the result.

That is a more credible and interesting claim than “weighted training creates alpha.” The paper usually acknowledges that its strongest evidence is about evaluation and execution. The present code does not yet support the cleaner causal language—“weighting alone,” a literal no-trade gate, and a valid conventional hysteresis benchmark—with which that argument is framed.

### 1.2 Actual pipeline map

| Layer | Actual path and role | Main cached product |
|---|---|---|
| Raw data | `scripts/00_fetch_data.py` fetches CRSP and Chen--Zimmermann inputs, constructs adjusted returns and trailing dollar volume. | `data/signed_predictors_all_wide.csv` and raw helpers |
| Processed panel | `scripts/01_process_data.py`, together with `src/data/loader.py`, merges risk-free rates, constructs the next-observation excess-return target, screens features, rank-normalizes within month, and retains raw liquidity helpers. | `data/processed_panel.parquet`, `data/feature_list.json`, feature categories |
| Motivation | `scripts/02`--`07` produce divergence, heterogeneity, ML fit/importance, progressive restriction, quintile-specific models, and regimes. | `outputs/motivation/step1*`--`step3*` |
| Formal training | `scripts/20_formal_run_experiment.py` calls the registry in `src/models/` and the rolling engine in `src/training/rolling.py`; weights come from `src/weighting/schemes.py`. | Predictions, tuning, importance, and diagnostics under `outputs/formalanalysis/experiment/` |
| Formal analysis | `scripts/21a`--`21e` compute R-squared, reallocation, restrictions, error differentials, and portfolio outputs. | `outputs/formalanalysis/analysis/{model}/{spec}/` |
| Evaluation realignment | `scripts/eval_realignment/41`--`47` compute deployment-weighted R-squared, capacity portfolios, the gate, two-by-two summaries, long-only results, bootstrap inference, and scale diagnostics. | `outputs/eval_realignment/analysis/xgboost/{spec}/` |
| Paper tables | `scripts/build_paper_tables.py` reads cached outputs and overwrites LaTeX tables. It does not rebuild upstream estimates. | All 25 current files in `paper/Tables/` |

I confirmed the critical builder contract: every current `.tex` file in `paper/Tables/` has an owner in the builder's `BUILDERS` mapping; captions and most table-specific assertions are in that script. The builder is therefore part of the empirical specification, not merely formatting code.

## 2. NUMBER AUDIT

### 2.1 Load-bearing values that reconcile

I compared the claims against the underlying files at their stored precision. The table displays enough digits to identify the source values without reproducing every machine-precision decimal.

| Claim | Audited cached value | Source | Status |
|---|---:|---|---|
| Conventional pooled zero-benchmark R-squared | 0.2700807% | `outputs/motivation/step3/xgboost/dvol/r2_by_quintile.csv` | Matches 0.27% |
| Fit in illiquid Q1 | 0.6029135% | same file | Matches 0.60%; Q2--Q5 are non-positive |
| Spanning all-characteristic / excluding-liquidity means | 0.9109846 / 0.5446683 | `outputs/motivation/step1/dvol/weight_regression_meta.json`, 431 months | Matches 0.911 / 0.545 |
| Focal heterogeneity joint statistic | 21.95338 | `outputs/motivation/step2/dvol/interaction_meta.json` | Matches F=22.0; inferential qualification in Section 4.3 below |
| Primary $500M cells: 1A / 1B / 2A / 2B | -0.1462002 / 0.3842279 / -0.0532096 / 0.4509491 | `outputs/eval_realignment/analysis/xgboost/tc_rank_lam3_500m/two_by_two_500M.csv` | Matches -0.15 / +0.38 / -0.05 / +0.45 |
| Training effect | +0.0929906, p=0.2507499 | same directory plus `inference_supplement.csv` | Matches +0.09, p=0.25 |
| Execution effect | +0.5304280, p=0.00019996 | same | Matches +0.53, p=0.0002 |
| Total effect | +0.5971493, p=0.00039992 | same | Matches +0.60, p=0.0004 |
| Interaction | -0.0262693, two-sided p=0.7866 | same | Matches -0.03 and insignificance |
| Dose response: conventional / deployment-weighted legs | -0.0859703 / +0.1833634 | `outputs/formalanalysis/analysis/xgboost/tc_rank_lam3_500m/{prediction_quantile,prediction_quantile_signal_weight}/stock_universe/full/two_by_two_500M.csv` | Matches -0.09 / +0.18 |
| Joint dose contrast | +0.2693337, p=0.00159968 | `outputs/eval_realignment/analysis/xgboost/tc_rank_lam3_500m/inference_supplement.csv` | Matches +0.27, p=0.0016 |
| Mean of four predicted-positive cells | +0.1904703, p=0.0747850 | same | Matches +0.19, p=0.075 |
| Gate medians | 48.7558 / 30.5982 / 44.8418 bps | `outputs/eval_realignment/analysis/xgboost/tc_rank_lam3_500m/gate_scale_diagnostics.csv` | Matches 49 / 31 / 45 bps; these are time-series means of monthly cross-sectional medians, as the paper says |
| Gate pass rates | 44.7831% / 28.1403% | same | Matches 44.8% / 28.1% |
| Signal dispersion ratio | 0.6847868 | same | Matches 0.685 |
| Dispersion-matched contrast | -0.0060755, p=0.51230 | same | Matches -0.006, p=0.51 |
| Conventional anatomy | gross SR 1.65613; net SR -0.50708; cached cost 223.299 bps/month | `outputs/formalanalysis/analysis/xgboost/tc_rank_lam3_500m/prediction_quantile/stock_universe/full/two_by_two_timeseries_500M.xlsx` | The Sharpe values match; the cost unit is misstated, below |

The primary capital grid also reconciles:

| Scenario | Training effect (p) | Execution effect (p) | Total effect (p) |
|---|---:|---:|---:|
| Half-spread only | -0.08231 (0.6751) | +0.30043 (0.0002) | +0.21332 (0.08018) |
| $100M | -0.00580 (0.5005) | +0.40158 (0.0002) | +0.38124 (0.005399) |
| $500M | +0.09299 (0.25075) | +0.53043 (0.0002) | +0.59715 (0.000400) |
| $1B | +0.16545 (0.12018) | +0.62435 (0.0002) | +0.75882 (0.0002) |

The point estimates are monotone. As explained below, the reported joint statistic tests the endpoints, not monotonicity.

I also reconciled the other main claim clusters, rather than limiting the check to the requested headline list:

| Claim cluster | Audited values | Source and assessment |
|---|---|---|
| Panel imbalance and divergence | Q1 contains 38.5% of evaluation stock-months; 105/113 characteristics diverge significantly | Step 1 outputs and generated motivation tables; matches |
| Slope heterogeneity | focal reversal slope 0.0260 in Q1 and 0.0031 in Q5; 10/15 focal interactions significant | `outputs/motivation/step2/dvol/quintile_fm_coefficients*.csv` and `interaction_regression.csv`; matches |
| Deployment-weighted prediction fit | full panel: 0.0496651% standard, 0.1284110% weighted, +0.0787458 pp; full-sample-breakpoint Q4--Q5 gain +0.1405570 pp | `outputs/eval_realignment/analysis/xgboost/tc_rank_lam3_500m/liquidity_breakpoints/*/deployment_weighted_r2.csv`; matches the paper's +0.08/+0.14 rounding |
| Feature reallocation | rule-based illiquidity cluster 12.80% to 5.85%, -6.95 pp, t=-11.28; native motivation ratio 11.8% importance versus 2.7% liquid predictive content | `outputs/formalanalysis/analysis/xgboost/tc_rank_lam3_500m/importance_shift.csv` plus builder group aggregation; matches |
| Long-only $500M two-by-two | cells 0.340111 / 0.470553 / 0.326871 / 0.492935; training -0.013240 (p=0.6465), execution +0.130442 (p=0.0002), total +0.152824 (p=0.0002) | `longonly_two_by_two_500M.csv` and full inference supplement; matches |
| Primary factor-alpha ranking | 2B CAPM alpha 2.02% per year, t=2.23; the 2B-minus-1B alpha differential is insignificant | primary `two_by_two_500M.csv`; matches and is not presented as a significant difference |
| Comparison weighting blocks | $500M total effects +0.83 (TC level) and +0.58 (volume-rank), each p<=0.001 | generated capacity table and corresponding spec outputs; matches |
| Conventional band effect | cached standard effect +0.87, p=0.0002; cost 223 to 85 bps and turnover 0.96 to 0.39 in cached units | formal full-universe time-series workbook; transcription matches, but both the hysteresis construction and level units are invalid as currently described |
| Gross-effect table | primary gross training/execution/total about -0.16/-0.18/-0.50; tested training and total p-values match | `paper/Tables/GrossEffects.tex` and its builder inputs; numerical entries match, while the universal significance claim exceeds the tests |

Across the remaining regime, alternative-weight, linear-benchmark, training-scale, and value-weighted tables, I found no additional output-to-paper transcription discrepancy. Their interpretation remains conditional on the common tuning, cache, cost-model, and portfolio-construction findings below.

### 2.2 Mismatches and unit errors

#### [MUST-FIX] N1. Conventional long--short levels are twice the value implied by the stated capital denominator

**Exact paper text:** `paper/Sections/04_main_results.tex:153` says the portfolio is “**allocating $A/2$ to each leg**.” `paper/Tables/ConventionalLadder.tex:20` says “**the long--short row is the dollar-neutral $Q5-Q1$ portfolio with $250M per leg**.” The prose then says “**a mean monthly cost drag of $223$ basis points against a gross spread of $176$**” (`04_main_results.tex:156`).

**Actual implementation:** `_selected_leg_weights` in `src/portfolio/construction.py:147-181` normalizes each leg to one. `build_long_short_portfolio` at lines 377-403 reports `ret_long-ret_short`, rather than one half of that spread. Physical impact is correctly sized at $A/2$ per leg by `_leg_aum`, but lines 960-1013 similarly sum both leg-normalized cost drags without the one-half portfolio-capital factor.

For the primary 1A book, the cached monthly gross/cost/net values are 176.085 / 223.299 / -47.213 bps and turnover 0.9621. Per dollar of the stated $500M **total gross**, the correct values are:

- gross return: **88.043 bps/month**;
- transaction-cost drag: **111.649 bps/month**;
- net return: **-23.607 bps/month**;
- one-way turnover: **0.4810**.

The same one-half correction applies to all conventional long--short mean-return, cost, turnover, certainty-equivalent, and alpha **levels**, including the banded rows. It does not apply to the standalone full-$A$ quantile portfolios. A common one-half rescaling leaves Sharpe ratios, Sharpe differences, bootstrap p-values, signs, and alpha t-statistics unchanged. Thus the quoted “SR 1.66 to -0.51” anatomy survives, while “223 against 176,” the cost/turnover ladder, and Panel C alpha magnitudes do not.

#### [SHOULD-FIX] N2. “Significant at every capital level” is needlessly ambiguous

**Exact paper text:** the abstract says the total effect is “**$+0.60$ ($p<0.001$), significant at every capital level**” (`paper/Main.tex:114`).

The $+0.60$ and $p<0.001$ correctly refer to $500M. The total effect is significant at 5% for all actual dollar-capital points—$100M, $500M, and $1B—with p-values 0.005399, 0.000400, and 0.000200. It is not significant in the half-spread-only benchmark, p=0.08018; Section 4 explicitly says no capital level applies to that scenario. Replace the sentence with: “The total effect is +0.60 at $500M (p=0.0004) and is significant at each dollar-capital point (p<=0.0054), but not in the half-spread-only benchmark (p=0.080).”

#### [SHOULD-FIX] N3. The displayed `N_train` values are not averages over realized rolling training windows

**Exact paper text:** `paper/Tables/ScreeningSplitting.tex:24` says “**$N_{train}$ averages over the rolling training windows**”; `paper/Sections/02_implementability_imbalance.tex:80` repeats that the 6,776 figure is “**the average across the rolling training windows**.”

Scripts 05 and 06 instead compute the mean raw monthly cross-sectional count from the whole input panel (`scripts/05_motivation_step3d_progressive_restriction.py:354-360`; `scripts/06_motivation_step3e_quintile_specific_models.py:216-232`). They do not count observations actually consumed in each 120-month fit. The displayed values should be relabeled “eligible names per month” or recomputed from the realized training blocks.

Apart from N1--N3 and the claim/test scope problems below, I found **no additional transcription mismatch** in the load-bearing values requested for audit. In particular, the paper correctly uses native XGBoost gain—not SHAP—for its motivation Figure 3, the two-by-two statistics come from the `two_by_two_timeseries_{AUM}.xlsx` sheets rather than the standalone quantile CSV, and the primary capacity portfolios use signal/deployment weighting rather than 21e's legacy equal-weight default.

## 3. INTERNAL CONSISTENCY

#### [MUST-FIX] IC1. “Weighting is the only treatment” contradicts the realized tuning experiment

**Exact paper text:** `paper/Sections/03_framework.tex:46` says “**the model class, feature set, tuning protocol, and estimation window are held identical ... so any measured difference is attributable to the weighting alone**.” Line 122 says “**the weighting is the only treatment**,” while its footnote calls the standard candidates “**a broader superset**” that “**if anything, favours the benchmark**.”

The appendix discloses different Cartesian spaces—6,480 combinations for standard and 1,944 for weighted—but the actual procedure draws 150 candidates separately from each space with `ParameterSampler` (`src/models/xgboost_model.py:195-202`; `src/training/rolling.py:61-83`). Reconstructing the seeded menus gives only **3 common candidates out of 150**. Only 55 of the 150 broad-space draws even lie in the narrow Cartesian universe; none of the 25 cached standard winners is in the realized weighted menu, and none of the 25 weighted winners is in the realized standard menu. Eight standard winners are outside weighted support altogether. A larger search space under a fixed 150-draw budget can help or hurt through menu composition and validation-selection error; it is not a conservative nested comparison.

The paper may describe the existing result as a comparison of two historical pipelines, but it cannot identify weighting alone. The definitive repair is to persist one candidate manifest and give exactly that realized menu to both arms, then regenerate every dependent output.

#### [MUST-FIX] IC2. The final executed portfolio does not satisfy the paper's gate equation

**Exact paper text:** Equation (gate) and the following sentence say that on failure “**the name is left at its drifted weight**” and a name with no inherited position “**never opens**” (`paper/Sections/04_main_results.tex:94-103`). The introduction says the gate “**trades only when a name's centred signal covers its half-spread**” (`01_introduction.tex:17`).

`src/analysis/eval_realignment/breakeven_capacity.py:123-148` constructs that raw target-or-drift vector, after which lines 151-168 and 218-220 rescale every surviving long and short to restore neutrality and unit gross. On the primary cache, the nominal pass rate is 44.7639%, yet 95.6302% of names move relative to drift and 92.0888% of nominal failures move. Failure-name trades account for 8.7986% of pooled traded notional. All such induced trades are charged, so this is not a hidden missing-cost bug; it is a mismatch between the stated economic rule and the implemented normalized projection. The appendix partially admits it: “**a name the gate holds at its drifted weight is moved slightly by [the rescale]**” (`paper/Appendix.tex:259`).

Define the piecewise vector as a raw $\tilde\theta$, define final holdings as an explicit normalization $N_t(\tilde\theta)$, and report nominal-pass versus induced-trade diagnostics. If literal no-trade failures are intended, redesign the projection and rerun scripts 43--47. Also call this a **half-spread gate**: the threshold omits price impact, as the paper itself admits at `04_main_results.tex:114` and `06_conclusion.tex:14`.

#### [MUST-FIX] IC3. An endpoint contrast is repeatedly called a jointly significant trend

**Exact paper text:** the introduction says the capital effect has “**a trend that is itself jointly significant**” (`01_introduction.tex:21`); Section 4 says “**The upward slope is itself jointly significant**” (`04_main_results.tex:140`); the conclusion says “**the capital-grid trend is itself jointly certified**” (`06_conclusion.tex:12`).

The stored statistics are explicitly `*_effect_1B_minus_PropTC_annualised` in `inference_supplement.csv`. They test $1B minus the half-spread-only endpoint. They do not test a linear slope, monotonicity, or an order restriction across the intermediate cells. The accurate statement is: “Point estimates rise monotonically; the end-to-end PropTC-to-$1B increase is significant, p=0.0002.” A true trend claim needs a slope or order-restricted joint statistic.

#### [MUST-FIX] IC4. “No gross effect anywhere” extends beyond the tests run

**Exact paper text:** the abstract says “**nowhere is a gross effect significantly positive**” (`paper/Main.tex:114`); the conclusion says “**no specification anywhere in the design earns a significantly positive gross effect**” (`06_conclusion.tex:12`). The generated caption simultaneously says “**Gross execution carries no separate pairwise test**” (`paper/Tables/GrossEffects.tex:13`).

The builder tests training and total gross contrasts for three Section 4 specifications at $500M. It does not test gross execution separately and does not span “anywhere in the design.” An untested effect cannot be declared insignificant. Either run the missing execution and wider-scope tests or narrow the claim to: “Across the three Section 4 specifications at $500M, neither tested training nor tested total gross contrast is significantly positive; gross execution is descriptive.”

#### [MUST-FIX] IC5. The final focal-15 list was not fixed before any regression

**Exact paper text:** the footnote at `paper/Sections/02_implementability_imbalance.tex:55` says “**The focal set was fixed on economic grounds before any regression was estimated**.”

Repository history shows that commit `9cf3eb9` created the initial set with `ChNAnalyst`; commit `8274855` then added Step 2 regressions and result files; commit `3061bc8` subsequently replaced that characteristic with `AnnouncementReturn`, explicitly because missingness was 93% rather than 35%. This is not evidence of t-statistic selection, and the all-113 results materially reduce the concern, but the literal provenance claim is false. Disclose the coverage-driven substitution and say the set was not selected on the reported coefficients unless external preregistration evidence exists.

#### [SHOULD-FIX] IC6. The consistency principle is stated more literally than the code implements

**Exact paper text:** the conclusion says “**one mean-one implementability weight, carried unchanged through the training loss, the evaluation metric, and the cost accounting**” (`paper/Sections/06_conclusion.tex:8`).

The same weight directly enters the loss, prediction metric, and target holdings. Realized cost is then computed from actual changes in those holdings using the same cost primitive at actual trade size; it is not an explicit mean-one-weighted cost sum, and its imputation convention differs from training-stage weight construction. Use the exact statement: “The same weight enters the loss, error metric, and target holdings; realized costs apply the same cost primitive to the resulting trades.”

#### [SHOULD-FIX] IC7. “Never a standalone repair” is too absolute

**Exact paper text:** the conclusion says weighted training “**is never a standalone repair**,” then immediately says that the plain cost tilt “**clears conventional significance at every capital level**” (`paper/Sections/06_conclusion.tex:12`).

The intended low-base caveat is reasonable, but the absolute and its exception conflict. Say “not a robust standalone repair under the primary and rank-based specifications; the plain-level tilt is the disclosed low-base exception.”

#### [MINOR] IC8. The momentum claim needs the focal-regression qualifier

**Exact paper text:** the introduction says “**momentum predicts returns only among liquid names**” (`paper/Sections/01_introduction.tex:13`). That is the focal-15 regression result. The appendix reports the opposite location in the all-113 specification. Add “in the focal specification.”

I found no classic difference-in-significance error in the primary four-cell discussion: the paper correctly distinguishes significant individual alphas from an insignificant 2B-minus-1B alpha difference, and it correctly labels the +0.09 training effect insignificant. The problems above are missing tests and overextended scopes, not merely one estimate being significant while another is not.

## 4. METHODOLOGY

### 4.1 Portfolio construction and identification

#### [MUST-FIX] M1. The conventional hysteresis portfolios frequently hold the same stock long and short

**Exact paper text:** the conventional portfolio is described as “**long the top quintile, short the bottom**” (`paper/Sections/04_main_results.tex:153`), and its band is said to execute replacement “**only when the entrant's edge over the incumbent covers the round-trip cost**” (`04_main_results.tex:159`; see also `paper/Appendix.tex:267`).

In `_apply_hysteresis_bands` (`src/portfolio/construction.py:187-239`), retained former longs are appended after the fresh short leg has already been selected. The function never removes those retained longs from the fresh short leg. The later “never hold a name on both legs” guard protects only newly retained shorts.

Read-only reconstruction from cached XGBoost predictions shows:

- standard model: overlap in **279/299 months**, 38,703 overlapping name-months, mean 129 per month;
- primary weighted model: overlap in **284/299 months**, 83,272 overlapping name-months, mean 279 per month;
- after netting duplicates, mean actual two-leg gross is 1.923 standard and 1.853 weighted, versus nominal gross 2; minima are 1.524 and 1.461.

Return algebra implicitly nets the opposing positions, while transaction-cost code charges the two leg positions separately. The exposure error differs by training arm. Consequently all conventional **B** cells, conventional two-by-two decompositions, hysteresis effects, dose-response total/execution claims, value-weighted variants, and conventional training-scale results require reconstruction. Standalone per-quantile portfolios and the long-only implementation do not use this conflicting two-leg path.

This is economically material, not merely a set-membership nicety. A diagnostic that only nets the cached duplicate positions changes the standard $500M B-cell Sharpe from 0.3598 to 0.3265 and the weighted B-cell from 0.4145 to 0.4214; a simple long-precedence no-overlap counterfactual moves the standard result to 0.3900. Those are not proposed corrected estimates—a proper conflict rule changes membership and must be rerun—but they show that the error is large enough to affect the reported decomposition.

Resolve conflicts deterministically, assert disjoint legs and signed gross every month, add transition tests, and rerun 21e/22/46 plus dependent paper tables. Until then, “a cost-scaled execution band restores [the spread]” is not established for the portfolio defined in the text.

#### [MUST-FIX] M2. The training-by-execution design does not isolate its training treatment

This is the econometric consequence of IC1. The paper's exact identifying sentence—“**any measured difference is attributable to the weighting alone**”—is false under materially different realized candidate menus. The primary +0.09 training effect is insignificant, but the confound also affects the weighted forecasts used in every 2A/2B, R-squared, dose-response, and mechanism comparison. A common candidate manifest is a prerequisite, not a robustness option.

#### [MUST-FIX] M3. The central gate is a normalized half-spread rule, not a literal breakeven/no-trade rule

This is the methodological consequence of IC2. The appendix's disclosure makes the cached results interpretable as “raw half-spread gate plus global leg re-truing, with all induced trades charged.” It does not make Equation (gate), “trades only,” or “left at drift” correct. Nor is $|\alpha|\geq\text{half-spread}$ the exact incremental trade condition from an arbitrary inherited position. The paper must choose and defend the actual constrained portfolio map.

### 4.2 Rolling protocol and leakage

The core rolling chronology is otherwise well designed: a 120-month training window precedes a 12-month validation block, which precedes each test month; retuning occurs annually and parameters are frozen between retunes. I found no direct use of future returns in fitting a past forecast.

#### [SHOULD-FIX] M4. The feature universe uses an unsupervised full-sample look-ahead

**Exact paper text:** `paper/Sections/03_framework.tex:28` openly says “**The missingness screen is computed once on the full 1989--2024 panel, including the out-of-sample years**,” while line 122 says “**no test-period observation feeds back into parameters or hyperparameters**.”

This is not target leakage, but it is pseudo-out-of-sample conditioning on future feature availability. Freeze the feature list using the initial training period, or show that a pre-2000/rolling-availability universe leaves results unchanged.

#### [SHOULD-FIX] M5. “One-month-ahead” targets can jump across multi-month gaps

**Exact paper text:** `paper/Sections/03_framework.tex:24` says “**The prediction target throughout is the one-month-ahead excess return ... month-t characteristics predict r(i,t+1)**.”

`src/data/loader.py:139-142` uses `groupby("permno").shift(-1)` without requiring the next row to be the next calendar month. In the current panel, 1,274 rows with a non-null shifted target cross a calendar gap, among 2,920,455 processed rows; examples include 201512 to 201809 and 200206 to 202103. The incidence is small (0.0436%) but the target contract is wrong. Require exact month adjacency. The “12-month” volatility helper is likewise 12 observations rather than necessarily 12 calendar months.

### 4.3 Inference

#### [SHOULD-FIX] M6. The focal joint F-test is iid Hotelling inference, not HAC time-series inference

**Exact paper text:** `paper/Sections/02_implementability_imbalance.tex:55` reports “**the joint test of gamma=0 delivers F=22.0 (p<0.001)**” immediately after describing Newey--West standard errors for the coefficient averages.

`_joint_gamma_f_test` in `src/analysis/motivation.py:1252-1287` uses the ordinary sample covariance of monthly coefficient vectors and the finite-sample Hotelling F transformation. Individual t-statistics use Newey--West; the joint test does not. The point statistic 21.953 is correctly reported, but its p-value is not serial-correlation robust. Use a HAC Wald test or a joint block bootstrap, or label the existing statistic as an iid Hotelling test.

#### [SHOULD-FIX] M7. The inference plan is only partly pre-specified and does not address the effective search

**Exact paper text:** `paper/Sections/03_framework.tex:71` says “**The tilt beta=3 was fixed ex ante and not tuned**”; Section 4 fixes $500M as primary; Sharpe contrasts are one-sided (`04_main_results.tex:112`).

Repository history supports that beta=3 and the $500M training tilt were in code before the primary weighted fit. It is not an external preregistration, however, and the gate/evaluation design was developed after cached forecast production. The paper examines multiple weight families, capital levels, portfolio objects, universes, learners, execution rules, and one-sided contrasts. The execution effect is large enough that modest multiplicity corrections will not erase it, but borderline training/mechanism claims require an explicit specification ledger, family definitions, and adjusted or holdout evidence.

The circular-block bootstrap is reproducible (B=5,000, block length 6, seeded), but six months is weakly motivated. For the 120-month-overlapping SHAP windows, the paper itself admits that six lags are “**conservative in name only**” (`04_main_results.tex:171`). That mechanism should be treated as a point-estimate result until inference uses a dependence horizon commensurate with the overlap. Front-matter p-values should also say they are one-sided.

### 4.4 Cost model and economic design

#### [MUST-FIX] M8. The dominant economic result is conditional on an unvalidated cost level

**Exact paper text:** `paper/Sections/03_framework.tex:83` says the high--low spread is “**necessarily a noisy proxy**,” that external spread-level validation “**is deferred to future work**,” and that net magnitudes “**are stated conditional on the cost model**.”

That caveat is candid, but it belongs in the abstract and conclusion because +0.53 of the +0.60 primary effect is execution, and both selection by the gate and realized cost accounting consume the same spread proxy. A shared error series can preserve within-design comparability while still mechanically amplifying an avoided-cost result. Validate levels against an independent effective-spread/implementation-shortfall source, vary impact calibration and execution horizon, and report break-even sensitivity. The claim should be “conditional on this cost model” until then.

#### [SHOULD-FIX] M9. The implementability weight is a disciplined heuristic, not an observed deployment density ratio

**Exact paper text:** the introduction calls the weight “**a tractable proxy for the density ratio between deployment and training distributions**” (`paper/Sections/01_introduction.tex:17`).

The rank-softmax at beta=3 is attractive because it controls concentration, but it is not estimated from institutional holdings, derived from an investor's optimum, or externally validated as a deployment distribution. The same chosen weight defines the weighted loss, weighted metric, and capacity holdings, so some consistency is mechanical by construction. Report results under an observed holdings/capacity proxy or derive the weight from an explicit investor problem. The raw-volume failure and alternative families are useful sensitivity checks, not external validation.

### 4.5 Reproducibility and engineering controls

#### [SHOULD-FIX] M10. Cache completeness is file-existence based and permitted mixed experimental generations

**Exact paper text:** the protocol says the standard fit “**is estimated once and shared across all weighted specifications**” (`paper/Sections/03_framework.tex:46,122`).

`scripts/04_motivation_step3_ml_diagnostics.py` and `scripts/20_formal_run_experiment.py` treat artifact existence as cache completeness. Metadata do not bind a dataset hash, feature-list hash, realized candidate manifest, code revision, or complete rolling contract. The present cache mixes a May 10 standard fit, May 30 weighted fits, and later configuration/code. That is the concrete mechanism behind the tuning confound. Every fitted directory should carry a provenance manifest, and consumers should fail closed on mismatch.

#### [MINOR] M11. Paper generation is complete but not atomic

**Exact project premise:** every table is “**GENERATED by scripts/build_paper_tables.py**.” This is correct for all 25 current tables. However, lines 1621-1625 overwrite them sequentially; a late assertion can leave a mixed-generation paper. Build in a staging directory, validate every table and source hash, then atomically publish the set.

#### [MINOR] M12. Several live documentation promises no longer describe the executed pipeline

**Exact text examples:** `docs/portfolio_construction.md:89` says “**the book trades only the top 40%**,” while the live restriction is top 60%; `scripts/eval_realignment/45_longonly_capacity_q5.py:28` promises “**Panel C = hysteresis diagnostics**,” while its call at line 303 passes `include_panel_c=False`; and `AGENTS.md:208` says each standalone quantile uses “**AUM / portfolio.n_quantiles**,” while live code and the current paper use full AUM for each standalone quantile. Legacy 2x3 language and an incomplete 41--47 run order also persist. The current parquet additionally contains 2,977 zero `liq_dvol_21d` observations although current preprocessing converts such zeros to missing. These do not overturn the headline cached calculations, but they reinforce the need for artifact provenance and a documentation refresh.

## 5. CONTRIBUTION

#### [SHOULD-FIX] C1. The description of Gu--Kelly--Xiu is selective

**Exact paper text:** `paper/Sections/01_introduction.tex:31` says Gu--Kelly--Xiu “**fixed the conventions this paper re-examines—the equal-weighted loss, the pooled R2, the equal-weighted decile spread**.”

That is a recognizable part of their empirical setup, but it understates their value-weighted portfolio evidence and their conclusion that ML is particularly valuable among larger, more liquid stocks. The positioning should acknowledge both, then explain why neither is the same as a declared deployment distribution carried into training and evaluation. See [Gu, Kelly, and Xiu (2020)](https://academic.oup.com/rfs/article/33/5/2223/5758276).

#### [MINOR] C2. The comparison with Jensen--Kelly--Malamud--Pedersen is unusually honest

**Exact paper text:** the introduction says their headline Portfolio-ML “**trains a characteristics-to-weights mapping end to end on utility net of trading costs, so costs enter their training objective as squarely as ours**” (`paper/Sections/01_introduction.tex:29`).

That is fair. The real distinction is also stated correctly: this paper retains a reusable return forecast and supplies optimizer-free diagnostic/evaluation objects, whereas Portfolio-ML directly learns portfolio weights. See the [RFS Portfolio-ML article](https://academic.oup.com/rfs/advance-article/doi/10.1093/rfs/hhag022/8524346). The problem is sufficiency, not misdescription: once the paper's significant result is mainly a cost-aware execution rule and the training effect is insignificant, much of the economics lies close to the existing cost-aware portfolio literature.

#### [SHOULD-FIX] C3. The distinction from Avramov--Cheng--Metzker is plausible but not yet demonstrated on a common design

**Exact paper text:** the introduction says their result is “**our motivation, not our result: restriction destroys transferable signal, so the correction belongs upstream**” (`paper/Sections/01_introduction.tex:31`).

The restriction-versus-reweighting distinction is real, and the restriction curves are useful. But the current comparison does not put both approaches through a common held-out cost model, tuning menu, and capacity portfolio. Given the tuning confound and the static full-sample feature universe, the evidence is not yet strong enough to conclude that upstream reweighting is the superior correction. See [Avramov, Cheng, and Metzker](https://si-cheng.net/wp-content/uploads/2023/05/2023-ms-avramov_cheng_metzker-machine-learning-vs.-economic-restrictions.pdf).

#### [MUST-FIX] C4. The marginal top-journal contribution is not yet isolated from implementation choices

**Exact paper text:** the conclusion's summary is “**it was the metric, not the method**” (`paper/Sections/06_conclusion.tex:8`).

The metric contribution is potentially valuable: the paper gives asset-pricing researchers a transparent way to declare the deployment measure and demonstrates how conventional evaluation can reverse the sign of a training comparison. But the statistically strong economic result is execution, not training; the gate is explicitly not claimed as novel; the weight is heuristic; and key conventional robustness portfolios are currently malformed. For a top finance journal, a careful accounting framework plus an insignificant training lever is not enough without an externally validated, genuinely held-out demonstration that the framework changes investment conclusions beyond choices made in this sample.

## 6. REFEREE VERDICT

### Strongest hostile-referee objections

1. **[MUST-FIX] The training experiment is not a treatment experiment.** The paper says “weighting is the only treatment,” but the realized XGBoost candidate menus barely overlap. **Does the text pre-empt it?** No. It discloses different Cartesian spaces, then incorrectly asserts the asymmetry favors standard. Only a same-manifest rerun resolves the objection.
2. **[MUST-FIX] The central execution object is not the object in the equation, and its economic magnitude depends on the same unvalidated spread proxy used to trigger it.** **Does the text pre-empt it?** Partly. The appendix discloses post-gate rescaling and Section 3 candidly conditions levels on the cost model, but the abstract, introduction, equation, and conclusion retain literal “trades only,” “left at drift,” “breakeven,” and unconditional headline language.
3. **[MUST-FIX] The conventional benchmark that is supposed to deliver the dose-response identification is not a valid disjoint, constant-gross Q5-minus-Q1 book, and its level units are wrong.** **Does the text pre-empt it?** No. Neither simultaneous long/short membership nor the factor-of-two denominator is disclosed; both are inconsistent with the stated construction.

### Recommendation

**Reject in its current form.** I would encourage a substantially rebuilt submission rather than dismiss the idea. The cached numerical transcription is unusually careful, the paper is frank that training alone is weak, and the deployment-consistent evaluation concept is worth pursuing. But correcting the three central design failures requires rerunning affected experiments, not revising prose around a stable empirical object.

**Single highest-value improvement:** conduct one frozen-pipeline replication on a genuinely untouched period or market. Before opening that holdout, lock (i) one identical realized hyperparameter manifest for both training arms, (ii) mathematically coherent gate and disjoint/constant-gross hysteresis code with invariant tests, and (iii) an externally validated cost calibration. Then run the full two-by-two once and report the complete specification ledger. That one exercise would simultaneously establish whether the training contrast is causal, whether the execution result survives corrected construction, and whether the economics travel beyond the design sample.

### Priority order for the authors/developers

1. Fix and test hysteresis leg disjointness and conventional total-gross normalization.
2. Choose the intended final gate map; make equation, code, diagnostic, and label identical.
3. Rerun standard and weighted XGBoost with one persisted candidate manifest.
4. Regenerate all dependent formal/evaluation outputs and all 25 paper tables from provenance-locked caches.
5. Replace endpoint-as-trend and untested-gross claims; correct the focal-set provenance and unit captions.
6. Validate cost levels externally and reserve a true holdout for the corrected, frozen pipeline.
