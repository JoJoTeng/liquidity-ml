# Step 1: The Distributions Differ

## Objective

Establish that the training distribution P_train(x) (equal-weighted cross-section) and the deployment distribution P_deploy(x) (implementability-weighted, i.e. dollar-volume-weighted) place mass on systematically different regions of the characteristic space. Crucially, the divergence must extend beyond trivially mechanical liquidity dimensions to non-liquidity characteristics like momentum, value, and profitability.

## Data and Setup

- **Sample:** Full CRSP–CZ merged panel, all available months (431 months used).
- **Liquidity measure:** 21-day trailing average dollar volume (`liq_dvol_21d`), following Eq. (2) of the motivation document.
- **Implementability weight:** w̃_it = DolVol_it / mean(DolVol_t), normalized to mean 1 within each cross-section (Eq. 3).
- **Characteristics:** ~113 Clear Predictor signals from Chen & Zimmermann (2022), plus CRSP-derived features (STreversal, Size, Price). Discrete/binary features excluded. All cross-sectionally rank-transformed to [0, 1] within each month (Gu et al. 2020).
- **Economic categories:** 8 broad groups (Liquidity, Value, Momentum, Profitability, Investment, Risk, Quality, Other) mapped from CZ `Cat.Economic` fine labels.
- **Statistical inference:** Newey-West standard errors with 6 lags throughout.

## Methods and Outputs

### Output 1.1 — Marginal Divergence Bar Chart

**Method:** For each characteristic j and each month t, compute:
- Equal-weighted mean: x̄^train_j,t = (1/N_t) Σ_i x_{ij,t}
- Dollar-volume-weighted mean: x̄^deploy_j,t = Σ_i w̃_it · x_{ij,t} / Σ_i w̃_it
- Divergence: d_j,t = x̄^deploy_j,t − x̄^train_j,t

Take the time-series average d̄_j = (1/T) Σ_t d_j,t and compute Newey-West t-statistics (6 lags).

**Output files:**
- `dvol/divergence_bar_chart.png` — Main paper: category-level bar chart (8 bars)
- `dvol/divergence_bar_chart_appendix.png` — Appendix: all ~113 characteristics, sorted by |d̄_j|, color-coded by category
- `dvol/divergence_stats.csv` — Full table: feature, d̄, std error, t-stat, p-value, |d̄|
- `dvol/divergence_monthly.parquet` — Monthly divergence panel (rows=months, cols=features)

### Output 1.2 — Divergence by Category Table

**Method:** For each of the 8 broad economic categories, aggregate the average absolute divergence |d̄_j| and count the number of characteristics with |t| > 2.

**Output file:** `dvol/divergence_by_category.csv`

### Output 1.3 — Fama-MacBeth Weight Regression

**Method:** Each month, run a cross-sectional OLS regression:

    log(w̃_it) = x'_it δ_t + ε_it

using all ~113 rank-transformed characteristics as regressors. Log transform reduces the influence of extreme right-skewed weights (a few mega-caps have w̃ > 100). Report:
- Time-series average R² across months
- Fama-MacBeth coefficients δ̄_j with Newey-West t-statistics (6 lags)
- Ranked table of top 15 characteristics by |δ̄_j|

**Output files:**
- `dvol/weight_regression_meta.json` — R̄², R² median, number of months
- `dvol/weight_regression_top15.csv` — Top 15 characteristics by |δ̄|
- `dvol/weight_regression_all.csv` — Full coefficient table
- `dvol/weight_regression_r2_monthly.csv` — Monthly R² time series

### Output 1.4 — Density Comparison Plots

**Method:** For 6 focal characteristics, overlay:
- Training distribution: flat line at y=1.0 (rank-transform to [0,1] is Uniform by construction)
- Deployment distribution: dollar-volume-weighted kernel density estimate

Organized as a 2-panel figure: Panel A (liquidity-related: Illiquidity, BM, STreversal) and Panel B (non-liquidity: IdioVol3F, Mom12m, AnnouncementReturn).

**Output file:** `dvol/density_comparison.png`

### Output 1.5 — Weight Distribution Histogram

**Method:** Histogram of log₁₀(w̃) pooled across all months, with percentile summary of the raw w̃ distribution (5th, 25th, 50th, 75th, 95th). Includes optional value-weight (market cap) overlay for comparison.

**Output file:** `dvol/weight_distribution.png`

## Results

### Distributional Divergence by Category

| Category      | Avg. \|d̄\| | # Significant (\|t\|>2) | # Characteristics |
|---------------|-------------|-------------------------|-------------------|
| Liquidity     | 0.2729      | 10/10                   | 10                |
| Value         | 0.1065      | 10/10                   | 10                |
| Momentum      | 0.0903      | 11/15                   | 15                |
| Other         | 0.0893      | 13/13                   | 13                |
| Profitability | 0.0825      | 14/15                   | 15                |
| Risk          | 0.0581      | 9/11                    | 11                |
| Quality       | 0.0567      | 11/11                   | 11                |
| Investment    | 0.0548      | 27/28                   | 28                |

**Total: 105 out of 113 characteristics have |t| > 2** (93%).

### Top 15 Divergences (by |d̄_j|)

| Rank | Feature         | d̄      | t-stat   |
|------|-----------------|---------|----------|
| 1    | DolVol          | −0.436  | −311.8   |
| 2    | Illiquidity     | −0.429  | −282.6   |
| 3    | Size            | −0.421  | −276.4   |
| 4    | VolSD           | −0.387  | −87.1    |
| 5    | Price           | −0.342  | −81.3    |
| 6    | PriceDelayRsq   | −0.280  | −37.7    |
| 7    | zerotrade12M    | −0.249  | −30.1    |
| 8    | zerotrade6M     | −0.247  | −32.4    |
| 9    | zerotrade1M     | −0.239  | −37.9    |
| 10   | std_turn        | −0.230  | −47.9    |
| 11   | FEPS            | +0.230  | +36.1    |
| 12   | CBOperProf      | +0.196  | +44.0    |
| 13   | BM              | −0.192  | −28.3    |
| 14   | IntanSP         | −0.180  | −19.3    |
| 15   | roaq            | +0.177  | +52.0    |

### Fama-MacBeth Weight Regression

- **Average R² = 0.913** (median 0.926), computed over 431 months.
- This means that the stock-characteristic vector explains over 91% of the cross-sectional variation in implementability weights — the covariate shift is highly systematic and predictable from the feature space.

### Top 15 Regression Coefficients (by |δ̄|)

| Rank | Feature       | δ̄       | t-stat  |
|------|---------------|---------|---------|
| 1    | DolVol        | −3.245  | −42.2   |
| 2    | Size          | −2.943  | −39.8   |
| 3    | zerotrade1M   | −1.627  | −102.9  |
| 4    | RealizedVol   | −1.391  | −14.1   |
| 5    | VolSD         | −1.222  | −25.3   |
| 6    | IdioVolAHT    | +0.957  | +32.1   |
| 7    | Price         | −0.684  | −12.3   |
| 8    | Illiquidity   | −0.433  | −3.6    |
| 9    | MaxRet        | +0.419  | +9.3    |
| 10   | IdioVol3F     | −0.328  | −5.3    |
| 11   | BidAskSpread  | −0.317  | −8.0    |
| 12   | STreversal    | −0.264  | −21.1   |
| 13   | AM            | −0.172  | −4.0    |
| 14   | Herf          | +0.142  | +16.5   |
| 15   | NOA           | +0.129  | +26.6   |

## Analysis

### The divergence is broad, not just mechanical

The PDF's diagnostic criterion requires statistical significance for a substantial number of non-liquidity characteristics (at least 20–30 out of ~130). Our results vastly exceed this threshold: **105 of 113 characteristics** show |t| > 2. Every single category — including Value (10/10), Profitability (14/15), Momentum (11/15), and Investment (27/28) — has the majority of its characteristics significantly diverging. This is not a liquidity-only phenomenon.

### Non-liquidity characteristics diverge substantially

While liquidity characteristics naturally show the largest divergences (avg |d̄| = 0.273), the shift extends deeply into non-liquidity dimensions:
- **Value** (avg |d̄| = 0.107): BM diverges by −0.192 (t = −28.3), meaning the deployment distribution substantially underweights high-BM (value) stocks. Liquid stocks are more growth-oriented.
- **Profitability** (avg |d̄| = 0.083): CBOperProf (+0.196, t = +44.0) and RoE (+0.171, t = +49.7) shift sharply positive — liquid stocks are more profitable on average.
- **Momentum** (avg |d̄| = 0.090): Mom12m shifts +0.113 (t = +12.7), indicating liquid stocks have higher recent returns. STreversal shifts −0.033 (t = −8.8), showing less short-term reversal among liquid stocks.
- **Investment** (avg |d̄| = 0.055): AssetGrowth shifts −0.091 (t = −19.8), consistent with liquid stocks being more mature and investing less aggressively.

### The covariate shift is systematic

The average R² of 0.913 from the Fama-MacBeth weight regression far exceeds the PDF's threshold of 0.5. This means implementability is almost entirely predictable from the characteristic space — it is not random which stocks get high weight. The top regression coefficients confirm that DolVol and Size dominate (as expected), but non-mechanical features like RealizedVol (rank 4), IdioVolAHT (rank 6), STreversal (rank 12), and NOA (rank 15) contribute significantly even after controlling for direct liquidity measures.

### Direction of divergence is economically interpretable

The negative signs on most divergences indicate that the deployment distribution (liquid stocks) has lower characteristic values than the training distribution (all stocks). Since characteristics are rank-transformed to [0, 1], a negative d̄ means liquid stocks concentrate in the lower ranks. For example:
- Low Illiquidity, low VolSD, low zerotrade → liquid by construction (mechanical)
- Low BM, low AssetGrowth → growth-oriented, mature firms (non-mechanical)
- High FEPS, high CBOperProf, high RoE → profitable, well-covered firms (non-mechanical)

This pattern is consistent with the known correlation structure of the cross-section: liquid stocks tend to be large, profitable, growth-oriented, well-covered, and less volatile.

### Implication for the ML training problem

A model trained with equal weights allocates equal attention to all regions of the characteristic space. But the investor deploys predictions in a sharply different region — one concentrated among profitable, low-volatility, growth-oriented, highly traded stocks. The R² = 0.913 result means this mismatch is almost perfectly predictable, and the breadth of significant divergences (93% of characteristics) means the covariate shift operates across the entire feature space, not just along a few liquidity-adjacent dimensions.

**However, Step 1 alone does not establish that this mismatch is harmful.** If the return-generating function E[r|x] is the same everywhere, the model still learns the correct function — just with unevenly distributed precision. Step 2 addresses whether the mismatch is harmful by testing for heterogeneous predictability.
