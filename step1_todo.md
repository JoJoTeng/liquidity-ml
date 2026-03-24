# Step 1: Distributional Divergence — Complete TODO List

## Overview

Step 1 establishes that P_train(x) ≠ P_deploy(x) across a broad range of characteristics,
not just mechanical liquidity dimensions. It produces 5 outputs (1.1–1.5).

**Prerequisites before any output:**
- Data must include all ~150+ CZ characteristics (not just the current 75)
- Dollar volume (dvol_21d) must be available
- Exchange code (exchcd) must be available for NYSE breakpoints
- All characteristics must be rank-transformed to [0, 1] within each month

---

## Phase 0: Data Restructuring (do this FIRST)

### Task 0.1 — Audit current data pipeline

**Status:** RECHECK existing code

- [ ] **0.1a** Open `scripts/00_fetch_data.py` and document:
  - Which CRSP tables are queried (msf? dsf? msenames?)
  - Which columns are pulled (ret, prc, vol, shrout, exchcd, shrcd, etc.)
  - How CZ signed predictors are merged (by permno + date?)
  - What the output file looks like (`data/signed_predictors_all_wide.csv`)
  - Does `exchcd` (exchange code) survive the merge? (Need it for NYSE breakpoints)

- [ ] **0.1b** Open `data/temp/signed_predictors_dl_wide.zip` (or the unzipped CSV) and document:
  - How many unique characteristic columns exist (expecting 209 from CZ + 3 from CRSP = 212)
  - List all column names — this is your master feature inventory
  - Which columns are the CZ "signed" predictors vs. identifiers (permno, date, etc.)
  - Check which CZ data release version you have:
    - Latest is **October 2025 (v2.0.0)** — first release using Python translation for all signals
    - If you have an older version, consider updating (bug fixes in ChNAnalyst,
      PriceDelayTstat, Recomm_ShortInterest)
  - Note: CZ signed predictors file does NOT include category labels inline.
    Category labels are in a **separate file** called `SignalDoc.csv` (see Task 0.3).
  - Note: CZ omits Price, Size, and STreversal from the signed predictors file
    — these must be downloaded separately from CRSP. You likely already have them
    from your `00_fetch_data.py` pipeline. Verify.

- [ ] **0.1b-alt** If you need to re-download or update CZ data, two options:
  - **Option A (recommended):** Use the `openassetpricing` Python package:
    ```bash
    pip install openassetpricing
    ```
    ```python
    import openassetpricing as oap
    openap = oap.OpenAP()              # initializes latest release
    df = openap.dl_all_signals('pandas')  # requires WRDS account
    ```
  - **Option B:** Manual download from Google Drive:
    https://www.openassetpricing.com/data/
    → "209 predictive firm-level characteristics in wide format" (1.6 GB zipped CSV)

- [ ] **0.1c** Open `src/data/loader.py` and document:
  - The `SELECTED_FEATURES` list (currently 75 features) — write down all 75 names
  - The `normalize_features()` function — what does it currently do?
    - Rank transform? Quantile transform? Rescale range?
    - Current range: [-0.5, 0.5] (changing to [-1, 1] per GKX convention)
  - The `load_panel()` function — what does it load and what columns are returned?
  - How is missing data handled? (currently: drop >50% missing rows, fill NaN with 0.0)

### Task 0.2 — Decide on characteristic set for motivation analysis

- [ ] **0.2a** Decision: use ALL available CZ predictors for Step 1, or curate a subset?
  - CZ provides **212 predictive firm-level characteristics** (209 in their file + 3 from CRSP)
  - Of these, CZ classifies them as:
    - **Clear Predictor:** 161 (clearly significant in original papers, long-short t > 4)
    - **Likely Predictor:** 44 (borderline evidence, t ≈ 2.5)
    - **Not-Predictor:** 14 (t < 1.96 in original papers)
    - **Indirect Signal:** ~100 (suggestive but not tested for predictability in original papers)
  - **Recommendation:** Use Clear + Likely = **~205 characteristics** for Step 1.
    This matches the document's "~150 characteristics" reference (the document was written
    before the exact count was pinned down). Exclude Not-Predictors and Indirect Signals
    unless you want to show them separately.
  - The 75 selected features for ML training remain separate — Step 1 is about demonstrating
    the covariate shift broadly, not about which features enter the ML model.
  - You can use SignalDoc.csv (see Task 0.3) to filter by `Cat.Predictor` column
    (or similar) to get Clear + Likely predictors only.

- [ ] **0.2b** Define the characteristic set:
  - Create a new constant `ALL_CZ_FEATURES` (or `MOTIVATION_FEATURES`) in loader.py
    that lists all usable characteristic columns from the CZ data
  - Exclude: identifiers (permno, date), return variables, mechanical liquidity measures
    used as the weighting variable itself (dvol_21d — you compute divergence w.r.t. it,
    so including it is circular)
  - Also exclude characteristics with very poor time-series coverage (e.g., options-based
    predictors that only start in 1996 — check `SampleStartYear` in SignalDoc.csv)

### Task 0.3 — Build category mapping from CZ SignalDoc.csv

**Source file:** `SignalDoc.csv` — the official CZ Signal Documentation file.
  - Download: https://drive.google.com/file/d/1Sev9s6cPFUGgxp1pFiej0lGzpsMqJCI2/view
  - Also browsable interactively: https://openassetpricing.com/SignalDoc-Browser.html
  - Contains one row per predictor with: Acronym, Category, Authors, Year, Journal,
    SampleStartYear, LongDescription, and hand-collected results from original papers.

- [ ] **0.3a** Download `SignalDoc.csv` and place in `data/` or `config/`:
  ```bash
  # Download from Google Drive link above, or:
  # It may also be available in the CZ GitHub repo:
  # https://github.com/OpenSourceAP/CrossSection
  ```

- [ ] **0.3b** Inspect the file and identify key columns:
  ```python
  import pandas as pd
  doc = pd.read_csv('SignalDoc.csv')
  print(doc.columns.tolist())
  # Look for columns like:
  #   'Acronym'      → characteristic name (matches column names in signed predictors)
  #   'Cat.Signal'   → economic category (THIS IS WHAT YOU NEED)
  #   'Cat.Form'     → portfolio formation category (less relevant)
  #   'Cat.Predictor'→ predictability classification (Clear/Likely/Not/Indirect)
  #   'SampleStartYear' → when the characteristic becomes available
  #   'Authors', 'Year', 'Journal' → original paper reference

  # Check unique categories:
  print(doc['Cat.Signal'].unique())
  # Expected: ~13-15 fine-grained categories like:
  # Momentum, Value, Profitability, Investment, Trading Frictions,
  # Intangibles, Short-Term Reversal, Seasonality, Accruals,
  # Low Risk, Low Leverage, Debt Issuance, Profit Growth, Size, etc.
  ```

- [ ] **0.3c** Build the fine-grained mapping (CZ categories):
  ```python
  # CZ Acronym → CZ Cat.Signal (fine-grained)
  fine_category_map = dict(zip(doc['Acronym'], doc['Cat.Signal']))
  # Example: {'Mom12m': 'Momentum', 'BM': 'Value', 'AssetGrowth': 'Investment', ...}
  ```

- [ ] **0.3d** Build the broad category mapping (for Table 2 in motivation document):
  The motivation document uses 8 broad categories. Map CZ fine categories to these:
  ```python
  CZ_TO_BROAD = {
      'Momentum':           'Momentum',
      'Short-Term Reversal': 'Momentum',     # group with momentum or keep separate
      'Seasonality':         'Momentum',      # or 'Other'
      'Value':               'Value',
      'Profitability':       'Profitability',
      'Profit Growth':       'Profitability',
      'Investment':          'Investment',
      'Debt Issuance':       'Investment',    # or keep separate
      'Trading Frictions':   'Liquidity',
      'Low Risk':            'Risk',
      'Accruals':            'Quality',
      'Intangibles':         'Quality',       # or 'Other'
      'Low Leverage':        'Quality',       # or 'Other'
      'Size':                'Other',
      # ... verify once you see actual Cat.Signal values
  }

  broad_category_map = {
      acronym: CZ_TO_BROAD.get(cat, 'Other')
      for acronym, cat in fine_category_map.items()
  }
  ```

- [ ] **0.3e** Save both mappings:
  ```python
  import json

  output = {
      'fine': fine_category_map,       # CZ original categories
      'broad': broad_category_map,     # 8 broad categories for Table 2
      'cz_to_broad': CZ_TO_BROAD,     # the mapping between category levels
  }
  with open('config/feature_categories.json', 'w') as f:
      json.dump(output, f, indent=2)
  ```

- [ ] **0.3f** Decide which category level to use where:
  - **Output 1.1 (bar chart):** Use CZ fine categories for color-coding (more informative,
    ~13 distinct colors). The bar chart has ~150+ bars, so fine categories help readers
    see patterns within groups.
  - **Output 1.2 (summary table):** Use the 8 broad categories (matches Table 2 template
    in the motivation document, cleaner presentation).
  - Both levels should be available; you can always switch.

### Task 0.4 — Expand data loading for motivation analysis

- [ ] **0.4a** Modify `scripts/00_fetch_data.py` OR create a new loading path:
  - Ensure `exchcd` (exchange code) is in the output data
    - NYSE: exchcd == 1
    - AMEX: exchcd == 2
    - NASDAQ: exchcd == 3
  - Ensure ALL CZ characteristic columns are preserved (not just the 75 selected)
  - Check: is the merged file `signed_predictors_all_wide.csv` already wide enough?
    If so, you may not need to change 00_fetch_data.py at all — just load more columns

- [ ] **0.4b** Create a motivation-specific data loader (or extend load_panel):
  ```python
  def load_motivation_panel():
      """
      Load full panel with ALL CZ characteristics for motivation analysis.
      Returns: DataFrame with columns:
        - permno, date (yyyymm), exchcd
        - ret, excess_ret (or RF for computing excess_ret)
        - dvol_21d (primary liquidity measure)
        - All ~150+ CZ characteristics (rank-transformed to [0, 1])
      """
  ```

- [ ] **0.4c** Implement rank transformation to [0, 1] for motivation analysis:
  - Note: This is DIFFERENT from your ML pipeline's normalization
  - For motivation: simple cross-sectional percentile rank within each month
  - For ML pipeline: rank → quantile → rescale to [-1, 1]
  - The motivation analysis uses the simpler [0, 1] rank following GKX (2020)
  ```python
  def rank_transform(panel, feature_cols):
      """Cross-sectional rank to [0, 1] within each month."""
      for col in feature_cols:
          panel[col] = panel.groupby('date')[col].rank(pct=True)
      return panel
  ```

### Task 0.5 — Compute implementability weights

- [ ] **0.5a** Recheck existing `compute_weights()` in `src/weighting/`:
  - Does it compute the normalized weight w̃_it = dvol_21d / mean(dvol_21d) per month?
  - Confirm: mean=1 normalization within each cross-section
  - For motivation analysis, you need the RAW normalized dollar volume weights,
    not the softmax-rank weights used for ML training

- [ ] **0.5b** Create (or verify) a function:
  ```python
  def compute_implementability_weights(panel, liquidity_col='dvol_21d'):
      """
      Compute w̃_it = dvol_it / mean(dvol_t) within each month.
      Returns Series with same index as panel.
      """
  ```

### Task 0.6 — Assign NYSE breakpoint quintiles

- [ ] **0.6a** New function needed:
  ```python
  def assign_nyse_quintiles(panel, sort_col='dvol_21d', exchange_col='exchcd'):
      """
      Each month:
        1. Compute quintile breakpoints using ONLY NYSE stocks (exchcd == 1)
        2. Apply those breakpoints to ALL stocks
        3. Q1 = most illiquid, Q5 = most liquid
      Returns: Series of quintile labels (1-5)
      """
  ```

- [ ] **0.6b** Verify NYSE exchange code values in your data:
  - Run: `panel[panel['exchcd'] == 1].shape` — should be substantial
  - Check if exchcd is integer or float (CRSP sometimes has NaN exchcd for delisted stocks)

### Task 0.7 — Define focal characteristics

- [ ] **0.7a** Map the 15 focal characteristics from Table 1 to CZ Acronyms.
  Use SignalDoc.csv or the Signal Browser (https://openassetpricing.com/SignalDoc-Browser.html)
  to look up exact acronyms:
  ```
  Document Name              → CZ Acronym          → In 75?  → In CZ data?  → Notes
  ──────────────────────────────────────────────────────────────────────────────────────
  Short-term reversal        → STreversal           → ?       → CRSP only    → Not in CZ file; download from CRSP
  Momentum (12-1 month)      → Mom12m               → ✓       → ✓            →
  Book-to-market             → BM (or BMdec)        → ✓       → ✓            → Check: CZ uses 'BM', your code uses 'BMdec'
  Earnings-to-price          → EP                   → ?       → ✓            → Verify in SignalDoc.csv
  Gross profitability        → GP                   → ✓       → ✓            →
  Asset growth               → AssetGrowth          → ✓       → ✓            →
  Return on equity           → RoE                  → ✓       → ✓            →
  Accruals                   → Accruals             → ?       → ✓            → CZ may have multiple accrual variants
  Idiosyncratic volatility   → IdioVol3F (or similar)→ ✓      → ✓            → Check exact CZ acronym
  Beta                       → Beta (or BetaFP)     → ?       → ✓            → CZ may use BetaFP (Frazzini-Pedersen)
  Amihud illiquidity         → AmiHud               → ?       → ✓            → Verify: may be in Trading Frictions category
  Zero-trading days          → zerotrade (variants) → ✓       → ✓            → CZ has ZeroTradingDays or similar
  Log market capitalization  → Size                 → ?       → CRSP only    → Not in CZ file; download from CRSP
  Analyst coverage           → ChNAnalyst (or nanalyst)→ ?    → ✓            → CZ name may differ from your code
  Bid-ask spread (Roll)      → BidAskSpread         → ✓       → ✓            → Corwin-Schultz in your pipeline
  ```

- [ ] **0.7b** Resolve name mismatches between CZ Acronyms and your code:
  - CZ uses specific acronyms (e.g., 'BM' not 'BMdec', 'ChNAnalyst' not 'nanalyst')
  - Your existing code may use different names for the same characteristic
  - Create an explicit rename mapping if needed:
    ```python
    CZ_TO_CODE_RENAME = {
        'BM': 'BMdec',          # if your code uses BMdec
        'ChNAnalyst': 'nanalyst', # if your code uses nanalyst
        # ... add as needed after checking
    }
    ```
  - The Signal Browser lets you search by keyword to find the exact CZ acronym

- [ ] **0.7c** Verify all 15 focal characteristics exist in your data:
  - Note: STreversal and Size are NOT in the CZ signed predictors file
    (CZ omits these because they can be downloaded directly from CRSP)
  - Verify that your `00_fetch_data.py` computes or pulls these from CRSP
  - If any focal characteristic is missing, either:
    (a) Add it to your data pipeline, OR
    (b) Substitute with the closest available CZ predictor

- [ ] **0.7d** Store as a constant:
  ```python
  FOCAL_CHARACTERISTICS = [
      'STreversal', 'Mom12m', 'BM', 'EP', 'GP',
      'AssetGrowth', 'RoE', 'Accruals', 'IdioVol3F', 'Beta',
      'AmiHud', 'ZeroTradingDays', 'Size', 'ChNAnalyst', 'BidAskSpread'
  ]
  # NOTE: Update these names after verifying exact CZ acronyms in Task 0.7a
  ```

### Task 0.8 — Validate the restructured data

- [ ] **0.8a** After loading, run these sanity checks:
  ```python
  # Shape: expecting ~2-4M stock-months (1972-2024, ~3000-6000 stocks/month)
  print(f"Panel shape: {panel.shape}")
  print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
  print(f"Avg stocks per month: {panel.groupby('date').size().mean():.0f}")
  
  # Features: count non-NaN for each characteristic
  feature_coverage = panel[all_features].notna().mean()
  print(f"Features with >50% coverage: {(feature_coverage > 0.5).sum()}")
  print(f"Features with >80% coverage: {(feature_coverage > 0.8).sum()}")
  
  # Weights: verify mean=1 per month
  w_means = panel.groupby('date')['w_tilde'].mean()
  assert (w_means - 1.0).abs().max() < 1e-6, "Weights not normalized!"
  
  # Quintiles: verify NYSE breakpoints work
  q_counts = panel.groupby(['date', 'liq_quintile']).size().unstack()
  print(f"Avg stocks per quintile per month:\n{q_counts.mean()}")
  # Q1 should have MANY more stocks than Q5 (NYSE breakpoints effect)
  
  # Rank transform: verify [0, 1] range
  for col in all_features[:5]:
      assert panel[col].min() >= 0 and panel[col].max() <= 1, f"{col} not in [0,1]!"
  ```

---

## Phase 1: Output 1.5 — Weight Distribution (Do This First)

**Why first:** This is the simplest output and validates your weight computation is correct.

### Task 1.5a — Compute log₁₀(w̃) distribution

- [ ] Pool w̃_it across all stock-months
- [ ] Compute log₁₀(w̃) — handle w̃ = 0 by setting floor at a small value (e.g., 1e-6)
- [ ] Compute percentiles of w̃ (not log): 5th, 25th, 50th, 75th, 95th

### Task 1.5b — Plot histogram

- [ ] Histogram of log₁₀(w̃) with ~50 bins
- [ ] Add vertical dashed line at log₁₀(1) = 0 (the "average" stock)
- [ ] Add text annotation with the 5 percentiles
- [ ] Expected: heavy left tail (many stocks with w̃ << 1), thin right tail (few mega-caps)
- [ ] Save to `outputs/motivation/step1/weight_distribution.png`

### Task 1.5c — Sanity check

- [ ] Median w̃ should be WELL below 1.0 (maybe 0.05–0.2)
  - Because most stocks are small; the mean is pulled up by mega-caps
- [ ] 95th percentile should be 10–100+
- [ ] If median ≈ 1.0, something is wrong — check if you're using equal-weight by accident

---

## Phase 2: Output 1.1 — Marginal Divergence Bar Chart (Headline Figure)

### Task 1.1a — Compute monthly divergences

- [ ] For each month t, for each characteristic j:
  ```
  x̄_train = mean(x_ij)                        # equal-weighted
  x̄_deploy = sum(w̃_it * x_ij) / sum(w̃_it)     # dollar-volume-weighted  
  d_jt = x̄_deploy - x̄_train
  ```
- [ ] Store as DataFrame: rows = months, columns = characteristics
- [ ] Handle NaN: for a given month, use only stocks where feature j is non-missing
  for BOTH the equal-weighted and volume-weighted means

### Task 1.1b — Compute time-series statistics

- [ ] For each characteristic j:
  - d̄_j = mean(d_jt) across all months
  - SE_j = Newey-West standard error with 6 lags
  - t_j = d̄_j / SE_j
- [ ] Recheck: does your existing `newey_west_tstat()` function accept a time series
  and return both the mean and the t-stat? If not, may need a wrapper.
- [ ] Verify NW implementation handles the 6-lag Bartlett kernel correctly

### Task 1.1c — Build the bar chart

- [ ] Sort characteristics by |d̄_j| (largest divergence at top)
- [ ] Horizontal bar chart: y-axis = characteristic names, x-axis = d̄_j (signed)
- [ ] Color-code bars by economic category (from Task 0.3)
- [ ] Mark bars where |t_j| > 2 (e.g., darker color, or add asterisk)
- [ ] Add vertical line at 0
- [ ] Figure size: tall and narrow (probably ~150 bars, so maybe 20+ inches tall)
  - Consider: show top 50 in main figure, full set in appendix
- [ ] Label axes clearly
- [ ] Save to `outputs/motivation/step1/divergence_bar_chart.png`

### Task 1.1d — Diagnostic check

- [ ] Count characteristics with |t| > 2:
  - Total significant: expecting 40–80 out of ~150
  - Significant NON-liquidity: expecting 20-30+ out of ~130 non-liquidity
  - If < 20 non-liquidity significant → weak result, document notes concern
- [ ] The top divergences should include both mechanical (Amihud, zero-trade, spread)
  AND non-mechanical (idiovol, momentum, analyst coverage, beta)

---

## Phase 3: Output 1.2 — Divergence by Category Table

### Task 1.2a — Aggregate by category

- [ ] For each economic category:
  - Average |d̄_j| across all characteristics in that category
  - Count # with |t| > 2
  - Total # characteristics in category
- [ ] Store as DataFrame

### Task 1.2b — Format table

- [ ] Columns: Category | Avg |d̄| | # Significant (|t|>2) | # Characteristics
- [ ] Sort by Avg |d̄| (descending)
- [ ] Save as CSV: `outputs/motivation/step1/divergence_by_category.csv`
- [ ] Optionally generate LaTeX: `outputs/motivation/step1/divergence_by_category.tex`

### Task 1.2c — Diagnostic check

- [ ] Liquidity category should have highest Avg |d̄| (mechanical, expected)
- [ ] But at least 3-4 OTHER categories should also show meaningful divergence
- [ ] If only Liquidity category is significant → weak result

---

## Phase 4: Output 1.3 — Weight Regression (Fama-MacBeth)

### Task 1.3a — Monthly cross-sectional regressions

- [ ] Each month t, run OLS:
  ```
  log(w̃_it) = x'_it δ_t + ε_it
  ```
  - Dependent variable: log of normalized dollar volume weight
  - Regressors: ALL ~150 characteristics (rank-transformed)
  - Handle: stocks with w̃ = 0 → drop or set floor before log
  - Handle: months with too few stocks or too many missing features
    → require minimum N (e.g., N > 200) and max missing features

- [ ] Store: R²_t for each month, δ̂_jt for each characteristic × month

### Task 1.3b — Compute Fama-MacBeth statistics

- [ ] R̄² = mean(R²_t) across all months
- [ ] For each j: δ̄_j = mean(δ̂_jt), with Newey-West t-statistic (6 lags)

### Task 1.3c — Format output

- [ ] Report R̄² prominently (target: > 0.5)
- [ ] Ranked table of top 15 characteristics by |δ̄_j|
  - Columns: Rank | Characteristic | Category | δ̄_j | t-stat
- [ ] Save as CSV: `outputs/motivation/step1/weight_regression.csv`
- [ ] Save R̄² as JSON metadata

### Task 1.3d — Diagnostic check

- [ ] R̄² > 0.5 → covariate shift is systematic and predictable from features
- [ ] R̄² < 0.3 → weak, the shift is noisy
- [ ] Top 15 should overlap with top characteristics from Output 1.1
  but may differ because regression controls for correlations

### Task 1.3e — Implementation note

- [ ] This is a LARGE regression (~150 regressors) run ~600 times (one per month)
- [ ] May need regularization if multicollinearity is severe
  - Option: Ridge regression instead of OLS (but document uses OLS)
  - Option: Drop features with >50% missing in that month
- [ ] Consider using `np.linalg.lstsq` or `statsmodels.OLS` for speed
- [ ] Progress bar recommended (600 regressions × ~3000 stocks each)

---

## Phase 5: Output 1.4 — Density Comparison Plots

### Task 1.4a — Select 6 characteristics

- [ ] From the focal list:
  1. Amihud illiquidity (AmiHud) — mechanical divergence
  2. Idiosyncratic volatility (IdioVol3F) — correlated with illiquidity
  3. 12-month momentum (Mom12m) — core anomaly
  4. Book-to-market (BMdec) — core anomaly
  5. Analyst coverage (nanalyst) — information environment
  6. Short-term reversal (STreversal) — microstructure

### Task 1.4b — Compute kernel density estimates

- [ ] For each of the 6 characteristics, pool across all months:
  - Equal-weighted KDE: each stock-month gets weight 1/N
  - Dollar-volume-weighted KDE: each stock-month gets weight w̃_it
- [ ] Use scipy.stats.gaussian_kde with `weights` parameter
- [ ] Evaluate on a grid: x = np.linspace(0, 1, 200)

### Task 1.4c — Plot 2×3 panel figure

- [ ] 2 rows × 3 columns of subplots
- [ ] Each subplot: two overlaid density curves
  - Solid line: equal-weighted (label: "Training distribution")
  - Dashed line: volume-weighted (label: "Deployment distribution")
- [ ] Shared legend (top or bottom)
- [ ] Each subplot titled with characteristic name
- [ ] x-axis: Characteristic rank [0, 1]
- [ ] y-axis: Density
- [ ] Save to `outputs/motivation/step1/density_comparison.png`

### Task 1.4d — Alternative: time-varying densities

- [ ] If densities shift substantially over time, show 3 representative months:
  2002, 2012, 2022 (or pick months in your data)
- [ ] This would be a 6×3 panel (6 characteristics × 3 time periods)
  → probably too busy; stick with pooled unless time variation is dramatic

### Task 1.4e — Diagnostic check

- [ ] Amihud: two curves should barely overlap (mechanical, large shift)
- [ ] Momentum/BM: visible but smaller shift
- [ ] If all 6 show large shifts → strong visual evidence
- [ ] If only Amihud shows a shift → weak (only mechanical divergence)

---

## Phase 6: Output 1.5 (already done in Phase 1 — verify)

- [ ] Revisit Output 1.5 after all other Step 1 outputs
- [ ] Add the percentile summary in a text box or table below the histogram
- [ ] Verify the numbers make sense in the context of what Outputs 1.1–1.4 show

---

## Summary: File/Function Inventory for Step 1

### New files to create:

```
data/SignalDoc.csv                     # CZ Signal Documentation (DOWNLOAD from openassetpricing.com)
config/feature_categories.json        # characteristic → category mapping (built from SignalDoc.csv)
                                      #   contains: fine (CZ categories), broad (8 categories),
                                      #   cz_to_broad (mapping between levels)
src/analysis/motivation.py             # all computational functions
scripts/04_motivation_analysis.py      # orchestration script
```

### New functions needed in `src/analysis/motivation.py`:

```python
# Data prep
load_motivation_panel()                # load full CZ panel for motivation
load_feature_categories()              # load config/feature_categories.json
resolve_cz_names(panel)                # rename CZ acronyms to match your code if needed
rank_transform(panel, feature_cols)    # cross-sectional rank to [0, 1]
compute_implementability_weights()     # w̃ = dvol / mean(dvol) per month
assign_nyse_quintiles()                # quintile assignment using NYSE breaks

# Output 1.1 + 1.2
compute_marginal_divergence()          # d̄_j and NW t-stats per feature
plot_divergence_bar_chart()            # ranked bar chart, color by category
summarize_divergence_by_category()     # category-level summary table

# Output 1.3
fama_macbeth_weight_regression()       # monthly OLS of log(w̃) on all features

# Output 1.4
plot_density_comparison()              # 2×3 panel of weighted vs unweighted KDE

# Output 1.5
plot_weight_distribution()             # histogram of log₁₀(w̃) + percentiles
```

### Existing functions to RECHECK:

```python
# src/data/loader.py
load_panel()                           # What columns does it return? Can it be extended?
normalize_features()                   # Does it do rank transform? What range?
SELECTED_FEATURES                      # List all 75; compare to CZ full set (212 predictors)

# src/weighting/schemes.py
compute_weights()                      # Does it have a raw dollar-volume option?

# src/evaluation/statistics.py
newey_west_tstat()                     # Verify: accepts time series, returns (mean, tstat)?
                                       # Verify: 6 lags, Bartlett kernel?

# scripts/00_fetch_data.py
                                       # Does exchcd survive the merge?
                                       # Are all CZ columns preserved (not just 75)?
                                       # Does it handle STreversal, Size, Price from CRSP?
                                       # Which CZ data release version was downloaded?
```

### External files to DOWNLOAD:

```
SignalDoc.csv                          # From: https://drive.google.com/file/d/1Sev9s6cPFUGgxp1pFiej0lGzpsMqJCI2/view
                                       # Contains: Acronym, Cat.Signal, Cat.Predictor,
                                       # Authors, Year, SampleStartYear, etc.
                                       # Use to build feature_categories.json (Task 0.3)

signed_predictors_dl_wide.csv          # From: https://www.openassetpricing.com/data/
                                       # Or: pip install openassetpricing → dl_all_signals()
                                       # Latest: October 2025 (v2.0.0), 209 characteristics
                                       # You may already have this in data/temp/
                                       # CHECK: is it the latest version?
```

---

## Execution Order (Recommended)

```
Phase 0 (Data)    ████████████████░░░░░░░░░░░░░░░░  ~3-5 days
  0.1  Audit existing code
  0.2  Decide characteristic set
  0.3  Build category mapping
  0.4  Expand data loading
  0.5  Compute weights
  0.6  NYSE quintiles
  0.7  Map focal characteristics
  0.8  Validate everything

Phase 1 (Out 1.5) ░░░░░░░░░░░░░░░░██░░░░░░░░░░░░░░  ~0.5 day
  1.5  Weight distribution histogram

Phase 2 (Out 1.1) ░░░░░░░░░░░░░░░░░░████░░░░░░░░░░  ~1-2 days
  1.1  Marginal divergence bar chart (headline figure)

Phase 3 (Out 1.2) ░░░░░░░░░░░░░░░░░░░░░░██░░░░░░░░  ~0.5 day
  1.2  Divergence by category table (aggregates from 1.1)

Phase 4 (Out 1.3) ░░░░░░░░░░░░░░░░░░░░░░░░████░░░░  ~1-2 days
  1.3  Fama-MacBeth weight regression (most complex)

Phase 5 (Out 1.4) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░██░░  ~1 day
  1.4  Density comparison plots

Total estimated time: ~7-10 days
```

---

## Notes

- Phase 0 is the CRITICAL foundation — don't rush it
- Output 1.5 first because it validates weights are computed correctly
- Output 1.1 second because it's the headline result and informs whether to proceed
- Output 1.3 (Fama-MacBeth) is the most computationally intensive
- All Step 1 outputs are INDEPENDENT of the ML pipeline — no need for trained models
- Results from Step 1 feed into Step 2 (Output 2.4 uses |d̄_j| from Output 1.1)

## Key Reference Links

| Resource | URL | What You Need It For |
|----------|-----|---------------------|
| CZ Data Page | https://www.openassetpricing.com/data/ | Download signed predictors + SignalDoc.csv |
| SignalDoc.csv | https://drive.google.com/file/d/1Sev9s6cPFUGgxp1pFiej0lGzpsMqJCI2/view | Category labels (Acronym → Cat.Signal) |
| Signal Browser | https://openassetpricing.com/SignalDoc-Browser.html | Interactive lookup of any predictor |
| CZ GitHub | https://github.com/OpenSourceAP/CrossSection | Source code for predictor construction |
| Python package | `pip install openassetpricing` | Automated data download (needs WRDS) |
| CZ Paper | Chen & Zimmermann (2022), Critical Finance Review 11(2): 207–264 | Citation for your dissertation |
