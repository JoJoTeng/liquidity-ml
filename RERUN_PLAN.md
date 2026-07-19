# From-Scratch Rerun Plan (shrcd 10/11, ElasticNet + XGBoost)

*Drafted 2026-07-10 from: the simulated RFS referee report / editor letter /
code-vs-paper audit (`paper/review/`), plus four line-by-line pipeline audits
(data construction; weighting/models/training; eval realignment; motivation +
formal scripts). Nothing in this plan has been executed yet. Code changes
begin only after the decision list in Section 2 is settled.*

**Scope locked by the author:** adopt the CRSP common-share filter
(`shrcd` in {10, 11}); regenerate every result from scratch; models
restricted to `elastic_net` and `xgboost`.

---

## 0. Why this rerun, in referee terms

The editor named two kill switches. The rerun must be designed so both are
answered by construction:

- **Kill switch (i) — cost levels (Major Comment 1).** Every headline gain is
  avoided measured cost, and the Corwin–Schultz spread *level* is validated
  nowhere. The rerun adds a spread-level validation module (TAQ-based
  effective spreads by year x dollar-volume quintile), a mismatched-cost
  circularity check (gate on CS, charge the alternative; and vice versa),
  re-priced 2x2 cells under level-corrected spreads, and by-sub-period cost
  diagnostics. See Section 5.
- **Kill switch (ii) — inference on the shape claims (Major Comments 2/4).**
  Already answered by script 46 (execution p = 0.0002, AUM trend, dose-response
  flip p = 0.0016) — the rerun must **re-produce** these on the new panel, so
  46/47 and the `21e --portfolio-weighting all` prerequisite are wired into the
  runbook (they currently are not; a naive rerun silently loses the
  dose-response grid).
- **Major Comment 6 (universe + delisting)** is answered head-on: 6(i) by the
  shrcd filter itself; 6(ii) by a new `ret_delist_imputed` flag in the
  processed panel plus a quintile-R2-excluding-imputed-delistings diagnostic.
- **Minors 7 (sigma horizon), 8 (cost-input missingness by quintile),
  10 (beta in {2,4})** all require retraining or refetching — they are folded
  into this rerun (Decisions D4, and Sections 3/6).

Already addressed pre-rerun and only needing regeneration: Major 2, 3, 4, 5, 7
(inference supplement, gate-scale diagnostics, JKMP positioning, mechanical
fixes).

---

## 1. The shrcd change — exact design

All four audits converge on one requirement: **the filter must bind at
formation month t, and the target must become calendar-aware first.**
Share codes change within a permno (reorganisations, ADR conversions), and the
current row-positional target shift would pair month-t features with returns
from months later once filtered rows create gaps. Today 410 latent gap pairs
exist and zero are harmful — only because CRSP happens to set `ret` missing
after coverage gaps. That accident must become an enforced invariant.

Implementation (one coordinated edit):

1. `scripts/00_fetch_data.py` — add `b.shrcd` to the msenames SELECT (the
   table is already joined range-based on `namedt <= date <= nameendt`, so
   shrcd is automatically time-varying, as exchcd already is). Extend the
   universe filter to `exchcd` set (Decision D2) `& shrcd.isin([10, 11])`.
   Carry `shrcd` into the output so it lands in the panel for diagnostics.
2. `src/data/loader.py` — replace `groupby("permno").shift(-1)` with a
   calendar join: target = excess return merged on `(permno, yyyymm + 1
   month)`. Add a post-build assertion: zero non-NaN targets more than one
   calendar month ahead.
3. `src/data/loader.py` — enforce the universe at month t: drop rows with
   NaN `exchcd` after the date filter (today 305 rows leak in with NaN
   exchcd because the filter is applied pre-merge; with shrcd, every
   reclassification would leak the same way).
4. Add `shrcd` (and `siccd`, `permco`, `dlstcd`, raw `prc`, `shrout`,
   monthly `vol` — pulled while we are at it) to `CORE_NON_FEATURE_COLS` so
   none can ever be picked up as a feature.
5. Rank-normalisation order fix: rank the full month-t cross-section
   **first**, drop NaN-target rows **last** (currently ranks are computed on
   the survivor-conditioned sample — a subtle forward-looking selection; the
   trainer masks NaN targets independently, so the reorder is safe).

Expected impact (to be measured, not assumed): the CZ master table already
leans common-share, so shrinkage may be well below the naive 20–25%. The
removed names are unusual for a shrcd screen — they include the *largest*
dollar-volume securities (AMEX/NASDAQ-listed ETFs such as QQQ; NYSE-Arca
listings are already excluded by exchcd). Consequences to reason from the new
run, never spliced with the old:

- ETFs currently carry enormous deployment weights **and** default 1% spreads
  (their cost inputs are missing) — simultaneously over-weighted and
  over-costed. Removal is a genuine distortion fix, direction unknowable.
- NYSE quintile breakpoints rise (NYSE loses low-dvol closed-end funds/REITs);
  full-sample quintiles lose ETFs at the top. Quintile-conditional numbers
  move through composition *and* breakpoints.
- Q_t = A/N_t participation rises mechanically ~(1/(1-shrink)); at 20–25%
  shrinkage, post-filter $500M behaves like pre-filter ~$625–665M. The paper
  must state this once (the equal-breadth convention is unchanged; its
  economic anchor moves).
- The Q5 missingness rate falls (ETFs carry ~21/113 characteristics), so the
  missingness gradient likely *steepens* — good for the design-choice
  paragraph, but every quoted number moves.
- The feature screen recomputes: the "113 characteristics" count itself may
  change (see D5 and Section 7).

---

## 2. Decisions to settle before any code changes

| # | Decision | Recommendation |
|---|----------|----------------|
| D1 | shrcd 10/11 filter | **Decided (author): adopt.** Design per Section 1. |
| D2 | exchcd set: keep {1,2,3,-1,-2} or clean {1,2,3} | **DECIDED (author, 2026-07-10): {1,2,3}.** Only 2,056 raw rows are halted codes; kills the suspended-code asymmetry and the delisting-fill gap; HXZ/JKP convention. The paper drops "including temporarily halted issues". |
| D3 | NASDAQ dollar-volume double-count (no Gao–Ritter adjustment today) | **DECIDED (author, 2026-07-10): disclose only.** Volume stays unadjusted everywhere — internally consistent with the CZ `DolVol` feature. Add a disclosure sentence to the S3.2 measurement text and carry this as a named referee exposure (a Gao–Ritter-adjusted robustness pass remains the natural revision response if asked). No code change. |
| D4 | sigma in the cost primitive: 12-month monthly SD / sqrt(21) (referee: "stale, horizon-mismatched") | **DECIDED (author, 2026-07-10): switch the primary to daily realised volatility** (rolling SD of daily returns from the dsf we already pull; 63-day window, min 21 obs), and refit one spec with the old sigma as a robustness row (`tc_rank_lam3_500m_sigmaM` in the Section 4 matrix). Horizon-matched to the daily impact term; answers Minor 7 head-on. |
| D5 | 70% missingness screen: full-panel (disclosed look-ahead) or pre-2000-only | **Pre-2000 only.** Kills the one remaining disclosed look-ahead. Feature availability decided on 1989–1999 information, static thereafter. The feature count will move off 113; every count in the paper updates from `feature_list.json` (Section 7). |
| D6 | tc_rank beta sweep at $500M (referee Minor 10) | **Yes:** add `tc_rank_lam2_500m` and `tc_rank_lam4_500m`, XGBoost only. |
| D7 | tc family AUM grid | **Keep `tc_500m` only.** `tc_10m/100m/1000m` feed only the legacy script 22 (not in the paper); dropping them saves ~27% of weighted-training compute. |
| D8 | ElasticNet scope | **`standard` + `tc_rank_lam3_500m` only** (the linear-benchmark table and its inference supplement are all the paper consumes). Also drop EN from motivation scripts 05/06 (never consumed). |
| D9 | Zero/missing-dvol names get *neutral* weight (rank 0.5 / mean dolvol / median TC) — economically backwards for implementability | **Conditional:** re-run `scripts/audit_liquidity_zeros.py` on the new panel first; zeros are likely concentrated in the removed non-common names. If they persist, map zero/missing dvol to the bottom rank (disclosed); if they vanish, document and move on. |
| D10 | Delisting handling | **Fix both edge cases + flag:** (a) attach dlret when the delist month has no msf row (currently lost — survivorship-flavoured, concentrated in illiquid names); (b) alignment of the fill sets with the D2 exchcd set; (c) new `ret_delist_imputed` column; (d) post-rerun diagnostic: Q1 share of imputed delistings + quintile R2 excluding them (referee 6ii). |
| D11 | Spread-level validation source (kill switch (i)) | **Author checks WRDS subscription.** Plan A: WRDS Intraday Indicators (precomputed TAQ effective spreads, daily, permno-level; Daily-TAQ-based from 2003-09, Monthly-TAQ-based earlier) — a plain SQL pull like the CRSP fetch. Plan B: run Chen's `hf-spreads-all` SAS job on the WRDS server (needs TAQ + ISSM access; no prebuilt public CSV exists). Plan C (no TAQ needed): Hasbrouck Gibbs estimates from CRSP daily — explicitly accepted by the referee. |
| D12 | Primary cost input | **Keep Corwin–Schultz as the primary spread**, validated by D11 and stress-tested by the mismatched-cost check + re-priced cells (Section 5). Switching the primary to an external composite would add a dependency that ends before 2024. |
| D13 | XGB tuning restructure (slice one 1000-tree fit instead of refitting per n_estimators; ~60% fewer tuning fits) | **No.** Keep the current design: an identical 150-candidate pool at every retune and across all specs is a *feature* (standard and weighted runs search the same pool; only the selection criterion differs). Compute is affordable after D7/D8 + dropping NN. Revisit only if Apocrita queueing becomes the binding constraint. Related: keep the identical-candidates design and fix the **paper wording** instead ("a fixed pool of 150 candidate configurations drawn once") — do not reseed per retune. |
| D14 | EN l1_ratio fixed at 0.01 (effectively ridge) | **Widen the search** over l1_ratio with an all-zero-coefficient collapse guard (reject candidates with ~0 nonzero coefficients — the model already counts them). Cheap (EN fits are fast) and makes "elastic net" honest. Fallback: keep and describe as ridge-dominant. |

---

## 3. Code fixes before any compute

### Tier 0 — must fix (correctness / rerun safety)

Tuning identification (Codex audit IC1, adversarially verified 2026-07-19):
- Persist ONE realized 150-candidate manifest (seed 42, the 1,944-combination
  grid) and give exactly that menu to BOTH training arms; refit the standard
  XGBoost benchmark under it (weighted arms already share this menu and need
  no refit in v1 terms; under shrcd 10/11 everything refits anyway, so the
  requirement is simply: both arms consume the same persisted manifest).
  Verified v1 facts: realized menus overlap 3/150; 0/25 winners cross-arm;
  8/25 standard winners outside the 1,944 support. The v1 asymmetry is an
  INTENDED design choice (author 2026-07-19) — benchmark given at least the
  weighted arms' search freedom — but v2 must deliver the common-menu
  comparison so "weighting is the only treatment" is literally identified.
- Cache provenance manifest per fitted dir (data hash, feature-list hash,
  candidate-manifest hash, code rev); consumers fail closed on mismatch
  (Codex M10).

Data construction:
- Calendar-aware target join + month-t universe enforcement + shrcd filter
  (Section 1, items 1–3) with the gap assertion.
- `STreversal`: drop the `.fillna(0)` — missing month-t returns must stay
  missing (currently fabricated zero-return signal values enter the ranks).
- Deterministic WRDS pulls: `ORDER BY permno, date` on both queries +
  explicit dedup rule in 00 (keep the loader dedup as a guarded assertion).
- Data manifest written by 00: CZ release tag, SHA256 of zip and
  SignalDoc.csv (must be same release), WRDS pull timestamp, row counts;
  assert the CZ file does not already contain locally-constructed columns
  (`STreversal`/`Price`/`Size`/`ret`/`exchcd`) — a future release adding them
  would silently drop those features via merge suffixes.
- Rank-before-drop reorder (Section 1 item 5).

Caches and provenance (all four audits, independently):
- **Wipe before rerun:** `outputs/` moved aside wholesale (Section 6 gate 2);
  never regenerate into the live tree (stale `top40/` dirs, renamed-metric
  CSVs, and `--skip-importance` header-only files are all lying in wait).
- **Panel fingerprint in every cache/meta:** sha of `processed_panel.parquet`
  + `feature_list.json` (+ git hash, argv, config hash, seed, package
  versions) written by 20/04 into `training_meta.json`/`meta.json` and by
  41–47/21a–e as `run_meta.json`; **refuse cache reuse on mismatch**. Unify
  script 20's two inconsistent required-artifact lists; make
  `--skip-importance` write sentinel-marked files the completeness check
  rejects.
- Prediction-panel merge coverage asserts (>= 99.9% matched) before any R2 in
  04/21a/41 — silent inner-merge row-dropping is how mixed-vintage numbers
  would sneak through.

Paper-critical flags become defaults / runbook lines:
- Script 04: `--importance native` becomes the **default** (or dual-write
  `_native`/`_shap` files); the documented default would silently clobber
  Figure 3's inputs with SHAP values under a caption that says "gain".
- Script 03: atomic JSON writes (tmp + `os.replace`) + merge-on-write so a
  focal-only run can never destroy the `_full` keys again.
- `21e --portfolio-weighting all` added to CLAUDE.md/runbook as a hard
  prerequisite of 46 Part C2 (otherwise the dose-response grid silently
  vanishes); 46 and 47 added to the documented run order.
- `generate_hpc_jobs.sh`: default `MODELS=(elastic_net xgboost)`; split
  `TC_AUMS=(500)` from `TC_RANK_AUMS=(10 100 500 1000)`; EN runs only
  scripts 04 + 20-standard + 20-tc_rank_lam3_500m; drop EN from 05/06;
  `#SBATCH -c 4` instead of `-n 4`.
- Script 46: AUM grid and paths derived from config (currently hard-coded and
  silently decoupled); fix the K->M plumbing (the `--quick` flag currently
  does not reduce Part A draws).
- Transaction-cost context: **raise** if a required `liq_*` column is missing
  from the panel (today a renamed column silently costs every trade at fixed
  defaults), and log the per-run fallback bind rates (also feeds referee
  Minor 8).

### Tier 1 — should fix (before the freeze, cheap)

- Determinism pinning: `tree_method="hist"` explicit, fixed `n_jobs`, seed +
  versions recorded in the standard meta too; regenerate gated artifacts on
  one canonical machine (bitwise stability verified on this machine, but
  cross-platform identity is not guaranteed).
- Loud failures replacing silent skips: 21b exits if 03 `--full` outputs are
  missing (today it silently omits the `group` column and the builder
  KeyErrors much later); 21c always writes the weighted-R2 row and exits
  non-zero on a missing restriction curve; script 05 collect mode exits
  non-zero on missing shards; 44/45 refuse to write the per-model workbook
  from a filtered spec run (clobber); `--strict` mode on 42–45.
- Rename the trap file: `prediction_quantile_timeseries_*` ->
  `quantile_diagnostics_*` (different economic object in the same directory
  as the true cell series), and write CSV twins of the load-bearing xlsx cell
  series.
- Add the execution LW p-value row to 44's table output (the headline
  p = 0.0002 currently lives only in 46's supplement).
- 21-series paths anchored to repo root (currently CWD-relative; running from
  another directory writes a stray `outputs/` tree).
- Reallocation outputs: emit per-window share columns alongside pooled ones
  (or rename pooled `share_pooled_*`) — the pooled/per-window confusion
  already forced a hand-rebuilt figure once.
- Emit the hand-computed text statistics as artifacts: one
  `composition_stats.json` from script 02 (quintile composition, 98%-of-dvol
  footnote, w-tilde distribution stats, missingness-by-quintile), the
  cluster-threshold grid {0.3,0.4,0.5,0.6} from 04, `n_obs_hist` in Table 3's
  CSV. Everything the paper quotes must regenerate from disk after this
  rerun.
- Consistency guards: quintile-table splice assert (04 vs 06), identical
  std/wt month sets in 42, n_obs parity in 41, factor-file availability
  assert (FF5/Mom silently degrades to FF3 today), book-month contiguity
  assert.
- Efficiency (results-identical): hoist the per-month `isin` + `fillna(0.5)`
  out of the rolling loop (dominant for EN; minutes for XGB); float32 feature
  arrays into XGB; date-filter the daily WRDS pull (`>= 1987-01-01`); emit
  parquet as 00's primary artifact; parquet-only for the multi-million-row
  diagnostics dumps.
- New tests (the audit found `rolling.py` has zero direct tests): synthetic-
  panel test asserting exact train/val month sets, retune cadence at months
  1/13/25, a **leakage canary** (predictions for month t invariant to
  mutating data at months >= t), weighted != standard, contiguity assertion;
  a schemes test pinning N_t = full cross-section and mean-one after injected
  NaNs; a tc-context missing-column failure test; a 46-vs-44 execution-p
  parity fixture.
- Doc drift: eval_realignment doc gains 46/47 sections (incl. the sign-test
  labelling and seed/block/M statement); stale top-40% comments in 45/21e;
  legacy naming in statistics docstrings; make_*_xlsx scripts derive specs
  and drop NN; delete the orphan `data/processed_panel_filter.parquet`
  (2.5 GB, referenced nowhere) and script 22 if truly legacy.

### Tier 2 — flagged, not blocking (disclose or leave)

- Breakeven-gate "closed-form myopic optimum" claim: qualify (exact for
  opens/closes; trims are constraint-driven) or add a sign-aware gate row to
  47.
- Joint F-test is i.i.d.-Hotelling while the t-stats are NW(6): add a HAC
  joint Wald or state the assumption in the caption.
- Historical-mean benchmark is a 132-row (not strictly 132-calendar-month)
  window with min 12 obs: caption note or month-indexed rolling.
- Month-1 setup cost included in net SR but excluded from reported TC means:
  one doc sentence.
- Weight-stage fallbacks are month-medians while eval-stage are fixed
  constants: one disclosure sentence.
- `N_train/month` label in 05/06 tables is the full-sample average, not the
  actual training-set size: footnote or recompute.
- VIX regime split uses the full-sample median: caption disclosure (script 07
  runs once — it is model-independent).
- 46's "AUM monotonicity" is an endpoint contrast: add a successive-
  differences joint statistic (cheap; machinery exists).
- The predicted-positive cell list in 46 is an ex-ante claim: re-affirm the
  list is frozen *before* seeing the new panel's grid; document as
  pre-registered.

---

## 4. Final run matrix

| Model | Specs | Count |
|---|---|---|
| XGBoost | standard; dolvol; softmax_rank_lam2; softmax_rank_lam3; tc_500m; tc_rank_lam3_{10m,100m,500m,1000m}; **new:** tc_rank_lam2_500m; tc_rank_lam4_500m; tc_rank_lam3_500m_sigmaM (old-sigma robustness, if D4 adopted) | 12 |
| ElasticNet | standard; tc_rank_lam3_500m | 2 |

Motivation/diagnostics: 02 (`--vw`), 03 (`--full`), 04 for both models
(`--importance native`), 05/06 XGBoost only, 07 once. Formal: 21a–e
(21e `--portfolio-weighting all`). Eval realignment: 41–45, 47, then 46.
New: spread-validation module (Section 5), panel-diff report (Section 6),
delisting diagnostic (D10d).

Removed relative to the old matrix: neural_network everywhere (~23 long
jobs), tc_{10,100,1000}m (x2 models), EN x {dolvol, softmax x2, tc x4,
tc_rank x3}, EN 05/06 shards (~9 long jobs). The additions are 3 XGB fits +
cheap evaluation modules — the net matrix is roughly **half** the old
compute despite the upgrades.

---

## 5. Kill-switch (i) module: spread-level validation

New script (proposed `scripts/eval_realignment/48_spread_level_validation.py`
+ a fetch step depending on D11):

1. **Level table:** CS half-spread vs TAQ effective half-spread, median and
   mean ratio by year x dollar-volume quintile, 2000–2024 (or the validation
   series' coverage). This is the referee's remedy (a) — ratios, not
   correlations.
2. **Re-priced cells:** recompute the $500M 2x2 and gate diagnostics charging
   level-corrected spreads (evaluation-side only — forecasts and weights are
   trained objects and stay fixed; the training weight is rank-based and
   level-robust by the paper's own argument). Report what fraction of the
   +execution effect survives. Remedy (b).
3. **Mismatched-cost circularity check:** gate on CS, charge TAQ-consistent
   costs; gate on TAQ-consistent, charge CS. Remedy (c).
4. **Sub-period diagnostics:** gate pass rates, cost drags, four cells for
   2000–2011 vs 2012–2024 (spreads fell an order of magnitude; a too-flat
   cost series would manufacture exactly the pattern in Figure 4). Remedy (d).

If TAQ-consistent levels collapse the execution wedge, the paper's headline
restates as conditional on the cost model — better discovered by us now than
by the referee later.

---

## 6. Sequencing (gates; heavy steps run by the author)

- **Gate 0 — decisions.** Settle Section 2. Then code changes begin.
- **Gate 1 — code freeze.** All Tier 0 + agreed Tier 1/2 fixes applied;
  `python -m pytest tests -q` green including the new tests;
  `python -m compileall -q scripts src tests`; CLAUDE.md + docs updated;
  generator regenerates the Section 4 matrix exactly.
- **Gate 2 — archive.** `mv outputs outputs_v1_preshrcd` (plus the old
  processed panel and raw CSV compressed to an archive location); git tag
  the pre-rerun commit. Old and new results must never share a directory.
- **Gate 3 — data + diff report.** Author runs 00/01 locally (WRDS
  credentials). New script emits a panel-diff report: rows/permnos by year
  and by shrcd class removed, dvol quintile breakpoint shifts, weight
  distribution stats, missingness gradient, feature-list delta (old 113 vs
  new), fallback bind rates. **Author sign-off here before any HPC
  submission** — this is where "shrinkage is 8% not 25%" or "a focal feature
  fell out of the screen" gets caught cheaply.
- **Gate 4 — training.** Author submits the generated SLURM matrix on
  Apocrita. Standard caches build first (04), weighted specs depend on them
  (existing dependency graph, kept).
- **Gate 5 — analyses.** 02/03/05/06/07 + 21a–e + 41–45 + 47 + 46, in the
  documented order, `--strict` where available. Spread-validation module in
  parallel once D11's data source is confirmed.
- **Gate 6 — paper.** Rebuild all tables/figures from the builders; run the
  full number re-verification pass against Section 7's inventory; update the
  paper text (universe sentence, feature counts, every quoted number,
  participation re-anchoring sentence); update `.claude/paper-disciplines.md`
  and memory. The 2026-07-10 "state the universe positively, no share-code
  mention" decision is **superseded** by D1: Section 3.1 will now say common
  shares (codes 10 and 11) — one clause, plus the delisting-diagnostic
  disclosure.

---

## 7. Post-rerun re-verification inventory (numbers that move)

Every number below is currently quoted in the paper and mechanically moves
under the new universe. None may be spliced from old outputs.

- Panel constants: 2,920,455 obs; 26,750 permnos; 5,447–9,046 range; 6,776 /
  6,336 / 6,327 averages; "113 characteristics" (and 102/11 split, category
  counts); 299 test months (only if the month range changes — it should not).
- S2 weight distribution: median w 0.056, p25 0.008, p95 4.0, 85% below 1,
  0.2% clipped tail; divergence counts 105/113, 94/102, category means;
  spanning R2 0.911 / ex-liquidity 0.545; composition 38.5% / 58.4% / 24.9%,
  2,438/6,336; the 98.4%-of-dollar-volume footnote.
- S3 data section: missingness gradient 37.2% -> 17.5%; DataDescriptives
  percentiles (dvol median $0.70M / p99 $374M / ratio 536; cost percentiles
  55.7 -> 64.4 bps, 767 -> 1,140 bps, "sixteen times"); the participation
  re-anchoring statement (new).
- S2/S5 ML diagnostics: R2-by-quintile (+0.60 Q1 etc.); utility-weighted R2
  incl. _q45 keys; Spearman -0.14; illiquidity cluster 11.8% / 2.7% / 4.3x
  and threshold grid 1.2/4.5/6.3; cluster membership itself; F = 22.0 / 19.3
  / 7.4, 10/15 and 54/113 significant; reversal slopes 0.032/-0.034,
  0.0260 -> 0.0031.
- 2x2 headline cells {-0.15, +0.38, -0.05, +0.45}; training +0.09 (p 0.25)
  -> +0.17@1B; execution +0.53 (p 0.0002); total +0.60; gross effects;
  alphas 1.73–2.03; gate diagnostics 49/45/31 bps, pass 44.8%/28.1%; 47's
  rank-corr 0.70 / dispersion 0.685 / matched-gate numbers; long-only cells;
  dose-response grid (incl. flip +0.27 P 0.0016) and predicted-positive mean
  (+0.19, p 0.075); AUM trend; training-scale matrix (all 16 cells);
  weight-sweep ESS/top-10/deltas; EN benchmark row; regime splits; restriction
  curve M_w vs M_all; error differential; capacity-CE appendix.
- Discipline updates: several `.claude/paper-disciplines.md` "settled
  numbers" entries become stale the moment the rerun lands; the file gets a
  full pass at Gate 6.

---

## 8. Open items owned by the author

1. ~~Settle D2, D3, D4, timeline~~ — **decided 2026-07-10** (see table and
   below). Still open to veto, otherwise proceeding as recommended:
   D5 (missingness screen pre-2000), D9 (zero-dvol conditional), D13 (keep
   tuning design, fix wording), D14 (widen EN l1_ratio with collapse guard),
   plus recommended-accept D6–D8, D10, D12.
2. Check WRDS subscription for TAQ / Intraday Indicators (D11) — determines
   the validation plan A/B/C.
3. Confirm script 22 (`22_prepare_portfolio_excel_tables.py`) is legacy and
   can be dropped with the tc_{10,100,1000}m specs.
4. Timeline — **DECIDED (author, 2026-07-10): rerun first, deadline
   flexible.** Fix code, rerun everything, re-verify, submit when the numbers
   are certified, even if that slips past 2026-07-20. No piecemeal paper
   edits before Gate 6.
