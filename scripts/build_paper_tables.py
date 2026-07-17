"""Build the curated Section 2 paper tables into paper/Tables.

This script owns the reference-styled (sdftc-paper) LaTeX tables that the
paper \\input directly: caption below the tabular, \\footnotesize \\textbf
notes, booktabs rules, negatives wrapped in math mode. It reads only verified
motivation/eval outputs, so the numeric rows regenerate if the underlying
analysis changes, while the curated captions (refined during section review)
are held here verbatim.

The raw per-script LaTeX dumps written by scripts 02/03/04 live under each
script's own ``outputs/motivation/.../tables`` directory and are data
artifacts, not paper assets -- they deliberately do not touch paper/Tables.

Tables produced (all XGBoost, dvol liquidity, OOS 2000-2024):
    DivergenceByCategory.tex   tab:divergence_by_category   (script 02 data)
    HeterogeneityFocal.tex     tab:heterogeneity            (script 03 data)
    R2ByQuintileML.tex         tab:r2_by_quintile           (script 04 data)
    EvaluationMeasures.tex     tab:evaluation_measures      (04 + 41 data)
    ScreeningSplitting.tex     tab:screening_splitting      (05 + 06 data)
    DataDescriptives.tex       tab:data_descriptives        (S4; panel + schemes.py)

Run:
    python scripts/build_paper_tables.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "Tables"
MOT = ROOT / "outputs" / "motivation"


def num(x, dp=2, plus=False):
    """Format a number for LaTeX, math-wrapping negatives (and signed values)."""
    s = f"{x:+.{dp}f}" if plus else f"{x:.{dp}f}"
    if s.startswith(("-", "+")):
        return f"${s}$"
    return s


def mnum(x, dp=2):
    """Fixed-decimal number, always math-wrapped.

    Used in right-aligned numeric columns: math-wrapping every value (not just
    the negatives) keeps digit metrics identical down the column, so the
    decimal points line up with the minus signs hanging into the left margin.
    """
    return f"${x:.{dp}f}$"


def tstat(x):
    """t-statistic in parentheses, with a math minus for negatives."""
    return f"({x:.2f})" if x >= 0 else f"($-${abs(x):.2f})"


def build_divergence_by_category():
    dc = pd.read_csv(MOT / "step1/dvol/divergence_by_category.csv")
    # The display CSV rounds the category average to 4dp, which flips a 3dp
    # rounding boundary for one category; recompute at full precision from the
    # per-characteristic file. Counts are exact integers, taken from dc.
    ds = pd.read_csv(MOT / "step1/dvol/divergence_stats.csv")
    broad = json.load(open(ROOT / "config/feature_categories.json"))["broad"]
    ds["cat"] = ds["feature"].map(broad)
    avg = ds.groupby("cat")["abs_d_bar"].mean()
    rows = "\n".join(
        f"{r['Category']} & {avg[r['Category']]:.3f} & "
        f"{int(r['# Significant (|t| > 2)'])} & {int(r['# Characteristics'])} \\\\"
        for _, r in dc.iterrows()
    )
    tot_sig = int(dc["# Significant (|t| > 2)"].sum())
    tot_n = int(dc["# Characteristics"].sum())
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lccc}}
\toprule
Category & Avg.\ $|\bar d_j|$ & \# Significant ($|t|>2$) & \# Characteristics \\
\midrule
{rows}
\midrule
All & & {tot_sig} & {tot_n} \\
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Distributional divergence by characteristic category.}} For each characteristic $j$ (rank-transformed to $[0,1]$ within month), $\bar d_j$ is the time-series average difference between the dollar-volume-weighted and equal-weighted cross-sectional means, $\bar d_j = T^{{-1}}\sum_t (\bar x^{{\mathrm{{deploy}}}}_{{j,t}} - \bar x^{{\mathrm{{train}}}}_{{j,t}})$. Characteristics are grouped into the economic categories of \citet{{chen2022open}}; the table reports the category average of $|\bar d_j|$ and the number of characteristics whose divergence is significant at $|t|>2$ with Newey--West standard errors (6 lags). The sample is all NYSE, AMEX, and NASDAQ stock-months, 1989--2024.}}
\label{{tab:divergence_by_category}}
\end{{table}}
"""


FOCAL_LABELS = {
    "STreversal": "Short-term reversal",
    "Mom12m": "Momentum (12--1)",
    "BM": "Book-to-market",
    "EP": "Earnings-to-price",
    "GP": "Gross profitability",
    "AssetGrowth": "Asset growth",
    "RoE": "Return on equity",
    "Accruals": "Accruals",
    "IdioVol3F": "Idiosyncratic volatility",
    "Beta": "Beta",
    "Illiquidity": "Amihud illiquidity",
    "zerotrade12M": "Zero-trading days",
    "Size": "Size",
    "AnnouncementReturn": "Announcement return",
    "BidAskSpread": "Bid--ask spread",
}


def build_heterogeneity_focal():
    ir = pd.read_csv(MOT / "step2/dvol/interaction_regression.csv")
    im = json.load(open(MOT / "step2/dvol/interaction_meta.json"))
    # t-statistics sit in their own labelled columns here, so they carry no
    # parentheses (parentheses are reserved for t-stats sharing a cell with a
    # coefficient, as in the alpha panel of tab:capacity_2x2). Negatives keep a
    # math minus so the sign of each interaction reads at a glance.
    #
    # Every value in a column carries the same number of decimals, so the
    # numeric columns are right-aligned (see mnum).
    rows = "\n".join(
        f"{FOCAL_LABELS.get(r['feature'], r['feature'])} & {mnum(r['beta_bar'], 3)} & "
        f"{mnum(r['beta_t'], 2)} & {mnum(r['gamma_bar'], 3)} & {mnum(r['gamma_t'], 2)} \\\\"
        for _, r in ir.iterrows()
    )
    n_sig = im["n_sig_gamma_continuous"]
    f_stat = im["f_test_stat_continuous"]
    n_sig_full = im["n_sig_gamma_full"]
    f_full = im["f_test_stat_full"]
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{l rrrr}}
\toprule
 & \multicolumn{{2}}{{c}}{{Level}} & \multicolumn{{2}}{{c}}{{Liquidity interaction}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}}
Characteristic & \multicolumn{{1}}{{c}}{{$\bar\beta_j$}} & \multicolumn{{1}}{{c}}{{$t$}} & \multicolumn{{1}}{{c}}{{$\bar\gamma_j$}} & \multicolumn{{1}}{{c}}{{$t$}} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Heterogeneous predictability across the liquidity spectrum.}} Monthly Fama--MacBeth regressions of next-month returns on fifteen focal characteristics and their interactions with the within-month dollar-volume percentile rank $L_{{i,t}}\in[0,1]$: $r_{{i,t+1}} = \alpha_t + x_{{i,t}}'\beta_t + (x_{{i,t}}L_{{i,t}})'\gamma_t + \varepsilon_{{i,t+1}}$. Since $L=0$ ($L=1$) marks the least (most) liquid stock, $\bar\gamma_j$ measures the change in the predictive slope of characteristic $j$ from the illiquid to the liquid end of the spectrum. $t$-statistics use Newey--West standard errors (6 lags); {n_sig} of 15 interactions are significant at $|t|>2$, and the joint time-series $F$-test of $\gamma=0$ is $F={f_stat:.1f}$ ($p<0.001$). Estimating the same regression on all 113 characteristics yields {n_sig_full} significant interactions ($F={f_full:.1f}$, $p<0.001$). Missing characteristic ranks are set to the neutral value $0.5$. The sample comprises all NYSE, AMEX, and NASDAQ stock-months from 1989 to 2024, with no size or liquidity screen (Section~\ref{{sec:data}}).}}
\label{{tab:heterogeneity}}
\end{{table}}
"""


def build_r2_by_quintile():
    t3 = pd.read_csv(MOT / "step3/xgboost/dvol/table3_r2_by_quintile.csv")
    # The display CSV rounds the zero-benchmark column to 3dp, which flips the
    # Q5 rounding boundary at 2dp (and would disagree with the same statistic
    # in tab:screening_splitting). Take the zero-benchmark quintile values from
    # the full-precision comparison file; keep CS/hist/N and the full-sample
    # row from the display table.
    rc = pd.read_csv(
        MOT / "step3_quintile/xgboost/dvol/global/baseline/r2_comparison.csv"
    )
    zero_full = dict(zip(rc["quintile"], rc["r2_pooled_pct"]))  # 'Q1'..'Q5'

    def r2zero(quintile_label, fallback):
        v = zero_full.get(quintile_label.split()[0])  # 'Q1 (Illiquid)' -> 'Q1'
        return mnum(v if v is not None else fallback)  # 'Full sample' -> fallback

    # Numeric columns are right-aligned so the decimal points line up (each
    # column carries a uniform number of decimals); see mnum.
    rows = "\n".join(
        f"{r['Quintile']} & {r2zero(r['Quintile'], r['R2_zero (%)'])} & "
        f"{mnum(r['R2_CS (%)'])} & {mnum(r['R2_hist (%)'])} & {int(r['Avg N/month']):,} \\\\"
        for _, r in t3.iterrows()
    )
    rows = rows.replace("Full sample &", r"\midrule" + "\nFull sample &")
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{l rrrr}}
\toprule
Quintile & \multicolumn{{1}}{{c}}{{$R^2_{{\mathrm{{zero}}}}$ (\%)}} & \multicolumn{{1}}{{c}}{{$R^2_{{\mathrm{{CS}}}}$ (\%)}} & \multicolumn{{1}}{{c}}{{$R^2_{{\mathrm{{hist}}}}$ (\%)}} & \multicolumn{{1}}{{c}}{{Avg.\ $N$/month}} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Out-of-sample $R^2$ by liquidity quintile.}} Pooled out-of-sample $R^2$ of the standard-loss model within NYSE-breakpoint dollar-volume quintiles ($Q1$ = least liquid), over the 2000--2024 test period. The three columns benchmark squared prediction errors against zero \citep{{gu2020empirical}}, the within-month cross-sectional mean return, and each stock's rolling historical mean over the preceding $132$ months (the model's own training lookback). $Q1$ is the only quintile with a positive $R^2_{{\mathrm{{zero}}}}$. The cross-sectional benchmark is harsher in every quintile; the $Q1$-specific fact is the change of sign from positive to negative, indicating that the positive zero-benchmark figure partly reflects return levels rather than cross-sectional ranking. The historical-mean column is largest at the liquid end because the benchmark, not the model, varies with it: forecasting each stock's own historical mean is worse than forecasting zero in every quintile, and the penalty is largest for the least volatile names, whose smaller squared returns make a non-zero historical mean proportionally more costly. The model's own sum of squared errors stays within $0.61\%$ of the zero-benchmark denominator in every quintile, so no column should be read as evidence that the model predicts liquid stocks well.}}
\label{{tab:r2_by_quintile}}
\end{{table}}
"""


def build_evaluation_measures():
    u = json.load(open(MOT / "step3/xgboost/dvol/utility_weighted_r2.json"))
    b = pd.read_csv(
        ROOT / "outputs/eval_realignment/analysis/xgboost/tc_rank_lam3_500m"
        "/liquidity_breakpoints/nyse/deployment_weighted_r2.csv"
    )
    dw_full = b.loc[b.universe == "full", "r2_weighted_std_pct"].iloc[0]
    dw_q45 = b.loc[b.universe == "liquid_q4q5", "r2_weighted_std_pct"].iloc[0]

    # Cross-pipeline integrity guard: the equal-weighted liquid-set cell from
    # script 04 must equal Table 4's Mall Q4-Q5 entry (same predictions, same
    # quintiles, independent code paths).
    restr = pd.read_csv(
        MOT / "step3_restriction/xgboost/dvol/global/baseline/restriction_comparison.csv"
    )
    q45_check = restr.loc[restr.model == "Mall", "r2_q45_pct"].iloc[0]
    assert abs(u["r2_standard_zero_q45"] * 100 - q45_check) < 1e-4, (
        "equal-weighted Q4-Q5 R2 disagrees with restriction_comparison"
    )

    rows = [
        ("Equal (the conventional pooled $R^2$)",
         u["r2_standard_zero"] * 100, u["r2_standard_zero_q45"] * 100),
        (r"Implementability ($\tilde w^{\mathrm{tcr}}$, \$500M)",
         dw_full, dw_q45),
        ("Dollar volume",
         u["r2_weighted_zero"] * 100, u["r2_weighted_zero_q45"] * 100),
        ("Value (market capitalisation)",
         u["r2_weighted_mcap_zero"] * 100, u["r2_weighted_mcap_zero_q45"] * 100),
    ]
    body = "\n".join(
        f"{lab} & {num(full, 2, plus=True)} & {num(q45, 2, plus=True)} \\\\"
        for lab, full, q45 in rows
    )
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{l rr}}
\toprule
 & \multicolumn{{2}}{{c}}{{$R^2_w$ (\%)}} \\
\cmidrule(lr){{2-3}}
Error weighting $w_{{i,t}}$ & \multicolumn{{1}}{{c}}{{Full cross-section}} & \multicolumn{{1}}{{c}}{{Liquid $Q4$--$Q5$}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{The same predictions under different measures.}} Out-of-sample $R^2_w$ of Equation~\eqref{{eq:weighted_r2}} (zero benchmark, 2000--2024) for the identical standard-loss prediction panel, under alternative cross-sectional weightings $w_{{i,t}}$ of the squared errors, on the full cross-section and restricted to the liquid $Q4$--$Q5$ names. All weights are normalised to mean one within each month over the full cross-section and are kept unrenormalised on the liquid subset; $R^2$ is a ratio, so the two columns are directly comparable. The implementability weight is the transaction-cost-rank weight of Section~\ref{{subsec:weighting_schemes}} at \$500M.}}
\label{{tab:evaluation_measures}}
\end{{table}}
"""


RESTRICT_UNIV = {
    "Mall": "Q1--Q5 (all)",
    "MQ2+": "Q2--Q5",
    "MQ3+": "Q3--Q5",
    "MQ4+": "Q4--Q5",
    "MQ5+": "Q5 only",
}


def build_screening_splitting():
    r = pd.read_csv(
        MOT / "step3_restriction/xgboost/dvol/global/baseline/restriction_comparison.csv"
    )
    q = pd.read_csv(
        MOT / "step3_quintile/xgboost/dvol/global/baseline/r2_comparison.csv"
    )
    # Numeric columns are right-aligned so the decimal points line up; see mnum.
    # Panel A has no Delta column, but we keep the slot empty rather than drop
    # it, so that N_train lands in the same column as Panel B's N_train.
    rows_a = "\n".join(
        f"\\quad {RESTRICT_UNIV[r_['model']]} & {mnum(r_['r2_q45_pct'])} & "
        f"{mnum(r_['r2_full_pct'])} & & {int(round(r_['N_train/month'])):,} \\\\"
        for _, r_ in r.iterrows()
    )
    rows_b = "\n".join(
        f"\\quad {q_['quintile']} & {mnum(q_['r2_pooled_pct'])} & {mnum(q_['r2_own_pct'])} & "
        f"{mnum(q_['delta_pp'])} & {int(q_['N_train/month']):,} \\\\"
        for _, q_ in q.iterrows()
    )
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{l rrrr}}
\toprule
\multicolumn{{5}}{{l}}{{\textit{{Panel A: Progressive restriction of the training universe}}}} \\[2pt]
Training universe & \multicolumn{{1}}{{c}}{{$R^2$ Q4--Q5 (\%)}} & \multicolumn{{1}}{{c}}{{$R^2$ full (\%)}} & & \multicolumn{{1}}{{c}}{{$N_{{\mathrm{{train}}}}$/month}} \\
\midrule
{rows_a}
\midrule
\multicolumn{{5}}{{l}}{{\textit{{Panel B: Quintile-specific models}}}} \\[2pt]
Quintile & \multicolumn{{1}}{{c}}{{$R^2$ pooled (\%)}} & \multicolumn{{1}}{{c}}{{$R^2$ own (\%)}} & \multicolumn{{1}}{{c}}{{$\Delta$ (pp)}} & \multicolumn{{1}}{{c}}{{$N_{{\mathrm{{train}}}}$/month}} \\
\midrule
{rows_b}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Neither screening nor splitting repairs the misalignment.}} Panel~A retrains the model on progressively more liquid training universes (holding the rolling protocol, characteristics, and tuned hyperparameters fixed) and evaluates all variants on the same full test cross-section; $R^2$ is the pooled out-of-sample $R^2$ (zero benchmark) on liquid $Q4$--$Q5$ stocks and on the full sample. Liquid-stock accuracy deteriorates monotonically as illiquid observations are discarded. Panel~B trains a separate model within each liquidity quintile and compares its within-quintile $R^2$ with the pooled model's; $\Delta$ is own minus pooled. Every $\Delta$ is negative: the pooled model dominates its own-quintile counterpart in every quintile, consistent with the loss of signal that transfers across the liquidity spectrum---though, as in Panel~A, the design does not separate this from the smaller training samples. In both panels $N_{{\mathrm{{train}}}}$ averages over the rolling training windows, which extend back into the larger cross-sections of the 1990s and therefore exceed the corresponding 2000--2024 evaluation-panel counts.}}
\label{{tab:screening_splitting}}
\end{{table}}
"""


BUILDERS = {
    "DivergenceByCategory.tex": build_divergence_by_category,
    "HeterogeneityFocal.tex": build_heterogeneity_focal,
    "R2ByQuintileML.tex": build_r2_by_quintile,
    "EvaluationMeasures.tex": build_evaluation_measures,
    "ScreeningSplitting.tex": build_screening_splitting,
}


def build_data_descriptives():
    """S4 descriptives of the liquidity and cost inputs (pooled 1989-2024).

    tau(A) is computed through the same code path as the tc/tc_rank weight
    families (src/weighting/schemes.py::_compute_tc_per_stock): lambda = 0.1,
    equal-breadth participation Q_t = A / N_t with N_t the full cross-section,
    and within-month median imputation of missing spread / sigma / dollar volume.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    import numpy as np

    from src.weighting.schemes import _compute_tc_per_stock

    panel = pd.read_parquet(
        ROOT / "data/processed_panel.parquet",
        columns=[
            "permno", "yyyymm",
            "liq_dvol_21d", "liq_BidAskSpread", "liq_excess_sigma_12m_daily",
        ],
    )
    Q = [1, 25, 50, 75, 95, 99]

    def pcell(v, dp):
        # A value that would render as 0.0...0 is shown as "<" the last digit.
        if v < 10 ** (-dp) / 2:
            return rf"$<${10 ** (-dp):.{dp}f}"
        return f"{v:,.{dp}f}"

    def prow(label, vals, dp=1):
        cells = " & ".join(pcell(np.percentile(vals, q), dp) for q in Q)
        ratio = np.percentile(vals, 99) / np.percentile(vals, 50)
        return f"{label} & {cells} & {ratio:,.0f} \\\\"

    dv = panel["liq_dvol_21d"].dropna()
    dv = dv[dv > 0] / 1e6                                   # $M
    sp = panel["liq_BidAskSpread"].dropna().abs() / 2 * 1e4  # half-spread, bps
    sg = panel["liq_excess_sigma_12m_daily"].dropna().abs() * 100  # %

    rows = [
        prow(r"Daily dollar volume (\$M)", dv.values, dp=2),
        prow(r"Half-spread (bps)", sp.values, dp=1),
        prow(r"Daily-scaled volatility $\sigma$ (\%)", sg.values, dp=2),
    ]
    for aum, label in [(100e6, r"\$100M"), (500e6, r"\$500M"), (1e9, r"\$1B")]:
        taus = panel.groupby("yyyymm", group_keys=False).apply(
            lambda g: _compute_tc_per_stock(
                g, aum=aum, lam=0.1, spread_col="liq_BidAskSpread",
                sigma_col="liq_excess_sigma_12m_daily", adv_col="liq_dvol_21d",
            ),
            include_groups=False,
        )
        rows.append(prow(rf"$\tau$ at {label} (bps)", taus.values * 1e4, dp=1))
    body = "\n".join(rows)

    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lccccccc}}
\toprule
 & P1 & P25 & P50 & P75 & P95 & P99 & P99/P50 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Liquidity and cost inputs: cross-sectional descriptives.}} Pooled percentiles across all stock-months, 1989--2024. Daily dollar volume is the 21-day trailing average of daily price times share volume from the CRSP daily file; the half-spread is one half of the \citet{{chen2022open}} \texttt{{BidAskSpread}} series (a \citet{{corwin2012simple}} high--low effective-spread estimate scaled by price); $\sigma$ is the 12-month rolling standard deviation of monthly excess returns rescaled to a daily horizon by $1/\sqrt{{21}}$. The one-way cost $\tau$ applies Equation~\eqref{{eq:tc_primitive}} with $\lambda=0.1$ and equal-breadth participation $Q_t = A/N_t$, where $A$ is the deployed capital and $N_t$ is the full cross-section in month $t$; missing spread, volatility, or volume inputs are imputed at within-month medians, matching the training-weight construction of Section~3.3. The last column reports the ratio of the 99th percentile to the median, computed from unrounded percentiles.}}
\label{{tab:data_descriptives}}
\end{{table}}
"""


BUILDERS["DataDescriptives.tex"] = build_data_descriptives


def _latex_escape(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def build_characteristics_table():
    """Appendix longtable of all 113 predictors (app:features).

    Mirrors the reference paper's asset-characteristics appendix table:
    acronym, description, original reference, and economic category, one row
    per predictor, sourced from data/SignalDoc.csv + data/feature_list.json +
    config/feature_categories.json.
    """
    feats = json.load(open(ROOT / "data/feature_list.json"))["features"]
    broad = json.load(open(ROOT / "config/feature_categories.json"))["broad"]
    sd = pd.read_csv(ROOT / "data/SignalDoc.csv").set_index("Acronym")

    rows = []
    for f in feats:
        r = sd.loc[f]
        year = int(r["Year"]) if pd.notna(r["Year"]) else ""
        ref = f"{_latex_escape(r['Authors'])} ({year})" if year else _latex_escape(r["Authors"])
        rows.append({
            "acr": _latex_escape(f),
            "desc": _latex_escape(r["LongDescription"]),
            "ref": ref,
            "cat": broad.get(f, "Other"),
        })
    # Category order by descending count (as in the prose lists), then acronym.
    order = pd.Series([r["cat"] for r in rows]).value_counts().index.tolist()
    rows.sort(key=lambda r: (order.index(r["cat"]), r["acr"].lower()))

    body = "\n".join(
        f"{r['acr']} & {r['desc']} & {r['ref']} & {r['cat']} \\\\" for r in rows
    )
    # Compact landscape longtable, mirroring the reference paper's
    # asset-characteristics appendix table (arraystretch 0.7, tabcolsep 2pt,
    # tiny body, midrule-only frame, "(continued)" markers).
    return rf"""\begin{{landscape}}
\renewcommand{{\arraystretch}}{{0.7}}
\setlength{{\tabcolsep}}{{2pt}}
\begin{{tiny}}
\begin{{longtable}}{{>{{\raggedright\arraybackslash}}p{{3.2cm}}>{{\raggedright\arraybackslash}}p{{8.4cm}}>{{\raggedright\arraybackslash}}p{{6.2cm}}>{{\raggedright\arraybackslash}}p{{3.0cm}}}}
\caption{{\footnotesize \textbf{{Stock-level predictors.}} The $113$ characteristics used throughout the paper, drawn from the open-source library of \citet{{chen2022open}} (see Section~\ref{{subsec:returns_chars}} for the selection rule). Descriptions are the library's short names; full construction details are documented in the library. The category column reports the broad economic grouping used in Sections~\ref{{sec:imbalance}} and~\ref{{sec:framework}}.}}
\label{{tab:characteristics}} \\
\midrule
\textbf{{Acronym}} & \textbf{{Description}} & \textbf{{Original reference}} & \textbf{{Category}} \\
\midrule
\endfirsthead
\multicolumn{{4}}{{c}}{{\small\itshape (continued)}} \\
\midrule
\textbf{{Acronym}} & \textbf{{Description}} & \textbf{{Original reference}} & \textbf{{Category}} \\
\midrule
\endhead
\midrule
\multicolumn{{4}}{{r}}{{\small\itshape (continued)}} \\
\endfoot
\midrule
\endlastfoot
{body}
\end{{longtable}}
\end{{tiny}}
\end{{landscape}}
"""


BUILDERS["CharacteristicsTable.tex"] = build_characteristics_table


# ════════════════════════════════════════════════════════════════════
# Section 5 tables (eval_realignment 41-45 + formal 21b), xgboost,
# primary spec tc_rank_lam3_500m.
# ════════════════════════════════════════════════════════════════════

EVAL = ROOT / "outputs/eval_realignment/analysis/xgboost/tc_rank_lam3_500m"
FORMAL_AN = ROOT / "outputs/formalanalysis/analysis/xgboost/tc_rank_lam3_500m"
FORMAL_EXP = ROOT / "outputs/formalanalysis/experiment/xgboost"
AUM_GRID = [("PropTC", "Prop.\\ TC"), ("100M", "\\$100M"),
            ("500M", "\\$500M"), ("1B", "\\$1B")]

# The three specifications reported side by side in Section 4 (author
# decision 2026-07-13). Labels match the WeightFamilySweep rows; the paper
# symbol for the softmax tilt is beta (lambda is the impact coefficient).
SEC4_SPECS = [
    ("tc_rank_lam3_500m", "TC-rank $\\beta{=}3$, \\$500M (primary)"),
    ("tc_500m", "TC level, \\$500M"),
    ("softmax_rank_lam2", "Softmax rank $\\beta{=}2$"),
]


def _two_by_two(aum, spec="tc_rank_lam3_500m"):
    f = ROOT / f"outputs/eval_realignment/analysis/xgboost/{spec}/two_by_two_{aum}.csv"
    t = pd.read_csv(f).set_index("metric")["value"]
    return t.to_dict()


def build_deployment_weighted_r2():
    LAB = {"Q1": "Q1 (Illiquid)", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4",
           "Q5": "Q5 (Liquid)", "liquid_q4q5": "Liquid (Q4--Q5)", "full": "Full cross-section"}
    ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5", "liquid_q4q5", "full"]
    blocks = []
    for spec, _ in SEC4_SPECS:
        base = ROOT / f"outputs/eval_realignment/analysis/xgboost/{spec}/liquidity_breakpoints/nyse"
        r2 = pd.read_csv(base / "deployment_weighted_r2.csv")
        st = pd.read_csv(base / "deployment_weighted_error_diff_stats.csv")
        blocks.append(
            r2.merge(st[["universe", "t_stat"]], on="universe").set_index("universe").loc[ORDER]
        )
    tmax = max(b["t_stat"].abs().max() for b in blocks)
    assert tmax < 1.96, "caption claims no differential is significant"
    rows = []
    for u in ORDER:
        pre = r"\midrule" + "\n" if u == "liquid_q4q5" else ""
        cells = []
        for b in blocks:
            r = b.loc[u]
            cells.append(
                f"{mnum(r['r2_weighted_std_pct'])} & {mnum(r['r2_weighted_wt_pct'])} & "
                f"{num(r['delta_pct'], plus=True)} & {mnum(r['t_stat'])}"
            )
        rows.append(pre + f"{LAB[u]} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    # The primary header is stacked on two lines: a \multicolumn wider than
    # its spanned columns dumps ALL excess width into the last spanned column
    # (the primary block's t column), opening a large visual gap.
    HEAD = {
        "tc_rank_lam3_500m": "\\shortstack{TC-rank $\\beta{=}3$, \\$500M\\\\(primary)}",
        "tc_500m": "TC level, \\$500M",
        "softmax_rank_lam2": "Softmax rank $\\beta{=}2$",
    }
    heads = " & ".join(rf"\multicolumn{{4}}{{c}}{{{HEAD[spec]}}}" for spec, _ in SEC4_SPECS)
    sub = " & ".join([r"\multicolumn{1}{c}{Std} & \multicolumn{1}{c}{Wt} & "
                      r"\multicolumn{1}{c}{$\Delta$} & \multicolumn{1}{c}{$t(D_t)$}"] * len(SEC4_SPECS))
    return rf"""\begin{{table}}[t!]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{l rrrr rrrr rrrr}}
\toprule
 & {heads} \\
\cmidrule(lr){{2-5}} \cmidrule(lr){{6-9}} \cmidrule(lr){{10-13}}
Universe & {sub} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Deployment-weighted out-of-sample $R^2$.}} The deployment-weighted $R^2$ of Equation~\eqref{{eq:dw_r2}} for the standard and implementability-weighted models under the three Section-4 specifications, within NYSE-breakpoint dollar-volume quintiles of the prediction universe ($Q1$ = least liquid), on the pooled liquid set, and on the full cross-section. Each block prices errors under its own specification's weight, so the blocks are three different metrics rather than one metric on three models: the standard model is the same in every block, and its measured accuracy varies only because the weight does. Within each block the global mean-one normalisation is held fixed across subsets. Std and Wt report $R^2_{{\tilde w}}$ in per cent; $\Delta$ is weighted minus standard in percentage points; $\Delta$ is computed on unrounded values, so it can differ from the printed cells in the last digit; $t(D_t)$ is the Newey--West $t$-statistic (6 lags) on the monthly weighted mean-squared-error differential $D_t$ (Appendix~\ref{{app:capacity}}). No individual differential is statistically significant in any block (the largest $|t|$ in the table is ${tmax:.2f}$). 299 months, 2000--2024.}}
\label{{tab:dw_r2}}
\end{{table}}
"""


def build_capacity_portfolio():
    def metrics(tag, aum, spec="tc_rank_lam3_500m"):
        f = (ROOT / f"outputs/eval_realignment/analysis/xgboost/{spec}/"
             f"capacity_portfolio{tag}_metrics_{aum}.csv")
        df = pd.read_csv(f)
        s = df[df.row_type == "standard"].iloc[0]
        w = df[df.row_type == "weighted"].iloc[0]
        d = df[df.row_type == "difference"].iloc[0]
        return s, w, d

    rows = []
    for spec, slab in SEC4_SPECS:
        s0, w0, _ = metrics("", "500M", spec)
        p_gross = _two_by_two("500M", spec)["LW p-val (training, gross)"]
        rows.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{slab}}}}} \\")
        rows.append(
            f"\\quad Gross (invariant to $A$) & {num(s0['gross_sr_annual'])} & {num(w0['gross_sr_annual'])} & "
            f"{num(w0['gross_sr_annual'] - s0['gross_sr_annual'], plus=True)} & {_pfmt(p_gross)} \\\\"
        )
        for aum, lab in AUM_GRID:
            s, w, d = metrics("", aum, spec)
            delta_ann = w["net_sr_annual"] - s["net_sr_annual"]
            tail = " \\\\[2pt]" if aum == AUM_GRID[-1][0] and spec != SEC4_SPECS[-1][0] else " \\\\"
            rows.append(
                f"\\quad Net at {lab} & {num(s['net_sr_annual'])} & {num(w['net_sr_annual'])} & "
                f"{num(delta_ann, plus=True)} & {_pfmt(d['net_sr_diff_pval'])}" + tail
            )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{l rrrr}}
\toprule
 & \multicolumn{{1}}{{c}}{{Standard}} & \multicolumn{{1}}{{c}}{{Weighted}} & \multicolumn{{1}}{{c}}{{$\Delta$SR}} & \multicolumn{{1}}{{c}}{{$p$}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{The capacity portfolio.}} Annualised Sharpe ratios of the centred, dollar-neutral, unit-gross portfolio of Equations~\eqref{{eq:book_center}}--\eqref{{eq:book_norm}}, one block per Section-4 specification, each trained, evaluated, and costed under its own weight. Within each block the first row reports the gross Sharpe ratio, which does not depend on deployed capital; the remaining rows report net Sharpe ratios as deployed capital $A$ sweeps the capital grid, with Prop.~TC the half-spread-only scenario, in which impact is off and net returns do not depend on $A$. $\Delta$SR is weighted minus standard; one-sided $p$-values are from the \citet{{ledoit2008robust}} studentised circular-block bootstrap on the monthly Sharpe difference, on the gross series for the gross row and the net series otherwise. $\Delta$SR is computed on unrounded values, so it can differ from the printed cells in the last digit. No primary-specification training difference is statistically significant at any capital level, the softmax-rank ($\beta{{=}}2$) difference is significant only at \$1B, and the TC-level difference is significant at every capital level and read against its base in Section~\ref{{subsec:twobytwo_results}}. The equal- and value-weight capacity benchmarks for the primary specification are tabulated with the certainty-equivalent companion in Appendix~\ref{{ia:capacity}} (Table~\ref{{tab:capacity_ce}}). 299 months, 2000--2024.}}
\label{{tab:capacity}}
\end{{table}}
"""


def _supplement_pvals(book, spec="tc_rank_lam3_500m"):
    """Pairwise p-values from the inference supplement (script 46), by AUM."""
    f = ROOT / f"outputs/eval_realignment/analysis/xgboost/{spec}/inference_supplement.csv"
    out = {}
    if f.exists():
        d = pd.read_csv(f)
        d = d[(d.part == "A_pairwise_LW") & (d.book == book)]
        for _, r in d.iterrows():
            out[(r["aum"], r["statistic"])] = r["p_value_one_sided"]
    return out


def build_capacity_two_by_two():
    t5 = _two_by_two("500M")

    def cell(c):
        return (f"{num(t5[f'SR_net_annualized({c})'])} & {num(t5[f'SR_gross_annualized({c})'])} & "
                f"{t5[f'TC mean monthly ({c})'] * 1e4:.0f} & {t5[f'Turnover ({c})']:.2f}")

    rows_b = []
    for spec, slab in SEC4_SPECS:
        sup = _supplement_pvals("long_short", spec)
        rows_b.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{slab}}}}} \\")
        for aum, lab in AUM_GRID:
            t = _two_by_two(aum, spec)
            p_exec = sup.get((aum, "execution_1B_vs_1A"))
            adopt = t["SR_net_annualized(2B)"] - t["SR_net_annualized(1B)"]
            p_adopt = sup.get((aum, "adoption_2B_vs_1B"))
            exec_cell = f"{num(t['Net portfolio effect annualized'], plus=True)}"
            if p_exec is not None:
                exec_cell += f" ({_pfmt(p_exec)})"
            adopt_cell = f"{num(adopt, plus=True)}"
            if p_adopt is not None:
                adopt_cell += f" ({_pfmt(p_adopt)})"
            tail = r" \\"
            rows_b.append(
                f"\\quad {lab} & "
                f"{num(t['Net training effect annualized'], plus=True)} ({_pfmt(t['LW p-val (training, net)'])}) & "
                f"{exec_cell} & {adopt_cell} & "
                f"{num(t['Net total effect annualized'], plus=True)} ({_pfmt(t['LW p-val (total, net)'])}) & "
                f"{num(t['Net interaction annualized'], plus=True)}" + tail
            )
    body_b = "\n".join(rows_b)

    rows_c = []
    for spec, slab in SEC4_SPECS:
        tsp = _two_by_two("500M", spec)
        rows_c.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{slab}}}}} \\")
        for key, lab in [("capm", "CAPM"), ("ff3", "FF3"), ("ff5", "FF5"), ("ff5_mom", "FF5+Mom")]:
            cells = []
            for c in ["1A", "1B", "2A", "2B"]:
                a = tsp[f"alpha_{key}({c})_annual"] * 100
                tt = tsp[f"alpha_{key}({c})_tstat"]
            # A third decimal when two would round onto the 1.96 boundary,
            # so the reader can resolve which side the statistic falls on.
                if f"{abs(tt):.2f}" == "1.96":
                    ts = f"({tt:.3f})" if tt >= 0 else f"($-${abs(tt):.3f})"
                else:
                    ts = tstat(tt)
                cells.append(f"{num(a)} {ts}")
            tail = r" \\"
            rows_c.append(f"\\quad {lab} & " + " & ".join(cells) + tail)
    body_c = "\n".join(rows_c)

    return rf"""\begin{{table}}[t!]
\centering
\footnotesize
\renewcommand{{\arraystretch}}{{0.85}}
\setlength{{\abovecaptionskip}}{{4pt}}
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lccccccc}}
\toprule
\multicolumn{{8}}{{l}}{{\textit{{Panel A: The four cells at \$500M (primary specification)}}}} \\[2pt]
 & Net SR & Gross SR & Cost (bps/mo) & Turnover & & & \\
\midrule
\multicolumn{{8}}{{l}}{{\emph{{Full rebalance ($A$)}}}} \\
\quad Standard training ($1A$) & {cell('1A')} & & & \\
\quad Weighted training ($2A$) & {cell('2A')} & & & \\
\multicolumn{{8}}{{l}}{{\emph{{Breakeven gate ($B$)}}}} \\
\quad Standard training ($1B$) & {cell('1B')} & & & \\
\quad Weighted training ($2B$) & {cell('2B')} & & & \\
\midrule
\multicolumn{{8}}{{l}}{{\textit{{Panel B: Decomposition across deployed capital (net, annualised)}}}} \\[2pt]
 & Training ($p$) & Execution ($p$) & $2B{{-}}1B$ ($p$) & Total ($p$) & Interaction & & \\
\midrule
{body_b}
\midrule
\multicolumn{{8}}{{l}}{{\textit{{Panel C: Annualised factor alphas at \$500M (\%, $t$-statistics in parentheses)}}}} \\[2pt]
 & $1A$ & $1B$ & $2A$ & $2B$ & & & \\
\midrule
{body_c}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Cost-aware execution and the training $\times$ execution decomposition.}} Panel~A reports the four cells of the two-by-two design at the primary \$500M scale---training loss (standard vs.\ implementability-weighted) crossed with execution (full monthly rebalancing vs.\ the breakeven gate of Equation~\eqref{{eq:gate}}): net and gross annualised Sharpe ratios, mean monthly cost drag in basis points, and monthly one-sided turnover. Panels~B and~C span the three Section-4 specifications, each trained, evaluated, and costed under its own weight; Panel~B decomposes the net gain of Equation~\eqref{{eq:decomp}} at each level of deployed capital. One-sided $p$-values in parentheses are from the \citet{{ledoit2008robust}} studentised circular-block bootstrap; the training, total, execution ($1B$ vs.\ $1A$), and adoption ($2B$ vs.\ $1B$, what weighted training adds on top of the gate) contrasts share the same seeded machinery, with the pairwise tests computed in the inference supplement of Appendix~\ref{{app:bootstrap}}. Effects are computed on unrounded values, so the printed identities can differ in the last digit. Turnover and gate composition do not vary with $A$ (the gate triggers on the half-spread alone), so the execution effect grows with capital purely through the price of the avoided trades. Panel~C reports annualised factor alphas of the four net return series at \$500M, per specification, with Newey--West $t$-statistics (6 lags). 299 months, 2000--2024.}}
\label{{tab:capacity_2x2}}
\end{{table}}
"""


def build_longonly_two_by_two():
    lt5 = pd.read_csv(EVAL / "longonly_two_by_two_500M.csv").set_index("metric")["value"].to_dict()

    def cell(c):
        return (f"{num(lt5[f'SR_net_annualized({c})'])} & "
                f"{lt5[f'TC mean monthly ({c})'] * 1e4:.0f} & {lt5[f'Turnover ({c})']:.2f}")

    def pfmt_lo(p):
        # Table-local: three decimals on [0.01, 0.10) so the borderline $1B
        # adoption p (0.097) is never printed as the bare 10% threshold.
        if p < 0.01:
            return f"{p:.4f}"
        if p < 0.10:
            return f"{p:.3f}"
        return f"{p:.2f}"

    rows_b = []
    for spec, slab in SEC4_SPECS:
        sup = _supplement_pvals("long_only", spec)
        rows_b.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{slab}}}}} \\")
        for aum, lab in AUM_GRID:
            t = pd.read_csv(
                ROOT / f"outputs/eval_realignment/analysis/xgboost/{spec}/longonly_two_by_two_{aum}.csv"
            ).set_index("metric")["value"].to_dict()
            inter = t.get("Net interaction annualized",
                          t["Net total effect annualized"] - t["Net training effect annualized"]
                          - t["Net portfolio effect annualized"])
            p_exec = sup.get((aum, "execution_1B_vs_1A"))
            adopt = t["SR_net_annualized(2B)"] - t["SR_net_annualized(1B)"]
            p_adopt = sup.get((aum, "adoption_2B_vs_1B"))
            exec_cell = f"{num(t['Net portfolio effect annualized'], plus=True)}"
            if p_exec is not None:
                exec_cell += f" ({pfmt_lo(p_exec)})"
            adopt_cell = f"{num(adopt, plus=True)}"
            if p_adopt is not None:
                adopt_cell += f" ({pfmt_lo(p_adopt)})"
            tail = r" \\[2pt]" if aum == AUM_GRID[-1][0] and spec != SEC4_SPECS[-1][0] else r" \\"
            rows_b.append(
                f"\\quad {lab} & "
                f"{num(t['Net training effect annualized'], plus=True)} ({pfmt_lo(t['LW p-val (training, net)'])}) & "
                f"{exec_cell} & {adopt_cell} & "
                f"{num(t['Net total effect annualized'], plus=True)} ({pfmt_lo(t['LW p-val (total, net)'])}) & "
                f"{num(inter, plus=True)}" + tail
            )
    body_b = "\n".join(rows_b)
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lccccccc}}
\toprule
\multicolumn{{8}}{{l}}{{\textit{{Panel A: The four cells at \$500M (primary specification)}}}} \\[2pt]
 & \multicolumn{{3}}{{c}}{{Plain membership ($A$)}} & \multicolumn{{3}}{{c}}{{Hysteresis band ($B$)}} & \\
\cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}}
 & Net SR & Cost (bps/mo) & Turnover & Net SR & Cost (bps/mo) & Turnover & \\
\midrule
\quad Standard training & {cell('1A')} & {cell('1B')} & \\
\quad Weighted training & {cell('2A')} & {cell('2B')} & \\
\midrule
\multicolumn{{8}}{{l}}{{\textit{{Panel B: Decomposition across deployed capital (net, annualised)}}}} \\[2pt]
 & Training ($p$) & Execution ($p$) & $2B{{-}}1B$ ($p$) & Total ($p$) & Interaction & & \\
\midrule
{body_b}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{The long-only capacity portfolio.}} The two-by-two design applied to the long-only portfolio of Equation~\eqref{{eq:longonly}}: rows vary the training loss, columns vary execution between plain monthly membership refresh and the cost-scaled membership-hysteresis band. Panel~A reports net annualised Sharpe ratios, mean monthly cost drag (bps), and monthly one-sided turnover at \$500M for the primary specification; Panel~B decomposes the net gain at each level of deployed capital, one block per Section-4 specification, each trained, evaluated, and costed under its own weight; one-sided \citet{{ledoit2008robust}} bootstrap $p$-values in parentheses cover the training, execution ($1B$ vs.\ $1A$), adoption ($2B$ vs.\ $1B$), and total contrasts (the execution and adoption tests are computed in the inference supplement of Appendix~\ref{{app:bootstrap}}). Effects are computed on unrounded values, so the printed decomposition identities can differ in the last digit. 299 months, 2000--2024.}}
\label{{tab:longonly_2x2}}
\end{{table}}
"""


def build_reallocation():
    """S5.5 mechanism table: per-window importance-share shifts with NW t.

    All inference in this table is computed here, per window, with the same
    Newey-West(6) convention as the 21b per-feature tests: for each group or
    feature, the monthly share of total |SHAP| importance is computed for the
    standard and weighted models on their common windows, and the t-statistic
    is on the time series of paired share differences.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    import numpy as np

    from src.evaluation.statistics import newey_west_tstat

    std = pd.read_csv(FORMAL_EXP / "standard/importance_shap.csv").set_index("yyyymm").sort_index()
    wt = pd.read_csv(FORMAL_EXP / "tc_rank_lam3_500m/importance_shap.csv").set_index("yyyymm").sort_index()
    common = std.index.intersection(wt.index)
    std, wt = std.loc[common], wt.loc[common]
    feats = [c for c in std.columns if c in wt.columns]
    tot_s = std[feats].abs().sum(axis=1)
    tot_w = wt[feats].abs().sum(axis=1)

    def share_diff(sub):
        s = std[sub].abs().sum(axis=1) / tot_s
        w = wt[sub].abs().sum(axis=1) / tot_w
        return s.mean(), w.mean(), (w - s)

    # Group definitions
    rel = pd.read_csv(MOT / "step3/xgboost/dvol/illiquidity_relatedness.csv", index_col=0)
    cluster = [f for f in rel[abs(rel[rel.columns[0]]) > 0.5].index if f in feats]
    import yaml

    cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
    econ = [f for f in cfg["data"]["illiquidity_features"] if f in feats]
    shift = pd.read_csv(FORMAL_AN / "importance_shift.csv")
    gmap = dict(zip(shift["feature"], shift["group"]))
    retain = [f for f in feats if gmap.get(f) in ("both", "Q5_only")]

    rows_a = []
    for lab, sub in [
        (rf"Illiquidity cluster ($|\bar\rho_j|>0.5$; {len(cluster)})", cluster),
        (rf"Illiquidity/microstructure group ({len(econ)})", econ),
        (rf"Liquid-signal group (quintile-based; {len(retain)})", retain),
    ]:
        ms, mw, d = share_diff(sub)
        r = newey_west_tstat(d.values, lags=6)
        rows_a.append(
            f"\\quad {lab} & {mnum(ms*100)} & {mnum(mw*100)} & "
            f"{num(r['mean']*100, plus=True)} & {mnum(r['t_stat'])} \\\\"
        )
    body_a = "\n".join(rows_a)

    # Per-feature: top movers by |mean per-window share diff|
    per = []
    for f in feats:
        ms, mw, d = share_diff([f])
        r = newey_west_tstat(d.values, lags=6)
        per.append((f, ms * 100, mw * 100, r["mean"] * 100, r["t_stat"]))
    per.sort(key=lambda x: x[3])
    losers, gainers = per[:5], sorted(per[-5:], key=lambda x: -x[3])
    rows_b = []
    for f, ms, mw, dd, tt in losers + gainers:
        rows_b.append(f"\\quad {_latex_escape(f)} & {mnum(ms)} & {mnum(mw)} & {num(dd, plus=True)} & {mnum(tt)} \\\\")
    rows_b.insert(5, r"\addlinespace")
    body_b = "\n".join(rows_b)

    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{l rrrr}}
\toprule
 & \multicolumn{{2}}{{c}}{{Share of importance (\%)}} & & \\
\cmidrule(lr){{2-3}}
 & \multicolumn{{1}}{{c}}{{Standard}} & \multicolumn{{1}}{{c}}{{Weighted}} & \multicolumn{{1}}{{c}}{{$\Delta$ (pp)}} & \multicolumn{{1}}{{c}}{{$t$}} \\
\midrule
\multicolumn{{5}}{{l}}{{\textit{{Panel A: Group shares}}}} \\[2pt]
{body_a}
\midrule
\multicolumn{{5}}{{l}}{{\textit{{Panel B: Largest individual share shifts}}}} \\[2pt]
{body_b}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Feature-importance reallocation.}} Shares of total SHAP importance under standard and implementability-weighted training (primary specification), averaged over the 299 rolling windows; $\Delta$ is the mean of the per-window paired share differences and $t$ its Newey--West statistic (6 lags), the same convention as the per-feature tests. In Panel~A, the illiquidity cluster is the rule-based group of Section~\ref{{sec:imbalance}} (characteristics whose average rank correlation with dollar volume exceeds $0.5$ in absolute value); the illiquidity/microstructure group is the broader economically defined list fixed ex ante; the liquid-signal group collects characteristics whose predictive slopes are significant in the most liquid quintile of the all-113-characteristic quintile-specific Fama--MacBeth regressions of Section~\ref{{subsec:heterogeneity}} (significant in $Q5$, whether or not also in $Q1$). Panel~B reports the five largest decreases and increases in per-window importance share. Out-of-sample period 2000--2024.}}
\label{{tab:reallocation}}
\end{{table}}
"""


def build_capacity_ce():
    """Appendix companion to tab:capacity: mean net returns, turnover, CE grid."""
    def rows_for(tag, aum, lab):
        df = pd.read_csv(EVAL / f"capacity_portfolio{tag}_metrics_{aum}.csv")
        out = []
        for rt, rlab in [("standard", "standard"), ("weighted", "weighted")]:
            r = df[df.row_type == rt].iloc[0]
            out.append(
                f"\\quad {lab}, {rlab} & {num(r['gross_sr_annual'])} & {num(r['net_sr_annual'])} & "
                f"{num(r['net_mean_annual']*100)} & "
                f"{r['turnover_mean']:.2f} & {num(r['net_ce_annual_g1']*100)} & "
                f"{num(r['net_ce_annual_g5']*100)} & {num(r['net_ce_annual_g10']*100)} \\\\"
            )
        return out

    rows_a = []
    for aum, lab in AUM_GRID:
        rows_a.extend(rows_for("", aum, lab))
        rows_a.append(r"[2pt]") if False else None
    # interleave the [2pt] spacing between AUM blocks
    spaced_a = []
    for i, r in enumerate(rows_a):
        spaced_a.append(r if (i % 2 == 0 or i == len(rows_a) - 1) else r + "[2pt]")
    body_a = "\n".join(spaced_a)
    rows_b = []
    for tag, lab in [("_equal", "Equal"), ("_value", "Value")]:
        rr = rows_for(tag, "500M", lab)
        rows_b.append(rr[0])
        rows_b.append(rr[1] + ("[2pt]" if tag == "_equal" else ""))
    body_b = "\n".join(rows_b)
    return rf"""\begin{{table}}[t!]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{l rrrrrrr}}
\toprule
 & \multicolumn{{1}}{{c}}{{Gross SR}} & \multicolumn{{1}}{{c}}{{Net SR}} & \multicolumn{{1}}{{c}}{{Net mean (\%)}} & \multicolumn{{1}}{{c}}{{Turnover}} & \multicolumn{{1}}{{c}}{{$\mathrm{{CE}}(1)$}} & \multicolumn{{1}}{{c}}{{$\mathrm{{CE}}(5)$}} & \multicolumn{{1}}{{c}}{{$\mathrm{{CE}}(10)$}} \\
\midrule
\multicolumn{{8}}{{l}}{{\textit{{Panel A: Signal-weighted portfolio across deployed capital}}}} \\[2pt]
{body_a}
\midrule
\multicolumn{{8}}{{l}}{{\textit{{Panel B: Capacity-weight benchmarks at \$500M}}}} \\[2pt]
{body_b}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Mean net returns, turnover, and certainty equivalents of the capacity portfolios.}} Companion to the primary (TC-rank, \$500M) block of Table~\ref{{tab:capacity}}: gross and net annualised Sharpe ratios, annualised mean net returns (\%), mean monthly one-sided turnover, and the annualised certainty-equivalent return $\mathrm{{CE}}(\gamma)$ of Equation~\eqref{{eq:ce}} (\%) evaluated on the monthly net series, for the full-rebalance capacity portfolio of Equations~\eqref{{eq:book_center}}--\eqref{{eq:book_norm}} under standard and implementability-weighted training. Panel~A sweeps deployed capital under the signal capacity weight $\tilde w$; turnover does not vary with $A$. Panel~B replaces the capacity weight with equal and value weights at \$500M, the benchmarks that isolate the deployment stage in Section~\ref{{subsec:capacity_results}}. Within Panel~A, the certainty-equivalent comparison of the two training losses reproduces the net-Sharpe comparison of Table~\ref{{tab:capacity}}'s primary block: the standard portfolio dominates under proportional costs, the two portfolios are within a few basis points of one another at \$100M, and the weighted portfolio dominates at \$500M and \$1B at every $\gamma$. In Panel~B the two criteria diverge for the equal-weighted portfolio: under weighted training it carries the higher mean net return and the higher certainty equivalent at every $\gamma$ although its net Sharpe ratio is lower---both means are negative, so the Sharpe ordering there is the familiar negative-mean pathology. 299 months, 2000--2024.}}
\label{{tab:capacity_ce}}
\end{{table}}
"""


# ════════════════════════════════════════════════════════════════════
# Section 6 tables (formal 21e track, spec/model sweeps, regimes).
# ════════════════════════════════════════════════════════════════════

FORMAL_AN_ROOT = ROOT / "outputs/formalanalysis/analysis/xgboost/tc_rank_lam3_500m"
EVAL_ROOT = ROOT / "outputs/eval_realignment/analysis"


def build_consistency_dose_response():
    """S6.1: training effect across the 21e leg-weighting x universe grid."""
    LEGS = [("prediction_quantile", "Equal legs"),
            ("prediction_quantile_signal_weight", "Deployment-weighted legs"),
            ("prediction_quantile_value_weight", "Value legs")]
    UNIS = [("full", "Full"), ("nyse", "NYSE"), ("top60", "Top-60\\%")]

    def grid(leg, uni):
        f = FORMAL_AN_ROOT / leg / "stock_universe" / uni / "two_by_two_500M.csv"
        if not f.exists():
            return None
        t = pd.read_csv(f).set_index("metric")["value"]
        return t

    rows_a = []
    ex_lo, ex_hi = float("inf"), float("-inf")
    max_total_p = 0.0
    for leg, llab in LEGS:
        cells = []
        for uni, _ in UNIS:
            t = grid(leg, uni)
            if t is None:
                cells.append("---")
                continue
            eff = t["Net training effect annualized"]
            p = t["LW p-val (training, net)"]
            cells.append(f"{num(eff, plus=True)} ({p:.2f})")
            ex = t["Net portfolio effect annualized"]
            ex_lo, ex_hi = min(ex_lo, ex), max(ex_hi, ex)
            max_total_p = max(max_total_p, t["LW p-val (total, net)"])
        rows_a.append(f"\\quad {llab} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows_a)
    p_bound = "0.001" if max_total_p <= 0.001 else ("0.002" if max_total_p <= 0.002 else f"{max_total_p:.4f}")
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lccc}}
\toprule
 & \multicolumn{{3}}{{c}}{{Stock universe}} \\
\cmidrule(lr){{2-4}}
Leg weighting & Full & NYSE & Top-60\% \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{The dose--response of consistency on the sorted portfolio.}} Net annualised training effect ($2A-1A$, weighted minus standard training on the plain sorted portfolio) for the prediction-quantile long--short portfolio of Section~\ref{{subsec:formal_2x2}}, across leg-weighting schemes (rows) and stock universes (columns), at the primary specification and \$500M (XGBoost); one-sided \citet{{ledoit2008robust}} bootstrap $p$-values in parentheses. The equal-legs/full-universe cell is the conventional scoreboard of the machine-learning asset-pricing literature; the NYSE universe and the deployment-weighted legs (within-leg positions proportional to $\tilde w$; the forecast determines membership only) bring the object into alignment with the trained objective, while the top-$60\%$ universe pre-screens away the illiquid margin the weighting corrects, so the substitutes logic of Section~\ref{{subsec:cost_sensitive}} predicts the attenuation toward zero observed in that column. The execution (hysteresis) effect is large in every cell (between ${ex_lo:+.2f}$ and ${ex_hi:+.2f}$), and the total effect is statistically significant in every cell ($p\le {p_bound}$); no individual training effect is significant, and the table is read as a pattern of point estimates, not as cell-level inference. The capacity-portfolio counterpart of the training effect at the same specification is $+0.09$ (Section~\ref{{sec:results}}). 299 months, 2000--2024.}}
\label{{tab:dose_response}}
\end{{table}}
"""


def _pfmt(p):
    """p-value display: two decimals unless p < 0.01, then four."""
    return f"{p:.4f}" if p < 0.01 else f"{p:.2f}"


def build_weight_family_sweep():
    """S6.3: 2x2 effects, deployment-R2, and weight-concentration by family."""
    import copy
    import sys

    sys.path.insert(0, str(ROOT))
    import yaml

    from src.data.loader import load_processed_panel
    from src.weighting.schemes import compute_weights

    cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
    out = load_processed_panel()
    panel = out[0] if isinstance(out, tuple) else out

    cfg2 = copy.deepcopy(cfg); cfg2["weighting"]["softmax_rank_lambda"] = 2.0
    cfg3 = copy.deepcopy(cfg); cfg3["weighting"]["softmax_rank_lambda"] = 3.0
    SPECS = [
        ("tc_rank_lam3_500m", "TC-rank $\\beta{=}3$ (primary)", "tc_rank", cfg, 500e6),
        ("tc_500m", "TC level (\\$500M)", "tc", cfg, 500e6),
        ("softmax_rank_lam2", "Softmax rank $\\beta{=}2$", "softmax_rank", cfg2, None),
        ("softmax_rank_lam3", "Softmax rank $\\beta{=}3$", "softmax_rank", cfg3, None),
        ("dolvol", "Dollar volume", "dolvol", cfg, None),
    ]
    rows = []
    for spec, lab, scheme, c, aum in SPECS:
        w = compute_weights(panel, scheme=scheme, config=c, aum=aum)
        df = pd.DataFrame({"yyyymm": panel["yyyymm"], "w": w}).dropna()
        df = df[df.w > 0]
        g = df.groupby("yyyymm")["w"]
        ess = g.apply(lambda x: (x.sum() ** 2 / (x ** 2).sum()) / len(x) * 100).mean()
        top10 = g.apply(lambda x: x.nlargest(10).sum() / x.sum() * 100).mean()
        t = pd.read_csv(EVAL_ROOT / f"xgboost/{spec}/two_by_two_500M.csv").set_index("metric")["value"]
        r2 = pd.read_csv(EVAL_ROOT / f"xgboost/{spec}/liquidity_breakpoints/nyse/deployment_weighted_r2.csv")
        dq45 = r2.loc[r2.universe == "liquid_q4q5", "delta_pct"].iloc[0]
        # Training p-values follow the Section 5 two-decimal convention, except
        # values that would round to 0.00 are printed at four decimals so the
        # table matches the precision quoted in the text (e.g. p=0.0004).
        p_tr = t["LW p-val (training, net)"]
        p_tr_txt = f"{p_tr:.4f}" if p_tr < 0.005 else f"{p_tr:.2f}"

        # Execution and total p-values print four decimals below 0.05 so that
        # exact bootstrap values (e.g. 0.0136) are quoted rather than "0.01".
        def pfmt4(p):
            return f"{p:.4f}" if p < 0.05 else f"{p:.2f}"

        exec_cell = f"{num(t['Net portfolio effect annualized'], plus=True)}"
        supf = EVAL_ROOT / f"xgboost/{spec}/inference_supplement.csv"
        if supf.exists():
            d = pd.read_csv(supf)
            d = d[(d.part == "A_pairwise_LW") & (d.book == "long_short")
                  & (d.aum == "500M") & (d.statistic == "execution_1B_vs_1A")]
            if len(d):
                exec_cell += f" ({pfmt4(d['p_value_one_sided'].iloc[0])})"
        rows.append(
            f"\\quad {lab} & {ess:.0f} & {top10:.1f} & {num(dq45, plus=True)} & "
            f"{num(t['Net training effect annualized'], plus=True)} ({p_tr_txt}) & "
            f"{exec_cell} & "
            f"{num(t['Net total effect annualized'], plus=True)} ({pfmt4(t['LW p-val (total, net)'])}) \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t!]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{2.7pt}}
\begin{{tabular}}{{lcccccc}}
\toprule
 & \multicolumn{{2}}{{c}}{{Weight concentration}} & & \multicolumn{{3}}{{c}}{{Two-by-two at \$500M (net)}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{5-7}}
Weight family & ESS (\%) & Top-10 (\%) & $\Delta R^2_{{\tilde w}}$ (pp) & Training ($p$) & Execution ($p$) & Total ($p$) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Across weighting families.}} Each row re-estimates the weighted model and the full evaluation pipeline under a different implementability-weight family (Section~\ref{{subsec:weighting_schemes}}). All rows use XGBoost; the first row is the primary specification of Section~\ref{{sec:results}}. ESS is the mean monthly Kish effective sample size $(\sum_i w_i)^2/\sum_i w_i^2$ as a percentage of the cross-section; Top-10 is the mean share of total weight carried by the ten largest names. $\Delta R^2_{{\tilde w}}$ is the deployment-weighted $R^2$ gain of Equation~\eqref{{eq:dw_r2}} on the liquid $Q4$--$Q5$ set, each family evaluated under its own weight. The two-by-two columns report the net annualised training, execution ($1B$ vs.\ $1A$), and total effects of Equation~\eqref{{eq:decomp}} at \$500M, with one-sided \citet{{ledoit2008robust}} bootstrap $p$-values for each contrast (the execution tests are computed in the inference supplement of Appendix~\ref{{app:bootstrap}}). The training and execution effects need not sum to the total; the remainder is the interaction of Equation~\eqref{{eq:decomp}}, reported for the three Section-4 specifications in Table~\ref{{tab:capacity_2x2}}, Panel~B ($-0.03$, $-0.18$, and $+0.16$ at \$500M) and equal to $+0.08$ and $+0.06$ for the softmax $\beta{{=}}3$ and dollar-volume rows. The concentration diagnostics are computed over the full 1989--2024 training panel (431 months); the $\Delta R^2_{{\tilde w}}$ and two-by-two columns cover the 299 test months, 2000--2024.}}
\label{{tab:weight_sweep}}
\end{{table}}
"""


def build_linear_benchmark():
    """S6.4: elastic net vs XGBoost at the primary specification."""
    rows = []
    for model, lab in [("xgboost", "XGBoost"), ("elastic_net", "Elastic net")]:
        base = EVAL_ROOT / f"{model}/tc_rank_lam3_500m"
        t = pd.read_csv(base / "two_by_two_500M.csv").set_index("metric")["value"]
        sup = {}
        f = base / "inference_supplement.csv"
        if f.exists():
            d = pd.read_csv(f)
            d = d[(d.part == "A_pairwise_LW") & (d.book == "long_short") & (d.aum == "500M")]
            sup = dict(zip(d["statistic"], d["p_value_one_sided"]))
        adopt = t["SR_net_annualized(2B)"] - t["SR_net_annualized(1B)"]
        exec_cell = f"{num(t['Net portfolio effect annualized'], plus=True)}"
        if "execution_1B_vs_1A" in sup:
            exec_cell += f" ({_pfmt(sup['execution_1B_vs_1A'])})"
        adopt_cell = f"{num(adopt, plus=True)}"
        if "adoption_2B_vs_1B" in sup:
            adopt_cell += f" ({_pfmt(sup['adoption_2B_vs_1B'])})"
        rows.append(
            f"\\quad {lab} & " + " & ".join(num(t[f"SR_net_annualized({c})"]) for c in ["1A", "1B", "2A", "2B"]) +
            f" & {num(t['Net training effect annualized'], plus=True)} ({t['LW p-val (training, net)']:.2f}) & "
            f"{exec_cell} & {adopt_cell} & "
            f"{num(t['Net total effect annualized'], plus=True)} ({t['LW p-val (total, net)']:.4f}) \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t!]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lcccccccc}}
\toprule
 & \multicolumn{{4}}{{c}}{{Net SR by cell}} & & & & \\
\cmidrule(lr){{2-5}}
Model & $1A$ & $1B$ & $2A$ & $2B$ & Training ($p$) & Execution ($p$) & $2B{{-}}1B$ ($p$) & Total ($p$) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Linear versus nonlinear learners.}} The two-by-two design of Section~\ref{{subsec:twobytwo_results}} at the primary specification and \$500M, for the gradient-boosted trees of the main analysis and the regularised linear benchmark (elastic net; Appendix~\ref{{app:models}}). Net annualised Sharpe ratios by cell, with the net training, execution ($1B$ vs.\ $1A$), adoption ($2B$ vs.\ $1B$), and total effects and one-sided \citet{{ledoit2008robust}} bootstrap $p$-values for each contrast (the execution and adoption tests are computed in the inference supplement of Appendix~\ref{{app:bootstrap}}). Effects are computed on unrounded values, so they can differ from the printed cells in the last digit. 299 months, 2000--2024.}}
\label{{tab:linear}}
\end{{table}}
"""


def build_regime_splits():
    """S6.5: 2x2 effects in friction/volatility regimes (descriptive)."""
    import numpy as np

    base = EVAL_ROOT / "xgboost/tc_rank_lam3_500m"
    fullr = pd.read_csv(base / "capacity_portfolio_monthly_500M.csv")
    gated = pd.read_csv(base / "capacity_breakeven_monthly_500M.csv")
    ri = pd.read_csv(ROOT / "data/regime_indicators.csv")

    def cells(df, rt):
        return df[df.row_type == rt].set_index("yyyymm")["ret_net"]

    c = pd.DataFrame({"1A": cells(fullr, "standard"), "2A": cells(fullr, "weighted"),
                      "1B": cells(gated, "standard"), "2B": cells(gated, "weighted")}).dropna()
    c = c.join(ri.set_index("yyyymm")[["vix", "recession"]], how="left")
    med = c["vix"].median()
    n_rec = int((c.recession == 1).sum())

    def sr(x):
        return x.mean() / x.std() * np.sqrt(12)

    rows = []
    for lab, mask in [("High-VIX months", c.vix > med), ("Low-VIX months", c.vix <= med),
                      ("NBER recessions", c.recession == 1), ("Expansions", c.recession == 0)]:
        s = c[mask]
        tr = sr(s["2A"]) - sr(s["1A"])
        ex = sr(s["1B"]) - sr(s["1A"])
        tot = sr(s["2B"]) - sr(s["1A"])
        rows.append(f"\\quad {lab} & {len(s)} & {num(sr(s['1A']))} & {num(tr, plus=True)} & "
                    f"{num(ex, plus=True)} & {num(tot, plus=True)} \\\\")
    rows.insert(2, r"\addlinespace")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lccccc}}
\toprule
Regime & Months & $1A$ net SR & Training & Execution & Total \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{The decomposition across market regimes.}} Net annualised Sharpe ratios and the training, execution, and total effects of Equation~\eqref{{eq:decomp}} for the XGBoost capacity portfolio at the primary specification and \$500M, computed within subsamples of the 299 test months: months with VIX above and below its sample median (${med:.1f}$), and NBER recession versus expansion months. The subsample effects are descriptive point estimates---no subsample bootstrap is run---and the recession sample contains only ${n_rec}$ months. VIX is the CBOE volatility index and recession months are NBER business-cycle dates (both from FRED).}}
\label{{tab:regimes}}
\end{{table}}
"""


def build_training_scale_matrix():
    """Appendix: 2x2 effects as the training-weight scale varies over the fitted grid."""
    SPECS = [("tc_rank_lam3_10m", "\\$10M"),
             ("tc_rank_lam3_100m", "\\$100M"),
             ("tc_rank_lam3_500m", "\\$500M (primary)"),
             ("tc_rank_lam3_1000m", "\\$1B")]
    AUMS = ["PropTC", "100M", "500M", "1B"]

    def t(spec, aum):
        f = EVAL_ROOT / f"xgboost/{spec}/two_by_two_{aum}.csv"
        return pd.read_csv(f).set_index("metric")["value"]

    def pfmt3(p):
        # Table-local: three decimals on [0.01, 0.10) so borderline values
        # (e.g. 0.052) are never printed as the bare 5% threshold.
        if p < 0.01:
            return f"{p:.4f}"
        if p < 0.10:
            return f"{p:.3f}"
        return f"{p:.2f}"

    rows_a, rows_b = [], []
    n_total_sig, n_bonf, min_train_p, min_total = 0, 0, 1.0, 1.0
    for spec, lab in SPECS:
        ca, cb = [], []
        for aum in AUMS:
            v = t(spec, aum)
            tr, p_tr = v["Net training effect annualized"], v["LW p-val (training, net)"]
            to, p_to = v["Net total effect annualized"], v["LW p-val (total, net)"]
            ca.append(f"{num(tr, plus=True)} ({pfmt3(p_tr)})")
            cb.append(f"{num(to, plus=True)} ({pfmt3(p_to)})")
            n_total_sig += int(p_to <= 0.05)
            n_bonf += int(p_to <= 0.05 / 16)
            min_train_p = min(min_train_p, p_tr)
            min_total = min(min_total, to)
        rows_a.append(f"\\quad {lab} & " + " & ".join(ca) + r" \\")
        rows_b.append(f"\\quad {lab} & " + " & ".join(cb) + r" \\")
    assert min_total > 0, "caption claims the total effect is positive in every cell"
    body_a = "\n".join(rows_a)
    body_b = "\n".join(rows_b)
    words = {9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
             14: "fourteen", 15: "fifteen", 16: "all sixteen"}
    n_sig_txt = words.get(n_total_sig, str(n_total_sig))
    n_bonf_txt = words.get(n_bonf, str(n_bonf))
    return rf"""\begin{{table}}[t!]
\centering
\footnotesize
\begin{{tabular}}{{lcccc}}
\toprule
 & \multicolumn{{4}}{{c}}{{Deployed capital (capital grid)}} \\
\cmidrule(lr){{2-5}}
Training scale & Prop.\ TC & \$100M & \$500M & \$1B \\
\midrule
\multicolumn{{5}}{{l}}{{\textit{{Panel A: Net training effect ($2A-1A$, annualised)}}}} \\[2pt]
{body_a}
\midrule
\multicolumn{{5}}{{l}}{{\textit{{Panel B: Net total effect ($2B-1A$, annualised)}}}} \\[2pt]
{body_b}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Training-scale sensitivity of the decomposition.}} Net annualised training effect ($2A-1A$, Panel~A) and total effect ($2B-1A$, Panel~B) of Equation~\eqref{{eq:decomp}} for the capacity portfolio, in annualised net Sharpe-ratio units, as the scale $A$ at which the training weight of Equation~\eqref{{eq:tcr}} is computed (rows) varies over the fitted grid of Section~\ref{{subsec:weighting_schemes}} and the capital at which the portfolio is charged costs (columns) varies over the capital grid of Table~\ref{{tab:capacity_2x2}}; the body of the paper reports the \$500M row. Each row is a self-contained pipeline---the row's weight enters the training loss, the portfolio tilt, and the deployment-weighted evaluation---so effects are comparable across rows, while the underlying cell Sharpe ratios (not shown) are not. Matched training and deployment scales lie on the diagonal, with the Prop.~TC column (half-spread only) approximating the \$10M zero-impact limit. One-sided \citet{{ledoit2008robust}} bootstrap $p$-values in parentheses, from the same seeded machinery as Table~\ref{{tab:capacity_2x2}}, unadjusted across the sixteen cells of each panel. The total effect is positive in every cell and nominally significant at the $5\%$ level in {n_sig_txt} of sixteen ({n_bonf_txt} of sixteen survive a Bonferroni correction at the same level); no training effect is individually significant even before adjustment (the smallest $p$, ${min_train_p:.3f}$, is the unadjusted minimum over the grid), and Panel~A is read as a pattern of point estimates, not as cell-level inference, as in Table~\ref{{tab:dose_response}}. 299 months, 2000--2024.}}
\label{{tab:train_scale}}
\end{{table}}
"""


BUILDERS["ConsistencyDoseResponse.tex"] = build_consistency_dose_response
BUILDERS["WeightFamilySweep.tex"] = build_weight_family_sweep
BUILDERS["LinearBenchmark.tex"] = build_linear_benchmark
BUILDERS["RegimeSplits.tex"] = build_regime_splits

BUILDERS["CapacityCE.tex"] = build_capacity_ce
def build_conventional_r2():
    """IA.6: conventional equal-weighted R2 by quintile, all three S4 specs."""
    LAB = {"1": "Q1 (Illiquid)", "2": "Q2", "3": "Q3", "4": "Q4",
           "5": "Q5 (Liquid)", "Q4-Q5": "Liquid (Q4--Q5)", "Full": "Full cross-section"}
    ORDER = ["1", "2", "3", "4", "5", "Q4-Q5", "Full"]
    blocks = []
    for spec, _ in SEC4_SPECS:
        d = pd.read_csv(ROOT / f"outputs/formalanalysis/analysis/xgboost/{spec}/"
                        "liquidity_breakpoints/nyse/r2_by_quintile.csv")
        d["quintile"] = d["quintile"].astype(str)
        blocks.append(d.set_index("quintile").loc[ORDER])
    # one standard model, one metric: the Std column must be identical across specs
    for b in blocks[1:]:
        assert (b["r2_std_pct"] - blocks[0]["r2_std_pct"]).abs().max() < 1e-9
    full = {spec: b.loc["Full", "delta_pct"] for (spec, _), b in zip(SEC4_SPECS, blocks)}
    assert full["tc_500m"] < full["tc_rank_lam3_500m"] < full["softmax_rank_lam2"] < 0.005
    # Table-local delta formatter: snap deltas that round to zero at two
    # decimals to an unsigned $0.00$ (a signed -0.00 cell would contradict the
    # caption, which quotes the same quantity unsigned via abs()).
    def dnum(v):
        return "$0.00$" if abs(round(v, 2)) == 0 else num(v, plus=True)

    rows = []
    for u in ORDER:
        pre = r"\midrule" + "\n" if u == "Q4-Q5" else ""
        cells = [mnum(blocks[0].loc[u, "r2_std_pct"])]
        for b in blocks:
            cells.append(f"{mnum(b.loc[u, 'r2_wt_pct'])} & {dnum(b.loc[u, 'delta_pct'])}")
        rows.append(pre + f"{LAB[u]} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    heads = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{lab}}}"
        for lab in ["\\shortstack{TC-rank $\\beta{=}3$\\\\(primary)}", "TC level, \\$500M", "Softmax rank $\\beta{=}2$"]
    )
    return rf"""\begin{{table}}[t!]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{l r rr rr rr}}
\toprule
 & & {heads} \\
\cmidrule(lr){{3-4}} \cmidrule(lr){{5-6}} \cmidrule(lr){{7-8}}
Universe & \multicolumn{{1}}{{c}}{{Std}} & \multicolumn{{1}}{{c}}{{Wt}} & \multicolumn{{1}}{{c}}{{$\Delta$}} & \multicolumn{{1}}{{c}}{{Wt}} & \multicolumn{{1}}{{c}}{{$\Delta$}} & \multicolumn{{1}}{{c}}{{Wt}} & \multicolumn{{1}}{{c}}{{$\Delta$}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Conventional equal-weighted out-of-sample $R^2$ by liquidity quintile.}} Companion to Table~\ref{{tab:dw_r2}}: the conventional equal-weighted zero-benchmark $R^2$ of Equation~\eqref{{eq:oos_r2}}, in per cent, within the same NYSE-breakpoint dollar-volume quintiles, for the standard model and for each of the three weighted specifications of Section~\ref{{sec:results}} (both cost-based specifications at \$500M). The metric and the standard model are identical across specifications, so the Std column is common; Wt and $\Delta$ (weighted minus standard, percentage points) vary only through the training weight. The conventional concession scales with the strength of the tilt: the plain transaction-cost tilt concedes ${abs(full['tc_500m']):.2f}$ percentage points on the full cross-section, the primary rank weight ${abs(full['tc_rank_lam3_500m']):.2f}$, and the mild volume-rank tilt essentially nothing (${abs(full['softmax_rank_lam2']):.2f}$). The deltas are read as a pattern of point estimates, under the same discipline as the dose--response grid of Table~\ref{{tab:dose_response}}; no cell-level inference is attached. 299 months, 2000--2024.}}
\label{{tab:conventional_r2}}
\end{{table}}
"""


def build_gate_gross_diagnostics():
    """S4.3 Table tab:gate_gross: forecast-scale/gate diagnostics + gross effects."""
    base = ROOT / "outputs/eval_realignment/analysis/xgboost/tc_rank_lam3_500m"
    g = pd.read_csv(base / "gate_scale_diagnostics.csv")

    def stat(name):
        return g.loc[g.statistic == name, "mean"].iloc[0]

    disp = stat("dispersion_ratio_wt_over_std")
    corr = stat("alpha_rank_correlation")
    pf_s, pf_w = stat("pass_frac_standard"), stat("pass_frac_weighted")
    pf_m = stat("pass_frac_matched_standard")
    v1 = g[g.part == "V1_matched"].dropna(subset=["sr_1B"]).iloc[0]
    v2 = g[g.part == "V2_topk_gate"].dropna(subset=["k"]).set_index("k")
    med = pd.read_csv(base / "capacity_breakeven_gate_diag.csv")
    sig_s = med[med.row_type == "standard"]["median_abs_alpha"].mean() * 1e4
    sig_w = med[med.row_type == "weighted"]["median_abs_alpha"].mean() * 1e4
    hs = med[med.row_type == "standard"]["median_half_spread"].mean() * 1e4
    # guards: the S4.3 prose quotes 49/31/45 bps, 0.70, 0.685, 44.8/28.1/24.4%
    assert abs(sig_s - 49) < 1.5 and abs(sig_w - 31) < 1.5 and abs(hs - 45) < 1.5
    assert abs(disp - 0.685) < 0.002 and abs(corr - 0.701) < 0.002
    assert abs(pf_s - 0.448) < 0.002 and abs(pf_w - 0.281) < 0.002

    def pg(p):
        if p < 0.01:
            return f"{p:.4f}"
        if p > 0.985:
            return f"{p:.3f}"
        return f"{p:.2f}"

    k_lo, k_hi = sorted(v2.index)
    rows_b = []
    for spec, slab in SEC4_SPECS:
        t = _two_by_two("500M", spec)
        rows_b.append(
            f"\\quad {slab} & {num(t['Gross training effect annualized'], plus=True)} ({pg(t['LW p-val (training, gross)'])}) & "
            f"{num(t['Gross portfolio effect annualized'], plus=True)} & "
            f"{num(t['Gross total effect annualized'], plus=True)} ({pg(t['LW p-val (total, gross)'])}) \\\\"
        )
    body_b = "\n".join(rows_b)
    return rf"""\begin{{table}}[t!]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{l rrr}}
\toprule
\multicolumn{{4}}{{l}}{{\textit{{Panel A: Forecast scale and the gate (primary specification, \$500M deployment scale)}}}} \\[2pt]
 & \multicolumn{{1}}{{r}}{{Standard}} & \multicolumn{{1}}{{r}}{{Weighted}} & \\
\midrule
\quad Median centred signal (bps per month) & {sig_s:.0f} & {sig_w:.0f} & \\
\quad Share of names clearing the gate (\%) & {pf_s*100:.1f} & {pf_w*100:.1f} & \\[2pt]
\multicolumn{{4}}{{l}}{{\emph{{Joint statistics of the two forecast panels}}}} \\
\quad Median half-spread (bps per month) & {hs:.0f} & & \\
\quad Rank correlation of centred signals & {corr:.2f} & & \\
\quad Dispersion ratio (weighted/standard) & {disp:.3f} & & \\[2pt]
\multicolumn{{4}}{{l}}{{\emph{{Counterfactual gates (net annualised Sharpe ratios)}}}} \\
 & \multicolumn{{1}}{{r}}{{SR}} & \multicolumn{{2}}{{r}}{{$\Delta$ ($p$)}} \\
\quad Gated standard ($1B$) & {v1['sr_1B']:.3f} & & \\
\quad Dispersion-matched standard, gated (passes {pf_m*100:.1f}\%) & {v1['sr_1B_matched']:.3f} & & \\
\quad Weighted-gated ($2B$); $\Delta=2B$ minus matched & {v1['sr_2B']:.3f} & \multicolumn{{2}}{{r}}{{${v1['d_2B_minus_matched']:+.3f}$ ({pg(v1['p_2B_vs_matched'])})}} \\
\quad Top-$k$ standard at $k={k_lo*100:.1f}\%$; $\Delta=$ weighted minus standard & {v2.loc[k_lo,'sr_1B_k']:.3f} & \multicolumn{{2}}{{r}}{{${v2.loc[k_lo,'d_annualised']:+.3f}$ ({pg(v2.loc[k_lo,'p_one_sided'])})}} \\
\quad Top-$k$ standard at $k={k_hi*100:.1f}\%$ & {v2.loc[k_hi,'sr_1B_k']:.3f} & \multicolumn{{2}}{{r}}{{${v2.loc[k_hi,'d_annualised']:+.3f}$ ({pg(v2.loc[k_hi,'p_one_sided'])})}} \\
\midrule
\multicolumn{{4}}{{l}}{{\textit{{Panel B: Gross-of-costs effects at the \$500M deployment scale}}}} \\[2pt]
 & \multicolumn{{1}}{{c}}{{Training ($p$)}} & \multicolumn{{1}}{{c}}{{Execution}} & \multicolumn{{1}}{{c}}{{Total ($p$)}} \\
\midrule
{body_b}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Diagnostics behind the decomposition.}} Two independent checks on the decomposition of Table~\ref{{tab:capacity_2x2}}: Panel~A asks whether the weighted model's margin at the gate is forecast scale rather than ranking information; Panel~B asks whether any of the net gains is bought with gross alpha. Panel~A tabulates the forecast-scale and gate diagnostics of the primary specification at the \$500M deployment scale: the gate medians and pass rates (time-series means of within-month statistics), the scale statistics of the two forecast panels, and the two counterfactual gates; the joint-statistics rows are single-valued by construction---the half-spread is a property of the shared trading universe, and the rank correlation and dispersion ratio compare the two forecast panels---dispersion-matched and scale-invariant top-$k$---read against the gated cells of Table~\ref{{tab:capacity_2x2}}; the constructions are given in Appendix~\ref{{ia:capacity}}. Panel~B reports the decomposition of Equation~\eqref{{eq:decomp}} computed on gross rather than net Sharpe ratios, one row per Section-4 specification. One-sided $p$-values in parentheses are from the same seeded \citet{{ledoit2008robust}} bootstrap; because it tests the positive tail, the primary gross total's $p=0.997$ is a rejection under the mirrored negative-tail test, and no gross effect is significantly positive anywhere in the design. Gross execution carries no separate pairwise test. 299 months, 2000--2024.}}
\label{{tab:gate_gross}}
\end{{table}}
"""


BUILDERS["DeploymentWeightedR2.tex"] = build_deployment_weighted_r2
BUILDERS["GateGrossDiagnostics.tex"] = build_gate_gross_diagnostics
BUILDERS["ConventionalR2.tex"] = build_conventional_r2
BUILDERS["CapacityPortfolio.tex"] = build_capacity_portfolio
BUILDERS["CapacityTwoByTwo.tex"] = build_capacity_two_by_two
BUILDERS["LongOnlyTwoByTwo.tex"] = build_longonly_two_by_two
BUILDERS["Reallocation.tex"] = build_reallocation
BUILDERS["TrainingScaleMatrix.tex"] = build_training_scale_matrix


def main(out_dir: Path = OUT):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        (out_dir / name).write_text(builder())
        print(f"  wrote {out_dir / name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build curated Section 2 paper tables")
    parser.add_argument(
        "--out-dir", type=Path, default=OUT,
        help="Destination directory (default: paper/Tables)",
    )
    args = parser.parse_args()
    print("Building curated paper tables:")
    main(args.out_dir)
