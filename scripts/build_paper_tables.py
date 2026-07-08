"""Build the curated Section 2 paper tables into paper/TablesNew.

This script owns the reference-styled (sdftc-paper) LaTeX tables that the
paper \\input directly: caption below the tabular, \\footnotesize \\textbf
notes, booktabs rules, negatives wrapped in math mode. It reads only verified
motivation/eval outputs, so the numeric rows regenerate if the underlying
analysis changes, while the curated captions (refined during section review)
are held here verbatim.

The raw per-script LaTeX dumps written by scripts 02/03/04 live under each
script's own ``outputs/motivation/.../tables`` directory and are data
artifacts, not paper assets -- they deliberately do not touch paper/TablesNew.

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
OUT = ROOT / "paper" / "TablesNew"
MOT = ROOT / "outputs" / "motivation"


def num(x, dp=2, plus=False):
    """Format a number for LaTeX, math-wrapping negatives (and signed values)."""
    s = f"{x:+.{dp}f}" if plus else f"{x:.{dp}f}"
    if s.startswith(("-", "+")):
        return f"${s}$"
    return s


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
\caption{{\footnotesize \textbf{{Distributional divergence by characteristic category.}} For each characteristic $j$ (rank-transformed to $[0,1]$ within month), $\bar d_j$ is the time-series average difference between the dollar-volume-weighted and equal-weighted cross-sectional means, $\bar d_j = T^{{-1}}\sum_t (\bar x^{{\mathrm{{deploy}}}}_{{j,t}} - \bar x^{{\mathrm{{train}}}}_{{j,t}})$. Characteristics are grouped into the economic categories of \citet{{chen2022open}}; the table reports the category average of $|\bar d_j|$ and the number of characteristics whose divergence is significant at $|t|>2$ with Newey--West standard errors (6 lags). The sample is the full CRSP cross-section, 1989--2024.}}
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
    rows = "\n".join(
        f"{FOCAL_LABELS.get(r['feature'], r['feature'])} & {num(r['beta_bar'], 3)} & "
        f"{tstat(r['beta_t'])} & {num(r['gamma_bar'], 3)} & {tstat(r['gamma_t'])} \\\\"
        for _, r in ir.iterrows()
    )
    n_sig = im["n_sig_gamma_continuous"]
    f_stat = im["f_test_stat_continuous"]
    n_sig_full = im["n_sig_gamma_full"]
    f_full = im["f_test_stat_full"]
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
 & \multicolumn{{2}}{{c}}{{Level}} & \multicolumn{{2}}{{c}}{{Liquidity interaction}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}}
Characteristic & $\bar\beta_j$ & $t$ & $\bar\gamma_j$ & $t$ \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Heterogeneous predictability across the liquidity spectrum.}} Monthly Fama--MacBeth regressions of next-month returns on fifteen focal characteristics and their interactions with the within-month dollar-volume percentile rank $L_{{i,t}}\in[0,1]$: $r_{{i,t+1}} = \alpha_t + x_{{i,t}}'\beta_t + (x_{{i,t}}L_{{i,t}})'\gamma_t + \varepsilon_{{i,t+1}}$. Since $L=0$ ($L=1$) marks the least (most) liquid stock, $\bar\gamma_j$ measures the change in the predictive slope of characteristic $j$ from the illiquid to the liquid end of the spectrum. $t$-statistics use Newey--West standard errors (6 lags); {n_sig} of 15 interactions are significant at $|t|>2$, and the joint time-series $F$-test of $\gamma=0$ is $F={f_stat:.1f}$ ($p<0.001$). Estimating the same regression on all 113 characteristics yields {n_sig_full} significant interactions ($F={f_full:.1f}$, $p<0.001$). The sample is the full CRSP cross-section, 1989--2024.}}
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
        return num(v) if v is not None else num(fallback)  # 'Full sample' -> fallback

    rows = "\n".join(
        f"{r['Quintile']} & {r2zero(r['Quintile'], r['R2_zero (%)'])} & "
        f"{num(r['R2_CS (%)'])} & {num(r['R2_hist (%)'])} & {int(r['Avg N/month']):,} \\\\"
        for _, r in t3.iterrows()
    )
    rows = rows.replace("Full sample &", r"\midrule" + "\nFull sample &")
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
Quintile & $R^2_{{\mathrm{{zero}}}}$ (\%) & $R^2_{{\mathrm{{CS}}}}$ (\%) & $R^2_{{\mathrm{{hist}}}}$ (\%) & Avg.\ $N$/month \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Out-of-sample $R^2$ by liquidity quintile.}} Pooled out-of-sample $R^2$ of the standard-loss model within NYSE-breakpoint dollar-volume quintiles (Q1 = least liquid), over the 2000--2024 test period. The three columns benchmark squared prediction errors against zero \citep{{gu2020empirical}}, the within-month cross-sectional mean return, and the expanding-window historical mean of each stock. Q1 is the only quintile with a positive $R^2_{{\mathrm{{zero}}}}$. The cross-sectional benchmark is harsher in every quintile; the Q1-specific fact is the change of sign from positive to negative, indicating that the positive zero-benchmark figure partly reflects return levels rather than cross-sectional ranking.}}
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
    pooled = u["r2_standard_zero"] * 100
    dvolw = u["r2_weighted_zero"] * 100
    mcapw = u["r2_weighted_mcap_zero"] * 100
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{llc}}
\toprule
Evaluation measure & Universe & $R^2$ (\%) \\
\midrule
Pooled, equal-weighted & Full cross-section & {num(pooled, 2, plus=True)} \\
Implementability-weighted ($\tilde w^{{\mathrm{{tcr}}}}$, \$500M) & Full cross-section & {num(dw_full, 2, plus=True)} \\
Implementability-weighted ($\tilde w^{{\mathrm{{tcr}}}}$, \$500M) & Liquid (Q4--Q5) & {num(dw_q45, 2, plus=True)} \\
Dollar-volume-weighted & Full cross-section & {num(dvolw, 2, plus=True)} \\
Value-weighted (market cap) & Full cross-section & {num(mcapw, 2, plus=True)} \\
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{The same predictions under different measures.}} Out-of-sample $R^2$ (zero benchmark, 2000--2024) of the identical standard-loss prediction panel, evaluated under alternative cross-sectional weightings of squared errors. Row 1 weights every stock-month equally. Rows 2--3 weight errors by the transaction-cost-rank implementability weight of Section~\ref{{sec:framework}} at \$500M; rows 4--5 weight by normalised dollar volume and market capitalisation. All weights are normalised to mean one within each month.}}
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
    rows_a = "\n".join(
        f"\\quad {RESTRICT_UNIV[r_['model']]} & {num(r_['r2_q45_pct'])} & "
        f"{num(r_['r2_full_pct'])} & {int(round(r_['N_train/month'])):,} \\\\"
        for _, r_ in r.iterrows()
    )
    rows_b = "\n".join(
        f"\\quad {q_['quintile']} & {num(q_['r2_pooled_pct'])} & {num(q_['r2_own_pct'])} & "
        f"{num(q_['delta_pp'])} & {int(q_['N_train/month']):,} \\\\"
        for _, q_ in q.iterrows()
    )
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
\multicolumn{{5}}{{l}}{{\textit{{Panel A: Progressive restriction of the training universe}}}} \\[2pt]
Training universe & $R^2$ Q4--Q5 (\%) & $R^2$ full (\%) & $N_{{\mathrm{{train}}}}$/month & \\
\midrule
{rows_a}
\midrule
\multicolumn{{5}}{{l}}{{\textit{{Panel B: Quintile-specific models}}}} \\[2pt]
Quintile & $R^2$ pooled (\%) & $R^2$ own (\%) & $\Delta$ (pp) & $N_{{\mathrm{{train}}}}$/month \\
\midrule
{rows_b}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Neither screening nor splitting repairs the misalignment.}} Panel~A retrains the model on progressively more liquid training universes (holding the rolling protocol, characteristics, and tuned hyperparameters fixed) and evaluates all variants on the same full test cross-section; $R^2$ is the pooled out-of-sample $R^2$ (zero benchmark) on liquid Q4--Q5 stocks and on the full sample. Liquid-stock accuracy deteriorates monotonically as illiquid observations are discarded. Panel~B trains a separate model within each liquidity quintile and compares its within-quintile $R^2$ with the pooled model's; $\Delta$ is own minus pooled. Every $\Delta$ is negative: segmentation destroys signal that transfers across the liquidity spectrum. In both panels $N_{{\mathrm{{train}}}}$ averages over the rolling training windows, which extend back into the larger cross-sections of the 1990s and therefore exceed the corresponding 2000--2024 evaluation-panel counts.}}
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
    and within-month median imputation of missing spread / sigma / ADV.
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
 & p1 & p25 & p50 & p75 & p95 & p99 & p99/p50 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Liquidity and cost inputs: cross-sectional descriptives.}} Pooled percentiles across all stock-months, 1989--2024. Daily dollar volume is the 21-day trailing average of daily price times share volume from the CRSP daily file; the half-spread is one half of the \citet{{chen2022open}} \texttt{{BidAskSpread}} series (a \citet{{corwin2012simple}} high--low effective-spread estimate scaled by price); $\sigma$ is the 12-month rolling standard deviation of monthly excess returns rescaled to a daily horizon by $1/\sqrt{{21}}$. The one-way cost $\tau$ applies Equation~\eqref{{eq:tc_primitive}} with $\lambda=0.1$ and equal-breadth participation $Q_t = A/N_t$, where $N_t$ is the full cross-section in month $t$; missing spread, volatility, or volume inputs are imputed at within-month medians, as in the weight construction. The last column reports the ratio of the 99th percentile to the median.}}
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
\caption{{\footnotesize \textbf{{Stock-level predictors.}} The $113$ characteristics used throughout the paper, drawn from the open-source library of \citet{{chen2022open}} (see Section~\ref{{subsec:returns_chars}} for the selection rule). Descriptions are the library's short names; full construction details are documented in the library. The category column reports the broad economic grouping used in Sections~\ref{{sec:imbalance}} and~\ref{{sec:data}}.}}
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


def _two_by_two(aum):
    t = pd.read_csv(EVAL / f"two_by_two_{aum}.csv").set_index("metric")["value"]
    return t.to_dict()


def build_deployment_weighted_r2():
    r2 = pd.read_csv(EVAL / "liquidity_breakpoints/nyse/deployment_weighted_r2.csv")
    st = pd.read_csv(EVAL / "liquidity_breakpoints/nyse/deployment_weighted_error_diff_stats.csv")
    m = r2.merge(st[["universe", "t_stat"]], on="universe")
    LAB = {"Q1": "Q1 (Illiquid)", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4",
           "Q5": "Q5 (Liquid)", "liquid_q4q5": "Liquid (Q4--Q5)", "full": "Full cross-section"}
    ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5", "liquid_q4q5", "full"]
    m = m.set_index("universe").loc[ORDER].reset_index()
    rows = []
    for _, r in m.iterrows():
        pre = r"\midrule" + "\n" if r["universe"] == "liquid_q4q5" else ""
        rows.append(
            pre + f"{LAB[r['universe']]} & {num(r['r2_weighted_std_pct'])} & "
            f"{num(r['r2_weighted_wt_pct'])} & {num(r['delta_pct'], plus=True)} & "
            f"{tstat(r['t_stat'])} \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
 & \multicolumn{{2}}{{c}}{{$R^2_{{\tilde w}}$ (\%)}} & & \\
\cmidrule(lr){{2-3}}
Universe & Standard & Weighted & $\Delta R^2_{{\tilde w}}$ (pp) & $t(D_t)$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Deployment-weighted out-of-sample $R^2$.}} The deployment-weighted $R^2$ of Equation~\eqref{{eq:dw_r2}} for the standard and implementability-weighted models, within NYSE-breakpoint dollar-volume quintiles of the prediction universe (Q1 = least liquid), on the pooled liquid set, and on the full cross-section; the primary specification's weight ($\tilde w^{{\mathrm{{tcr}}}}$ at \$500M) prices the errors throughout, with the global mean-one normalisation held fixed across subsets. $\Delta R^2_{{\tilde w}}$ is weighted minus standard in percentage points. The last column reports the Newey--West $t$-statistic (6 lags) on the monthly weighted mean-squared-error differential $D_t$ (Appendix~\ref{{app:capacity}}); no individual differential is statistically significant. Out-of-sample period 2000--2024, 299 months.}}
\label{{tab:dw_r2}}
\end{{table}}
"""


def build_capacity_portfolio():
    def metrics(tag, aum):
        f = EVAL / f"capacity_portfolio{tag}_metrics_{aum}.csv"
        df = pd.read_csv(f)
        s = df[df.row_type == "standard"].iloc[0]
        w = df[df.row_type == "weighted"].iloc[0]
        d = df[df.row_type == "difference"].iloc[0]
        return s, w, d

    rows_a = []
    for aum, lab in AUM_GRID:
        s, w, d = metrics("", aum)
        delta_ann = w["net_sr_annual"] - s["net_sr_annual"]
        rows_a.append(
            f"\\quad {lab} & {num(s['net_sr_annual'])} & {num(w['net_sr_annual'])} & "
            f"{num(delta_ann, plus=True)} & {d['net_sr_diff_pval']:.2f} \\\\"
        )
    s0, w0, _ = metrics("", "500M")
    rows_b = []
    for tag, lab in [("", "Signal ($\\tilde w$)"), ("_equal", "Equal"), ("_value", "Value")]:
        s, w, d = metrics(tag, "500M")
        delta_ann = w["net_sr_annual"] - s["net_sr_annual"]
        rows_b.append(
            f"\\quad {lab} & {num(s['gross_sr_annual'])} & {num(w['gross_sr_annual'])} & "
            f"{num(s['net_sr_annual'])} & {num(w['net_sr_annual'])} & "
            f"{num(delta_ann, plus=True)} & {d['net_sr_diff_pval']:.2f} \\\\"
        )
    body_a = "\n".join(rows_a)
    body_b = "\n".join(rows_b)
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lcccccc}}
\toprule
\multicolumn{{7}}{{l}}{{\textit{{Panel A: Signal-weighted book across deployed capital (net Sharpe ratios)}}}} \\[2pt]
 & Standard & Weighted & $\Delta$SR & $p$ & & \\
\midrule
{body_a}
\midrule
\multicolumn{{7}}{{l}}{{\textit{{Panel B: Capacity-weight benchmarks at \$500M}}}} \\[2pt]
 & \multicolumn{{2}}{{c}}{{Gross SR}} & \multicolumn{{2}}{{c}}{{Net SR}} & & \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}}
Capacity weight & Std & Wt & Std & Wt & $\Delta$SR & $p$ \\
\midrule
{body_b}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{The capacity portfolio.}} Annualised Sharpe ratios of the centred, dollar-neutral, unit-gross book of Equations~\eqref{{eq:book_center}}--\eqref{{eq:book_norm}} under standard and implementability-weighted training. Panel~A holds the signal capacity weight $\tilde w$ fixed and sweeps deployed capital $A$ over the book grid; the gross Sharpe ratio does not depend on $A$ ($1.48$ standard, $1.32$ weighted). Panel~B holds $A=\$500$M fixed and replaces the capacity weight in the book construction with equal and value weights, isolating the deployment stage. $\Delta$SR is weighted minus standard on the net series; one-sided $p$-values are from the \citet{{ledoit2008robust}} studentised circular-block bootstrap on the monthly Sharpe difference. $\Delta$SR is computed on unrounded values, so it can differ from the printed cells in the last digit. No training-stage difference is statistically significant. 299 months, 2000--2024.}}
\label{{tab:capacity}}
\end{{table}}
"""


def _supplement_pvals(book):
    """Pairwise p-values from the inference supplement (script 46), by AUM."""
    f = EVAL / "inference_supplement.csv"
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
        return (f"{num(t5[f'SR_net_annualized({c})'])} & ({num(t5[f'SR_gross_annualized({c})'])}) & "
                f"{t5[f'TC mean monthly ({c})'] * 1e4:.0f} & {t5[f'Turnover ({c})']:.2f}")

    sup = _supplement_pvals("long_short")
    rows_b = []
    for aum, lab in AUM_GRID:
        t = _two_by_two(aum)
        p_exec = sup.get((aum, "execution_1B_vs_1A"))
        adopt = t["SR_net_annualized(2B)"] - t["SR_net_annualized(1B)"]
        p_adopt = sup.get((aum, "adoption_2B_vs_1B"))
        exec_cell = f"{num(t['Net portfolio effect annualized'], plus=True)}"
        if p_exec is not None:
            exec_cell += f" ({_pfmt(p_exec)})"
        adopt_cell = f"{num(adopt, plus=True)}"
        if p_adopt is not None:
            adopt_cell += f" ({_pfmt(p_adopt)})"
        rows_b.append(
            f"\\quad {lab} & "
            f"{num(t['Net training effect annualized'], plus=True)} ({_pfmt(t['LW p-val (training, net)'])}) & "
            f"{exec_cell} & {adopt_cell} & "
            f"{num(t['Net total effect annualized'], plus=True)} ({_pfmt(t['LW p-val (total, net)'])}) & "
            f"{num(t['Net interaction annualized'], plus=True)} \\\\"
        )
    body_b = "\n".join(rows_b)

    rows_c = []
    for key, lab in [("capm", "CAPM"), ("ff3", "FF3"), ("ff5", "FF5"), ("ff5_mom", "FF5+Mom")]:
        cells = []
        for c in ["1A", "1B", "2A", "2B"]:
            a = t5[f"alpha_{key}({c})_annual"] * 100
            tt = t5[f"alpha_{key}({c})_tstat"]
            cells.append(f"{num(a)} {tstat(tt)}")
        rows_c.append(f"\\quad {lab} & " + " & ".join(cells) + r" \\")
    body_c = "\n".join(rows_c)

    return rf"""\begin{{table}}[t!]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lccccccc}}
\toprule
\multicolumn{{8}}{{l}}{{\textit{{Panel A: The four cells at \$500M}}}} \\[2pt]
 & Net SR & (Gross SR) & Cost (bps/mo) & Turnover & & & \\
\midrule
\multicolumn{{8}}{{l}}{{\emph{{Full rebalance ($A$)}}}} \\
\quad Standard training ($1A$) & {cell('1A')} & & & \\
\quad Weighted training ($2A$) & {cell('2A')} & & & \\[2pt]
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
\caption{{\footnotesize \textbf{{Cost-aware execution and the training $\times$ execution decomposition.}} Panel~A reports the four cells of the two-by-two design at the primary \$500M scale: net and gross annualised Sharpe ratios, the mean monthly transaction-cost drag in basis points, and monthly one-sided turnover. Rows vary the training loss (standard vs.\ implementability-weighted); blocks vary execution (full monthly rebalancing vs.\ the breakeven gate of Equation~\eqref{{eq:gate}}). Panel~B decomposes the net gain of Equation~\eqref{{eq:decomp}} at each level of deployed capital; one-sided $p$-values in parentheses are from the \citet{{ledoit2008robust}} studentised circular-block bootstrap: the training, total, execution ($1B$ vs.\ $1A$), and adoption ($2B$ vs.\ $1B$, what weighted training adds on top of the gate) contrasts are all tested with the same seeded machinery (the pairwise execution and adoption tests are computed in the inference supplement of Appendix~\ref{{app:bootstrap}}). Effects are computed on unrounded values, so the printed decomposition identities can differ in the last digit. Turnover and gate composition do not vary with $A$ because the gate triggers on the half-spread alone, so the execution effect grows with capital purely through the price of the avoided trades. Panel~C reports annualised factor alphas of the four net return series at \$500M with Newey--West $t$-statistics (6 lags). 299 months, 2000--2024.}}
\label{{tab:capacity_2x2}}
\end{{table}}
"""


def build_longonly_two_by_two():
    lt5 = pd.read_csv(EVAL / "longonly_two_by_two_500M.csv").set_index("metric")["value"].to_dict()

    def cell(c):
        return (f"{num(lt5[f'SR_net_annualized({c})'])} & "
                f"{lt5[f'TC mean monthly ({c})'] * 1e4:.0f} & {lt5[f'Turnover ({c})']:.2f}")

    sup = _supplement_pvals("long_only")
    rows_b = []
    for aum, lab in AUM_GRID:
        t = pd.read_csv(EVAL / f"longonly_two_by_two_{aum}.csv").set_index("metric")["value"].to_dict()
        inter = t.get("Net interaction annualized",
                      t["Net total effect annualized"] - t["Net training effect annualized"]
                      - t["Net portfolio effect annualized"])
        p_exec = sup.get((aum, "execution_1B_vs_1A"))
        adopt = t["SR_net_annualized(2B)"] - t["SR_net_annualized(1B)"]
        p_adopt = sup.get((aum, "adoption_2B_vs_1B"))
        exec_cell = f"{num(t['Net portfolio effect annualized'], plus=True)}"
        if p_exec is not None:
            exec_cell += f" ({_pfmt(p_exec)})"
        adopt_cell = f"{num(adopt, plus=True)}"
        if p_adopt is not None:
            adopt_cell += f" ({_pfmt(p_adopt)})"
        rows_b.append(
            f"\\quad {lab} & "
            f"{num(t['Net training effect annualized'], plus=True)} ({_pfmt(t['LW p-val (training, net)'])}) & "
            f"{exec_cell} & {adopt_cell} & "
            f"{num(t['Net total effect annualized'], plus=True)} ({_pfmt(t['LW p-val (total, net)'])}) & "
            f"{num(inter, plus=True)} \\\\"
        )
    body_b = "\n".join(rows_b)
    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lccccccc}}
\toprule
\multicolumn{{8}}{{l}}{{\textit{{Panel A: The four cells at \$500M---net SR, mean monthly cost (bps), turnover}}}} \\[2pt]
 & \multicolumn{{3}}{{c}}{{Plain membership ($A$)}} & \multicolumn{{3}}{{c}}{{Hysteresis band ($B$)}} & \\
\cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}}
\quad Standard training & {cell('1A')} & {cell('1B')} & \\
\quad Weighted training & {cell('2A')} & {cell('2B')} & \\
\midrule
\multicolumn{{8}}{{l}}{{\textit{{Panel B: Decomposition across deployed capital (net, annualised)}}}} \\[2pt]
 & Training ($p$) & Execution ($p$) & $2B{{-}}1B$ ($p$) & Total ($p$) & Interaction & & \\
\midrule
{body_b}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{The long-only capacity book.}} The two-by-two design applied to the long-only book of Equation~\eqref{{eq:longonly}}: rows vary the training loss, columns vary execution between plain monthly membership refresh and the cost-scaled membership-hysteresis band. Panel~A reports net annualised Sharpe ratios, mean monthly cost drag (bps), and monthly one-sided turnover at \$500M; Panel~B decomposes the net gain at each level of deployed capital; one-sided \citet{{ledoit2008robust}} bootstrap $p$-values in parentheses cover the training, execution ($1B$ vs.\ $1A$), adoption ($2B$ vs.\ $1B$), and total contrasts (the execution and adoption tests are computed in the inference supplement described in Appendix~\ref{{app:bootstrap}}). Effects are computed on unrounded values, so the printed decomposition identities can differ in the last digit. 299 months, 2000--2024.}}
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
            f"\\quad {lab} & {ms*100:.1f} & {mw*100:.1f} & "
            f"{num(r['mean']*100, plus=True)} & {tstat(r['t_stat'])} \\\\"
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
        rows_b.append(f"\\quad {_latex_escape(f)} & {ms:.2f} & {mw:.2f} & {num(dd, plus=True)} & {tstat(tt)} \\\\")
    rows_b.insert(5, r"\addlinespace")
    body_b = "\n".join(rows_b)

    return rf"""\begin{{table}}[t!]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
 & \multicolumn{{2}}{{c}}{{Share of importance (\%)}} & & \\
\cmidrule(lr){{2-3}}
 & Standard & Weighted & $\Delta$ (pp) & $t$ \\
\midrule
\multicolumn{{5}}{{l}}{{\textit{{Panel A: Group shares}}}} \\[2pt]
{body_a}
\midrule
\multicolumn{{5}}{{l}}{{\textit{{Panel B: Largest individual share shifts}}}} \\[2pt]
{body_b}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Feature-importance reallocation.}} Shares of total SHAP importance under standard and implementability-weighted training (primary specification), averaged over the 299 rolling windows; $\Delta$ is the mean of the per-window paired share differences and $t$ its Newey--West statistic (6 lags), the same convention as the per-feature tests. In Panel~A, the illiquidity cluster is the rule-based group of Section~\ref{{sec:imbalance}} (characteristics whose average rank correlation with dollar volume exceeds $0.5$ in absolute value); the illiquidity/microstructure group is the broader economically defined list fixed ex ante; the liquid-signal group collects characteristics whose predictive slopes are significant in the most liquid quintile of the all-113-characteristic quintile-specific Fama--MacBeth regressions of Section~\ref{{subsec:heterogeneity}} (significant in Q5, whether or not also in Q1). Panel~B reports the five largest decreases and increases in per-window importance share.}}
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
                f"\\quad {lab}, {rlab} & {num(r['net_mean_annual']*100)} & "
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
\begin{{tabular}}{{lccccc}}
\toprule
 & Net mean (\%) & Turnover & $\mathrm{{CE}}(1)$ & $\mathrm{{CE}}(5)$ & $\mathrm{{CE}}(10)$ \\
\midrule
\multicolumn{{6}}{{l}}{{\textit{{Panel A: Signal-weighted book across deployed capital}}}} \\[2pt]
{body_a}
\midrule
\multicolumn{{6}}{{l}}{{\textit{{Panel B: Capacity-weight benchmarks at \$500M}}}} \\[2pt]
{body_b}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Mean net returns, turnover, and certainty equivalents of the capacity books.}} Companion to Table~\ref{{tab:capacity}}: annualised mean net returns (\%), mean monthly one-sided turnover, and the annualised certainty-equivalent return $\mathrm{{CE}}(\gamma)$ of Equation~\eqref{{eq:ce}} (\%) evaluated on the monthly net series, for the full-rebalance capacity book of Equations~\eqref{{eq:book_center}}--\eqref{{eq:book_norm}} under standard and implementability-weighted training. Panel~A sweeps deployed capital under the signal capacity weight $\tilde w$; turnover does not vary with $A$. Panel~B replaces the capacity weight with equal and value weights at \$500M. Within Panel~A, the certainty-equivalent comparison of the two training losses reproduces the net-Sharpe comparison of Table~\ref{{tab:capacity}}: the standard book dominates under proportional costs, the two books are within a few basis points of one another at \$100M, and the weighted book dominates at \$500M and \$1B at every $\gamma$. In Panel~B the two criteria diverge for the equal book: the weighted equal book carries the higher mean net return and the higher certainty equivalent at every $\gamma$ although its net Sharpe ratio is lower---both means are negative, so the Sharpe ordering there is the familiar negative-mean pathology. 299 months, 2000--2024.}}
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
\caption{{\footnotesize \textbf{{The dose--response of consistency on the sorted book.}} Net annualised training effect ($2A-1A$, weighted minus standard training on the plain sorted long--short book) for the prediction-quantile long--short portfolio of Section~\ref{{subsec:formal_2x2}}, across leg-weighting schemes (rows) and stock universes (columns), at \$500M; one-sided \citet{{ledoit2008robust}} bootstrap $p$-values in parentheses. The equal-legs/full-universe cell is the conventional scoreboard of the machine-learning asset-pricing literature; the NYSE universe and the deployment-weighted legs (within-leg positions proportional to $\tilde w$; the forecast determines membership only) bring the object into alignment with the trained objective, while the top-$60\%$ universe pre-screens away the illiquid margin the weighting corrects, so the substitutes logic of Section~\ref{{subsec:cost_sensitive}} predicts the attenuation toward zero observed in that column. The execution (hysteresis) effect is large in every cell (between ${ex_lo:+.2f}$ and ${ex_hi:+.2f}$), and the total effect is statistically significant in every cell ($p\le {p_bound}$); no individual training effect is significant, and the table is read as a pattern of point estimates, not as cell-level inference. The capacity-book counterpart of the training effect at the same specification is $+0.09$ (Section~\ref{{sec:results}}).}}
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
        ("tc_rank_lam3_500m", "TC-rank $\\beta{=}3$ (\\$500M)", "tc_rank", cfg, 500e6),
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
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lcccccc}}
\toprule
 & \multicolumn{{2}}{{c}}{{Weight concentration}} & & \multicolumn{{3}}{{c}}{{Two-by-two at \$500M (net)}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{5-7}}
Weight family & ESS (\%) & Top-10 (\%) & $\Delta R^2_{{\tilde w}}$ (pp) & Training ($p$) & Execution ($p$) & Total ($p$) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{Across weighting families.}} Each row re-estimates the weighted model and the full evaluation pipeline under a different implementability-weight family (Section~\ref{{subsec:weighting_schemes}}). ESS is the mean monthly Kish effective sample size $(\sum_i w_i)^2/\sum_i w_i^2$ as a percentage of the cross-section; Top-10 is the mean share of total weight carried by the ten largest names. $\Delta R^2_{{\tilde w}}$ is the deployment-weighted $R^2$ gain of Equation~\eqref{{eq:dw_r2}} on the liquid $Q4$--$Q5$ set, each family evaluated under its own weight. The two-by-two columns report the net annualised training, execution ($1B$ vs.\ $1A$), and total effects of Equation~\eqref{{eq:decomp}} at \$500M, with one-sided \citet{{ledoit2008robust}} bootstrap $p$-values for each contrast (the execution tests are computed in the inference supplement of Appendix~\ref{{app:bootstrap}}). 299 months, 2000--2024.}}
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
\caption{{\footnotesize \textbf{{The decomposition across market regimes.}} Net annualised Sharpe ratios and the training, execution, and total effects of Equation~\eqref{{eq:decomp}} for the \$500M capacity book, computed within subsamples of the 299 test months: months with VIX above and below its sample median ($17.9$), and NBER recession versus expansion months. The subsample effects are descriptive point estimates---no subsample bootstrap is run---and the recession sample contains only $28$ months. VIX is the CBOE volatility index and recession months are NBER business-cycle dates (both from FRED).}}
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
 & \multicolumn{{4}}{{c}}{{Deployed capital (book grid)}} \\
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
\caption{{\footnotesize \textbf{{Training-scale sensitivity of the decomposition.}} Net annualised training effect ($2A-1A$, Panel~A) and total effect ($2B-1A$, Panel~B) of Equation~\eqref{{eq:decomp}} for the capacity book, in annualised net Sharpe-ratio units, as the scale $A$ at which the training weight of Equation~\eqref{{eq:tcr}} is computed (rows) varies over the fitted grid of Section~\ref{{subsec:weighting_schemes}} and the capital at which the book is charged costs (columns) varies over the book grid of Table~\ref{{tab:capacity_2x2}}; the main text reports the \$500M row. Each row is a self-contained pipeline---the row's weight enters the training loss, the book tilt, and the deployment-weighted evaluation---so effects are comparable across rows, while the underlying cell Sharpe ratios (not shown) are not. Matched training and deployment scales lie on the diagonal, with the Prop.~TC column (half-spread only) approximating the \$10M zero-impact limit. One-sided \citet{{ledoit2008robust}} bootstrap $p$-values in parentheses, from the same seeded machinery as Table~\ref{{tab:capacity_2x2}}, unadjusted across the sixteen cells of each panel. The total effect is positive in every cell and nominally significant at the $5\%$ level in {n_sig_txt} of sixteen ({n_bonf_txt} of sixteen survive a Bonferroni correction at the same level); no training effect is individually significant even before adjustment (the smallest $p$, ${min_train_p:.3f}$, is the unadjusted minimum over the grid), and Panel~A is read as a pattern of point estimates, not as cell-level inference, as in Table~\ref{{tab:dose_response}}. 299 months, 2000--2024.}}
\label{{tab:train_scale}}
\end{{table}}
"""


BUILDERS["ConsistencyDoseResponse.tex"] = build_consistency_dose_response
BUILDERS["WeightFamilySweep.tex"] = build_weight_family_sweep
BUILDERS["LinearBenchmark.tex"] = build_linear_benchmark
BUILDERS["RegimeSplits.tex"] = build_regime_splits

BUILDERS["CapacityCE.tex"] = build_capacity_ce
BUILDERS["DeploymentWeightedR2.tex"] = build_deployment_weighted_r2
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
        help="Destination directory (default: paper/TablesNew)",
    )
    args = parser.parse_args()
    print("Building curated paper tables:")
    main(args.out_dir)
