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
\caption{{\footnotesize \textbf{{Heterogeneous predictability across the liquidity spectrum.}} Monthly Fama--MacBeth regressions of next-month returns on fifteen focal characteristics and their interactions with the within-month dollar-volume percentile rank $L_{{it}}\in[0,1]$: $r_{{i,t+1}} = \alpha_t + x_{{it}}'\beta_t + (x_{{it}}L_{{it}})'\gamma_t + \varepsilon_{{i,t+1}}$. Since $L=0$ ($L=1$) marks the least (most) liquid stock, $\bar\gamma_j$ measures the change in the predictive slope of characteristic $j$ from the illiquid to the liquid end of the spectrum. $t$-statistics use Newey--West standard errors (6 lags); {n_sig} of 15 interactions are significant at $|t|>2$, and the joint time-series $F$-test of $\gamma=0$ is $F={f_stat:.1f}$ ($p<0.001$). Estimating the same regression on all 113 characteristics yields {n_sig_full} significant interactions ($F={f_full:.1f}$, $p<0.001$). The sample is the full CRSP cross-section, 1989--2024.}}
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
Implementability-weighted ($\tilde w^{{\mathrm{{tcr}}}}$, \$500m) & Full cross-section & {num(dw_full, 2, plus=True)} \\
Implementability-weighted ($\tilde w^{{\mathrm{{tcr}}}}$, \$500m) & Liquid (Q4--Q5) & {num(dw_q45, 2, plus=True)} \\
Dollar-volume-weighted & Full cross-section & {num(dvolw, 2, plus=True)} \\
Value-weighted (market cap) & Full cross-section & {num(mcapw, 2, plus=True)} \\
\bottomrule
\end{{tabular}}
\caption{{\footnotesize \textbf{{The same predictions under different measures.}} Out-of-sample $R^2$ (zero benchmark, 2000--2024) of the identical standard-loss prediction panel, evaluated under alternative cross-sectional weightings of squared errors. Row 1 weights every stock-month equally. Rows 2--3 weight errors by the transaction-cost-rank implementability weight of Section~\ref{{sec:framework}} at \$500m; rows 4--5 weight by normalised dollar volume and market capitalisation. All weights are normalised to mean one within each month.}}
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
        prow(r"Half bid--ask spread (bps)", sp.values, dp=1),
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
\caption{{\footnotesize \textbf{{Liquidity and cost inputs: cross-sectional descriptives.}} Pooled percentiles across all stock-months, 1989--2024. Daily dollar volume is the 21-day trailing average of daily price times share volume from the CRSP daily file; the half spread is one half of the \citet{{chen2022open}} \texttt{{BidAskSpread}} series (a \citet{{corwin2012simple}} high--low effective-spread estimate scaled by price); $\sigma$ is the 12-month rolling standard deviation of monthly excess returns rescaled to a daily horizon by $1/\sqrt{{21}}$. The one-way cost $\tau$ applies Equation~\eqref{{eq:tc_primitive}} with $\lambda=0.1$ and equal-breadth participation $Q_t = A/N_t$, where $N_t$ is the full cross-section in month $t$; missing spread, volatility, or volume inputs are imputed at within-month medians, as in the weight construction. The last column reports the ratio of the 99th percentile to the median.}}
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
    return rf"""{{\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{longtable}}{{p{{3.4cm}}p{{5.2cm}}p{{4.4cm}}p{{1.9cm}}}}
\caption{{\footnotesize \textbf{{Stock-level predictors.}} The $113$ characteristics used throughout the paper, drawn from the open-source library of \citet{{chen2022open}} (see Section~\ref{{subsec:returns_chars}} for the selection rule). Descriptions are the library's short names; full construction details are documented in the library. The category column reports the broad economic grouping used in Sections~\ref{{sec:imbalance}} and~\ref{{sec:data}}.}}
\label{{tab:characteristics}} \\
\toprule
Acronym & Description & Original reference & Category \\
\midrule
\endfirsthead
\multicolumn{{4}}{{l}}{{\footnotesize\emph{{Table~\ref{{tab:characteristics}}, continued}}}} \\
\toprule
Acronym & Description & Original reference & Category \\
\midrule
\endhead
\bottomrule
\endfoot
{body}
\end{{longtable}}
}}
"""


BUILDERS["CharacteristicsTable.tex"] = build_characteristics_table


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
