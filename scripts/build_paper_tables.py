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
