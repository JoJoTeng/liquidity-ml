"""Build curated paper figures into paper/Figures.

Owns paper figures that are derived transformations of pipeline outputs
(analogous to scripts/build_paper_tables.py for tables). Pipeline PNGs that
are used as-is are copied from outputs/ directly; figures built here exist
because the paper needs a different convention or styling than the pipeline
default.

Figures produced:
    importance_reallocation.png  fig:reallocation (S5.5)
        Per-window importance-share shifts, the SAME convention as
        tab:reallocation (mean over rolling windows of each feature's share
        of total |SHAP| importance, weighted minus standard), replacing the
        pipeline's pooled-share figure which disagrees with the table.

Run:
    python scripts/build_paper_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "Figures"
FORMAL_EXP = ROOT / "outputs/formalanalysis/experiment/xgboost"
FORMAL_AN = ROOT / "outputs/formalanalysis/analysis/xgboost/tc_rank_lam3_500m"


def _paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def build_importance_reallocation(n_each: int = 7):
    """fig:reallocation — per-window share shifts, matching tab:reallocation."""
    _paper_style()
    std = pd.read_csv(FORMAL_EXP / "standard/importance_shap.csv").set_index("yyyymm").sort_index()
    wt = pd.read_csv(FORMAL_EXP / "tc_rank_lam3_500m/importance_shap.csv").set_index("yyyymm").sort_index()
    common = std.index.intersection(wt.index)
    std, wt = std.loc[common], wt.loc[common]
    feats = [c for c in std.columns if c in wt.columns]
    tot_s = std[feats].abs().sum(axis=1)
    tot_w = wt[feats].abs().sum(axis=1)

    deltas = {}
    for f in feats:
        d = (wt[f].abs() / tot_w) - (std[f].abs() / tot_s)
        deltas[f] = d.mean() * 100  # pp
    ser = pd.Series(deltas).sort_values()
    movers = pd.concat([ser.head(n_each), ser.tail(n_each)])

    shift = pd.read_csv(FORMAL_AN / "importance_shift.csv")
    gmap = dict(zip(shift["feature"], shift["group"]))
    GROUP_COLOR = {
        "Q1_only": "#b0533b",   # illiquid-end signal
        "both": "#2c6b9c",      # significant at both ends
        "Q5_only": "#3f7d5a",   # liquid-end signal
        "neither": "0.55",
    }
    GROUP_LABEL = {
        "Q1_only": "significant in Q1 only",
        "both": "significant in both Q1 and Q5",
        "Q5_only": "significant in Q5 only",
        "neither": "significant in neither",
    }

    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = [GROUP_COLOR.get(gmap.get(f, "neither"), "0.55") for f in movers.index]
    ax.barh(range(len(movers)), movers.values, color=colors, alpha=0.85, height=0.65)
    ax.set_yticks(range(len(movers)))
    ax.set_yticklabels(movers.index, fontsize=10)
    ax.axvline(0, color="0.2", lw=0.9)
    ax.set_xlabel("Change in share of total importance, weighted $-$ standard (pp)")

    seen = []
    handles = []
    for f in movers.index:
        g = gmap.get(f, "neither")
        if g not in seen:
            seen.append(g)
            handles.append(plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g], alpha=0.85,
                                         label=GROUP_LABEL[g]))
    ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="lower right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "importance_reallocation.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return movers


def build_importance_vs_liquid_r2():
    """fig:importance_liquid (S2.3) — misallocation scatter, cluster-marked.

    Rebuilds the pipeline scatter to match its paper caption: no in-figure
    title/stats, and the HIGHLIGHTED points are the S2 illiquidity-proxy
    cluster (|avg rank corr with dollar volume| > 0.5), not the focal list.
    """
    _paper_style()
    iv = pd.read_csv(ROOT / "outputs/motivation/step3/xgboost/dvol/importance_vs_liquid_r2.csv")
    rel = pd.read_csv(ROOT / "outputs/motivation/step3/xgboost/dvol/illiquidity_relatedness.csv", index_col=0)
    cluster = set(rel[abs(rel[rel.columns[0]]) > 0.5].index)

    in_c = iv[iv.feature.isin(cluster)]
    out_c = iv[~iv.feature.isin(cluster)]

    # Compact canvas (displayed at 0.82\textwidth): the exhibit is a
    # supporting diagnostic. Cluster labels take per-feature offsets so the
    # Price/Size pair and the zerotrade stack do not collide.
    LABEL_OFFSETS = {
        "Price": (5, 5, "left"),
        "Size": (5, -11, "left"),
        "zerotrade1M": (6, 4, "left"),
        "zerotrade6M": (-7, -4, "right"),
        "zerotrade12M": (6, -11, "left"),
        "VolSD": (7, 0, "left"),
    }
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.scatter(out_c.liquid_r2, out_c.avg_importance, s=26, color="#2c6b9c",
               alpha=0.45, edgecolors="none", label="Other characteristics")
    ax.scatter(in_c.liquid_r2, in_c.avg_importance, s=42, color="#b0533b",
               alpha=0.95, edgecolors="0.2", linewidths=0.4,
               label=r"Illiquidity cluster ($|\bar\rho_j| > 0.5$)")
    for _, r in in_c.iterrows():
        dx, dy, ha = LABEL_OFFSETS.get(r.feature, (4, 3, "left"))
        ax.annotate(r.feature, (r.liquid_r2, r.avg_importance),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, color="#7a3826", ha=ha)
    ax.set_xlabel(r"Univariate predictive $R^2$ among liquid $Q4$--$Q5$ stocks")
    ax.set_ylabel("Average native (gain) importance")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    path = OUT / "importance_vs_liquid_r2.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def build_breakeven_cumret():
    """S5.3 exhibit — cumulative net returns of the four 2x2 cells at $500m."""
    _paper_style()
    base = ROOT / "outputs/eval_realignment/analysis/xgboost/tc_rank_lam3_500m"
    fullr = pd.read_csv(base / "capacity_portfolio_monthly_500M.csv")
    gated = pd.read_csv(base / "capacity_breakeven_monthly_500M.csv")

    def series(df, rt):
        s = df[df.row_type == rt].set_index("yyyymm")["ret_net"].sort_index()
        s.index = pd.to_datetime(s.index.astype(str), format="%Y%m")
        return s.cumsum()

    SPECS = [
        (series(fullr, "standard"), "Standard, full rebalance ($1A$)", "#2c6b9c", (0, (4, 2)), 1.2),
        (series(fullr, "weighted"), "Weighted, full rebalance ($2A$)", "#b0533b", (0, (4, 2)), 1.2),
        (series(gated, "standard"), "Standard, breakeven gate ($1B$)", "#2c6b9c", "solid", 1.6),
        (series(gated, "weighted"), "Weighted, breakeven gate ($2B$)", "#b0533b", "solid", 1.6),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for s, lab, col, ls, lw in SPECS:
        ax.plot(s.index, s.values, label=lab, color=col, linestyle=ls, linewidth=lw)
    ax.axhline(0, color="0.3", lw=0.7, ls=":")
    ax.set_ylabel("Cumulative net return")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncols=2)
    fig.tight_layout()
    path = OUT / "capacity_breakeven_net_cumret_500M.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    print("Building curated paper figures:")
    build_importance_reallocation()
    build_importance_vs_liquid_r2()
    build_breakeven_cumret()
