"""Build curated paper figures into paper/FiguresNew.

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
OUT = ROOT / "paper" / "FiguresNew"
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


if __name__ == "__main__":
    print("Building curated paper figures:")
    build_importance_reallocation()
