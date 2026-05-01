"""Restriction-curve comparison for formal weighted models."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.formal.common import assign_liquidity_quintiles

logger = logging.getLogger(__name__)


def compare_weighted_model_to_restriction_curve(
    preds_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    restriction_csv_path: Path,
) -> dict:
    """Compare weighted-model Q4-Q5 R2 with the hard-restriction curve."""
    target = config["data"]["target_col"]
    panel_sub = panel[["permno", "yyyymm", target]].copy()
    panel_sub["liq_quintile"] = assign_liquidity_quintiles(panel, config)

    merged = preds_weighted.merge(panel_sub, on=["permno", "yyyymm"], how="inner")
    q45 = merged[merged["liq_quintile"] >= 4]

    if len(q45) == 0:
        return {
            "restriction_df": None,
            "weighted_r2_q45": np.nan,
            "dominates": False,
        }

    ss_pred = ((q45[target] - q45["prediction"]) ** 2).sum()
    ss_total = (q45[target] ** 2).sum()
    r2_q45 = (1.0 - ss_pred / ss_total) * 100 if ss_total > 0 else np.nan

    if not restriction_csv_path.exists():
        logger.warning(
            "Restriction curve CSV not found at %s; skipping overlay",
            restriction_csv_path,
        )
        return {
            "restriction_df": None,
            "weighted_r2_q45": r2_q45,
            "dominates": False,
        }

    curve = pd.read_csv(restriction_csv_path)
    best_on_curve = curve["r2_q45_pct"].max()
    dominates = r2_q45 > best_on_curve

    out_df = curve.copy()
    new_row = {col: np.nan for col in out_df.columns}
    new_row["model"] = "M_w (weighted)"
    new_row["r2_q45_pct"] = r2_q45
    out_df = pd.concat([out_df, pd.DataFrame([new_row])], ignore_index=True)

    return {
        "restriction_df": out_df,
        "weighted_r2_q45": r2_q45,
        "best_on_curve": best_on_curve,
        "dominates": dominates,
    }


def plot_restriction_curve_comparison(
    result: dict,
    out_path: Path,
    model: str,
    weight_family: str,
) -> None:
    """Plot the restriction curve with the weighted model highlighted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = result["restriction_df"]
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue"] * (len(df) - 1) + ["coral"]
    ax.bar(df["model"], df["r2_q45_pct"], color=colors)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylabel("OOS R2 on Q4-Q5 (%)")
    ax.set_title(f"Weighted model vs restriction curve ({model} / {weight_family})")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)

