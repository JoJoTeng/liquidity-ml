"""Liquid-stock squared-error differentials for formal experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis.formal.common import assign_liquidity_quintiles

logger = logging.getLogger(__name__)


def compute_liquid_squared_error_differential(
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Compute monthly Q4-Q5 squared-error difference: SE(std) - SE(weighted)."""
    target = config["data"]["target_col"]
    panel_sub = panel[["permno", "yyyymm", target]].copy()
    panel_sub["liq_quintile"] = assign_liquidity_quintiles(panel, config)

    std_merged = preds_standard.merge(panel_sub, on=["permno", "yyyymm"], how="inner")
    wt_merged = preds_weighted.merge(panel_sub, on=["permno", "yyyymm"], how="inner")

    std_liquid = std_merged[std_merged["liq_quintile"] >= 4]
    wt_liquid = wt_merged[wt_merged["liq_quintile"] >= 4]

    std_se = std_liquid.groupby("yyyymm").apply(
        lambda g: ((g[target] - g["prediction"]) ** 2).mean()
    ).rename("se_std")
    wt_se = wt_liquid.groupby("yyyymm").apply(
        lambda g: ((g[target] - g["prediction"]) ** 2).mean()
    ).rename("se_wt")

    diff = pd.DataFrame({"se_std": std_se, "se_wt": wt_se}).dropna()
    diff["se_diff"] = diff["se_std"] - diff["se_wt"]
    diff["cumulative_diff"] = diff["se_diff"].cumsum()
    return diff.reset_index()


def plot_liquid_squared_error_differential(
    diff_df: pd.DataFrame,
    out_path: Path,
    model: str,
) -> None:
    """Plot the cumulative Q4-Q5 squared-error differential."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        range(len(diff_df)),
        diff_df["cumulative_diff"],
        color="steelblue",
        lw=1.5,
    )
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylabel("Cumulative SE(M_std) - SE(M_w)")
    ax.set_title(f"Liquid-stock squared-error differential ({model}, Q4-Q5)")

    n_obs = len(diff_df)
    tick_every = max(n_obs // 10, 1)
    ax.set_xticks(range(0, n_obs, tick_every))
    ax.set_xticklabels(
        diff_df["yyyymm"].astype(str).iloc[::tick_every],
        rotation=45,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)
