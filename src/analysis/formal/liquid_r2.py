"""Liquid-stock R2 diagnostics for formal experiments."""

from __future__ import annotations

import pandas as pd

from src.analysis.formal.common import (
    assign_liquidity_quintiles,
    compute_formal_utility_weights,
    formal_weight_label,
    predictions_for_utility_r2,
)
from src.analysis.motivation import compute_quintile_oos_r2, compute_utility_weighted_r2


def evaluate_liquid_stock_r2(
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
    include_extra_benchmarks: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compare standard and weighted OOS R2 across liquidity quintiles.

    The saved quintile table includes Q1-Q5 separately, a pooled Q4-Q5
    liquid-stock row, and the full sample.
    """
    target = config["data"]["target_col"]
    liq_col = f"liq_{config['liquidity']['primary']}"

    panel_sub = panel[["permno", "yyyymm", target, liq_col]].copy()
    panel_sub["liq_quintile"] = assign_liquidity_quintiles(panel, config)

    hist_window = None
    if include_extra_benchmarks:
        hist_window = (
            config["training"]["train_window"]
            + config["training"]["validation_window"]
        )

    r2_standard = compute_quintile_oos_r2(
        preds_standard,
        panel_sub,
        "liq_quintile",
        return_col=target,
        hist_window=hist_window,
        pooled_quintile_groups={"Q4-Q5": [4, 5]},
    )
    r2_weighted = compute_quintile_oos_r2(
        preds_weighted,
        panel_sub,
        "liq_quintile",
        return_col=target,
        hist_window=hist_window,
        pooled_quintile_groups={"Q4-Q5": [4, 5]},
    )

    r2_table = r2_standard.merge(
        r2_weighted,
        on="quintile",
        suffixes=("_std", "_wt"),
    )
    for benchmark in ["zero", "cs", "hist"]:
        std_col = f"pooled_r2_{benchmark}_std"
        wt_col = f"pooled_r2_{benchmark}_wt"
        if std_col not in r2_table or wt_col not in r2_table:
            continue
        r2_table[f"r2_{benchmark}_std_pct"] = r2_table[std_col] * 100
        r2_table[f"r2_{benchmark}_wt_pct"] = r2_table[wt_col] * 100
        r2_table[f"delta_{benchmark}_pct"] = (
            r2_table[wt_col] - r2_table[std_col]
        ) * 100

    full_row = r2_table[r2_table["quintile"] == "Full"].iloc[0]
    r2_std_full = full_row["r2_zero_std_pct"] / 100
    r2_wt_full = full_row["r2_zero_wt_pct"] / 100

    r2_table["r2_std_pct"] = r2_table["r2_zero_std_pct"]
    r2_table["r2_wt_pct"] = r2_table["r2_zero_wt_pct"]
    r2_table["delta_pct"] = r2_table["delta_zero_pct"]
    r2_table["n_obs"] = r2_table["n_obs_std"]

    keep_cols = [
        "quintile",
        "r2_std_pct",
        "r2_wt_pct",
        "delta_pct",
        "avg_n_month_std",
        "avg_n_month_wt",
        "n_obs",
    ]
    if include_extra_benchmarks:
        keep_cols[4:4] = [
            "r2_zero_std_pct",
            "r2_zero_wt_pct",
            "delta_zero_pct",
            "r2_cs_std_pct",
            "r2_cs_wt_pct",
            "delta_cs_pct",
            "r2_hist_std_pct",
            "r2_hist_wt_pct",
            "delta_hist_pct",
        ]
    r2_table = r2_table[[c for c in keep_cols if c in r2_table.columns]]

    panel_utility = panel_sub.copy()
    panel_utility["w_tilde"] = compute_formal_utility_weights(panel, config, spec)
    std_eval = predictions_for_utility_r2(preds_standard, panel_sub, target)
    wt_eval = predictions_for_utility_r2(preds_weighted, panel_sub, target)
    r2_utility_standard = compute_utility_weighted_r2(
        std_eval,
        panel_utility,
    )["r2_weighted_zero"]
    r2_utility_weighted = compute_utility_weighted_r2(
        wt_eval,
        panel_utility,
    )["r2_weighted_zero"]
    utility_label = formal_weight_label(spec, config)

    utility_r2 = pd.DataFrame(
        [
            {
                "metric": "Unweighted evaluation",
                "r2_std_pct": r2_std_full * 100,
                "r2_wt_pct": r2_wt_full * 100,
            },
            {
                "metric": f"Utility-weighted evaluation ({utility_label})",
                "r2_std_pct": r2_utility_standard * 100,
                "r2_wt_pct": r2_utility_weighted * 100,
            },
            {
                "metric": f"Gap (unweighted - {utility_label})",
                "r2_std_pct": (r2_std_full - r2_utility_standard) * 100,
                "r2_wt_pct": (r2_wt_full - r2_utility_weighted) * 100,
            },
        ]
    )

    return {"quintile_r2": r2_table, "utility_weighted_r2": utility_r2}
