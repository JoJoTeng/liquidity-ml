"""Portfolio tables for formal experiment analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.analysis.formal.common import assign_liquidity_quintiles
from src.evaluation.statistics import compute_effect_decomposition, sharpe_ratio
from src.portfolio.construction import build_portfolio_timeseries, compute_net_returns

logger = logging.getLogger(__name__)


def compute_within_quintile_portfolio_table(
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    aum: int | float,
    config: dict,
) -> pd.DataFrame:
    """Compute gross and net Sharpe ratios inside each liquidity quintile."""
    panel_work = panel.copy()
    panel_work["liq_quintile"] = assign_liquidity_quintiles(panel_work, config)

    rows = []
    for quintile in [1, 2, 3, 4, 5]:
        panel_q = panel_work[panel_work["liq_quintile"] == quintile].copy()
        if len(panel_q) < 100:
            continue

        for label, preds in [("std", preds_standard), ("wt", preds_weighted)]:
            ret_df, pos_hist = build_portfolio_timeseries(
                panel_q,
                preds,
                tc_penalised=False,
                config=config,
            )
            if len(ret_df) < 12:
                continue

            gross_sr = sharpe_ratio(ret_df["ret_long_short"].dropna())
            net_df = compute_net_returns(
                ret_df,
                pos_hist,
                panel,
                aum=aum,
                config=config,
            )
            net_sr = sharpe_ratio(net_df["ret_long_short_net"].dropna())

            rows.append(
                {
                    "quintile": f"Q{quintile}",
                    "model_type": label,
                    "gross_sr": gross_sr,
                    "net_sr": net_sr,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    pivot = df.pivot(
        index="quintile",
        columns="model_type",
        values=["gross_sr", "net_sr"],
    )
    pivot.columns = [f"{metric}_{model_type}" for metric, model_type in pivot.columns]
    return pivot.reset_index()


def compute_two_by_two_decomposition(
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    aum: int | float,
    config: dict,
) -> dict:
    """Build the 2x2 training-vs-portfolio decomposition at one AUM level."""
    logger.info("Building 2x2 portfolios at AUM=$%.0fM", aum / 1e6)

    cells = {}
    for cell_name, preds, tc_penalised in [
        ("1A", preds_standard, False),
        ("1B", preds_standard, True),
        ("2A", preds_weighted, False),
        ("2B", preds_weighted, True),
    ]:
        ret_df, pos_hist = build_portfolio_timeseries(
            panel,
            preds,
            tc_penalised=tc_penalised,
            aum=aum if tc_penalised else None,
            config=config,
        )
        net_df = compute_net_returns(ret_df, pos_hist, panel, aum=aum, config=config)
        cells[cell_name] = net_df

    common_months = set(cells["1A"]["yyyymm"])
    for cell_name in ["1B", "2A", "2B"]:
        common_months &= set(cells[cell_name]["yyyymm"])
    common_months = sorted(common_months)

    aligned = {}
    for cell_name, df in cells.items():
        aligned[cell_name] = df[df["yyyymm"].isin(common_months)].sort_values("yyyymm")

    return_summary = {}
    for cell_name, df in aligned.items():
        return_summary[cell_name] = {
            "gross_return_monthly": df["ret_long_short"].mean(),
            "gross_return_annual": df["ret_long_short"].mean() * 12,
            "net_return_monthly": df["ret_long_short_net"].mean(),
            "net_return_annual": df["ret_long_short_net"].mean() * 12,
            "gross_sr_monthly": sharpe_ratio(
                df["ret_long_short"].values,
                annualize=False,
            ),
            "net_sr_monthly": sharpe_ratio(
                df["ret_long_short_net"].values,
                annualize=False,
            ),
            "gross_sr_annual": sharpe_ratio(
                df["ret_long_short"].values,
                annualize=True,
            ),
            "net_sr_annual": sharpe_ratio(
                df["ret_long_short_net"].values,
                annualize=True,
            ),
            "tc_mean_monthly": df["transaction_cost"].mean(),
            "tc_median_monthly": df["transaction_cost"].median(),
        }

    decomp_net = compute_effect_decomposition(
        returns_1a=aligned["1A"]["ret_long_short_net"].values,
        returns_1b=aligned["1B"]["ret_long_short_net"].values,
        returns_2a=aligned["2A"]["ret_long_short_net"].values,
        returns_2b=aligned["2B"]["ret_long_short_net"].values,
        yyyymm=aligned["1A"]["yyyymm"].values,
        config=config,
    )

    decomp_gross = compute_effect_decomposition(
        returns_1a=aligned["1A"]["ret_long_short"].values,
        returns_1b=aligned["1B"]["ret_long_short"].values,
        returns_2a=aligned["2A"]["ret_long_short"].values,
        returns_2b=aligned["2B"]["ret_long_short"].values,
        yyyymm=aligned["1A"]["yyyymm"].values,
        config=config,
    )

    total = decomp_net["total_effect"]
    training_share = (
        decomp_net["training_effect"] / total * 100
        if abs(total) > 1e-10
        else np.nan
    )

    turnover = {}
    raw_trade_sum = {}
    for cell_name, df in aligned.items():
        turnover_values = (
            df["turnover"].iloc[1:] if "turnover" in df else pd.Series(dtype=float)
        )
        raw_trade_values = (
            df["raw_trade_sum"].iloc[1:]
            if "raw_trade_sum" in df
            else pd.Series(dtype=float)
        )
        turnover[cell_name] = (
            turnover_values.mean() if len(turnover_values) else np.nan
        )
        raw_trade_sum[cell_name] = (
            raw_trade_values.mean() if len(raw_trade_values) else np.nan
        )

    return {
        "decomposition": decomp_net,
        "decomposition_gross": decomp_gross,
        "training_share": training_share,
        "cells": aligned,
        "return_summary": return_summary,
        "turnover": turnover,
        "raw_trade_sum": raw_trade_sum,
    }


def format_two_by_two_decomposition_rows(result: dict) -> list[dict]:
    """Flatten a 2x2 decomposition result into metric/value rows for CSV export."""
    rows = []
    ret_summary = result["return_summary"]
    decomp_net = result["decomposition"]
    decomp_gross = result["decomposition_gross"]

    for cell in ["1A", "1B", "2A", "2B"]:
        rows.extend(
            [
                {
                    "metric": f"Gross return monthly ({cell})",
                    "value": ret_summary[cell]["gross_return_monthly"],
                },
                {
                    "metric": f"Gross return annualized ({cell})",
                    "value": ret_summary[cell]["gross_return_annual"],
                },
                {
                    "metric": f"Net return monthly ({cell})",
                    "value": ret_summary[cell]["net_return_monthly"],
                },
                {
                    "metric": f"Net return annualized ({cell})",
                    "value": ret_summary[cell]["net_return_annual"],
                },
                {
                    "metric": f"TC mean monthly ({cell})",
                    "value": ret_summary[cell]["tc_mean_monthly"],
                },
                {
                    "metric": f"TC median monthly ({cell})",
                    "value": ret_summary[cell]["tc_median_monthly"],
                },
                {
                    "metric": f"SR_gross_monthly({cell})",
                    "value": ret_summary[cell]["gross_sr_monthly"],
                },
                {
                    "metric": f"SR_net_monthly({cell})",
                    "value": ret_summary[cell]["net_sr_monthly"],
                },
                {
                    "metric": f"SR_gross_annualized({cell})",
                    "value": ret_summary[cell]["gross_sr_annual"],
                },
                {
                    "metric": f"SR_net_annualized({cell})",
                    "value": ret_summary[cell]["net_sr_annual"],
                },
            ]
        )

    rows.extend(
        [
            {"metric": "Net training effect", "value": decomp_net["training_effect"]},
            {"metric": "Net portfolio effect", "value": decomp_net["portfolio_effect"]},
            {"metric": "Net total effect", "value": decomp_net["total_effect"]},
            {"metric": "Net interaction", "value": decomp_net["interaction"]},
            {"metric": "Training share (%)", "value": result["training_share"]},
            {
                "metric": "LW p-val (training, net)",
                "value": decomp_net["lw_training"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (total, net)",
                "value": decomp_net["lw_total"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (H1, net)",
                "value": decomp_net["lw_h3"].get("p_value", np.nan),
            },
            {
                "metric": "Gross training effect",
                "value": decomp_gross["training_effect"],
            },
            {
                "metric": "Gross portfolio effect",
                "value": decomp_gross["portfolio_effect"],
            },
            {
                "metric": "Gross total effect",
                "value": decomp_gross["total_effect"],
            },
            {
                "metric": "LW p-val (training, gross)",
                "value": decomp_gross["lw_training"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (total, gross)",
                "value": decomp_gross["lw_total"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (H1, gross)",
                "value": decomp_gross["lw_h3"].get("p_value", np.nan),
            },
        ]
    )

    alphas = decomp_net.get("factor_alphas", {})
    for model_name in ["capm", "ff3", "ff5", "ff5_mom"]:
        if model_name not in alphas:
            continue
        for cell in ["1A", "1B", "2A", "2B"]:
            alpha = alphas[model_name].get(cell, {})
            rows.extend(
                [
                    {
                        "metric": f"alpha_{model_name}({cell})_annual",
                        "value": alpha.get("alpha_annual", np.nan),
                    },
                    {
                        "metric": f"alpha_{model_name}({cell})_tstat",
                        "value": alpha.get("alpha_tstat", np.nan),
                    },
                    {
                        "metric": f"alpha_{model_name}({cell})_pvalue",
                        "value": alpha.get("alpha_pvalue", np.nan),
                    },
                ]
            )

    for cell in ["1A", "1B", "2A", "2B"]:
        rows.extend(
            [
                {
                    "metric": f"Turnover ({cell})",
                    "value": result["turnover"].get(cell, np.nan),
                },
                {
                    "metric": f"Raw trade sum ({cell})",
                    "value": result["raw_trade_sum"].get(cell, np.nan),
                },
            ]
        )

    return rows
