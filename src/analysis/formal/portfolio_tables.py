"""Portfolio tables for formal experiment analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.analysis.formal.common import (
    aum_label,
    is_proportional_tc_scenario,
    scenario_sizing_aum,
)
from src.evaluation.statistics import compute_effect_decomposition, sharpe_ratio
from src.portfolio.construction import (
    build_portfolio_timeseries,
    build_prediction_quantile_timeseries,
    compute_net_returns,
    prepare_transaction_cost_context,
)

logger = logging.getLogger(__name__)


def _build_long_short_cell(
    panel, preds, *, hysteresis, sizing_aum, tc_context, config,
    portfolio_weighting, value_weight_col, signal_weight_col,
    liquidity_screen_pct, liquidity_screen_col, leg_capital, proportional_tc_only,
):
    """One long-short cell as a TRUE dollar-neutral book (build_portfolio_timeseries).

    A = plain prediction sort (hysteresis=False); B = symmetric membership
    hysteresis. $A gross by default via ``leg_capital``. Returns the per-month
    frame [yyyymm, ret_long, ret_short, ret_long_short, transaction_cost,
    raw_trade_sum, turnover, ret_long_short_net, n_long, n_short].
    """
    returns_df, positions_history = build_portfolio_timeseries(
        panel,
        preds,
        config=config,
        portfolio_weighting=portfolio_weighting,
        value_weight_col=value_weight_col,
        signal_weight_col=signal_weight_col,
        liquidity_screen_pct=liquidity_screen_pct,
        liquidity_screen_col=liquidity_screen_col,
        hysteresis=hysteresis,
    )
    if returns_df.empty:
        return pd.DataFrame()
    net = compute_net_returns(
        returns_df,
        positions_history,
        panel,
        aum=sizing_aum,
        config=config,
        tc_context=tc_context,
        proportional_tc_only=proportional_tc_only,
        leg_capital=leg_capital,
    )
    return net.sort_values("yyyymm").reset_index(drop=True)


def compute_two_by_two_decomposition(
    preds_standard,
    preds_weighted,
    preds_tc_target_standard=None,   # deprecated: tc-target 'C' column dropped
    preds_tc_target_weighted=None,   # deprecated: retained for call-sig compat
    panel=None,
    aum=None,
    config=None,
    portfolio_mode: str = "long_short",
    portfolio_weighting: str = "equal",
    portfolio_design: str = "prediction_quantile",
    value_weight_col: str = "liq_me_raw",
    signal_weight_col: str = "w_tilde",
    liquidity_screen_pct: float | None = None,
    liquidity_screen_col: str = "liq_dvol_21d",
    leg_capital: str = "gross",
    proportional_tc_only: bool = False,
) -> dict:
    """Build the 2x2 training-vs-device decomposition for the formal long-short.

    Rows = training (1=standard, 2=weighted); columns = device (A = plain
    prediction sort, B = cost-scaled membership hysteresis). The long-short cells
    are a TRUE dollar-neutral long-short book (``build_portfolio_timeseries``;
    ``$A`` gross by default via ``leg_capital``) with signal/equal/value legs. The
    per-quantile timeseries (standalone Q1-Q5 + Table 14) is built separately with
    A=plain quantiles and B=per-quantile sticky hysteresis. The tc-target 'C'
    column is dropped; ``preds_tc_target_*`` are ignored (kept for call-signature
    compatibility).
    """
    if panel is None or aum is None or config is None:
        raise ValueError("panel, aum, and config are required")
    tc_label = "PropTC" if proportional_tc_only else f"${aum / 1e6:.0f}M"
    logger.info(
        "Building 2x2 long-short (true book): tc_scenario=%s, weighting=%s",
        tc_label,
        portfolio_weighting,
    )
    tc_context = prepare_transaction_cost_context(panel, config)
    sizing_aum = 0.0 if proportional_tc_only else float(aum)

    cells = {}
    quantile_timeseries = {}
    for cell_name, preds, hysteresis in [
        ("1A", preds_standard, False),
        ("1B", preds_standard, True),
        ("2A", preds_weighted, False),
        ("2B", preds_weighted, True),
    ]:
        cells[cell_name] = _build_long_short_cell(
            panel,
            preds,
            hysteresis=hysteresis,
            sizing_aum=sizing_aum,
            tc_context=tc_context,
            config=config,
            portfolio_weighting=portfolio_weighting,
            value_weight_col=value_weight_col,
            signal_weight_col=signal_weight_col,
            liquidity_screen_pct=liquidity_screen_pct,
            liquidity_screen_col=liquidity_screen_col,
            leg_capital=leg_capital,
            proportional_tc_only=proportional_tc_only,
        )
        quantile_timeseries[cell_name] = build_prediction_quantile_timeseries(
            panel,
            preds,
            sort_mode="prediction",
            aum=aum,
            config=config,
            tc_context=tc_context,
            portfolio_weighting=portfolio_weighting,
            value_weight_col=value_weight_col,
            signal_weight_col=signal_weight_col,
            liquidity_screen_pct=liquidity_screen_pct,
            liquidity_screen_col=liquidity_screen_col,
            proportional_tc_only=proportional_tc_only,
            hysteresis=hysteresis,
        )

    common_months = set(cells["1A"]["yyyymm"])
    for cell_name in ["1B", "2A", "2B"]:
        common_months &= set(cells[cell_name]["yyyymm"])
    common_months = sorted(common_months)

    aligned = {
        cell_name: df[df["yyyymm"].isin(common_months)].sort_values("yyyymm")
        for cell_name, df in cells.items()
    }

    return_summary = {}
    for cell_name, df in aligned.items():
        return_summary[cell_name] = {
            "gross_return_monthly": df["ret_long_short"].mean(),
            "gross_return_annual": df["ret_long_short"].mean() * 12,
            "net_return_monthly": df["ret_long_short_net"].mean(),
            "net_return_annual": df["ret_long_short_net"].mean() * 12,
            "gross_sr_monthly": sharpe_ratio(df["ret_long_short"].values, annualize=False),
            "net_sr_monthly": sharpe_ratio(df["ret_long_short_net"].values, annualize=False),
            "gross_sr_annual": sharpe_ratio(df["ret_long_short"].values, annualize=True),
            "net_sr_annual": sharpe_ratio(df["ret_long_short_net"].values, annualize=True),
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

    turnover = {}
    raw_trade_sum = {}
    for cell_name, df in aligned.items():
        tv = df["turnover"].iloc[1:] if "turnover" in df else pd.Series(dtype=float)
        rv = df["raw_trade_sum"].iloc[1:] if "raw_trade_sum" in df else pd.Series(dtype=float)
        turnover[cell_name] = tv.mean() if len(tv) else np.nan
        raw_trade_sum[cell_name] = rv.mean() if len(rv) else np.nan

    return {
        "decomposition": decomp_net,
        "decomposition_gross": decomp_gross,
        "tc_target_effects": {},
        "cells": aligned,
        "quantile_timeseries": quantile_timeseries,
        "return_summary": return_summary,
        "turnover": turnover,
        "raw_trade_sum": raw_trade_sum,
        "portfolio_mode": portfolio_mode,
        "portfolio_weighting": portfolio_weighting,
        "portfolio_design": portfolio_design,
        "liquidity_screen_pct": liquidity_screen_pct,
        "liquidity_screen_col": liquidity_screen_col,
    }


# Backward-compatible alias: the formal long-short is now a 2x2 (tc-target 'C'
# column dropped); 21e/22b still import the historical name.
compute_two_by_three_decomposition = compute_two_by_two_decomposition


def _derive_long_short_from_quantile_timeseries(
    quantile_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Derive configured Q_high-Q_low long-short returns from quantile portfolios."""
    if quantile_df.empty:
        return pd.DataFrame()

    portfolio_cfg = config["portfolio"]
    n_q = int(portfolio_cfg["n_quantiles"])
    long_q = int(portfolio_cfg.get("long_quantile", n_q))
    short_q = int(portfolio_cfg.get("short_quantile", 1))

    long = quantile_df[quantile_df["prediction_quantile"] == long_q].copy()
    short = quantile_df[quantile_df["prediction_quantile"] == short_q].copy()
    merged = long.merge(
        short,
        on="yyyymm",
        how="inner",
        suffixes=("_long", "_short"),
    )
    if merged.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "yyyymm": merged["yyyymm"],
        "ret_long": merged["gross_return_long"],
        "ret_short": merged["gross_return_short"],
        "ret_long_short": (
            merged["gross_return_long"] - merged["gross_return_short"]
        ),
        "transaction_cost_long": merged["transaction_cost_long"],
        "transaction_cost_short": merged["transaction_cost_short"],
        "transaction_cost": (
            merged["transaction_cost_long"] + merged["transaction_cost_short"]
        ),
        "raw_trade_sum_long": merged["raw_trade_sum_long"],
        "raw_trade_sum_short": merged["raw_trade_sum_short"],
        "raw_trade_sum": (
            merged["raw_trade_sum_long"] + merged["raw_trade_sum_short"]
        ),
        "n_long": merged["n_stocks_long"],
        "n_short": merged["n_stocks_short"],
        "long_quantile": long_q,
        "short_quantile": short_q,
    })
    out["turnover"] = 0.5 * out["raw_trade_sum"]
    out["ret_long_short_net"] = out["ret_long_short"] - out["transaction_cost"]
    return out.sort_values("yyyymm")


def format_two_by_three_decomposition_rows(result: dict) -> list[dict]:
    """Flatten a 2x3 decomposition result into metric/value rows for CSV export."""
    rows = []
    ret_summary = result["return_summary"]
    decomp_net = result["decomposition"]
    decomp_gross = result["decomposition_gross"]
    tc_target_effects = result.get("tc_target_effects", {})
    net_sr_annual = {
        cell: ret_summary[cell]["net_sr_annual"]
        for cell in ["1A", "1B", "2A", "2B"]
    }
    gross_sr_annual = {
        cell: ret_summary[cell]["gross_sr_annual"]
        for cell in ["1A", "1B", "2A", "2B"]
    }
    net_training_effect_annual = net_sr_annual["2A"] - net_sr_annual["1A"]
    net_portfolio_effect_annual = net_sr_annual["1B"] - net_sr_annual["1A"]
    net_total_effect_annual = net_sr_annual["2B"] - net_sr_annual["1A"]
    net_interaction_annual = (
        net_total_effect_annual
        - net_training_effect_annual
        - net_portfolio_effect_annual
    )
    gross_training_effect_annual = gross_sr_annual["2A"] - gross_sr_annual["1A"]
    gross_portfolio_effect_annual = gross_sr_annual["1B"] - gross_sr_annual["1A"]
    gross_total_effect_annual = gross_sr_annual["2B"] - gross_sr_annual["1A"]
    training_share_annual = (
        net_training_effect_annual / net_total_effect_annual * 100.0
        if abs(net_total_effect_annual) > 1e-10
        else np.nan
    )

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
            {
                "metric": "Net training effect annualized",
                "value": net_training_effect_annual,
            },
            {
                "metric": "Net portfolio effect annualized",
                "value": net_portfolio_effect_annual,
            },
            {
                "metric": "Net total effect annualized",
                "value": net_total_effect_annual,
            },
            {
                "metric": "Net interaction annualized",
                "value": net_interaction_annual,
            },
            {
                "metric": "Training share annualized (%)",
                "value": training_share_annual,
            },
            {
                "metric": "LW p-val (training, net)",
                "value": decomp_net["lw_training"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (portfolio, net)",
                "value": decomp_net["lw_portfolio"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (total, net)",
                "value": decomp_net["lw_total"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (H3, net)",
                "value": decomp_net["lw_h3"].get("p_value", np.nan),
            },
            {
                "metric": "Gross training effect annualized",
                "value": gross_training_effect_annual,
            },
            {
                "metric": "Gross portfolio effect annualized",
                "value": gross_portfolio_effect_annual,
            },
            {
                "metric": "Gross total effect annualized",
                "value": gross_total_effect_annual,
            },
            {
                "metric": "LW p-val (training, gross)",
                "value": decomp_gross["lw_training"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (portfolio, gross)",
                "value": decomp_gross["lw_portfolio"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (total, gross)",
                "value": decomp_gross["lw_total"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (H3, gross)",
                "value": decomp_gross["lw_h3"].get("p_value", np.nan),
            },
        ]
    )

    for metric, value in tc_target_effects.items():
        rows.append({"metric": metric, "value": value})

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


def format_two_by_three_timeseries_rows(
    result: dict,
    aum: int | float,
    label: str,
) -> pd.DataFrame:
    """Flatten monthly 2x3 portfolio returns and trading costs for CSV export."""
    cell_meta = {
        "1A": ("standard", "prediction"),
        "1B": ("standard", "hysteresis"),
        "2A": ("weighted", "prediction"),
        "2B": ("weighted", "hysteresis"),
    }
    frames = []
    portfolio_design = result.get("portfolio_design", "prediction_quantile")
    portfolio_weighting = result.get("portfolio_weighting", "equal")

    for cell, df in result["cells"].items():
        training_model, portfolio_sort = cell_meta[cell]
        out = df.copy()
        out["cell"] = cell
        out["training_model"] = training_model
        out["portfolio_sort"] = portfolio_sort
        out["aum"] = aum
        out["aum_label"] = label
        out["portfolio_design"] = portfolio_design
        out["portfolio_weighting"] = portfolio_weighting
        out["gross_return"] = out["ret_long_short"]
        out["net_return"] = out["ret_long_short_net"]
        out["gross_return_pct"] = out["gross_return"] * 100.0
        out["transaction_cost_pct"] = out["transaction_cost"] * 100.0
        out["net_return_pct"] = out["net_return"] * 100.0
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    columns = [
        "yyyymm",
        "cell",
        "training_model",
        "portfolio_sort",
        "aum",
        "aum_label",
        "portfolio_design",
        "portfolio_weighting",
        "gross_return",
        "transaction_cost",
        "transaction_cost_long",
        "transaction_cost_short",
        "net_return",
        "gross_return_pct",
        "transaction_cost_pct",
        "net_return_pct",
        "turnover",
        "raw_trade_sum",
        "raw_trade_sum_long",
        "raw_trade_sum_short",
        "ret_long",
        "ret_short",
        "n_long",
        "n_short",
        "long_quantile",
        "short_quantile",
    ]
    return out[[c for c in columns if c in out.columns]].sort_values(
        ["yyyymm", "cell"]
    )


def format_prediction_quantile_timeseries_rows(
    result: dict,
    aum: int | float,
    label: str,
) -> pd.DataFrame:
    """Flatten monthly prediction-quantile returns and trading costs for export."""
    cell_meta = {
        "1A": ("standard", "prediction"),
        "1B": ("standard", "hysteresis"),
        "2A": ("weighted", "prediction"),
        "2B": ("weighted", "hysteresis"),
    }
    frames = []
    portfolio_design = result.get("portfolio_design", "prediction_quantile")
    portfolio_weighting = result.get("portfolio_weighting", "equal")

    for cell, df in result["quantile_timeseries"].items():
        training_model, portfolio_sort = cell_meta[cell]
        out = df.copy()
        out["cell"] = cell
        out["training_model"] = training_model
        out["portfolio_sort"] = portfolio_sort
        out["aum"] = aum
        out["aum_label"] = label
        out["portfolio_design"] = portfolio_design
        out["portfolio_weighting"] = portfolio_weighting
        out["gross_return_pct"] = out["gross_return"] * 100.0
        out["transaction_cost_pct"] = out["transaction_cost"] * 100.0
        out["net_return_pct"] = out["net_return"] * 100.0
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    columns = [
        "yyyymm",
        "cell",
        "training_model",
        "portfolio_sort",
        "prediction_quantile",
        "aum",
        "aum_label",
        "portfolio_design",
        "portfolio_weighting",
        "gross_return",
        "transaction_cost",
        "net_return",
        "gross_return_pct",
        "transaction_cost_pct",
        "net_return_pct",
        "turnover",
        "raw_trade_sum",
        "n_stocks",
        "weight_sum",
        "score_mean",
    ]
    return out[[c for c in columns if c in out.columns]].sort_values(
        ["yyyymm", "cell", "prediction_quantile"]
    )


def _compute_tc_target_score_effects(aligned: dict[str, pd.DataFrame]) -> dict:
    """Compute Sharpe-ratio deltas involving TC-target score cells."""
    pairs = [
        ("standard_tc_target_score_vs_prediction", "1C", "1A"),
        ("standard_tc_target_score_vs_tc_sort", "1C", "1B"),
        ("weighted_tc_target_score_vs_prediction", "2C", "2A"),
        ("weighted_tc_target_score_vs_tc_sort", "2C", "2B"),
        ("tc_target_training_effect", "2C", "1C"),
        ("tc_target_total_effect_vs_baseline", "2C", "1A"),
    ]
    columns = {
        "net": "ret_long_short_net",
        "gross": "ret_long_short",
    }
    effects = {}
    for return_type, col in columns.items():
        for name, lhs, rhs in pairs:
            effects[f"{return_type}_sr_delta_{name}"] = (
                sharpe_ratio(aligned[lhs][col].values)
                - sharpe_ratio(aligned[rhs][col].values)
            )
            effects[f"{return_type}_sr_delta_{name}_annualized"] = (
                sharpe_ratio(aligned[lhs][col].values, annualize=True)
                - sharpe_ratio(aligned[rhs][col].values, annualize=True)
            )
            effects[f"{return_type}_mean_return_delta_{name}"] = (
                aligned[lhs][col].mean() - aligned[rhs][col].mean()
            )
    return effects
