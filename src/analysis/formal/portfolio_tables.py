"""Portfolio tables for formal experiment analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.formal.common import (
    assign_liquidity_quintiles,
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


def compute_quintile_aum_scaling(
    preds_reference: pd.DataFrame,
    panel: pd.DataFrame,
    base_aum: int | float,
    config: dict,
    reference_quintile: int = 1,
) -> pd.DataFrame:
    """Compute quintile-specific AUM scales from average selected-leg size.

    The reference quintile keeps ``base_aum``. Other quintiles receive
    ``base_aum * mean_leg_n_q / mean_leg_n_ref`` so each selected stock trades
    approximately the same dollars as in the reference quintile.
    """
    panel_work = panel.copy()
    panel_work["liq_quintile"] = assign_liquidity_quintiles(panel_work, config)
    pred_keys = preds_reference[["permno", "yyyymm"]].drop_duplicates()
    n_quantiles = config["portfolio"]["n_quantiles"]

    rows = []
    for quintile in [1, 2, 3, 4, 5]:
        panel_q = panel_work[panel_work["liq_quintile"] == quintile]
        panel_pred = panel_q.merge(pred_keys, on=["permno", "yyyymm"], how="inner")
        monthly_n = panel_pred.groupby("yyyymm")["permno"].nunique()
        mean_universe_n = float(monthly_n.mean()) if len(monthly_n) else np.nan
        mean_leg_n = mean_universe_n / n_quantiles if mean_universe_n > 0 else np.nan
        rows.append(
            {
                "quintile": quintile,
                "Quintile": f"Q{quintile}",
                "Mean_Universe_N": mean_universe_n,
                "Mean_Leg_N": mean_leg_n,
            }
        )

    out = pd.DataFrame(rows)
    ref = out.loc[out["quintile"] == reference_quintile, "Mean_Leg_N"]
    if ref.empty or not np.isfinite(ref.iloc[0]) or ref.iloc[0] <= 0:
        raise ValueError("Reference quintile has no valid leg count for AUM scaling")

    ref_leg_n = float(ref.iloc[0])
    out["Reference_Quintile"] = f"Q{reference_quintile}"
    out["Reference_Mean_Leg_N"] = ref_leg_n
    out["AUM_Scale"] = out["Mean_Leg_N"] / ref_leg_n
    out["Base_AUM"] = float(base_aum)
    out["Effective_AUM"] = out["Base_AUM"] * out["AUM_Scale"]
    out["Effective_AUM_Label"] = out["Effective_AUM"].map(_aum_label_precise)
    return out


def compute_within_quintile_portfolio_table(
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    aum: int | float,
    config: dict,
    portfolio_mode: str = "long_short",
    portfolio_weighting: str = "equal",
) -> pd.DataFrame:
    """Compute gross and net Sharpe ratios inside each liquidity quintile."""
    panel_work = panel.copy()
    panel_work["liq_quintile"] = assign_liquidity_quintiles(panel_work, config)
    tc_context = prepare_transaction_cost_context(panel, config)

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
                portfolio_mode=portfolio_mode,
                portfolio_weighting=portfolio_weighting,
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
                tc_context=tc_context,
                portfolio_mode=portfolio_mode,
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


def compute_quintile_sr_scissors_tables(
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    aum_scenarios: list[int | float | str],
    config: dict,
    tc_sort_aum: int | float | None = None,
    quintile_aum_scale: dict[int, float] | None = None,
    portfolio_mode: str = "long_short",
    portfolio_weighting: str = "equal",
) -> dict[str, pd.DataFrame]:
    """Compute old-style annualized gross/net Sharpe tables by liquidity quintile.

    The default ``std`` and ``weighted`` tables use the plain prediction sort.
    If ``tc_sort_aum`` is provided, an additional ``weighted_tc_sort`` table is
    produced using weighted predictions and the TC-aware portfolio sort.
    If ``quintile_aum_scale`` is provided, net-return AUMs and TC-sort AUMs are
    multiplied by the quintile-specific scale.
    ``portfolio_mode`` is the internal Q-high-minus-Q-low construction mode;
    only ``long_short`` is currently active. ``portfolio_weighting`` controls
    weights within each selected leg: equal-weighted or value-weighted.
    """
    panel_work = panel.copy()
    panel_work["liq_quintile"] = assign_liquidity_quintiles(panel_work, config)
    tc_context = prepare_transaction_cost_context(panel, config)

    model_specs = [
        ("std", preds_standard, False, None),
        ("weighted", preds_weighted, False, None),
    ]
    if tc_sort_aum is not None:
        model_specs.append(
            ("weighted_tc_sort", preds_weighted, True, tc_sort_aum)
        )

    rows_by_model = {key: [] for key, _, _, _ in model_specs}
    for quintile in [1, 2, 3, 4, 5]:
        panel_q = panel_work[panel_work["liq_quintile"] == quintile].copy()
        if len(panel_q) < 100:
            continue
        aum_scale = (
            float(quintile_aum_scale.get(quintile, 1.0))
            if quintile_aum_scale is not None
            else 1.0
        )
        if not np.isfinite(aum_scale) or aum_scale <= 0:
            aum_scale = 1.0

        for key, preds, tc_penalised, sort_aum in model_specs:
            effective_sort_aum = (
                float(sort_aum) * aum_scale
                if tc_penalised and sort_aum is not None
                else None
            )
            ret_df, pos_hist = build_portfolio_timeseries(
                panel_q,
                preds,
                tc_penalised=tc_penalised,
                aum=effective_sort_aum,
                config=config,
                portfolio_mode=portfolio_mode,
                portfolio_weighting=portfolio_weighting,
            )
            if len(ret_df) < 12:
                continue

            gross_returns = ret_df["ret_long_short"].dropna()
            mean_leg_n = ((ret_df["n_long"] + ret_df["n_short"]) / 2.0).mean()
            row = {
                "Quintile": f"Q{quintile}",
                "Gross_SR": sharpe_ratio(gross_returns, annualize=True),
                "Avg_Ret_pct": gross_returns.mean() * 100,
                "N_months": int(gross_returns.shape[0]),
                "Mean_Leg_N": mean_leg_n,
                "Portfolio_Mode": portfolio_mode,
                "Portfolio_Weighting": portfolio_weighting,
            }
            if quintile_aum_scale is not None:
                row["AUM_Scale"] = aum_scale
            if tc_penalised and effective_sort_aum is not None:
                row["Sort_AUM"] = (
                    _aum_label_precise(effective_sort_aum)
                    if quintile_aum_scale is not None
                    else aum_label(sort_aum)
                )

            for aum in aum_scenarios:
                label = aum_label(aum)
                proportional_tc_only = is_proportional_tc_scenario(aum)
                effective_aum = (
                    0.0
                    if proportional_tc_only
                    else scenario_sizing_aum(aum) * aum_scale
                )
                net_df = compute_net_returns(
                    ret_df,
                    pos_hist,
                    panel,
                    aum=effective_aum,
                    config=config,
                    tc_context=tc_context,
                    portfolio_mode=portfolio_mode,
                    proportional_tc_only=proportional_tc_only,
                )
                if quintile_aum_scale is not None and not proportional_tc_only:
                    row[f"Effective_AUM_{label}"] = effective_aum
                    row[f"Effective_AUM_{label}_Label"] = _aum_label_precise(
                        effective_aum
                    )
                row[f"Net_SR_{label}"] = sharpe_ratio(
                    net_df["ret_long_short_net"].dropna(),
                    annualize=True,
                )
                row[f"Avg_TC_{label}_pct"] = (
                    net_df["transaction_cost"].dropna().mean() * 100
                )

            rows_by_model[key].append(row)

    return {key: pd.DataFrame(rows) for key, rows in rows_by_model.items()}


def format_within_quintile_from_scissors(
    scissors_tables: dict[str, pd.DataFrame],
    primary_aum: int | float,
) -> pd.DataFrame:
    """Convert annualized scissors tables to the existing monthly-SR CSV shape."""
    std = scissors_tables.get("std", pd.DataFrame()).copy()
    weighted = scissors_tables.get("weighted", pd.DataFrame()).copy()
    if std.empty or weighted.empty:
        return pd.DataFrame()

    label = aum_label(primary_aum)
    net_col = f"Net_SR_{label}"
    if net_col not in std or net_col not in weighted:
        return pd.DataFrame()

    annualizer = np.sqrt(12.0)
    std_keep = std[["Quintile", "Gross_SR", net_col]].rename(
        columns={
            "Quintile": "quintile",
            "Gross_SR": "gross_sr_std",
            net_col: "net_sr_std",
        }
    )
    wt_keep = weighted[["Quintile", "Gross_SR", net_col]].rename(
        columns={
            "Quintile": "quintile",
            "Gross_SR": "gross_sr_wt",
            net_col: "net_sr_wt",
        }
    )
    out = std_keep.merge(wt_keep, on="quintile", how="inner")
    for col in ["gross_sr_std", "net_sr_std", "gross_sr_wt", "net_sr_wt"]:
        out[col] = out[col] / annualizer
    return out[["quintile", "gross_sr_std", "gross_sr_wt", "net_sr_std", "net_sr_wt"]]


def plot_sr_scissors_table(
    table: pd.DataFrame,
    out_path: Path,
    model: str,
    training_label: str,
    aum_scenarios: list[int | float | str],
) -> None:
    """Plot gross and net Sharpe by quintile for one training regime."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if table.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    _plot_sr_scissors_lines(ax, table, aum_scenarios)
    ax.set_xlabel("Liquidity Quintile")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title(
        f"Gross vs. Net Sharpe Ratio by Liquidity Quintile "
        f"({_model_display_name(model)}) - {training_label}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sr_scissors_comparison(
    std_table: pd.DataFrame,
    weighted_table: pd.DataFrame,
    out_path: Path,
    model: str,
    aum_scenarios: list[int | float | str],
    left_title: str = "Standard Training",
    right_title: str = "Weighted Training",
) -> None:
    """Plot side-by-side standard and weighted Sharpe scissors figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if std_table.empty or weighted_table.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15.9, 6.15), dpi=150, sharey=True)
    _plot_sr_scissors_lines(axes[0], std_table, aum_scenarios)
    _plot_sr_scissors_lines(axes[1], weighted_table, aum_scenarios)
    axes[0].set_title(left_title)
    axes[1].set_title(right_title)
    axes[0].set_ylabel("Annualized Sharpe Ratio")
    for ax in axes:
        ax.set_xlabel("Liquidity Quintile")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper right")
    axes[0].legend().remove()
    fig.suptitle(
        f"Gross vs. Net Sharpe Ratio: {left_title} vs. {right_title} "
        f"({_model_display_name(model)})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sr_scissors_lines(
    ax,
    table: pd.DataFrame,
    aum_scenarios: list[int | float | str],
) -> None:
    labels = table["Quintile"].tolist()
    x = np.arange(len(labels))
    ax.plot(
        x,
        table["Gross_SR"],
        marker="o",
        lw=2,
        color="#4C72B0",
        label="Gross SR",
    )
    colors = ["#64C2A6", "#DD8452", "#E782C2", "#EAC086", "#8DA0CB"]
    markers = ["^", "s", "D", "v", "P"]
    linestyles = ["--", "--", ":", ":", "-."]
    for i, aum in enumerate(aum_scenarios):
        label = aum_label(aum)
        col = f"Net_SR_{label}"
        if col not in table:
            continue
        legend_label = (
            "Net SR (spread/2 only)"
            if is_proportional_tc_scenario(aum)
            else f"Net SR (${label})"
        )
        ax.plot(
            x,
            table[col],
            marker=markers[i % len(markers)],
            lw=1.8,
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % len(colors)],
            label=legend_label,
        )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)


def _quintile_label(quintile: int) -> str:
    if quintile == 1:
        return "Q1 (Most Illiquid)"
    if quintile == 5:
        return "Q5 (Most Liquid)"
    return f"Q{quintile}"


def _aum_label_precise(aum: int | float) -> str:
    """Format AUM labels for non-integer scaled-AUM scenarios."""
    aum = float(aum)
    if aum < 1_000_000_000:
        return f"{aum / 1_000_000:.1f}M"
    return f"{aum / 1_000_000_000:.2f}B"


def _model_display_name(model: str) -> str:
    return {
        "elastic_net": "ElasticNet",
        "xgboost": "XGBoost",
        "neural_network": "Neural Network",
    }.get(model, model)


def compute_two_by_three_decomposition(
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    preds_tc_target_standard: pd.DataFrame,
    preds_tc_target_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    aum: int | float,
    config: dict,
    portfolio_mode: str = "long_short",
    portfolio_weighting: str = "equal",
    portfolio_design: str = "prediction_quantile",
    proportional_tc_only: bool = False,
) -> dict:
    """Build the 2x3 training-vs-portfolio-vs-target decomposition.

    The first two columns reproduce the existing 2x2 design. The third column
    treats the TC-adjusted target model's prediction as the ranking score, with
    long candidates sorted on ``s`` and short candidates sorted on
    ``s + 2 * tc``. Realized gross/net returns are still measured on the normal
    return target through ``panel``.
    """
    portfolio_cfg = config["portfolio"]
    n_q = int(portfolio_cfg["n_quantiles"])
    long_q = int(portfolio_cfg.get("long_quantile", n_q))
    short_q = int(portfolio_cfg.get("short_quantile", 1))
    tc_label = "PropTC" if proportional_tc_only else f"${aum / 1e6:.0f}M"
    logger.info(
        "Building 2x3 prediction-quantile portfolios: tc_scenario=%s, "
        "derived=Q%d-Q%d, weighting=%s",
        tc_label,
        long_q,
        short_q,
        portfolio_weighting,
    )
    tc_context = prepare_transaction_cost_context(panel, config)

    cells = {}
    quantile_timeseries = {}
    for cell_name, preds, sort_mode in [
        ("1A", preds_standard, "prediction"),
        ("1B", preds_standard, "tc_net"),
        ("1C", preds_tc_target_standard, "tc_target_score"),
        ("2A", preds_weighted, "prediction"),
        ("2B", preds_weighted, "tc_net"),
        ("2C", preds_tc_target_weighted, "tc_target_score"),
    ]:
        q_df = build_prediction_quantile_timeseries(
            panel,
            preds,
            sort_mode=sort_mode,
            aum=aum,
            config=config,
            tc_context=tc_context,
            portfolio_weighting=portfolio_weighting,
            proportional_tc_only=proportional_tc_only,
        )
        quantile_timeseries[cell_name] = q_df
        cells[cell_name] = _derive_long_short_from_quantile_timeseries(
            q_df,
            config=config,
        )

    common_months = set(cells["1A"]["yyyymm"])
    for cell_name in ["1B", "1C", "2A", "2B", "2C"]:
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

    tc_target_effects = _compute_tc_target_score_effects(aligned)

    return {
        "decomposition": decomp_net,
        "decomposition_gross": decomp_gross,
        "tc_target_effects": tc_target_effects,
        "cells": aligned,
        "quantile_timeseries": quantile_timeseries,
        "return_summary": return_summary,
        "turnover": turnover,
        "raw_trade_sum": raw_trade_sum,
        "portfolio_mode": portfolio_mode,
        "portfolio_weighting": portfolio_weighting,
        "portfolio_design": portfolio_design,
    }


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
    tc_target_effects = result["tc_target_effects"]
    net_sr_annual = {
        cell: ret_summary[cell]["net_sr_annual"]
        for cell in ["1A", "1B", "1C", "2A", "2B", "2C"]
    }
    gross_sr_annual = {
        cell: ret_summary[cell]["gross_sr_annual"]
        for cell in ["1A", "1B", "1C", "2A", "2B", "2C"]
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

    for cell in ["1A", "1B", "1C", "2A", "2B", "2C"]:
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
                "metric": "LW p-val (total, net)",
                "value": decomp_net["lw_total"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (H1, net)",
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
                "metric": "LW p-val (total, gross)",
                "value": decomp_gross["lw_total"].get("p_value", np.nan),
            },
            {
                "metric": "LW p-val (H1, gross)",
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

    for cell in ["1A", "1B", "1C", "2A", "2B", "2C"]:
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
        "1B": ("standard", "tc_net_score"),
        "1C": ("standard_tc_target", "tc_target_score"),
        "2A": ("weighted", "prediction"),
        "2B": ("weighted", "tc_net_score"),
        "2C": ("weighted_tc_target", "tc_target_score"),
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
        "1B": ("standard", "tc_net_score"),
        "1C": ("standard_tc_target", "tc_target_score"),
        "2A": ("weighted", "prediction"),
        "2B": ("weighted", "tc_net_score"),
        "2C": ("weighted_tc_target", "tc_target_score"),
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
