"""
Portfolio Construction (LiquidityML v3, Section 9)
===================================================
Long-short decile portfolios for the 2x2 experimental framework.

The 2x2 design:
                    Sort on r_hat       Sort on r_hat - TC
  Standard train    1A (Baseline)       1B
  Weighted train    2A                  2B (Combined)

Column 1: standard sort on predicted returns.
Column 2: TC-penalised sort - sort on r_hat - TC(AUM).
  No explicit liquidity filter - the TC penalty naturally pushes
  illiquid stocks down the ranking.

Portfolio rules:
  - Decile sort (long Q10, short Q1)
  - Equal-dollar within each leg
  - Monthly rebalancing

Net return computation:
  - Uses actual turnover-adjusted trade size DeltaQ_it
    (target position minus drifted position) for market impact,
    NOT the full Q_it = AUM/N_leg.
  - Tracks position drift from returns between rebalancing months.

Transaction cost model: Frazzini et al. (2018) Eq. 25.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

_cfg = load_config()


# -- Helpers ----------------------------------------------------------


def _assign_deciles(values: pd.Series, n_quantiles: int = 10) -> pd.Series:
    """Assign stocks to decile bins (1 = lowest, 10 = highest).

    Uses rank-based assignment to handle tied predictions.
    """
    return pd.qcut(
        values.rank(method="first"),
        q=n_quantiles,
        labels=False,
    ) + 1  # 1-indexed


# -- Core: Build Single-Month Portfolio -------------------------------


def build_long_short_portfolio(
    df: pd.DataFrame,
    predictions: pd.Series,
    tc_penalised: bool = False,
    tc_per_stock: pd.Series | None = None,
    n_quantiles: int = 10,
) -> dict[str, Any]:
    """Build a long-short decile portfolio for a single cross-section.

    Parameters
    ----------
    df : DataFrame for one month with columns: permno, ret.
    predictions : Model return predictions aligned to df rows.
    tc_penalised : If True, sort on (predictions - tc_per_stock) for Column 2.
    tc_per_stock : One-way TC per stock (required if tc_penalised=True).
    n_quantiles : Number of quantiles (default 10 for deciles).

    Returns
    -------
    dict with keys:
        ret_long, ret_short, ret_long_short : gross returns
        n_long, n_short : stock counts
        positions_long, positions_short : {permno: weight} dicts
        permnos_long, permnos_short : sets of permnos in each leg
    """
    work = df.copy()
    work["_pred"] = predictions.values if isinstance(predictions, pd.Series) else predictions

    # -- Sorting signal ------
    if tc_penalised:
        if tc_per_stock is None:
            raise ValueError("tc_per_stock required for TC-penalised sort")
        work["_signal"] = work["_pred"] - tc_per_stock.reindex(work.index).fillna(0.0)
    else:
        work["_signal"] = work["_pred"]

    if len(work) < n_quantiles * 2:
        return _empty_result()

    # -- Decile assignment ------
    work["_decile"] = _assign_deciles(work["_signal"], n_quantiles)

    long_df = work[work["_decile"] == n_quantiles]   # Q10 = highest signal
    short_df = work[work["_decile"] == 1]             # Q1 = lowest signal

    if len(long_df) < 2 or len(short_df) < 2:
        return _empty_result()

    # -- Equal-dollar weights ------
    w_long = pd.Series(1.0 / len(long_df), index=long_df.index)
    w_short = pd.Series(1.0 / len(short_df), index=short_df.index)

    # -- Gross returns ------
    ret_long = (w_long * long_df["ret"]).sum()
    ret_short = (w_short * short_df["ret"]).sum()

    # -- Store positions for TC computation ------
    pos_long = dict(zip(long_df["permno"], w_long))
    pos_short = dict(zip(short_df["permno"], w_short))

    return {
        "ret_long": ret_long,
        "ret_short": ret_short,
        "ret_long_short": ret_long - ret_short,
        "n_long": len(long_df),
        "n_short": len(short_df),
        "positions_long": pos_long,
        "positions_short": pos_short,
        "permnos_long": set(long_df["permno"]),
        "permnos_short": set(short_df["permno"]),
    }


def _empty_result() -> dict[str, Any]:
    """Return NaN result for months with insufficient data."""
    return {
        "ret_long": np.nan,
        "ret_short": np.nan,
        "ret_long_short": np.nan,
        "n_long": 0,
        "n_short": 0,
        "positions_long": {},
        "positions_short": {},
        "permnos_long": set(),
        "permnos_short": set(),
    }


# -- Time Series Portfolio Builder ------------------------------------


def build_portfolio_timeseries(
    panel: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
    tc_penalised: bool = False,
    aum: float | None = None,
    config: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build monthly long-short portfolios over the full test period.

    Parameters
    ----------
    panel : Full panel with yyyymm, permno, ret, liq_* columns.
    predictions : Either
        - pd.DataFrame with columns [permno, yyyymm, prediction], or
        - pd.Series indexed by panel row labels (legacy support).
        Merged with panel via (permno, yyyymm) keys.
    tc_penalised : If True, use TC-penalised sort (Column 2).
    aum : AUM in dollars. Required if tc_penalised=True.
    config : Override config dict.

    Returns
    -------
    (returns_df, positions_history)
    returns_df : DataFrame with yyyymm, ret_long, ret_short, ret_long_short.
    positions_history : {yyyymm: {"long": {permno: w}, "short": {permno: w}}}
    """
    if config is None:
        config = _cfg
    n_q = config["portfolio"]["n_quantiles"]

    # Normalise predictions to a DataFrame with [permno, yyyymm, prediction]
    if isinstance(predictions, pd.Series):
        # Legacy path: index is assumed to be a subset of panel.index
        common = panel.index.intersection(predictions.index)
        pred_df = pd.DataFrame({
            "permno": panel.loc[common, "permno"].values,
            "yyyymm": panel.loc[common, "yyyymm"].values,
            "prediction": predictions.loc[common].values,
        })
    else:
        pred_df = predictions[["permno", "yyyymm", "prediction"]].copy()

    # Inner merge with panel on (permno, yyyymm) — guarantees alignment
    panel_pred = panel.merge(pred_df, on=["permno", "yyyymm"], how="inner")
    panel_pred = panel_pred.reset_index(drop=True)

    # Precompute TC for sorting if needed — on the merged panel
    tc_for_sort = None
    if tc_penalised:
        if aum is None:
            raise ValueError("aum required for TC-penalised sort")
        from src.weighting.schemes import compute_tc_for_sorting
        tc_for_sort = compute_tc_for_sorting(panel_pred, aum=aum, config=config)

    pred_months = sorted(panel_pred["yyyymm"].unique())
    records = []
    positions_history = {}

    for yyyymm in pred_months:
        mask = panel_pred["yyyymm"] == yyyymm
        month_panel = panel_pred[mask]

        if len(month_panel) < n_q * 2:
            continue

        month_preds = pd.Series(
            month_panel["prediction"].values, index=month_panel.index,
        )

        tc_month = None
        if tc_penalised and tc_for_sort is not None:
            tc_month = tc_for_sort.loc[month_panel.index]

        result = build_long_short_portfolio(
            df=month_panel,
            predictions=month_preds,
            tc_penalised=tc_penalised,
            tc_per_stock=tc_month,
            n_quantiles=n_q,
        )

        result["yyyymm"] = yyyymm
        records.append(result)
        positions_history[yyyymm] = {
            "long": result["positions_long"],
            "short": result["positions_short"],
        }

    if not records:
        logger.warning("No valid months for portfolio construction")
        return pd.DataFrame(), {}

    returns_df = pd.DataFrame(records)
    cols = ["yyyymm", "ret_long", "ret_short", "ret_long_short", "n_long", "n_short"]
    returns_df = returns_df[[c for c in cols if c in returns_df.columns]]

    return returns_df, positions_history


# -- Transaction Cost: Turnover-Adjusted (Net Returns) ----------------


def compute_net_returns(
    gross_returns: pd.DataFrame,
    positions_history: dict[int, dict],
    panel: pd.DataFrame,
    aum: float,
    config: dict | None = None,
) -> pd.DataFrame:
    """Compute net returns using turnover-adjusted trade sizes.

    For each stock held or traded:
      1. Compute drifted position from last month's holding + actual return
      2. DeltaQ_it = |target_position - drifted_position|
      3. TC_it = Spread/2 + lambda * sigma * sqrt(DeltaQ_it / ADV)
      4. Portfolio TC = sum over all traded stocks
    """
    if config is None:
        config = _cfg
    tc_cfg = config["transaction_costs"]
    lam = tc_cfg["lambda_market_impact"]
    spread_col = f"liq_{tc_cfg['spread_col']}"
    sigma_col = f"liq_{tc_cfg['sigma_col']}"
    adv_col = f"liq_{tc_cfg['adv_col']}"

    # Build lookup: (permno, yyyymm) -> {spread, sigma, adv, ret}
    lookup_cols = ["permno", "yyyymm", "ret"]
    for col in [spread_col, sigma_col, adv_col]:
        if col in panel.columns:
            lookup_cols.append(col)
    lookup = panel[list(set(lookup_cols))].copy()
    lookup = lookup.set_index(["permno", "yyyymm"])

    months = sorted(positions_history.keys())
    tc_series = {}

    # Track drifted positions (evolve with actual returns)
    prev_pos_long: dict[int, float] = {}
    prev_pos_short: dict[int, float] = {}

    for i, yyyymm in enumerate(months):
        pos = positions_history[yyyymm]
        target_long = pos["long"]
        target_short = pos["short"]

        # Compute drifted positions from last month
        if i == 0:
            drifted_long: dict[int, float] = {}
            drifted_short: dict[int, float] = {}
        else:
            prev_yyyymm = months[i - 1]
            drifted_long = _drift_positions(prev_pos_long, prev_yyyymm, yyyymm, lookup)
            drifted_short = _drift_positions(prev_pos_short, prev_yyyymm, yyyymm, lookup)

        # Compute TC for each leg
        total_tc = 0.0
        for target, drifted, leg_name in [
            (target_long, drifted_long, "long"),
            (target_short, drifted_short, "short"),
        ]:
            all_permnos = set(target) | set(drifted)
            for permno in all_permnos:
                target_w = target.get(permno, 0.0)
                drifted_w = drifted.get(permno, 0.0)
                delta_w = abs(target_w - drifted_w)

                if delta_w < 1e-10:
                    continue

                # Dollar trade amount
                delta_q = delta_w * aum

                # Look up stock characteristics
                try:
                    row = lookup.loc[(permno, yyyymm)]
                except KeyError:
                    total_tc += delta_w * 0.005  # 50bps default
                    continue

                spread = _safe_val(row, spread_col, 0.01)
                sigma = _safe_val(row, sigma_col, 0.02)
                adv = _safe_val(row, adv_col, 1e6)

                spread = abs(spread)
                adv = max(adv, 1e3)

                # Frazzini et al. Eq. 25 with actual DeltaQ
                half_spread = spread / 2.0
                market_impact = lam * sigma * np.sqrt(delta_q / adv)
                tc_i = half_spread + market_impact

                total_tc += delta_w * tc_i

        tc_series[yyyymm] = total_tc

        # Store current positions for next month's drift
        prev_pos_long = target_long.copy()
        prev_pos_short = target_short.copy()

    # Merge TC into gross returns
    result = gross_returns.copy()
    result["transaction_cost"] = result["yyyymm"].map(tc_series).fillna(0.0)
    result["ret_long_short_net"] = result["ret_long_short"] - result["transaction_cost"]
    return result


def _drift_positions(
    positions: dict[int, float],
    from_yyyymm: int,
    to_yyyymm: int,
    lookup: pd.DataFrame,
) -> dict[int, float]:
    """Drift positions forward by one month using actual returns."""
    if not positions:
        return {}

    drifted = {}
    total_value = 0.0

    for permno, w in positions.items():
        try:
            row = lookup.loc[(permno, from_yyyymm)]
            ret = row["ret"] if not pd.isna(row["ret"]) else 0.0
        except KeyError:
            ret = 0.0
        new_val = w * (1.0 + ret)
        drifted[permno] = new_val
        total_value += new_val

    # Renormalise to sum to ~1
    if total_value > 0:
        for permno in drifted:
            drifted[permno] /= total_value

    return drifted


def _safe_val(row, col: str, default: float) -> float:
    """Safely extract a value from a lookup row."""
    try:
        val = row[col] if col in row.index else default
    except (KeyError, TypeError):
        val = default
    if pd.isna(val) or val <= 0:
        val = default
    return val


def compute_net_returns_all_aum(
    gross_returns: pd.DataFrame,
    positions_history: dict[int, dict],
    panel: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Compute net returns for all AUM scenarios from config.

    Returns
    -------
    DataFrame with columns: ret_ls_net_{aum_label} for each AUM.
    """
    if config is None:
        config = _cfg
    aum_scenarios = config["transaction_costs"]["aum_scenarios"]

    result = gross_returns.copy()
    for aum in aum_scenarios:
        net_df = compute_net_returns(
            gross_returns, positions_history, panel, aum=aum, config=config,
        )
        label = f"{aum // 1_000_000}M" if aum < 1_000_000_000 else f"{aum // 1_000_000_000}B"
        result[f"ret_ls_net_{label}"] = net_df["ret_long_short_net"]

    return result
