"""
Portfolio Construction
======================
Long-short decile portfolios for the 2×2 experimental framework.

Supports both equal-weighted (standard) and liquidity-weighted portfolios.
Transaction cost model follows Frazzini et al. (2018) Eq. 12:

    TC_i = Spread_i / 2  +  λ · σ_i · √(Q_i / ADV_i)

where λ=0.1, σ_i is daily volatility, Q_i is trade size, ADV_i is
average daily volume.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

# ── Module-level constants (from config) ─────────────────────

_cfg = load_config()
_portfolio_cfg = _cfg["portfolio"]
_tc_cfg = _cfg["transaction_costs"]
_liq_cfg = _cfg["liquidity"]

N_QUANTILES: int = _portfolio_cfg["n_quantiles"]
LONG_QUANTILE: int = _portfolio_cfg["long_quantile"]
SHORT_QUANTILE: int = _portfolio_cfg["short_quantile"]
POSITION_CAP: float = _portfolio_cfg["position_cap"]
MIN_LIQUIDITY_PCTILE: float = _portfolio_cfg["min_liquidity_pctile"]
PRIMARY_LIQ_COL: str = f"liq_{_liq_cfg['primary']}"

# Weighting scheme for liquidity-weighted portfolios
_weighting_cfg = _cfg["weighting"]
DEFAULT_WEIGHT_COL: str = f"weight_{_weighting_cfg['primary']}"

# Transaction cost parameters (Frazzini et al. 2018, Eq. 12)
TC_LAMBDA: float = _tc_cfg["lambda_market_impact"]
TC_SPREAD_COL: str = _tc_cfg["spread_col"]
TC_SIGMA_COL: str = _tc_cfg["sigma_col"]
TC_ADV_COL: str = _tc_cfg["adv_col"]
AUM_SCENARIOS: list[int] = _tc_cfg["aum_scenarios"]


# ── Helpers ──────────────────────────────────────────────────


def _apply_position_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Clip individual weights at `cap` and renormalize to sum=1.

    Uses iterative clip-redistribute: excess weight from capped positions
    is redistributed proportionally to uncapped ones. If all positions
    exceed the cap (too few stocks), returns equal weights.
    """
    n = len(weights)
    if n == 0:
        return weights
    # Cap infeasible if n * cap < 1 (not enough room)
    if n * cap < 1.0:
        return pd.Series(1.0 / n, index=weights.index)

    w = weights / weights.sum()  # ensure sum=1
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if under.sum() == 0:
            break
        w[under] = w[under] + excess * (w[under] / w[under].sum())
    return w


def _assign_quantiles(
    predictions: pd.Series, n_quantiles: int
) -> pd.Series:
    """Assign stocks to quantile bins (1 = lowest, n = highest).

    Always uses rank-based assignment to handle tied predictions
    (common with tree models producing near-constant outputs).
    """
    quantiles = pd.qcut(
        predictions.rank(method="first"),
        q=n_quantiles,
        labels=False,
    ) + 1  # 1-indexed
    return quantiles


# ── Core: Single Month Portfolio ─────────────────────────────


def build_long_short_portfolio(
    df: pd.DataFrame,
    predictions: np.ndarray | pd.Series,
    weighted: bool = False,
    weight_col: str | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Build a long-short decile portfolio for a single cross-section.

    Parameters
    ----------
    df : DataFrame for one month with columns: permno, ret, liq_* columns,
         and weight_* columns (if weighted=True).
    predictions : Model return predictions aligned to df rows.
    weighted : If True, use liquidity weights within deciles.
               If False, equal-weight within deciles.
    weight_col : Weight column name (default: weight_softmax_rank).
    config : Override config dict.

    Returns
    -------
    dict with keys:
        ret_long        — weighted return of long decile
        ret_short       — weighted return of short decile
        ret_long_short  — long minus short return
        n_long          — number of stocks in long decile
        n_short         — number of stocks in short decile
        positions_long  — {permno: weight} for long leg
        positions_short — {permno: weight} for short leg
    """
    if config is None:
        config = _cfg
    pcfg = config["portfolio"]
    n_q = pcfg["n_quantiles"]
    long_q = pcfg["long_quantile"]
    short_q = pcfg["short_quantile"]
    cap = pcfg["position_cap"]
    min_liq_pctile = pcfg["min_liquidity_pctile"]

    if weight_col is None:
        weight_col = DEFAULT_WEIGHT_COL

    work = df.copy()
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions, index=df.index)
    work["_pred"] = predictions.values

    # ── 1. Liquidity filter ──────────────────────────────
    if PRIMARY_LIQ_COL in work.columns:
        liq_threshold = work[PRIMARY_LIQ_COL].quantile(min_liq_pctile)
        liq_mask = work[PRIMARY_LIQ_COL] >= liq_threshold
        n_filtered = (~liq_mask).sum()
        work = work[liq_mask]
        if n_filtered > 0:
            logger.debug(
                "Liquidity filter: removed %d stocks (%.0f%% pctile threshold)",
                n_filtered, min_liq_pctile * 100,
            )

    if len(work) < n_q * 2:
        logger.warning(
            "Too few stocks (%d) after liquidity filter for %d quantiles",
            len(work), n_q,
        )
        return _empty_result()

    # ── 2. Assign quantiles ──────────────────────────────
    work["_quantile"] = _assign_quantiles(work["_pred"], n_q)

    long_df = work[work["_quantile"] == long_q]
    short_df = work[work["_quantile"] == short_q]

    if len(long_df) < 2 or len(short_df) < 2:
        logger.warning(
            "Decile too small: long=%d, short=%d", len(long_df), len(short_df)
        )
        return _empty_result()

    # ── 3. Compute weights within each decile ────────────
    if weighted and weight_col in work.columns:
        w_long = long_df[weight_col] / long_df[weight_col].sum()
        w_short = short_df[weight_col] / short_df[weight_col].sum()
    else:
        w_long = pd.Series(1.0 / len(long_df), index=long_df.index)
        w_short = pd.Series(1.0 / len(short_df), index=short_df.index)

    # ── 4. Apply position cap ────────────────────────────
    w_long = _apply_position_cap(w_long, cap)
    w_short = _apply_position_cap(w_short, cap)

    # ── 5. Compute returns ───────────────────────────────
    ret_long = (w_long * long_df["ret"]).sum()
    ret_short = (w_short * short_df["ret"]).sum()
    ret_ls = ret_long - ret_short

    # ── 6. Store positions (for turnover / TC computation)
    pos_long = dict(zip(long_df["permno"], w_long))
    pos_short = dict(zip(short_df["permno"], w_short))

    return {
        "ret_long": ret_long,
        "ret_short": ret_short,
        "ret_long_short": ret_ls,
        "n_long": len(long_df),
        "n_short": len(short_df),
        "positions_long": pos_long,
        "positions_short": pos_short,
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
    }


# ── High-Level: Full Time Series ────────────────────────────


def build_portfolio_timeseries(
    panel: pd.DataFrame,
    predictions: pd.Series,
    weighted: bool = False,
    weight_col: str | None = None,
    config: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build monthly long-short portfolios over the full test period.

    Uses a no-trade buffer zone: stocks in the previous long/short leg
    are kept if they remain within ``buffer_quantiles`` deciles of the
    target. This reduces turnover from noisy month-to-month re-rankings.

    Parameters
    ----------
    panel : Full test panel with columns: permno, yyyymm, ret, weight_*, liq_*.
    predictions : Model predictions aligned to panel.index.
    weighted : If True, use liquidity weights within deciles.
    weight_col : Weight column name.
    config : Override config dict.

    Returns
    -------
    results_df : DataFrame with monthly rows:
        yyyymm, ret_long, ret_short, ret_long_short, n_long, n_short, turnover
    positions_history : {yyyymm: {"long": {permno: w}, "short": {permno: w}}}
    """
    if config is None:
        config = _cfg
    pcfg = config["portfolio"]
    buffer_q = pcfg.get("buffer_quantiles", 0)

    results: list[dict] = []
    positions_history: dict[int, dict] = {}
    prev_positions: dict[str, dict[int, float]] = {"long": {}, "short": {}}
    prev_long_permnos: set[int] = set()
    prev_short_permnos: set[int] = set()

    for yyyymm, group in panel.groupby("yyyymm"):
        group_preds = predictions.loc[group.index]

        if buffer_q > 0 and (prev_long_permnos or prev_short_permnos):
            month_result = _build_portfolio_with_buffer(
                group, group_preds,
                prev_long_permnos, prev_short_permnos,
                buffer_q=buffer_q,
                weighted=weighted, weight_col=weight_col, config=config,
            )
        else:
            month_result = build_long_short_portfolio(
                group, group_preds, weighted=weighted,
                weight_col=weight_col, config=config,
            )

        # Compute turnover vs previous month
        turnover = _compute_turnover(
            prev_positions["long"], month_result["positions_long"],
        ) + _compute_turnover(
            prev_positions["short"], month_result["positions_short"],
        )

        month_result["yyyymm"] = yyyymm
        month_result["turnover"] = turnover
        results.append(month_result)

        # Update previous positions
        positions_history[yyyymm] = {
            "long": month_result["positions_long"],
            "short": month_result["positions_short"],
        }
        prev_positions = positions_history[yyyymm]
        prev_long_permnos = set(month_result["positions_long"].keys())
        prev_short_permnos = set(month_result["positions_short"].keys())

    results_df = pd.DataFrame(results)
    # Drop position dicts from the DataFrame (kept in positions_history)
    results_df = results_df.drop(
        columns=["positions_long", "positions_short"], errors="ignore"
    )

    logger.info(
        "Portfolio timeseries: %d months, mean LS return=%.4f, "
        "mean turnover=%.2f",
        len(results_df),
        results_df["ret_long_short"].mean(),
        results_df["turnover"].mean(),
    )

    return results_df, positions_history


def _build_portfolio_with_buffer(
    df: pd.DataFrame,
    predictions: np.ndarray | pd.Series,
    prev_long_permnos: set[int],
    prev_short_permnos: set[int],
    buffer_q: int = 1,
    weighted: bool = False,
    weight_col: str | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Build long-short portfolio with no-trade buffer zone.

    Stocks already in the long (short) leg are kept if their new quantile
    is within ``buffer_q`` deciles of the target. New entries must strictly
    qualify for the target decile. This reduces turnover from noisy
    month-to-month re-rankings without distorting the portfolio signal.
    """
    if config is None:
        config = _cfg
    pcfg = config["portfolio"]
    n_q = pcfg["n_quantiles"]
    long_q = pcfg["long_quantile"]
    short_q = pcfg["short_quantile"]
    cap = pcfg["position_cap"]
    min_liq_pctile = pcfg["min_liquidity_pctile"]

    if weight_col is None:
        weight_col = DEFAULT_WEIGHT_COL

    work = df.copy()
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions, index=df.index)
    work["_pred"] = predictions.values

    # Liquidity filter
    if PRIMARY_LIQ_COL in work.columns:
        liq_threshold = work[PRIMARY_LIQ_COL].quantile(min_liq_pctile)
        work = work[work[PRIMARY_LIQ_COL] >= liq_threshold]

    if len(work) < n_q * 2:
        return _empty_result()

    # Assign quantiles
    work["_quantile"] = _assign_quantiles(work["_pred"], n_q)

    # Buffer logic: keep previous holdings if within buffer range
    long_mask = work["_quantile"] == long_q  # strict qualifiers
    short_mask = work["_quantile"] == short_q

    # Existing holdings get a wider acceptance band
    in_prev_long = work["permno"].isin(prev_long_permnos)
    in_prev_short = work["permno"].isin(prev_short_permnos)

    long_buffer_mask = in_prev_long & (work["_quantile"] >= long_q - buffer_q)
    short_buffer_mask = in_prev_short & (work["_quantile"] <= short_q + buffer_q)

    long_df = work[long_mask | long_buffer_mask]
    short_df = work[short_mask | short_buffer_mask]

    if len(long_df) < 2 or len(short_df) < 2:
        return _empty_result()

    # Compute weights
    if weighted and weight_col in work.columns:
        w_long = long_df[weight_col] / long_df[weight_col].sum()
        w_short = short_df[weight_col] / short_df[weight_col].sum()
    else:
        w_long = pd.Series(1.0 / len(long_df), index=long_df.index)
        w_short = pd.Series(1.0 / len(short_df), index=short_df.index)

    w_long = _apply_position_cap(w_long, cap)
    w_short = _apply_position_cap(w_short, cap)

    ret_long = (w_long * long_df["ret"]).sum()
    ret_short = (w_short * short_df["ret"]).sum()

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
    }


def _compute_turnover(
    prev_pos: dict[int, float], curr_pos: dict[int, float]
) -> float:
    """Compute one-way turnover between two position dicts.

    Turnover = Σ|w_i,t - w_i,t-1| summed over all stocks in either period.
    Returns 0.0 if previous positions are empty (first month).
    """
    if not prev_pos:
        return 0.0

    all_permnos = set(prev_pos) | set(curr_pos)
    turnover = sum(
        abs(curr_pos.get(p, 0.0) - prev_pos.get(p, 0.0))
        for p in all_permnos
    )
    return turnover


# ── Transaction Cost Model (Frazzini et al. 2018, Eq. 12) ───


def compute_transaction_costs(
    positions_history: dict[int, dict],
    panel: pd.DataFrame,
    aum: float,
    config: dict | None = None,
) -> pd.Series:
    """Compute monthly transaction costs using Frazzini et al. (2018) Eq. 12.

    TC_i = Spread_i/2 + λ · σ_i · √(Q_i / ADV_i)

    Parameters
    ----------
    positions_history : {yyyymm: {"long": {permno: w}, "short": {permno: w}}}
    panel : Full panel for spread, sigma, adv lookup by (permno, yyyymm).
    aum : Assets under management in dollars.
    config : Override config dict.

    Returns
    -------
    pd.Series indexed by yyyymm with monthly TC as decimal fraction.
    """
    if config is None:
        config = _cfg
    tc_cfg = config["transaction_costs"]
    lam = tc_cfg["lambda_market_impact"]
    spread_col = tc_cfg["spread_col"]
    sigma_col = tc_cfg["sigma_col"]
    adv_col = tc_cfg["adv_col"]
    spread_scale = tc_cfg.get("spread_scale", 1.0)
    spread_cap = tc_cfg.get("spread_cap", 0.05)

    # Build lookup: (permno, yyyymm) → {spread, sigma, adv}
    # Use liq_ prefixed columns if available, fall back to raw
    liq_spread = f"liq_{spread_col}" if f"liq_{spread_col}" in panel.columns else spread_col
    liq_sigma = f"liq_{sigma_col}" if f"liq_{sigma_col}" in panel.columns else sigma_col
    liq_adv = f"liq_{adv_col}" if f"liq_{adv_col}" in panel.columns else adv_col

    lookup_cols = ["permno", "yyyymm"]
    for col in [liq_spread, liq_sigma, liq_adv]:
        if col in panel.columns:
            lookup_cols.append(col)

    lookup = panel[lookup_cols].set_index(["permno", "yyyymm"])

    months = sorted(positions_history.keys())
    tc_series = {}

    for i, yyyymm in enumerate(months):
        pos = positions_history[yyyymm]
        if i == 0:
            # No turnover in first month
            tc_series[yyyymm] = 0.0
            continue

        prev_yyyymm = months[i - 1]
        prev_pos = positions_history[prev_yyyymm]

        total_tc = 0.0

        for leg in ["long", "short"]:
            curr = pos[leg]
            prev = prev_pos[leg]
            all_permnos = set(curr) | set(prev)

            for permno in all_permnos:
                delta_w = abs(curr.get(permno, 0.0) - prev.get(permno, 0.0))
                if delta_w < 1e-10:
                    continue

                # Look up stock characteristics
                try:
                    row = lookup.loc[(permno, yyyymm)]
                except KeyError:
                    # Stock not in panel this month, use conservative defaults
                    total_tc += delta_w * 0.005  # 50bps default
                    continue

                spread = row.get(liq_spread, 0.01)
                sigma = row.get(liq_sigma, 0.02)
                adv = row.get(liq_adv, 1e6)

                # Handle NaN/invalid values
                if pd.isna(spread) or spread <= 0:
                    spread = 0.01
                if pd.isna(sigma) or sigma <= 0:
                    sigma = 0.02
                if pd.isna(adv) or adv <= 0:
                    adv = 1e6

                # Calibrate spread: CZ Corwin-Schultz → effective spread
                spread = spread * spread_scale
                # Clip extreme outliers
                spread = min(spread, spread_cap)

                # Trade size in dollars
                q_i = delta_w * aum

                # Frazzini et al. (2018) Eq. 12
                half_spread = spread / 2.0
                market_impact = lam * sigma * np.sqrt(q_i / adv)
                tc_i = half_spread + market_impact

                # Portfolio cost = |Δw_i| × TC_i
                total_tc += delta_w * tc_i

        tc_series[yyyymm] = total_tc

    return pd.Series(tc_series, name="transaction_cost")


def compute_net_returns(
    gross_returns: pd.DataFrame,
    positions_history: dict[int, dict],
    panel: pd.DataFrame,
    aum: float,
    config: dict | None = None,
) -> pd.DataFrame:
    """Add net-of-cost return columns to gross returns DataFrame.

    Parameters
    ----------
    gross_returns : From build_portfolio_timeseries (has yyyymm, ret_long_short).
    positions_history : Position history from build_portfolio_timeseries.
    panel : Full panel for TC lookup.
    aum : AUM in dollars.
    config : Override config dict.

    Returns
    -------
    DataFrame with additional column: ret_long_short_net.
    """
    tc = compute_transaction_costs(
        positions_history, panel, aum=aum, config=config,
    )

    result = gross_returns.copy()
    result["transaction_cost"] = result["yyyymm"].map(tc).fillna(0.0)
    result["ret_long_short_net"] = result["ret_long_short"] - result["transaction_cost"]
    return result


def compute_net_returns_all_aum(
    gross_returns: pd.DataFrame,
    positions_history: dict[int, dict],
    panel: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Compute net returns for all AUM scenarios from config.

    Returns
    -------
    DataFrame with columns: ret_long_short_net_{aum} for each AUM scenario.
    """
    if config is None:
        config = _cfg
    aum_scenarios = config["transaction_costs"]["aum_scenarios"]

    result = gross_returns.copy()
    for aum in aum_scenarios:
        tc = compute_transaction_costs(
            positions_history, panel, aum=aum, config=config,
        )
        label = f"{aum // 1_000_000}M" if aum < 1_000_000_000 else f"{aum // 1_000_000_000}B"
        col_name = f"ret_ls_net_{label}"
        result[col_name] = result["yyyymm"].map(tc).fillna(0.0)
        result[col_name] = result["ret_long_short"] - result[col_name]

    return result
