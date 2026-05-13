"""
Portfolio Construction (LiquidityML v3, Section 9)
===================================================
Long-short quantile portfolios for the 2x2 experimental framework.

The 2x2 design:
                    Sort on r_hat       TC-aware sort
  Standard train    1A (Baseline)       1B
  Weighted train    2A                  2B (Combined)

Column 1: standard sort on predicted returns.
Column 2: TC-penalised sort:
  - Long candidates sort on r_hat - spread/2.
  - Short candidates sort on r_hat + spread/2, because short-leg expected
    profit is -r_hat - spread/2.
  No explicit liquidity filter - the TC penalty naturally pushes
  illiquid stocks down the ranking.

Portfolio rules:
  - Configured quantile sort (currently long Q5, short Q1)
  - Equal-dollar within each leg
  - Monthly rebalancing
  - Performance returns use the model target return, currently excess_ret

Net return computation:
  - Uses actual turnover-adjusted trade size DeltaQ_it
    (target position minus drifted position) for market impact,
    NOT the full Q_it = AUM/N_leg.
  - Interprets AUM as total gross strategy capital. The long-short portfolio
    uses AUM/2 per leg.
  - Tracks position drift from raw returns between rebalancing months.

Transaction cost model: Frazzini et al. (2018) Eq. 25.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.weighting.schemes import resolve_market_impact_lambda

logger = logging.getLogger(__name__)

_cfg = load_config()
PORTFOLIO_MODES = {"long_short"}


# -- Helpers ----------------------------------------------------------


def _validate_portfolio_mode(portfolio_mode: str) -> str:
    """Validate and normalize the portfolio mode."""
    if portfolio_mode not in PORTFOLIO_MODES:
        raise ValueError(
            f"portfolio_mode must be one of {sorted(PORTFOLIO_MODES)}, "
            f"got {portfolio_mode!r}"
        )
    return portfolio_mode


def _leg_aum(aum: float, portfolio_mode: str = "long_short") -> float:
    """Capital allocated to the traded leg for TC sizing."""
    _validate_portfolio_mode(portfolio_mode)
    return float(aum) / 2.0


def _assign_quantiles(values: pd.Series, n_quantiles: int = 5) -> pd.Series:
    """Assign stocks to rank quantile bins (1 = lowest, n = highest).

    Uses rank-based assignment to handle tied predictions.
    """
    return pd.qcut(
        values.rank(method="first"),
        q=n_quantiles,
        labels=False,
    ) + 1  # 1-indexed


_assign_deciles = _assign_quantiles


def _select_quantile(
    work: pd.DataFrame,
    signal_col: str,
    quantile: int,
    n_quantiles: int,
    exclude_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Select one signal quantile, optionally filling around excluded names."""
    quantiles = _assign_quantiles(work[signal_col], n_quantiles)
    selected_index = quantiles[quantiles == quantile].index
    target_n = len(selected_index)

    if exclude_index is not None and len(exclude_index) > 0:
        selected_index = selected_index.difference(exclude_index, sort=False)
        fill_n = target_n - len(selected_index)
        if fill_n > 0:
            used_index = selected_index.union(exclude_index, sort=False)
            fill_pool = work.loc[work.index.difference(used_index, sort=False)]
            ascending = quantile <= (n_quantiles + 1) / 2
            fill_index = fill_pool.sort_values(
                signal_col,
                ascending=ascending,
            ).index[:fill_n]
            selected_index = selected_index.append(fill_index)

    return work.loc[selected_index]


# -- Core: Build Single-Month Portfolio -------------------------------


def build_long_short_portfolio(
    df: pd.DataFrame,
    predictions: pd.Series,
    tc_penalised: bool = False,
    tc_per_stock: pd.Series | None = None,
    n_quantiles: int = 5,
    long_quantile: int | None = None,
    short_quantile: int = 1,
    return_col: str = "excess_ret",
    portfolio_mode: str = "long_short",
) -> dict[str, Any]:
    """Build a long-short quantile portfolio for one cross-section.

    Parameters
    ----------
    df : DataFrame for one month with columns: permno and return_col.
    predictions : Model return predictions aligned to df rows.
    tc_penalised : If True, use TC-aware two-sided sorting for Column 2:
        long candidates sort on predictions - tc_per_stock, while short
        candidates sort on predictions + tc_per_stock.
    tc_per_stock : One-way proportional cost penalty per stock, currently
        half bid-ask spread only (required if tc_penalised=True).
    n_quantiles : Number of signal quantiles (default 5 for quintiles).
    long_quantile : Quantile to hold long. Defaults to the top quantile.
    short_quantile : Quantile to hold short. Defaults to the bottom quantile.
    return_col : Realized return column used for portfolio performance.
        Defaults to excess_ret to match the model target.
    portfolio_mode : Must be ``"long_short"``.

    Returns
    -------
    dict with keys:
        ret_long, ret_short, ret_long_short : gross returns
        n_long, n_short : stock counts
        positions_long, positions_short : {permno: weight} dicts
        permnos_long, permnos_short : sets of permnos in each leg
    """
    _validate_portfolio_mode(portfolio_mode)
    work = df.copy()
    if return_col not in work.columns:
        raise KeyError(f"Portfolio return column {return_col!r} not found")
    work["_pred"] = predictions.values if isinstance(predictions, pd.Series) else predictions

    # -- Sorting signal ------
    if tc_penalised:
        if tc_per_stock is None:
            raise ValueError("tc_per_stock required for TC-penalised sort")
        tc = tc_per_stock.reindex(work.index).fillna(0.0)
        work["_long_signal"] = work["_pred"] - tc
        work["_short_signal"] = work["_pred"] + tc
    else:
        work["_signal"] = work["_pred"]

    if long_quantile is None:
        long_quantile = n_quantiles
    if not (1 <= long_quantile <= n_quantiles):
        raise ValueError("long_quantile must be between 1 and n_quantiles")
    if not (1 <= short_quantile <= n_quantiles):
        raise ValueError("short_quantile must be between 1 and n_quantiles")
    if long_quantile == short_quantile:
        raise ValueError("long_quantile and short_quantile must differ")

    if len(work) < n_quantiles * 2:
        return _empty_result()

    # -- Security selection ------
    if tc_penalised:
        long_df = _select_quantile(
            work,
            signal_col="_long_signal",
            quantile=long_quantile,
            n_quantiles=n_quantiles,
        )
        short_df = _select_quantile(
            work,
            signal_col="_short_signal",
            quantile=short_quantile,
            n_quantiles=n_quantiles,
            exclude_index=long_df.index,
        )
    else:
        work["_rank_quantile"] = _assign_quantiles(work["_signal"], n_quantiles)
        long_df = work[work["_rank_quantile"] == long_quantile]
        short_df = work[work["_rank_quantile"] == short_quantile]

    if len(long_df) < 2 or len(short_df) < 2:
        return _empty_result()

    # -- Equal-dollar weights ------
    w_long = pd.Series(1.0 / len(long_df), index=long_df.index)
    w_short = pd.Series(1.0 / len(short_df), index=short_df.index)

    # -- Gross returns ------
    ret_long = (w_long * long_df[return_col]).sum()
    ret_short = (w_short * short_df[return_col]).sum()
    portfolio_ret = ret_long - ret_short

    # -- Store positions for TC computation ------
    pos_long = dict(zip(long_df["permno"], w_long))
    pos_short = dict(zip(short_df["permno"], w_short))

    return {
        "ret_long": ret_long,
        "ret_short": ret_short,
        "ret_long_short": portfolio_ret,
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
    portfolio_mode: str = "long_short",
) -> tuple[pd.DataFrame, dict]:
    """Build monthly long-short portfolios over the full test period.

    Parameters
    ----------
    panel : Full panel with yyyymm, permno, model target return, ret, liq_* columns.
    predictions : Either
        - pd.DataFrame with columns [permno, yyyymm, prediction], or
        - pd.Series indexed by panel row labels (legacy support).
        Merged with panel via (permno, yyyymm) keys.
    tc_penalised : If True, use TC-penalised sort (Column 2).
    aum : AUM in dollars. Required if tc_penalised=True.
        Interpreted as total strategy capital; transaction-cost sizing uses
        half this amount per leg.
    config : Override config dict.
    portfolio_mode : Must be ``"long_short"``.

    Returns
    -------
    (returns_df, positions_history)
    returns_df : DataFrame with yyyymm, ret_long, ret_short, ret_long_short.
    positions_history : {yyyymm: {"long": {permno: w}, "short": {permno: w}}}
    """
    portfolio_mode = _validate_portfolio_mode(portfolio_mode)
    if config is None:
        config = _cfg
    portfolio_cfg = config["portfolio"]
    n_q = portfolio_cfg["n_quantiles"]
    long_q = portfolio_cfg.get("long_quantile", n_q)
    short_q = portfolio_cfg.get("short_quantile", 1)
    return_col = config["data"].get("target_col", "excess_ret")

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
        tc_for_sort = compute_tc_for_sorting(
            panel_pred,
            aum=_leg_aum(aum, portfolio_mode=portfolio_mode),
            config=config,
        )

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
            long_quantile=long_q,
            short_quantile=short_q,
            return_col=return_col,
            portfolio_mode=portfolio_mode,
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
    tc_context: dict[str, Any] | None = None,
    portfolio_mode: str = "long_short",
) -> pd.DataFrame:
    """Compute net returns using turnover-adjusted trade sizes.

    For each stock held or traded:
      1. Compute drifted position from last month's holding + actual return
      2. DeltaQ_it = |target_position - drifted_position|
      3. TC_it = Spread/2 + lambda * sigma * sqrt(DeltaQ_it / ADV)
      4. Portfolio TC = sum over all traded stocks

    The returned ``raw_trade_sum`` is the long-leg plus short-leg sum of
    absolute weight changes. Reported ``turnover`` follows the standard
    convention, ``0.5 * raw_trade_sum``; transaction costs still use the full
    raw trade sum because one-way costs are paid on both buys and sells.

    ``aum`` is interpreted as total strategy capital. The long-short portfolio
    uses ``aum / 2`` for each leg.
    """
    portfolio_mode = _validate_portfolio_mode(portfolio_mode)
    if config is None:
        config = _cfg
    if tc_context is None:
        tc_context = prepare_transaction_cost_context(panel, config)
    return _compute_net_returns_with_context(
        gross_returns,
        positions_history,
        aum=aum,
        tc_context=tc_context,
        portfolio_mode=portfolio_mode,
    )


def prepare_transaction_cost_context(
    panel: pd.DataFrame,
    config: dict | None = None,
) -> dict[str, Any]:
    """Prepare reusable transaction-cost lookup data for net-return calls."""
    if config is None:
        config = _cfg
    tc_cfg = config["transaction_costs"]
    spread_col = f"liq_{tc_cfg['spread_col']}"
    sigma_col = f"liq_{tc_cfg['sigma_col']}"
    adv_col = f"liq_{tc_cfg['adv_col']}"
    lam = resolve_market_impact_lambda(panel, config, sigma_col=sigma_col)

    # Build lookup: (permno, yyyymm) -> {spread, sigma, adv, ret}
    lookup_cols = ["permno", "yyyymm", "ret"]
    for col in [spread_col, sigma_col, adv_col]:
        if col in panel.columns:
            lookup_cols.append(col)
    lookup = panel[list(dict.fromkeys(lookup_cols))].copy()
    lookup = lookup.set_index(["permno", "yyyymm"])

    return {
        "lookup": lookup,
        "spread_col": spread_col,
        "sigma_col": sigma_col,
        "adv_col": adv_col,
        "lam": lam,
    }


def _compute_net_returns_with_context(
    gross_returns: pd.DataFrame,
    positions_history: dict[int, dict],
    aum: float,
    tc_context: dict[str, Any],
    portfolio_mode: str = "long_short",
) -> pd.DataFrame:
    """Compute net returns from a prebuilt transaction-cost context."""
    portfolio_mode = _validate_portfolio_mode(portfolio_mode)
    lookup = tc_context["lookup"]
    spread_col = tc_context["spread_col"]
    sigma_col = tc_context["sigma_col"]
    adv_col = tc_context["adv_col"]
    lam = tc_context["lam"]
    leg_aum = _leg_aum(aum, portfolio_mode=portfolio_mode)

    months = sorted(positions_history.keys())
    tc_series = {}
    raw_trade_sum_series = {}
    turnover_series = {}

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
            drifted_long = _drift_positions(prev_pos_long, prev_yyyymm, lookup)
            drifted_short = _drift_positions(prev_pos_short, prev_yyyymm, lookup)

        # Compute TC for each leg
        total_tc = 0.0
        raw_trade_sum = 0.0
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

                raw_trade_sum += delta_w

                # Dollar trade amount
                delta_q = delta_w * leg_aum

                # Look up stock characteristics. If an eliminated holding is
                # absent from the current cross-section, use its previous-month
                # TC inputs rather than a blunt default.
                fallback_yyyymm = None
                if target_w == 0.0 and drifted_w > 0.0 and i > 0:
                    fallback_yyyymm = months[i - 1]
                try:
                    row = _lookup_trade_row(lookup, permno, yyyymm, fallback_yyyymm)
                except KeyError:
                    total_tc += delta_w * 0.005  # 50bps default
                    continue

                spread = _safe_val(row, spread_col, 0.01)
                sigma = _safe_val(row, sigma_col, 0.02)
                adv = _safe_val(row, adv_col, 1e6)

                spread = abs(spread)

                # Frazzini et al. Eq. 25 with actual DeltaQ
                half_spread = spread / 2.0
                market_impact = lam * sigma * np.sqrt(delta_q / adv)
                tc_i = half_spread + market_impact

                total_tc += delta_w * tc_i

        tc_series[yyyymm] = total_tc
        raw_trade_sum_series[yyyymm] = raw_trade_sum
        turnover_series[yyyymm] = 0.5 * raw_trade_sum

        # Store current positions for next month's drift
        prev_pos_long = target_long.copy()
        prev_pos_short = target_short.copy()

    # Merge TC into gross returns
    result = gross_returns.copy()
    result["transaction_cost"] = result["yyyymm"].map(tc_series).fillna(0.0)
    result["raw_trade_sum"] = result["yyyymm"].map(raw_trade_sum_series).fillna(0.0)
    result["turnover"] = result["yyyymm"].map(turnover_series).fillna(0.0)
    result["ret_long_short_net"] = result["ret_long_short"] - result["transaction_cost"]
    return result


def _lookup_trade_row(
    lookup: pd.DataFrame,
    permno: int,
    yyyymm: int,
    fallback_yyyymm: int | None = None,
):
    """Lookup TC inputs, optionally falling back for eliminated holdings."""
    try:
        return lookup.loc[(permno, yyyymm)]
    except KeyError:
        if fallback_yyyymm is not None:
            return lookup.loc[(permno, fallback_yyyymm)]
        raise


def _drift_positions(
    positions: dict[int, float],
    from_yyyymm: int,
    lookup: pd.DataFrame,
) -> dict[int, float]:
    """Compute pre-rebalance weights after holding-period return drift.

    ``load_panel`` forward-shifts ``ret``, so ``lookup[(permno, from_yyyymm)]``
    is the raw return earned between ``from_yyyymm`` and the next rebalance.
    The final division normalises within the leg, matching
    w_i(1 + r_i) / sum_j w_j(1 + r_j).
    """
    if not positions:
        return {}

    drifted = {}
    leg_value_after_returns = 0.0

    for permno, w in positions.items():
        try:
            row = lookup.loc[(permno, from_yyyymm)]
            ret = row["ret"] if not pd.isna(row["ret"]) else 0.0
        except KeyError:
            ret = 0.0
        value_after_return = w * (1.0 + ret)
        drifted[permno] = value_after_return
        leg_value_after_returns += value_after_return

    if leg_value_after_returns > 0:
        for permno in drifted:
            drifted[permno] /= leg_value_after_returns

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
    portfolio_mode: str = "long_short",
) -> pd.DataFrame:
    """Compute net returns for all AUM scenarios from config.

    Returns
    -------
    DataFrame with columns: ret_ls_net_{aum_label} for each total-gross AUM.
    """
    portfolio_mode = _validate_portfolio_mode(portfolio_mode)
    if config is None:
        config = _cfg
    aum_scenarios = config["transaction_costs"]["aum_scenarios"]
    tc_context = prepare_transaction_cost_context(panel, config)

    result = gross_returns.copy()
    for aum in aum_scenarios:
        net_df = compute_net_returns(
            gross_returns,
            positions_history,
            panel,
            aum=aum,
            config=config,
            tc_context=tc_context,
            portfolio_mode=portfolio_mode,
        )
        label = f"{aum // 1_000_000}M" if aum < 1_000_000_000 else f"{aum // 1_000_000_000}B"
        result[f"ret_ls_net_{label}"] = net_df["ret_long_short_net"]

    return result
