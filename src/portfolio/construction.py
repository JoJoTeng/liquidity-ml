"""
Portfolio Construction (LiquidityML v3, Section 9)
===================================================
Prediction-sorted quantile portfolios for the formal framework.

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

TC-target score sort:
  - For a prediction trained on s = r_hat - spread/2, long candidates sort on s.
  - Short candidates sort on s + spread, which recovers r_hat + spread/2 for
    the short-leg cost ranking.

Portfolio rules:
  - Configured prediction quantile sort (currently Q1-Q5)
  - Equal-dollar within each quantile by default; value-weighted quantiles are optional
  - Monthly rebalancing
  - Performance returns use the model target return, currently excess_ret

Net return computation:
  - Uses actual turnover-adjusted trade size DeltaQ_it
    (target position minus drifted position) for market impact,
    NOT the full Q_it = AUM/N_leg.
  - Interprets AUM as total portfolio capital. The legacy long-short helper
    uses AUM/2 per leg; the quantile-first formal helper treats each
    prediction quantile as a standalone long-only portfolio with the full
    scenario AUM.
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
PORTFOLIO_WEIGHTINGS = {"equal", "value", "dolvol", "signal"}
PROPORTIONAL_TC_ALIASES = {"proportional", "proportional_tc", "prop_tc", "spread_only"}


# -- Helpers ----------------------------------------------------------


def _validate_portfolio_mode(portfolio_mode: str) -> str:
    """Validate and normalize the portfolio mode."""
    if portfolio_mode not in PORTFOLIO_MODES:
        raise ValueError(
            f"portfolio_mode must be one of {sorted(PORTFOLIO_MODES)}, "
            f"got {portfolio_mode!r}"
        )
    return portfolio_mode


def _validate_portfolio_weighting(portfolio_weighting: str) -> str:
    """Validate and normalize the selected-stock weighting method."""
    if portfolio_weighting not in PORTFOLIO_WEIGHTINGS:
        raise ValueError(
            f"portfolio_weighting must be one of {sorted(PORTFOLIO_WEIGHTINGS)}, "
            f"got {portfolio_weighting!r}"
        )
    return portfolio_weighting


def _is_proportional_tc_scenario(value: object) -> bool:
    """Return True for the spread-only TC reporting scenario."""
    return isinstance(value, str) and value.strip().lower() in PROPORTIONAL_TC_ALIASES


def _leg_aum(
    aum: float, portfolio_mode: str = "long_short", leg_capital: str = "gross"
) -> float:
    """Capital allocated to the traded leg for TC sizing.

    ``gross`` (default): each leg trades ``aum / 2``, so the long-short book is
    ``$A`` gross (|long| + |short| = $A) — comparable with the capacity book.
    ``full``: each leg trades the full ``aum``, the legacy ``$2A``-gross
    convention ($A long + $A short).
    """
    _validate_portfolio_mode(portfolio_mode)
    if leg_capital == "full":
        return float(aum)
    if leg_capital != "gross":
        raise ValueError(f"leg_capital must be 'gross' or 'full', got {leg_capital!r}")
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


def _selected_leg_weights(
    leg_df: pd.DataFrame,
    portfolio_weighting: str,
    value_weight_col: str,
    dolvol_col: str = "liq_dvol_21d",
    signal_weight_col: str = "w_tilde",
) -> pd.Series:
    """Return selected-leg weights that sum to one.

    ``equal`` gives equal-dollar weights; ``value`` weights by ``value_weight_col``
    (market cap by default); ``dolvol`` weights by ``dolvol_col`` (dollar volume);
    ``signal`` weights by ``signal_weight_col`` (the spec's deployment weight
    w-tilde, which the formal layer merges onto the panel before construction).
    """
    portfolio_weighting = _validate_portfolio_weighting(portfolio_weighting)
    if len(leg_df) == 0:
        return pd.Series(dtype=float)

    if portfolio_weighting == "equal":
        return pd.Series(1.0 / len(leg_df), index=leg_df.index)

    weight_col = {
        "dolvol": dolvol_col,
        "signal": signal_weight_col,
        "value": value_weight_col,
    }.get(portfolio_weighting, value_weight_col)
    if weight_col not in leg_df.columns:
        raise KeyError(f"Selected-leg weight column {weight_col!r} not found")

    raw = pd.to_numeric(leg_df[weight_col], errors="coerce")
    raw = raw.where(np.isfinite(raw) & (raw > 0.0))
    total = raw.sum(skipna=True)
    if not np.isfinite(total) or total <= 0.0:
        return pd.Series(1.0 / len(leg_df), index=leg_df.index)
    return raw.fillna(0.0) / total


# -- Core: Build Single-Month Portfolio -------------------------------


def _apply_hysteresis_bands(
    work: pd.DataFrame,
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    signal_col: str,
    prev_long: set[int],
    prev_short: set[int],
    tc_half_spread: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain held leg members inside a cost-scaled no-trade band (both legs).

    Mirrors the long-only membership hysteresis of ``longonly_capacity`` for a
    long-short book. A held long name (in ``prev_long``) that fell below the fresh
    top-quantile cutoff ``b_hi`` is retained while its signal is within the round-
    trip cost of replacing it: ``r_hat >= b_hi - (hs_i + cm_hi)``, where ``hs_i``
    is the name's own half-spread and ``cm_hi`` the median half-spread of the
    fresh long leg (the entrant toll). The short leg mirrors this at the bottom
    cutoff ``b_lo`` with ``r_hat <= b_lo + (hs_i + cm_lo)``. The band is
    parameter-free and vanishes as half-spreads -> 0 (it then coincides with the
    plain book). Held names absent from ``work`` (left the universe/screen) are
    not retained, i.e. force-sold.
    """
    if tc_half_spread is not None:
        hs = tc_half_spread.reindex(work.index).abs()
        hs = hs.fillna(hs.median() if hs.notna().any() else 0.0)
    else:
        hs = pd.Series(0.0, index=work.index)
    sig = work[signal_col]
    permno = work["permno"]

    if len(long_df):
        b_hi = float(long_df[signal_col].min())
        cm_hi = float(hs.reindex(long_df.index).median())
        held = work[permno.isin(prev_long) & ~work.index.isin(long_df.index)]
        if len(held):
            thr = b_hi - (hs.reindex(held.index) + cm_hi)
            keep = held[sig.reindex(held.index) >= thr]
            long_df = pd.concat([long_df, keep])

    if len(short_df):
        b_lo = float(short_df[signal_col].max())
        cm_lo = float(hs.reindex(short_df.index).median())
        held = work[
            permno.isin(prev_short)
            & ~work.index.isin(short_df.index)
            & ~work.index.isin(long_df.index)  # never hold a name on both legs
        ]
        if len(held):
            thr = b_lo + (hs.reindex(held.index) + cm_lo)
            keep = held[sig.reindex(held.index) <= thr]
            short_df = pd.concat([short_df, keep])

    return long_df, short_df


def build_long_short_portfolio(
    df: pd.DataFrame,
    predictions: pd.Series,
    tc_penalised: bool = False,
    tc_target_score: bool = False,
    tc_per_stock: pd.Series | None = None,
    n_quantiles: int = 5,
    long_quantile: int | None = None,
    short_quantile: int = 1,
    return_col: str = "excess_ret",
    portfolio_mode: str = "long_short",
    portfolio_weighting: str = "equal",
    value_weight_col: str = "liq_me_raw",
    signal_weight_col: str = "w_tilde",
    liquidity_screen_pct: float | None = None,
    liquidity_screen_col: str = "liq_dvol_21d",
    hysteresis: bool = False,
    prev_long: set[int] | None = None,
    prev_short: set[int] | None = None,
    tc_half_spread: pd.Series | None = None,
) -> dict[str, Any]:
    """Build a long-short quantile portfolio for one cross-section.

    Parameters
    ----------
    df : DataFrame for one month with columns: permno and return_col.
    predictions : Model return predictions aligned to df rows.
    tc_penalised : If True, use TC-aware two-sided sorting for Column 2:
        long candidates sort on predictions - tc_per_stock, while short
        candidates sort on predictions + tc_per_stock.
    tc_target_score : If True, treat predictions as a TC-adjusted target score
        ``s = r_hat - tc``. Long candidates sort on ``s``; short candidates
        sort on ``s + 2 * tc``.
    tc_per_stock : One-way proportional cost penalty per stock, currently
        half bid-ask spread only (required for any TC-aware sort).
    n_quantiles : Number of signal quantiles (default 5 for quintiles).
    long_quantile : Quantile to hold long. Defaults to the top quantile.
    short_quantile : Quantile to hold short. Defaults to the bottom quantile.
    return_col : Realized return column used for portfolio performance.
        Defaults to excess_ret to match the model target.
    portfolio_mode : Must be ``"long_short"``.
    portfolio_weighting : ``"equal"`` for equal-dollar selected-leg weights or
        ``"value"`` for market-cap weights within each selected leg.
    value_weight_col : Column used for value-weighted selected-leg weights.

    Returns
    -------
    dict with keys:
        ret_long, ret_short, ret_long_short : gross returns
        n_long, n_short : stock counts
        positions_long, positions_short : {permno: weight} dicts
        permnos_long, permnos_short : sets of permnos in each leg
    """
    _validate_portfolio_mode(portfolio_mode)
    portfolio_weighting = _validate_portfolio_weighting(portfolio_weighting)
    if tc_penalised and tc_target_score:
        raise ValueError("tc_penalised and tc_target_score are mutually exclusive")
    if hysteresis and (tc_penalised or tc_target_score):
        raise ValueError(
            "hysteresis replaces the TC-aware sort; they are mutually exclusive"
        )
    work = df.copy()
    if return_col not in work.columns:
        raise KeyError(f"Portfolio return column {return_col!r} not found")
    work["_pred"] = predictions.values if isinstance(predictions, pd.Series) else predictions

    # -- Sorting signal ------
    two_sided_sort = tc_penalised or tc_target_score
    if two_sided_sort:
        if tc_per_stock is None:
            raise ValueError("tc_per_stock required for TC-aware sort")
        tc = tc_per_stock.reindex(work.index).fillna(0.0)
        if tc_penalised:
            work["_long_signal"] = work["_pred"] - tc
            work["_short_signal"] = work["_pred"] + tc
        else:
            work["_long_signal"] = work["_pred"]
            work["_short_signal"] = work["_pred"] + 2.0 * tc
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

    # -- Liquidity screen (applied before quantile assignment) ------
    if liquidity_screen_pct is not None:
        if liquidity_screen_col not in work.columns:
            raise KeyError(
                f"Liquidity screen column {liquidity_screen_col!r} not found"
            )
        keep = work[liquidity_screen_col].rank(pct=True) >= liquidity_screen_pct
        work = work[keep]

    if len(work) < n_quantiles * 2:
        return _empty_result()

    # -- Security selection ------
    if two_sided_sort:
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
        if hysteresis and (prev_long or prev_short):
            long_df, short_df = _apply_hysteresis_bands(
                work,
                long_df,
                short_df,
                signal_col="_signal",
                prev_long=prev_long or set(),
                prev_short=prev_short or set(),
                tc_half_spread=tc_half_spread,
            )

    if len(long_df) < 2 or len(short_df) < 2:
        return _empty_result()

    # -- Selected-leg weights ------
    w_long = _selected_leg_weights(
        long_df,
        portfolio_weighting=portfolio_weighting,
        value_weight_col=value_weight_col,
        signal_weight_col=signal_weight_col,
    )
    w_short = _selected_leg_weights(
        short_df,
        portfolio_weighting=portfolio_weighting,
        value_weight_col=value_weight_col,
        signal_weight_col=signal_weight_col,
    )

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
    tc_target_score: bool = False,
    aum: float | None = None,
    config: dict | None = None,
    portfolio_mode: str = "long_short",
    portfolio_weighting: str = "equal",
    value_weight_col: str = "liq_me_raw",
    signal_weight_col: str = "w_tilde",
    liquidity_screen_pct: float | None = None,
    liquidity_screen_col: str = "liq_dvol_21d",
    hysteresis: bool = False,
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
    tc_target_score : If True, treat predictions as ``r_hat - spread/2`` scores
        and use the legacy two-sided TC-target long/short ranking.
    aum : AUM in dollars. Required for TC-aware sorting.
        Interpreted as total strategy capital; transaction-cost sizing uses
        half this amount per leg.
    config : Override config dict.
    portfolio_mode : Must be ``"long_short"``.
    portfolio_weighting : ``"equal"`` for equal-dollar selected-leg weights or
        ``"value"`` for market-cap weights within each selected leg.
    value_weight_col : Column used for value-weighted selected-leg weights.

    Returns
    -------
    (returns_df, positions_history)
    returns_df : DataFrame with yyyymm, ret_long, ret_short, ret_long_short.
    positions_history : {yyyymm: {"long": {permno: w}, "short": {permno: w}}}
    """
    portfolio_mode = _validate_portfolio_mode(portfolio_mode)
    portfolio_weighting = _validate_portfolio_weighting(portfolio_weighting)
    if tc_penalised and tc_target_score:
        raise ValueError("tc_penalised and tc_target_score are mutually exclusive")
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
    if tc_penalised or tc_target_score:
        if aum is None:
            raise ValueError("aum required for TC-aware sort")
        from src.weighting.schemes import compute_tc_for_sorting
        tc_for_sort = compute_tc_for_sorting(
            panel_pred,
            aum=_leg_aum(aum, portfolio_mode=portfolio_mode),
            config=config,
        )

    # Half-spread for the hysteresis no-trade band (spread/2, same source as the
    # TC sort; AUM-independent).
    hs_all = None
    if hysteresis:
        from src.weighting.schemes import compute_tc_for_sorting
        hs_all = compute_tc_for_sorting(panel_pred, aum=0.0, config=config)

    pred_months = sorted(panel_pred["yyyymm"].unique())
    records = []
    positions_history = {}
    prev_long: set[int] = set()
    prev_short: set[int] = set()

    for yyyymm in pred_months:
        mask = panel_pred["yyyymm"] == yyyymm
        month_panel = panel_pred[mask]

        if len(month_panel) < n_q * 2:
            continue

        month_preds = pd.Series(
            month_panel["prediction"].values, index=month_panel.index,
        )

        tc_month = None
        if (tc_penalised or tc_target_score) and tc_for_sort is not None:
            tc_month = tc_for_sort.loc[month_panel.index]
        hs_month = hs_all.loc[month_panel.index] if hs_all is not None else None

        result = build_long_short_portfolio(
            df=month_panel,
            predictions=month_preds,
            tc_penalised=tc_penalised,
            tc_target_score=tc_target_score,
            tc_per_stock=tc_month,
            n_quantiles=n_q,
            long_quantile=long_q,
            short_quantile=short_q,
            return_col=return_col,
            portfolio_mode=portfolio_mode,
            portfolio_weighting=portfolio_weighting,
            value_weight_col=value_weight_col,
            signal_weight_col=signal_weight_col,
            liquidity_screen_pct=liquidity_screen_pct,
            liquidity_screen_col=liquidity_screen_col,
            hysteresis=hysteresis,
            prev_long=prev_long,
            prev_short=prev_short,
            tc_half_spread=hs_month,
        )

        result["yyyymm"] = yyyymm
        records.append(result)
        positions_history[yyyymm] = {
            "long": result["positions_long"],
            "short": result["positions_short"],
        }
        if hysteresis:
            prev_long = result["permnos_long"]
            prev_short = result["permnos_short"]

    if not records:
        logger.warning("No valid months for portfolio construction")
        return pd.DataFrame(), {}

    returns_df = pd.DataFrame(records)
    cols = ["yyyymm", "ret_long", "ret_short", "ret_long_short", "n_long", "n_short"]
    returns_df = returns_df[[c for c in cols if c in returns_df.columns]]

    return returns_df, positions_history


def _apply_quantile_hysteresis(
    month_panel: pd.DataFrame,
    score_col: str,
    quantile_col: str,
    prev_quantile: dict[int, int],
    tc_half_spread: pd.Series | None,
    n_q: int,
) -> pd.Series:
    """Sticky per-quantile membership band (one quantile per stock).

    Keep a held name in its prior quantile ``q`` unless its score has moved beyond
    the round-trip cost of switching bins: a held member whose fresh bin differs
    from ``q`` is retained in ``q`` if its score is within ``hs_i + cm_q`` of q's
    edge toward the new bin (``cm_q`` = median half-spread of fresh bin q). Each
    stock stays in exactly one quantile (no adjacent-bin double-counting); the band
    vanishes as half-spreads -> 0. The per-quantile analogue of the long-short
    membership hysteresis, used for the Table-14 'B' device.
    """
    q_fresh = month_panel[quantile_col]
    score = month_panel[score_col]
    if tc_half_spread is not None:
        hs = tc_half_spread.reindex(month_panel.index).abs()
        hs = hs.fillna(hs.median() if hs.notna().any() else 0.0)
    else:
        hs = pd.Series(0.0, index=month_panel.index)

    lo: dict[int, float] = {}
    hi: dict[int, float] = {}
    cm: dict[int, float] = {}
    for q in range(1, n_q + 1):
        bin_q = month_panel[q_fresh == q]
        if len(bin_q):
            lo[q] = float(bin_q[score_col].min())
            hi[q] = float(bin_q[score_col].max())
            cm[q] = float(hs.reindex(bin_q.index).median())

    out = q_fresh.copy()
    permno = month_panel["permno"]
    for idx in month_panel.index:
        prev_q = prev_quantile.get(int(permno.loc[idx]))
        if prev_q is None or prev_q not in lo:
            continue
        fq = int(q_fresh.loc[idx])
        if fq == prev_q:
            continue
        s = float(score.loc[idx])
        h = float(hs.loc[idx])
        if fq < prev_q:  # drifted below prev_q's band
            if s >= lo[prev_q] - (h + cm[prev_q]):
                out.loc[idx] = prev_q
        else:            # drifted above prev_q's band
            if s <= hi[prev_q] + (h + cm[prev_q]):
                out.loc[idx] = prev_q
    return out


def build_prediction_quantile_timeseries(
    panel: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
    sort_mode: str = "prediction",
    aum: float | None = None,
    config: dict | None = None,
    tc_context: dict[str, Any] | None = None,
    portfolio_weighting: str = "equal",
    value_weight_col: str = "liq_me_raw",
    liquidity_screen_pct: float | None = None,
    liquidity_screen_col: str = "liq_dvol_21d",
    proportional_tc_only: bool = False,
    hysteresis: bool = False,
) -> pd.DataFrame:
    """Build monthly prediction-quantile portfolios and net returns.

    ``sort_mode`` controls the score used to assign stocks to quantiles:
    ``"prediction"`` uses raw model predictions, ``"tc_net"`` uses
    prediction minus the one-way TC penalty, and ``"tc_target_score"`` uses
    the supplied TC-target prediction score directly. Each quantile is treated
    as a standalone long-only portfolio and receives the full scenario ``aum``
    for transaction-cost sizing unless ``proportional_tc_only`` is true, in
    which case realized net returns use spread/2 only and ignore market impact.
    """
    portfolio_weighting = _validate_portfolio_weighting(portfolio_weighting)
    if sort_mode not in {"prediction", "tc_net", "tc_target_score"}:
        raise ValueError("sort_mode must be one of prediction, tc_net, tc_target_score")
    if aum is None and not proportional_tc_only:
        raise ValueError("aum is required for quantile portfolio net returns")
    if config is None:
        config = _cfg
    if tc_context is None:
        tc_context = prepare_transaction_cost_context(panel, config)

    portfolio_cfg = config["portfolio"]
    n_q = int(portfolio_cfg["n_quantiles"])
    return_col = config["data"].get("target_col", "excess_ret")
    quantile_aum = 0.0 if proportional_tc_only else float(aum)

    pred_df = _predictions_to_frame(panel, predictions)
    panel_pred = panel.merge(pred_df, on=["permno", "yyyymm"], how="inner")
    panel_pred = panel_pred.reset_index(drop=True)

    if return_col not in panel_pred.columns:
        raise KeyError(f"Portfolio return column {return_col!r} not found")

    score = panel_pred["prediction"].astype(float).copy()
    if sort_mode == "tc_net":
        from src.weighting.schemes import compute_tc_for_sorting

        score = score - compute_tc_for_sorting(
            panel_pred,
            aum=quantile_aum,
            config=config,
        )
    panel_pred["_quantile_score"] = score

    hs_all = None
    if hysteresis:
        from src.weighting.schemes import compute_tc_for_sorting
        hs_all = compute_tc_for_sorting(panel_pred, aum=0.0, config=config)

    records: list[dict[str, Any]] = []
    positions_by_quantile: dict[int, dict[int, dict[str, dict[int, float]]]] = {
        q: {} for q in range(1, n_q + 1)
    }
    prev_quantile: dict[int, int] = {}

    for yyyymm, month_panel in panel_pred.groupby("yyyymm", sort=True):
        if len(month_panel) < n_q * 2:
            continue

        month_panel = month_panel.copy()
        if liquidity_screen_pct is not None:
            if liquidity_screen_col not in month_panel.columns:
                raise KeyError(
                    f"Liquidity screen column {liquidity_screen_col!r} not found"
                )
            keep = (
                month_panel[liquidity_screen_col].rank(pct=True)
                >= liquidity_screen_pct
            )
            month_panel = month_panel[keep]
            if len(month_panel) < n_q * 2:
                continue
        month_panel["_prediction_quantile"] = _assign_quantiles(
            month_panel["_quantile_score"],
            n_q,
        )
        if hysteresis and prev_quantile:
            hs_month = (
                hs_all.loc[month_panel.index] if hs_all is not None else None
            )
            month_panel["_prediction_quantile"] = _apply_quantile_hysteresis(
                month_panel,
                score_col="_quantile_score",
                quantile_col="_prediction_quantile",
                prev_quantile=prev_quantile,
                tc_half_spread=hs_month,
                n_q=n_q,
            )
        if hysteresis:
            prev_quantile = dict(
                zip(
                    month_panel["permno"].astype(int),
                    month_panel["_prediction_quantile"].astype(int),
                )
            )

        for q in range(1, n_q + 1):
            q_df = month_panel[month_panel["_prediction_quantile"] == q].copy()
            if len(q_df) < 2:
                continue

            weights = _selected_leg_weights(
                q_df,
                portfolio_weighting=portfolio_weighting,
                value_weight_col=value_weight_col,
            )
            gross_return = float((weights * q_df[return_col]).sum())
            positions = dict(zip(q_df["permno"], weights))
            records.append(
                {
                    "yyyymm": yyyymm,
                    "prediction_quantile": q,
                    "gross_return": gross_return,
                    "n_stocks": int(len(q_df)),
                    "weight_sum": float(weights.sum()),
                    "score_mean": float(q_df["_quantile_score"].mean()),
                }
            )
            positions_by_quantile[q][yyyymm] = {
                "long": positions,
                "short": {},
            }

    if not records:
        logger.warning("No valid months for prediction-quantile portfolio construction")
        return pd.DataFrame()

    gross = pd.DataFrame(records)
    frames = []
    for q in range(1, n_q + 1):
        gross_q = gross[gross["prediction_quantile"] == q].copy()
        if gross_q.empty:
            continue

        net_input = gross_q[["yyyymm", "gross_return"]].rename(
            columns={"gross_return": "ret_long_short"}
        )
        net_q = _compute_net_returns_with_context(
            net_input,
            positions_by_quantile[q],
            aum=0.0 if proportional_tc_only else float(aum),
            tc_context=tc_context,
            leg_aum_override=quantile_aum,
            proportional_tc_only=proportional_tc_only,
        )
        out = gross_q.merge(
            net_q[
                [
                    "yyyymm",
                    "transaction_cost",
                    "raw_trade_sum",
                    "turnover",
                    "ret_long_short_net",
                ]
            ],
            on="yyyymm",
            how="left",
        )
        out["net_return"] = out["ret_long_short_net"]
        frames.append(out.drop(columns=["ret_long_short_net"]))

    return pd.concat(frames, ignore_index=True).sort_values(
        ["yyyymm", "prediction_quantile"]
    )


def _predictions_to_frame(
    panel: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    """Normalize predictions to permno, yyyymm, prediction columns."""
    if isinstance(predictions, pd.Series):
        common = panel.index.intersection(predictions.index)
        return pd.DataFrame({
            "permno": panel.loc[common, "permno"].values,
            "yyyymm": panel.loc[common, "yyyymm"].values,
            "prediction": predictions.loc[common].values,
        })
    return predictions[["permno", "yyyymm", "prediction"]].copy()


# -- Transaction Cost: Turnover-Adjusted (Net Returns) ----------------


def compute_net_returns(
    gross_returns: pd.DataFrame,
    positions_history: dict[int, dict],
    panel: pd.DataFrame,
    aum: float,
    config: dict | None = None,
    tc_context: dict[str, Any] | None = None,
    portfolio_mode: str = "long_short",
    proportional_tc_only: bool = False,
    leg_capital: str = "gross",
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
    uses ``aum / 2`` for each leg. If ``proportional_tc_only`` is true, only
    spread/2 is applied and the market-impact term is set to zero.
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
        proportional_tc_only=proportional_tc_only,
        leg_capital=leg_capital,
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
    leg_aum_override: float | None = None,
    proportional_tc_only: bool = False,
    leg_capital: str = "gross",
) -> pd.DataFrame:
    """Compute net returns from a prebuilt transaction-cost context."""
    portfolio_mode = _validate_portfolio_mode(portfolio_mode)
    lookup = tc_context["lookup"]
    spread_col = tc_context["spread_col"]
    sigma_col = tc_context["sigma_col"]
    adv_col = tc_context["adv_col"]
    lam = tc_context["lam"]
    leg_aum = (
        float(leg_aum_override)
        if leg_aum_override is not None
        else _leg_aum(aum, portfolio_mode=portfolio_mode, leg_capital=leg_capital)
    )

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

                # Dollar trade amount; only used for market-impact sizing.
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

                # Frazzini et al. Eq. 25 with actual DeltaQ. The
                # proportional-only reporting case keeps only spread/2.
                half_spread = spread / 2.0
                market_impact = (
                    0.0
                    if proportional_tc_only
                    else lam * sigma * np.sqrt(delta_q / adv)
                )
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
    leg_capital: str = "gross",
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
        proportional_tc_only = _is_proportional_tc_scenario(aum)
        sizing_aum = 0.0 if proportional_tc_only else float(aum)
        net_df = compute_net_returns(
            gross_returns,
            positions_history,
            panel,
            aum=sizing_aum,
            config=config,
            tc_context=tc_context,
            portfolio_mode=portfolio_mode,
            proportional_tc_only=proportional_tc_only,
            leg_capital=leg_capital,
        )
        label = (
            "PropTC"
            if proportional_tc_only
            else (
                f"{int(aum // 1_000_000)}M"
                if aum < 1_000_000_000
                else f"{int(aum // 1_000_000_000)}B"
            )
        )
        result[f"ret_ls_net_{label}"] = net_df["ret_long_short_net"]

    return result
