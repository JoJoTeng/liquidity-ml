"""
Implementability Weighting Schemes (LiquidityML v3)
====================================================
Two weighting families for importance-weighted ML training.

Family 1 - Dollar Volume (Eq. 21):
    w_it = DolVol_it / median(DolVol_t)
    Motivated by covariate shift: proxy for density ratio.
    AUM-independent.

Family 2 - Transaction-Cost-Based (Eq. 23):
    w_it = exp(-alpha_t * TC_it)
    where TC follows Frazzini et al. (2018) and alpha_t = median(TC_t).
    Motivated by cost-sensitive learning. AUM-dependent (via trade size).

Both schemes:
  - Operate per cross-section (within each yyyymm)
  - Normalize weights to mean=1.0 within each cross-section (Option A)
  - Handle NaN by assigning neutral weight (~1.0)
  - Return a pd.Series aligned to the input DataFrame index
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -- Helper -----------------------------------------------------------


def _normalize_to_mean_one(weights: pd.Series) -> pd.Series:
    """Rescale weights so their mean equals 1.0.

    This is the Option A normalisation (pre-normalise before passing
    to sample_weight). Ensures the data-fitting vs. regularisation
    balance in XGBoost/NN is identical to the unweighted case.
    """
    mean_w = weights.mean()
    if mean_w == 0 or np.isnan(mean_w):
        return pd.Series(1.0, index=weights.index)
    return weights / mean_w


# -- Family 1: Dollar Volume Weights (Eq. 21) ------------------------


def _dolvol_weights(group: pd.DataFrame, dolvol_col: str) -> pd.Series:
    """w_it = DolVol_it / median(DolVol_t), normalised to mean=1.

    Parameters
    ----------
    group : single cross-section (one yyyymm).
    dolvol_col : column name for dollar volume (e.g. 'liq_dvol_21d').
    """
    liq = group[dolvol_col].copy()
    median_val = liq.median()
    if np.isnan(median_val) or median_val <= 0:
        median_val = 1.0
    # NaN -> median (neutral); clip to avoid zero/negative
    liq = liq.fillna(median_val).clip(lower=1e-8)
    # Eq. 21: divide by median
    raw = liq / median_val
    return _normalize_to_mean_one(raw)


# -- Family 2: Transaction-Cost-Based Weights (Eq. 23) ---------------


def _compute_tc_per_stock(
    group: pd.DataFrame,
    aum: float,
    lam: float,
    spread_col: str,
    sigma_col: str,
    adv_col: str,
) -> pd.Series:
    """Compute one-way TC per stock (Frazzini et al. 2018, Eq. 22/25).

    TC_it = Spread_it/2 + lambda * sigma_it * sqrt(Q_it / ADV_it)

    where Q_it = AUM / N (equal-dollar trade size, N = number of stocks
    in the cross-section - used as approximation at the training stage).
    """
    n_stocks = len(group)
    if n_stocks == 0:
        return pd.Series(dtype=float)
    q_per_stock = aum / n_stocks

    spread = group[spread_col].copy()
    sigma = group[sigma_col].copy()
    adv = group[adv_col].copy()

    # Fill missing with cross-sectional median
    spread = spread.fillna(spread.median()).abs().clip(lower=0.0)
    sigma = sigma.fillna(sigma.median()).abs().clip(lower=1e-8)
    adv = adv.fillna(adv.median()).clip(lower=1e3)

    half_spread = spread / 2.0
    market_impact = lam * sigma * np.sqrt(q_per_stock / adv)
    tc = half_spread + market_impact
    return tc


def _tc_weights(
    group: pd.DataFrame,
    aum: float,
    lam: float,
    spread_col: str,
    sigma_col: str,
    adv_col: str,
) -> pd.Series:
    """w_it = exp(-alpha_t * TC_it), normalised to mean=1.

    Eq. 23: alpha_t = median(TC_it) controls the decay rate.

    Parameters
    ----------
    group : single cross-section (one yyyymm).
    aum : assets under management in dollars.
    lam : market impact coefficient (lambda = 0.1).
    spread_col, sigma_col, adv_col : column names for TC inputs.
    """
    tc = _compute_tc_per_stock(group, aum, lam, spread_col, sigma_col, adv_col)
    if len(tc) == 0:
        return pd.Series(dtype=float)

    alpha_t = tc.median()
    if np.isnan(alpha_t) or alpha_t <= 0:
        alpha_t = 1.0

    raw = np.exp(-alpha_t * tc)
    return _normalize_to_mean_one(raw)


# -- Registry ---------------------------------------------------------

AVAILABLE_SCHEMES = ["dolvol", "tc"]


def get_available_schemes() -> list[str]:
    """Return list of available weighting scheme names."""
    return AVAILABLE_SCHEMES.copy()


# -- Public API -------------------------------------------------------


def compute_weights(
    df: pd.DataFrame,
    scheme: str,
    config: dict,
    aum: float | None = None,
) -> pd.Series:
    """Compute implementability weights for a panel (multiple months).

    Weights are computed independently per cross-section (yyyymm)
    and normalised to mean=1.0 within each cross-section.

    Parameters
    ----------
    df : DataFrame with 'yyyymm' and required liquidity columns.
    scheme : 'dolvol' (Eq. 21) or 'tc' (Eq. 23).
    config : full config dict (for column names, lambda, etc.).
    aum : AUM in dollars. Required if scheme='tc'.

    Returns
    -------
    pd.Series of weights aligned to df.index, mean ~1.0 per cross-section.
    """
    if scheme not in AVAILABLE_SCHEMES:
        raise ValueError(
            f"Unknown scheme: {scheme!r}. Available: {AVAILABLE_SCHEMES}"
        )

    liq_cfg = config["liquidity"]
    tc_cfg = config["transaction_costs"]
    dolvol_col = f"liq_{liq_cfg['primary']}"

    if scheme == "tc" and aum is None:
        raise ValueError("aum is required for scheme='tc'")

    spread_col = f"liq_{tc_cfg['spread_col']}"
    sigma_col = f"liq_{tc_cfg['sigma_col']}"
    adv_col = f"liq_{tc_cfg['adv_col']}"
    lam = tc_cfg["lambda_market_impact"]

    parts: list[pd.Series] = []
    for yyyymm, group in df.groupby("yyyymm"):
        if scheme == "dolvol":
            w = _dolvol_weights(group, dolvol_col)
        else:  # tc
            w = _tc_weights(
                group, aum=aum, lam=lam,
                spread_col=spread_col, sigma_col=sigma_col, adv_col=adv_col,
            )
        parts.append(w)

    weights = pd.concat(parts)
    weights = weights.loc[df.index]

    logger.info(
        "compute_weights(scheme=%s, aum=%s): %d obs, %d months, "
        "mean=%.3f, std=%.3f, min=%.4f, max=%.4f",
        scheme,
        f"${aum/1e6:.0f}M" if aum else "N/A",
        len(weights),
        df["yyyymm"].nunique(),
        weights.mean(),
        weights.std(),
        weights.min(),
        weights.max(),
    )
    return weights


def compute_tc_for_sorting(
    df: pd.DataFrame,
    aum: float,
    config: dict,
) -> pd.Series:
    """Compute per-stock TC for the TC-penalised sort (Column 2).

    Uses Q_it = AUM / N_leg (estimated from N_total / 10 for decile sort)
    since the portfolio is not yet known at the sorting stage.

    Parameters
    ----------
    df : single cross-section or panel with yyyymm + liquidity columns.
    aum : AUM in dollars.
    config : full config dict.

    Returns
    -------
    pd.Series of one-way TC per stock, aligned to df.index.
    """
    tc_cfg = config["transaction_costs"]
    lam = tc_cfg["lambda_market_impact"]
    spread_col = f"liq_{tc_cfg['spread_col']}"
    sigma_col = f"liq_{tc_cfg['sigma_col']}"
    adv_col = f"liq_{tc_cfg['adv_col']}"

    parts: list[pd.Series] = []
    for yyyymm, group in df.groupby("yyyymm"):
        tc = _compute_tc_per_stock(
            group, aum=aum, lam=lam,
            spread_col=spread_col, sigma_col=sigma_col, adv_col=adv_col,
        )
        parts.append(tc)

    tc_all = pd.concat(parts)
    return tc_all.loc[df.index]
