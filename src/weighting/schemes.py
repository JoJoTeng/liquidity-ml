"""
Implementability Weighting Schemes (LiquidityML v3)
====================================================
Three weighting families for importance-weighted ML training.

Family 1 - Dollar Volume:
    w_it = DolVol_it / mean(DolVol_t)
    Motivated by covariate shift: proxy for density ratio.
    AUM-independent.

Family 2 - Softmax on Dollar-Volume Rank:
    w_it = exp(lambda * rank_it) / mean(exp(lambda * rank_t))
    where rank_it is the within-month percentile rank of DolVol.
    AUM-independent and smoother than raw dollar-volume weights.

Family 3 - Transaction-Cost-Based (Eq. 23):
    w_it = exp(-alpha_t * TC_it)
    where TC follows Frazzini et al. (2018) and
    alpha_t = scale / median(TC_t).
    Motivated by cost-sensitive learning. AUM-dependent (via trade size).

Both schemes:
  - Operate per cross-section (within each yyyymm)
  - Keep weights mean-normalized within each cross-section
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


def resolve_market_impact_lambda(
    df: pd.DataFrame,
    config: dict,
    sigma_col: str | None = None,
) -> float:
    """Resolve the market-impact lambda, optionally using PDF-style calibration.

    If ``transaction_costs.lambda_calibration.enabled`` is true, lambda is set so
    that a trade of ``anchor_trade_frac_adv`` ADV produces ``anchor_impact`` for
    a stock with average volatility:

        anchor_impact = lambda * sigma_bar * sqrt(anchor_trade_frac_adv)

    Otherwise this returns the fixed ``lambda_market_impact`` from config.
    """
    tc_cfg = config["transaction_costs"]
    base_lam = float(tc_cfg["lambda_market_impact"])
    cal_cfg = tc_cfg.get("lambda_calibration", {})
    if not cal_cfg.get("enabled", False):
        return base_lam

    sigma_col = sigma_col or f"liq_{tc_cfg['sigma_col']}"
    if sigma_col not in df.columns:
        logger.warning(
            "Lambda calibration requested but %s is missing; using fixed lambda %.4f",
            sigma_col,
            base_lam,
        )
        return base_lam

    sigma = pd.to_numeric(df[sigma_col], errors="coerce")
    sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    if sigma.empty:
        logger.warning(
            "Lambda calibration requested but %s has no positive values; "
            "using fixed lambda %.4f",
            sigma_col,
            base_lam,
        )
        return base_lam

    stat = cal_cfg.get("sigma_bar_stat", "median")
    if stat == "mean":
        sigma_bar = float(sigma.mean())
    elif stat == "median":
        sigma_bar = float(sigma.median())
    else:
        raise ValueError("lambda_calibration.sigma_bar_stat must be 'median' or 'mean'")

    anchor_trade_frac = float(cal_cfg.get("anchor_trade_frac_adv", 0.01))
    anchor_impact = float(cal_cfg.get("anchor_impact", 0.001))
    if sigma_bar <= 0 or anchor_trade_frac <= 0 or anchor_impact <= 0:
        logger.warning(
            "Invalid lambda calibration inputs; using fixed lambda %.4f",
            base_lam,
        )
        return base_lam

    lam = anchor_impact / (np.sqrt(anchor_trade_frac) * sigma_bar)
    logger.info(
        "Calibrated market-impact lambda: %.4f "
        "(anchor_impact=%.4g, anchor_trade_frac_adv=%.4g, %s(%s)=%.4f)",
        lam,
        anchor_impact,
        anchor_trade_frac,
        stat,
        sigma_col,
        sigma_bar,
    )
    return float(lam)


# -- Family 1: Dollar Volume Weights ---------------------------------


def _dolvol_weights(group: pd.DataFrame, dolvol_col: str) -> pd.Series:
    """w_it = DolVol_it / mean(DolVol_t).

    Parameters
    ----------
    group : single cross-section (one yyyymm).
    dolvol_col : column name for dollar volume (e.g. 'liq_dvol_21d').
    """
    liq = group[dolvol_col].copy()
    mean_val = liq.mean()
    if np.isnan(mean_val) or mean_val <= 0:
        mean_val = 1.0
    # NaN -> mean (neutral); clip to avoid zero/negative weights.
    liq = liq.fillna(mean_val).clip(lower=1e-8)
    return liq / mean_val


# -- Family 2: Softmax Rank Weights ----------------------------------


def _softmax_rank_weights(
    group: pd.DataFrame,
    dolvol_col: str,
    lam: float,
) -> pd.Series:
    """Smooth liquidity-rank weights, mean-normalized for sample_weight.

    The paper-style softmax formula sums to one. For ML sample weights we use
    the equivalent mean-one version so the average observation has weight 1.
    """
    liq = group[dolvol_col].copy()
    rank = liq.where(liq > 0).rank(pct=True, method="average").fillna(0.5)
    raw = np.exp(float(lam) * rank)
    return _normalize_to_mean_one(raw)


# -- Family 3: Transaction-Cost-Based Weights (Eq. 23) ---------------


def _compute_tc_per_stock(
    group: pd.DataFrame,
    aum: float,
    lam: float,
    spread_col: str,
    sigma_col: str,
    adv_col: str,
    leg_fraction: float = 1.0,
) -> pd.Series:
    """Compute one-way TC per stock (Frazzini et al. 2018, Eq. 22/25).

    TC_it = Spread_it/2 + lambda * sigma_it * sqrt(Q_it / ADV_it)

    where Q_it = AUM / N_effective.

    leg_fraction controls N_effective relative to the cross-section size:
      - leg_fraction=1.0 (default, training weights):
          N_effective = N (full cross-section, used at training stage)
      - leg_fraction=0.1 (decile-sort portfolios):
          N_effective = N * 0.1 = N_leg, matching Section 9.2 Q_it = AUM/N_leg
          for a long-short decile portfolio (one leg holds N/10 stocks).
    """
    n_stocks = len(group)
    if n_stocks == 0:
        return pd.Series(dtype=float)
    n_effective = max(1, int(n_stocks * leg_fraction))
    q_per_stock = aum / n_effective

    spread = group[spread_col].copy()
    sigma = group[sigma_col].copy()
    adv = group[adv_col].copy()

    # Fill missing/invalid values with cross-sectional medians. Do not floor
    # positive ADV values; tiny ADV should remain costly rather than clipped.
    spread = spread.fillna(spread.median()).abs().clip(lower=0.0)
    sigma = sigma.fillna(sigma.median()).abs().clip(lower=1e-8)
    adv = adv.where(adv > 0)
    adv_median = adv.median()
    if np.isnan(adv_median) or adv_median <= 0:
        adv_median = 1e6
    adv = adv.fillna(adv_median)

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
    alpha_cfg: dict | None = None,
) -> pd.Series:
    """w_it = exp(-alpha_t * TC_it), normalised to mean=1.

    Eq. 23: alpha_t controls the decay rate. The current primary setting is
    alpha_t = scale / median(TC_it), so the config scale directly controls the
    raw penalty for a median-cost stock.

    Parameters
    ----------
    group : single cross-section (one yyyymm).
    aum : assets under management in dollars.
    lam : market impact coefficient (lambda = 0.1).
    spread_col, sigma_col, adv_col : column names for TC inputs.
    alpha_cfg : transaction_costs.weight_alpha config block.
    """
    tc = _compute_tc_per_stock(group, aum, lam, spread_col, sigma_col, adv_col)
    if len(tc) == 0:
        return pd.Series(dtype=float)

    alpha_cfg = alpha_cfg or {}
    mode = alpha_cfg.get("mode", "inverse_median")
    scale = float(alpha_cfg.get("scale", 3.0))

    median_tc = tc.median()
    if np.isnan(median_tc) or median_tc <= 0:
        alpha_t = 1.0
    elif mode == "inverse_median":
        alpha_t = scale / median_tc
    elif mode == "median":
        alpha_t = scale * median_tc
    else:
        raise ValueError(
            "transaction_costs.weight_alpha.mode must be "
            "'inverse_median' or 'median'"
        )

    raw = np.exp(-alpha_t * tc)
    return _normalize_to_mean_one(raw)


# -- Registry ---------------------------------------------------------

AVAILABLE_SCHEMES = ["dolvol", "softmax_rank", "tc"]


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
    scheme : 'dolvol', 'softmax_rank', or 'tc' (Eq. 23).
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
    wt_cfg = config.get("weighting", {})
    dolvol_col = f"liq_{liq_cfg['primary']}"

    if scheme == "tc" and aum is None:
        raise ValueError("aum is required for scheme='tc'")

    if scheme == "tc":
        spread_col = f"liq_{tc_cfg['spread_col']}"
        sigma_col = f"liq_{tc_cfg['sigma_col']}"
        adv_col = f"liq_{tc_cfg['adv_col']}"
        lam = resolve_market_impact_lambda(df, config, sigma_col=sigma_col)

    parts: list[pd.Series] = []
    for yyyymm, group in df.groupby("yyyymm"):
        if scheme == "dolvol":
            w = _dolvol_weights(group, dolvol_col)
        elif scheme == "softmax_rank":
            w = _softmax_rank_weights(
                group,
                dolvol_col=dolvol_col,
                lam=wt_cfg.get("softmax_rank_lambda", 2.0),
            )
        else:  # tc
            w = _tc_weights(
                group, aum=aum, lam=lam,
                spread_col=spread_col, sigma_col=sigma_col, adv_col=adv_col,
                alpha_cfg=tc_cfg.get("weight_alpha", {}),
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
    """Compute per-stock TC for the TC-penalised sort (Column 2, Section 9.2).

    Uses Q_it = AUM / N_leg where N_leg = N / n_quantiles (typically N/10
    for decile sort). This matches the actual trade size of a long-short
    decile portfolio at the sorting stage.

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
    spread_col = f"liq_{tc_cfg['spread_col']}"
    sigma_col = f"liq_{tc_cfg['sigma_col']}"
    adv_col = f"liq_{tc_cfg['adv_col']}"
    lam = resolve_market_impact_lambda(df, config, sigma_col=sigma_col)

    # Decile sort: leg_fraction = 1 / n_quantiles (0.1 for deciles)
    n_quantiles = config["portfolio"]["n_quantiles"]
    leg_fraction = 1.0 / n_quantiles

    parts: list[pd.Series] = []
    for yyyymm, group in df.groupby("yyyymm"):
        tc = _compute_tc_per_stock(
            group, aum=aum, lam=lam,
            spread_col=spread_col, sigma_col=sigma_col, adv_col=adv_col,
            leg_fraction=leg_fraction,
        )
        parts.append(tc)

    tc_all = pd.concat(parts)
    return tc_all.loc[df.index]
