"""
Liquidity Weighting Schemes (Equations 8–11)
=============================================
Four cross-sectional weighting schemes for implementability-weighted
ML training and portfolio construction.

All schemes:
  - Operate per cross-section (within each yyyymm)
  - Normalize weights to mean=1.0 within each cross-section
  - Handle NaN by assigning neutral weight (~1.0)
  - Return a pd.Series aligned to the input DataFrame index
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

# ── Module-level constants (from config) ─────────────────────

_cfg = load_config()
_weighting_cfg = _cfg["weighting"]
_liquidity_cfg = _cfg["liquidity"]

PRIMARY_SCHEME: str = _weighting_cfg["primary"]
PRIMARY_LIQ_COL: str = (
    f"liq_{_liquidity_cfg['primary']}"  # config-driven, e.g. liq_dvol_21d
)


# ── Helper ───────────────────────────────────────────────────


def _normalize_to_mean_one(weights: pd.Series) -> pd.Series:
    """Rescale weights so their mean equals 1.0.

    If all weights are zero or NaN, returns uniform weights (all 1.0).
    """
    mean_w = weights.mean()
    if mean_w == 0 or np.isnan(mean_w):
        return pd.Series(1.0, index=weights.index)
    return weights / mean_w


# ── Scheme A: Softmax on Rank (Eq. 8) ───────────────────────


def _softmax_rank(group: pd.DataFrame, lam: float | None = None) -> pd.Series:
    """w_i = exp(λ · percentile_i) / Σ exp(λ · percentile_j), normalized to mean=1.

    Parameters
    ----------
    group : single cross-section (one yyyymm) with PRIMARY_LIQ_COL column.
    lam : softmax temperature. Default from config (2.0).
    """
    if lam is None:
        lam = _weighting_cfg["softmax_rank"]["lambda"]

    liq = group[PRIMARY_LIQ_COL]
    pctile = liq.rank(pct=True)
    pctile = pctile.fillna(0.5)  # NaN → neutral (median rank)

    # Softmax with subtract-max for numerical stability
    scaled = lam * pctile
    scaled = scaled - scaled.max()
    raw = np.exp(scaled)

    return _normalize_to_mean_one(raw)


# ── Scheme B: Linear Dollar Volume (Eq. 9) ──────────────────


def _linear_dolvol(group: pd.DataFrame) -> pd.Series:
    """w_i = DolVol_i / mean(DolVol), so mean=1 by construction.

    Parameters
    ----------
    group : single cross-section with PRIMARY_LIQ_COL column.
    """
    liq = group[PRIMARY_LIQ_COL].copy()
    median_val = liq.median()
    if np.isnan(median_val):
        median_val = 1.0
    liq = liq.fillna(median_val)
    liq = liq.clip(lower=1e-8)  # avoid zero/negative

    return _normalize_to_mean_one(liq)


# ── Scheme C: Spread-Based (Eq. 10) ──────────────────────────


def _bid_ask_spread(group: pd.DataFrame, spread_col: str | None = None) -> pd.Series:
    """w_i = 1 / (1 + spread_scale * Spread_i), normalized to mean=1.

    Parameters
    ----------
    group : single cross-section with liq_{spread_col} column.
    spread_col : column base name. Default from config ('BidAskSpread').
    """
    tc_cfg = _weighting_cfg.get("bid_ask_spread", {})
    if spread_col is None:
        spread_col = tc_cfg["spread_col"]
    spread_scale = tc_cfg.get("spread_scale", 1.0)
    col = f"liq_{spread_col}"

    spread = group[col].copy()
    median_val = spread.median()

    if np.isnan(median_val):
        # Use mean spread from non-NaN values across the full panel as fallback
        median_val = spread.quantile(0.5) if spread.notna().any() else 0.01

    spread = spread.fillna(median_val)
    spread = spread.abs()  # safety: CZ signed predictors may be negated
    spread = spread.clip(lower=0.0)
    spread = spread * spread_scale

    raw = 1.0 / (1.0 + spread)
    return _normalize_to_mean_one(raw)


# ── Scheme D: Quintile Discrete (Eq. 11) ─────────────────────


def _quintile_discrete(
    group: pd.DataFrame,
    quintile_weights: dict[str, float] | None = None,
) -> pd.Series:
    """Assign fixed weight by liquidity quintile (PRIMARY_LIQ_COL), normalized to mean=1.

    Q1 (least liquid) → 0.1, Q2 → 0.3, Q3 → 1.0, Q4 → 3.0, Q5 → 5.0

    Parameters
    ----------
    group : single cross-section with PRIMARY_LIQ_COL column.
    quintile_weights : override mapping {Q1: ..., Q5: ...}. Default from config.
    """
    if quintile_weights is None:
        quintile_weights = _weighting_cfg["quintile_discrete"]["weights"]

    liq = group[PRIMARY_LIQ_COL]
    pctile = liq.rank(pct=True)

    # Cut percentile ranks into 5 bins → Q1–Q5
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    quintile = pd.cut(pctile, bins=bins, labels=labels, include_lowest=True)

    # Map quintile labels to weights; NaN → neutral (Q3)
    neutral = quintile_weights["Q3"]
    raw = quintile.map(quintile_weights).astype(float)
    raw = raw.fillna(neutral)

    return _normalize_to_mean_one(raw)


# ── Scheme E: TC-Stock Full (Eq. 13 extended) ────────────────


def _tc_stock(group: pd.DataFrame) -> pd.Series:
    """w_i = 1 / (1 + κ · TC_stock_i), normalized to mean=1.

    TC_stock_i = S_i/2 + λ · σ_i / √ADV_i

    Captures spread, volatility, and volume — the full stock-specific
    component of the Frazzini et al. (2018) TC model. AUM-invariant
    because the constant c = λ·√(AUM/n) is absorbed by normalization.
    """
    tc_cfg = _weighting_cfg.get("tc_stock", {})
    kappa = tc_cfg.get("kappa", 1.0)
    lam = tc_cfg.get("lambda_market_impact", 0.1)
    spread_col = tc_cfg.get("spread_col", "BidAskSpread")
    sigma_col = tc_cfg.get("sigma_col", "daily_sigma")
    adv_col = tc_cfg.get("adv_col", "dvol_21d")

    spread = group[f"liq_{spread_col}"].copy()
    sigma = group[f"liq_{sigma_col}"].copy()
    adv = group[f"liq_{adv_col}"].copy()

    # Fill missing with cross-sectional median
    spread = spread.fillna(spread.median()).abs().clip(lower=0.0)
    sigma = sigma.fillna(sigma.median()).abs().clip(lower=1e-8)
    adv = adv.fillna(adv.median()).clip(lower=1e3)

    # TC_stock = S/2 + lambda * sigma / sqrt(ADV)
    tc_stock = spread / 2.0 + lam * sigma / np.sqrt(adv)

    raw = 1.0 / (1.0 + kappa * tc_stock)
    return _normalize_to_mean_one(raw)


# ── Registry ─────────────────────────────────────────────────

SCHEME_REGISTRY: dict[str, callable] = {
    "softmax_rank": _softmax_rank,
    "linear_dolvol": _linear_dolvol,
    "bid_ask_spread": _bid_ask_spread,
    "tc_stock": _tc_stock,
    "quintile_discrete": _quintile_discrete,
}


def get_available_schemes() -> list[str]:
    """Return list of available weighting scheme names."""
    return list(SCHEME_REGISTRY.keys())


# ── Public API ───────────────────────────────────────────────


def compute_weights(
    df: pd.DataFrame,
    scheme: str | None = None,
    **kwargs,
) -> pd.Series:
    """Compute liquidity weights for a panel (multiple months).

    Weights are computed independently per cross-section (yyyymm)
    and normalized to mean=1.0 within each cross-section.

    Parameters
    ----------
    df : DataFrame with 'yyyymm' and required liq_* columns.
    scheme : one of 'softmax_rank', 'linear_dolvol', 'spread',
             'quintile_discrete'. Default: primary scheme from config.
    **kwargs : passed to the scheme function (e.g., lam=3.0).

    Returns
    -------
    pd.Series of weights aligned to df.index, mean ~1.0 per cross-section.
    """
    if scheme is None:
        scheme = PRIMARY_SCHEME

    if scheme not in SCHEME_REGISTRY:
        raise ValueError(
            f"Unknown scheme: {scheme!r}. Available: {get_available_schemes()}"
        )

    scheme_fn = SCHEME_REGISTRY[scheme]

    # Apply per cross-section
    parts: list[pd.Series] = []
    for yyyymm, group in df.groupby("yyyymm"):
        n_nan = (
            group[PRIMARY_LIQ_COL].isna().sum()
            if PRIMARY_LIQ_COL in group.columns
            else 0
        )
        if len(group) > 0 and n_nan / len(group) > 0.2:
            logger.warning(
                "Scheme %s, month %s: %.0f%% NaN in liquidity column",
                scheme,
                yyyymm,
                100 * n_nan / len(group),
            )
        w = scheme_fn(group, **kwargs)
        parts.append(w)

    weights = pd.concat(parts)
    weights = weights.loc[df.index]  # ensure alignment

    logger.info(
        "compute_weights(scheme=%s): %d obs, %d months, "
        "mean=%.3f, std=%.3f, min=%.4f, max=%.4f",
        scheme,
        len(weights),
        df["yyyymm"].nunique(),
        weights.mean(),
        weights.std(),
        weights.min(),
        weights.max(),
    )
    return weights


def compute_all_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 4 weighting schemes and return as a DataFrame.

    Returns
    -------
    pd.DataFrame with columns: weight_softmax_rank, weight_linear_dolvol,
        weight_bid_ask_spread, weight_quintile_discrete.
    Index aligned to df.index.
    """
    result = pd.DataFrame(index=df.index)
    for name in SCHEME_REGISTRY:
        result[f"weight_{name}"] = compute_weights(df, scheme=name)
    return result
