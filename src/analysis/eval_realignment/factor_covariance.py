"""JKMP-style characteristic-factor covariance model (monthly adaptation).

Implements the Barra-style risk model of Jensen, Kelly, Malamud, and Pedersen
(2024) on monthly data:

    Sigma_t = X_t * Omega_t * X_t' + D_t

* X_t: per-stock exposures to K = 1 (intercept/market) + 8 broad characteristic
  themes (Value, Momentum, Profitability, Investment, Quality, Risk, Liquidity,
  Other — `config/feature_categories.json["broad"]`). A theme exposure is the
  mean of the stock's available rank-normalized features in that theme,
  z-scored cross-sectionally within month.
* Factor returns f_s: monthly cross-sectional OLS of the (forward-shifted)
  excess return on X_s. f_s realizes at s+1, so at a rebalance t the history
  {f_s : s <= t-1} is known — the track's no-lookahead convention.
* Omega_t: EWMA of factor returns with the monthly translation of JKMP's
  half-lives — variances 6 months (their 126 days), correlations 18 months
  (their 378 days) — recombined as diag(sigma) * C * diag(sigma).
* D_t: per-name EWMA of squared residuals (half-life 6 months), floored;
  names with fewer than 12 residual months receive the cross-sectional median
  idiosyncratic variance (the analogue of JKMP's 12-month validity rule).

Sigma is never formed densely: solves of (gamma*Sigma + Lambda)^{-1} v use the
Woodbury identity with a K-dimensional inner system, O(N*K^2) per month, which
is what allows the optimizer universe to expand to the top-60% DolVol screen.

Fidelity gaps vs the paper, by design: no industry factors (the panel carries
no industry codes) and monthly rather than daily estimation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VAR_HALFLIFE = 6     # months ~ JKMP 126 trading days
CORR_HALFLIFE = 18   # months ~ JKMP 378 trading days
IDIO_HALFLIFE = 6
IDIO_MIN_OBS = 12    # months of residual history before a name's own idio is used
IDIO_FLOOR = 1e-8


def load_theme_map(
    categories_path: str | Path = "config/feature_categories.json",
    feature_list_path: str | Path = "data/feature_list.json",
) -> dict[str, str]:
    """Feature -> broad theme for the model's selected features (unmapped -> Other)."""
    broad = json.load(open(categories_path))["broad"]
    feats = json.load(open(feature_list_path))
    if isinstance(feats, dict):
        feats = feats["features"]
    return {f: broad.get(f, "Other") for f in feats}


def theme_columns(theme_map: dict[str, str]) -> list[str]:
    """Stable theme ordering (alphabetical) used for the exposure matrix."""
    return sorted(set(theme_map.values()))


def build_exposures(
    month_df: pd.DataFrame,
    theme_map: dict[str, str],
    themes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(permnos, X) for one month: intercept + z-scored theme means.

    A stock's raw theme exposure is the mean of its non-missing rank-normalized
    features in that theme; exposures are z-scored cross-sectionally (zero-std
    or all-NaN themes become 0), and an intercept column is prepended.
    """
    if themes is None:
        themes = theme_columns(theme_map)
    permnos = month_df["permno"].to_numpy(dtype=np.int64)
    n = len(permnos)
    X = np.zeros((n, 1 + len(themes)), dtype=np.float64)
    X[:, 0] = 1.0
    for k, theme in enumerate(themes):
        cols = [f for f, th in theme_map.items() if th == theme and f in month_df.columns]
        if not cols:
            continue
        raw = month_df[cols].mean(axis=1, skipna=True).to_numpy(dtype=np.float64)
        mean = np.nanmean(raw)
        std = np.nanstd(raw)
        if not np.isfinite(mean):
            continue
        z = np.where(np.isfinite(raw), raw - mean, 0.0)
        if std > 1e-12:
            z = z / std
        X[:, 1 + k] = z
    return permnos, X


def woodbury_solve(
    a_diag: np.ndarray,
    X: np.ndarray,
    C: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve (diag(a) + X C X') theta = rhs via the K-dimensional Woodbury identity."""
    a_inv = 1.0 / a_diag
    a_inv_rhs = a_inv * rhs
    a_inv_X = X * a_inv[:, None]
    inner = np.linalg.inv(C) + X.T @ a_inv_X            # (K, K)
    middle = np.linalg.solve(inner, X.T @ a_inv_rhs)    # (K,)
    return a_inv_rhs - a_inv_X @ middle


class FactorRiskModel:
    """Sequential estimator of (X_t, Omega_t, D_t); update AFTER each month's solve.

    Usage per chronological month t:
      1. ``exposures(month_df)`` — X_t from month-t features (known at rebalance).
      2. ``sigma_parts(permnos)`` — (Omega, d_vec) reflecting history through t-1.
      3. After the books are solved: ``update(X, permnos, y)`` with the month-t
         forward returns (they realize at t+1, so folding them in after the
         solve preserves the no-lookahead convention).
    """

    def __init__(self, theme_map: dict[str, str]):
        self.theme_map = theme_map
        self.themes = theme_columns(theme_map)
        self.k = 1 + len(self.themes)
        self._a_var = 1.0 - 0.5 ** (1.0 / VAR_HALFLIFE)
        self._a_corr = 1.0 - 0.5 ** (1.0 / CORR_HALFLIFE)
        self._a_idio = 1.0 - 0.5 ** (1.0 / IDIO_HALFLIFE)
        self.var_fast: np.ndarray | None = None     # EWMA of f^2 (hl 6m)
        self.cov_slow: np.ndarray | None = None     # EWMA of f f' (hl 18m)
        self.idio: dict[int, float] = {}
        self.idio_n: dict[int, int] = {}
        self.n_updates = 0

    def exposures(self, month_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return build_exposures(month_df, self.theme_map, self.themes)

    @property
    def ready(self) -> bool:
        return self.n_updates >= 2 and self.cov_slow is not None

    def omega(self) -> np.ndarray:
        """Omega = diag(sigma_fast) * Corr_slow * diag(sigma_fast)."""
        cov = self.cov_slow
        d_slow = np.sqrt(np.maximum(np.diag(cov), 1e-16))
        corr = cov / np.outer(d_slow, d_slow)
        np.fill_diagonal(corr, 1.0)
        sig = np.sqrt(np.maximum(self.var_fast, 1e-16))
        return np.outer(sig, sig) * corr

    def idio_var(self, permnos: np.ndarray) -> np.ndarray:
        """Per-name idio variance; median fallback for short-history names."""
        seasoned = [
            v for p, v in self.idio.items() if self.idio_n.get(p, 0) >= IDIO_MIN_OBS
        ]
        median = float(np.median(seasoned)) if seasoned else 0.01
        out = np.empty(len(permnos), dtype=np.float64)
        for i, p in enumerate(permnos):
            p = int(p)
            if self.idio_n.get(p, 0) >= IDIO_MIN_OBS:
                out[i] = self.idio[p]
            else:
                out[i] = median
        return np.maximum(out, IDIO_FLOOR)

    def sigma_parts(self, permnos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(Omega_t, d_vec) for the given names, from history through t-1."""
        return self.omega(), self.idio_var(permnos)

    def update(self, X: np.ndarray, permnos: np.ndarray, y: np.ndarray) -> None:
        """Fold month-t factor return and residuals into the EWMA states."""
        ok = np.isfinite(y)
        if ok.sum() <= self.k:
            return
        Xv, yv = X[ok], y[ok]
        f, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        resid = yv - Xv @ f

        outer = np.outer(f, f)
        if self.cov_slow is None:
            self.cov_slow = outer
            self.var_fast = f * f
        else:
            self.cov_slow = (1 - self._a_corr) * self.cov_slow + self._a_corr * outer
            self.var_fast = (1 - self._a_var) * self.var_fast + self._a_var * f * f

        e2 = resid * resid
        for p, v in zip(permnos[ok], e2):
            p = int(p)
            prev = self.idio.get(p)
            self.idio[p] = (
                float(v) if prev is None
                else (1 - self._a_idio) * prev + self._a_idio * float(v)
            )
            self.idio_n[p] = self.idio_n.get(p, 0) + 1
        self.n_updates += 1
