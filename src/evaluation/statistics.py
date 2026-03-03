"""
Statistical Inference
=====================
Statistical tests for the 2×2 experimental framework.

Implements:
  1. Sharpe ratio computation (annualized)
  2. Newey-West HAC t-statistics for mean returns
  3. Factor alpha regressions (CAPM, FF3, FF5) with Newey-West SEs
  4. GRS test (Gibbons, Ross, Shanken 1989)
  5. Ledoit-Wolf (2008) bootstrap Sharpe ratio comparison
  6. Out-of-sample R² (Campbell & Thompson 2008)
  7. Paired t-tests for feature importance differences
  8. 2×2 effect decomposition with all statistical tests

References:
  - Ledoit & Wolf (2008), "Robust performance hypothesis testing
    with the Sharpe ratio", JEF.
  - Gibbons, Ross & Shanken (1989), "A test of the efficiency of
    a given portfolio", Econometrica.
  - Campbell & Thompson (2008), "Predicting excess stock returns
    out of sample", RFS.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import CircularBlockBootstrap
from scipy import stats
from statsmodels.regression.linear_model import OLS

from src.config import get_data_dir, load_config

logger = logging.getLogger(__name__)

# ── Module-level constants (from config) ─────────────────────

_cfg = load_config()
_inf_cfg = _cfg["inference"]

NEWEY_WEST_LAGS: int = _inf_cfg["newey_west_lags"]
BOOTSTRAP_SAMPLES: int = _inf_cfg["bootstrap_samples"]
SIGNIFICANCE_LEVEL: float = _inf_cfg["significance_level"]
ONE_SIDED_TEST: bool = _inf_cfg["one_sided_test"]
CALIBRATE_BLOCK_LENGTH: bool = _inf_cfg.get("calibrate_block_length", False)
BLOCK_GRID: list[int] = _inf_cfg.get("block_grid", [2, 4, 6, 8, 10])

FACTOR_MODELS: dict[str, list[str]] = _inf_cfg.get("factor_models", {
    "capm": ["Mkt-RF"],
    "ff3": ["Mkt-RF", "SMB", "HML"],
    "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
})

ANNUALIZE_FACTOR: int = 12  # monthly → annual


# ══════════════════════════════════════════════════════════════
# 1. Data Loading
# ══════════════════════════════════════════════════════════════


def load_ff_factors(
    n_factors: int = 5, config: dict | None = None
) -> pd.DataFrame:
    """Load Fama-French factor data.

    Parameters
    ----------
    n_factors : 3 or 5. If 3, loads Mkt-RF, SMB, HML from the local
        FFResearch_Data_Factors.csv. If 5, adds RMW and CMA via
        pandas_datareader (with local CSV fallback).
    config : Override config.

    Returns
    -------
    DataFrame with columns: yyyymm, Mkt-RF, SMB, HML [, RMW, CMA].
    All factor values in decimal (e.g. 0.01 = 1%).
    """
    if config is None:
        config = _cfg
    data_dir = get_data_dir()

    if n_factors <= 3:
        # Use existing local file
        ff_path = data_dir / config["data"]["ff_factors_csv"]
        ff = pd.read_csv(ff_path)
        ff = ff.rename(columns={"Date": "yyyymm"})
        for col in ["Mkt-RF", "SMB", "HML"]:
            ff[col] = ff[col] / 100.0
        return ff[["yyyymm", "Mkt-RF", "SMB", "HML"]]

    # n_factors == 5: try pandas_datareader, fall back to local CSV
    try:
        import pandas_datareader.data as web

        ff_all = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3",
            "famafrench",
            start="1972-01-01",
            end="2024-12-31",
        )
        ff = ff_all[0] / 100.0  # percent → decimal
        ff.index = ff.index.to_timestamp("M")
        # Convert index to yyyymm integer
        ff["yyyymm"] = ff.index.year * 100 + ff.index.month
        ff = ff.reset_index(drop=True)
        cols = ["yyyymm", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        ff = ff[cols]
        logger.info("Loaded FF5 factors via pandas_datareader (%d rows)", len(ff))
        return ff

    except Exception as e:
        logger.warning("pandas_datareader FF5 fetch failed (%s); trying local CSV", e)

    # Fallback: local CSV
    ff5_path = data_dir / "ff5_factors.csv"
    if ff5_path.exists():
        ff = pd.read_csv(ff5_path)
        ff = ff.rename(columns={"Date": "yyyymm"})
        for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
            if col in ff.columns:
                ff[col] = ff[col] / 100.0
        return ff[["yyyymm", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]]

    # Last resort: return FF3 with warning
    logger.warning("FF5 data unavailable; returning FF3 only")
    return load_ff_factors(n_factors=3, config=config)


# ══════════════════════════════════════════════════════════════
# 2. Core Statistics
# ══════════════════════════════════════════════════════════════


def sharpe_ratio(
    returns: np.ndarray | pd.Series, annualize: bool = True
) -> float:
    """Annualized Sharpe ratio.

    Parameters
    ----------
    returns : Monthly excess returns (already excess or long-short).
    annualize : If True, multiply by sqrt(12).

    Returns
    -------
    float : Sharpe ratio. Returns 0.0 if std ≈ 0, NaN if < 2 obs.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return np.nan
    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    if sigma < 1e-12:
        return 0.0
    sr = mu / sigma
    if annualize:
        sr *= np.sqrt(ANNUALIZE_FACTOR)
    return float(sr)


def newey_west_tstat(
    returns: np.ndarray | pd.Series,
    lags: int | None = None,
    config: dict | None = None,
) -> dict[str, float]:
    """Newey-West HAC t-statistic for mean return ≠ 0.

    Parameters
    ----------
    returns : Monthly return series.
    lags : Newey-West lags (default from config: 6).

    Returns
    -------
    dict: mean, std_err, t_stat, p_value, n_obs
    """
    if lags is None:
        lags = NEWEY_WEST_LAGS

    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]

    if len(r) < 3:
        return {"mean": np.nan, "std_err": np.nan, "t_stat": np.nan,
                "p_value": np.nan, "n_obs": len(r)}

    X = np.ones((len(r), 1))
    result = OLS(r, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    return {
        "mean": float(result.params[0]),
        "std_err": float(result.bse[0]),
        "t_stat": float(result.tvalues[0]),
        "p_value": float(result.pvalues[0]),
        "n_obs": len(r),
    }


# ══════════════════════════════════════════════════════════════
# 3. Factor Alpha Regressions
# ══════════════════════════════════════════════════════════════


def factor_alpha(
    returns_df: pd.DataFrame | pd.Series,
    model: str = "ff3",
    ff_factors: pd.DataFrame | None = None,
    lags: int | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Factor alpha regression with Newey-West SEs.

    Regresses portfolio returns on a factor model:
        r_t = α + Σ β_k · F_k,t + ε_t

    Parameters
    ----------
    returns_df : DataFrame with columns [yyyymm, ret_long_short]
        or Series with yyyymm as index.
    model : "capm", "ff3", or "ff5".
    ff_factors : Factor data (if None, loaded automatically).
    lags : Newey-West lag count (default from config: 6).

    Returns
    -------
    dict: alpha, alpha_annual, alpha_se, alpha_tstat, alpha_pvalue,
          betas, beta_tstats, beta_pvalues, r_squared, n_obs, residuals
    """
    if lags is None:
        lags = NEWEY_WEST_LAGS

    factor_cols = FACTOR_MODELS.get(model)
    if factor_cols is None:
        raise ValueError(f"Unknown model: {model!r}. Available: {list(FACTOR_MODELS)}")

    n_factors_needed = 5 if model == "ff5" else 3
    if ff_factors is None:
        ff_factors = load_ff_factors(n_factors=n_factors_needed, config=config)

    # Prepare returns with yyyymm column
    if isinstance(returns_df, pd.Series):
        if returns_df.index.name == "yyyymm":
            ret_df = returns_df.reset_index()
            ret_df.columns = ["yyyymm", "ret"]
        else:
            raise ValueError("Series must have yyyymm as index name")
    else:
        ret_df = returns_df.copy()
        # Auto-detect return column
        ret_col = "ret_long_short"
        if ret_col not in ret_df.columns:
            # Try other common names
            for candidate in ["ret_ls", "ret", "excess_ret"]:
                if candidate in ret_df.columns:
                    ret_col = candidate
                    break
            else:
                raise ValueError(
                    f"Cannot find return column. Expected 'ret_long_short'. "
                    f"Columns: {list(ret_df.columns)}"
                )
        ret_df = ret_df[["yyyymm", ret_col]].rename(columns={ret_col: "ret"})

    # Merge with factors
    merged = ret_df.merge(ff_factors, on="yyyymm", how="inner")

    n_lost = len(ret_df) - len(merged)
    if n_lost > 0:
        logger.debug("factor_alpha: %d rows lost in merge (%.1f%%)",
                      n_lost, 100 * n_lost / len(ret_df))

    if len(merged) < 10:
        logger.warning("factor_alpha: only %d observations after merge", len(merged))
        return _empty_alpha_result(factor_cols)

    y = merged["ret"].values
    X = merged[factor_cols].values
    Xc = sm.add_constant(X)

    result = OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    # Extract results
    alpha = float(result.params[0])
    betas = {col: float(result.params[i + 1]) for i, col in enumerate(factor_cols)}
    beta_tstats = {col: float(result.tvalues[i + 1]) for i, col in enumerate(factor_cols)}
    beta_pvalues = {col: float(result.pvalues[i + 1]) for i, col in enumerate(factor_cols)}

    return {
        "alpha": alpha,
        "alpha_annual": alpha * ANNUALIZE_FACTOR,
        "alpha_se": float(result.bse[0]),
        "alpha_tstat": float(result.tvalues[0]),
        "alpha_pvalue": float(result.pvalues[0]),
        "betas": betas,
        "beta_tstats": beta_tstats,
        "beta_pvalues": beta_pvalues,
        "r_squared": float(result.rsquared),
        "n_obs": len(merged),
        "residuals": np.asarray(result.resid),
    }


def _empty_alpha_result(factor_cols: list[str]) -> dict[str, Any]:
    """Return NaN result when there's insufficient data."""
    return {
        "alpha": np.nan, "alpha_annual": np.nan,
        "alpha_se": np.nan, "alpha_tstat": np.nan, "alpha_pvalue": np.nan,
        "betas": {c: np.nan for c in factor_cols},
        "beta_tstats": {c: np.nan for c in factor_cols},
        "beta_pvalues": {c: np.nan for c in factor_cols},
        "r_squared": np.nan, "n_obs": 0,
        "residuals": np.array([]),
    }


# ══════════════════════════════════════════════════════════════
# 4. GRS Test (Gibbons, Ross, Shanken 1989)
# ══════════════════════════════════════════════════════════════


def grs_test(
    alpha_vec: np.ndarray,
    resid_matrix: np.ndarray,
    factor_data: np.ndarray,
    T: int,
) -> dict[str, float]:
    """GRS F-test: H₀ — all portfolio alphas are jointly zero.

    Parameters
    ----------
    alpha_vec : (N,) monthly alphas for N portfolios.
    resid_matrix : (T, N) regression residuals.
    factor_data : (T, K) factor returns used in the regressions.
    T : Number of time periods.

    Returns
    -------
    dict: f_stat, p_value, n_portfolios, n_factors
    """
    alpha_vec = np.asarray(alpha_vec, dtype=np.float64)
    resid_matrix = np.asarray(resid_matrix, dtype=np.float64)
    factor_data = np.asarray(factor_data, dtype=np.float64)

    N = len(alpha_vec)
    K = factor_data.shape[1]

    if T <= N + K:
        logger.warning("GRS: T=%d ≤ N+K=%d, test undefined", T, N + K)
        return {"f_stat": np.nan, "p_value": np.nan,
                "n_portfolios": N, "n_factors": K}

    resid_cov = np.cov(resid_matrix, rowvar=False)  # (N, N)
    fbar = factor_data.mean(axis=0)                   # (K,)
    sigma_f = np.cov(factor_data, rowvar=False)        # (K, K)

    # Handle scalar case for CAPM (K=1)
    if sigma_f.ndim == 0:
        sigma_f = sigma_f.reshape(1, 1)
    if fbar.ndim == 0:
        fbar = fbar.reshape(1)

    try:
        inv_resid = np.linalg.inv(resid_cov)
        inv_sigma_f = np.linalg.inv(sigma_f)
    except np.linalg.LinAlgError:
        logger.warning("GRS: singular matrix, cannot compute test")
        return {"f_stat": np.nan, "p_value": np.nan,
                "n_portfolios": N, "n_factors": K}

    f_num = ((T - N - K) / N) * float(alpha_vec @ inv_resid @ alpha_vec)
    f_den = 1.0 + float(fbar @ inv_sigma_f @ fbar)
    f_stat = f_num / f_den
    p_val = 1.0 - stats.f.cdf(f_stat, N, T - N - K)

    return {
        "f_stat": float(f_stat),
        "p_value": float(p_val),
        "n_portfolios": N,
        "n_factors": K,
    }


# ══════════════════════════════════════════════════════════════
# 5. Ledoit-Wolf (2008) Bootstrap Sharpe Test
# ══════════════════════════════════════════════════════════════
#
# Studentised circular-block bootstrap (Politis-Romano 1992)
# with prewhitened Parzen kernel HAC estimation.
# Adapted from user reference implementation.
# ──────────────────────────────────────────────────────────────


def _moment_components(r: pd.Series) -> tuple[float, float, float]:
    """Return (μ, γ=E[r²], σ²=γ-μ²)."""
    mu = r.mean()
    gamma = (r**2).mean()
    return mu, gamma, gamma - mu**2


def _sharpe_from_moments(mu: float, gamma: float) -> float:
    """Sharpe ratio from (μ, γ) parameterization."""
    var = gamma - mu**2
    return 0.0 if var <= 0 else mu / np.sqrt(var)


def _grad_sharpe_mu_gamma(mu: float, gamma: float) -> np.ndarray:
    """Gradient of SR w.r.t. (μ, γ)."""
    var = gamma - mu**2
    if var <= 0:
        return np.array([0.0, 0.0])
    denom = var**1.5
    return np.array([gamma / denom, -0.5 * mu / denom])


# ── Parzen kernel HAC ────────────────────────────────────────


def _parzen_kernel(x: float) -> float:
    x = abs(x)
    if x <= 0.5:
        return 1 - 6 * x**2 + 6 * x**3
    elif x <= 1:
        return 2 * (1 - x) ** 3
    return 0.0


def _compute_optimal_bandwidth_parzen(y: np.ndarray) -> float:
    """Andrews (1991) optimal bandwidth for Parzen kernel."""
    T, p = y.shape
    alpha_2 = 0.0
    sigma_sq = 0.0

    for j in range(p):
        yj = y[:, j]
        if np.var(yj) > 0:
            rho = np.corrcoef(yj[:-1], yj[1:])[0, 1] if T > 1 else 0.0
            rho = np.clip(rho, -0.97, 0.97)
            sigma2_j = np.var(yj) * (1 - rho**2)
            alpha_2 += 4 * rho**2 * sigma2_j**2 / (1 - rho) ** 8
            sigma_sq += sigma2_j**2 / (1 - rho) ** 4

    if sigma_sq > 0:
        bw = 2.6614 * ((alpha_2 / sigma_sq) * T) ** 0.2
    else:
        bw = 2.6614 * T**0.2

    return float(np.clip(bw, 1, T**0.5))


def _compute_hac_matrix_parzen(y: np.ndarray, bandwidth: float) -> np.ndarray:
    """HAC covariance matrix using Parzen kernel."""
    T, p = y.shape
    Psi = (y.T @ y) / T

    max_lag = int(bandwidth)
    for lag in range(1, max_lag + 1):
        if lag < T:
            Gamma_lag = (y[lag:].T @ y[:-lag]) / T
            weight = _parzen_kernel(lag / bandwidth)
            if weight > 0:
                Psi += weight * (Gamma_lag + Gamma_lag.T)

    # Ensure positive definiteness
    eigenvalues = np.linalg.eigvalsh(Psi)
    if np.min(eigenvalues) < 1e-10:
        Psi += np.eye(p) * (1e-10 - np.min(eigenvalues))

    return Psi


def _compute_se_parzen_pw(r_i: pd.Series, r_n: pd.Series) -> float:
    """Prewhitened Parzen kernel HAC standard error for SR difference.

    Steps: VAR(1) prewhitening → Parzen HAC on residuals → post-coloring.
    """
    mu_i, gamma_i, _ = _moment_components(r_i)
    mu_n, gamma_n, _ = _moment_components(r_n)

    # Gradient of Δ w.r.t. (μ_i, μ_n, γ_i, γ_n)
    gradient = np.zeros(4)
    if gamma_i > mu_i**2:
        gradient[0] = gamma_i / (gamma_i - mu_i**2) ** 1.5
        gradient[2] = -0.5 * mu_i / (gamma_i - mu_i**2) ** 1.5
    if gamma_n > mu_n**2:
        gradient[1] = -gamma_n / (gamma_n - mu_n**2) ** 1.5
        gradient[3] = 0.5 * mu_n / (gamma_n - mu_n**2) ** 1.5

    # Data matrix: [r_i, r_n, r_i², r_n²]
    y = np.column_stack([
        r_i.values, r_n.values, r_i.values**2, r_n.values**2
    ])
    y_centered = y - y.mean(axis=0)
    T = len(r_i)

    # Prewhiten using VAR(1)
    if T > 10:
        try:
            X = np.column_stack([np.ones(T - 1), y_centered[:-1]])
            Y = y_centered[1:]
            B = np.linalg.lstsq(X, Y, rcond=None)[0]
            A = B[1:].T  # VAR coefficient matrix (4×4)

            eigenvalues = np.linalg.eigvals(A)
            if np.max(np.abs(eigenvalues)) < 0.97:
                residuals = Y - X @ B
                residuals = np.vstack([np.zeros(4), residuals])
                bandwidth = _compute_optimal_bandwidth_parzen(residuals)
                Psi_pw = _compute_hac_matrix_parzen(residuals, bandwidth)
                I_minus_A = np.eye(4) - A
                inv_I_minus_A = np.linalg.inv(I_minus_A)
                Psi = inv_I_minus_A @ Psi_pw @ inv_I_minus_A.T
            else:
                bandwidth = _compute_optimal_bandwidth_parzen(y_centered)
                Psi = _compute_hac_matrix_parzen(y_centered, bandwidth)
        except Exception:
            bandwidth = _compute_optimal_bandwidth_parzen(y_centered)
            Psi = _compute_hac_matrix_parzen(y_centered, bandwidth)
    else:
        bandwidth = _compute_optimal_bandwidth_parzen(y_centered)
        Psi = _compute_hac_matrix_parzen(y_centered, bandwidth)

    variance = (gradient @ Psi @ gradient) / T
    return float(np.sqrt(max(variance, 0)))


# ── Bootstrap machinery ──────────────────────────────────────


def _delta_hat(r_i: pd.Series, r_n: pd.Series) -> float:
    """Observed Sharpe ratio difference: SR(i) - SR(n)."""
    mu_i, gamma_i, _ = _moment_components(r_i)
    mu_n, gamma_n, _ = _moment_components(r_n)
    return _sharpe_from_moments(mu_i, gamma_i) - _sharpe_from_moments(mu_n, gamma_n)


def _bootstrap_statistic(
    r_i: pd.Series, r_n: pd.Series, delta_obs: float,
    b: int, rng: np.random.Generator, one_sided: bool = False,
) -> float:
    """Single studentized bootstrap statistic."""
    data = np.column_stack([r_i.values, r_n.values])
    boot = next(
        CircularBlockBootstrap(b, data, seed=rng.integers(2**32)).bootstrap(1)
    )[0]
    if isinstance(boot, tuple):
        boot = np.column_stack(boot)

    r_i_star = boot[:, 0]
    r_n_star = boot[:, 1]

    mu_i_star = r_i_star.mean()
    mu_n_star = r_n_star.mean()
    gamma_i_star = (r_i_star**2).mean()
    gamma_n_star = (r_n_star**2).mean()

    delta_star = (
        _sharpe_from_moments(mu_i_star, gamma_i_star)
        - _sharpe_from_moments(mu_n_star, gamma_n_star)
    )

    # Gradient at bootstrap estimates
    gradient = np.zeros(4)
    var_i = gamma_i_star - mu_i_star**2
    var_n = gamma_n_star - mu_n_star**2
    if var_i > 0:
        gradient[0] = gamma_i_star / var_i**1.5
        gradient[2] = -0.5 * mu_i_star / var_i**1.5
    if var_n > 0:
        gradient[1] = -gamma_n_star / var_n**1.5
        gradient[3] = 0.5 * mu_n_star / var_n**1.5

    y_star = np.column_stack([
        r_i_star - mu_i_star,
        r_n_star - mu_n_star,
        r_i_star**2 - gamma_i_star,
        r_n_star**2 - gamma_n_star,
    ])

    T = len(r_i_star)
    n_blocks = T // b

    if n_blocks < 2:
        return np.inf

    Psi_hat_star = np.zeros((4, 4))
    for j in range(n_blocks):
        block = y_star[j * b : (j + 1) * b]
        zeta_star = np.sqrt(b) * block.mean(axis=0)
        Psi_hat_star += np.outer(zeta_star, zeta_star)
    Psi_hat_star /= n_blocks

    se_star = np.sqrt(max(gradient @ Psi_hat_star @ gradient / T, 0))
    if se_star <= 1e-10:
        return np.inf

    if one_sided:
        return (delta_star - delta_obs) / se_star
    return abs(delta_star - delta_obs) / se_star


def _choose_block_length(
    r_i: pd.Series, r_n: pd.Series, alpha: float = 0.05,
    grid: tuple = (2, 4, 6, 8, 10), K: int = 1000,
    M_inner: int = 199, b_av: int = 5, T_start: int = 50,
    seed: int | None = None, one_sided: bool = False,
) -> int:
    """Block length calibration — Ledoit-Wolf (2008) Algorithm 3.1."""
    rng = np.random.default_rng(seed)
    T = len(r_i)
    delta_obs = _delta_hat(r_i, r_n)

    # Fit VAR(1)
    X = np.column_stack([np.ones(T - 1), r_i.values[:-1], r_n.values[:-1]])
    coef1 = np.linalg.lstsq(X, r_i.values[1:], rcond=None)[0]
    coef2 = np.linalg.lstsq(X, r_n.values[1:], rcond=None)[0]
    resid1 = r_i.values[1:] - X @ coef1
    resid2 = r_n.values[1:] - X @ coef2
    resid_mat = np.column_stack([resid1, resid2])

    emp_reject_probs = np.zeros(len(grid))

    for _ in range(K):
        # Simulate VAR(1) with stationary bootstrap residuals
        var_data = np.zeros((T_start + T, 2))
        var_data[0] = [r_i.iloc[0], r_n.iloc[0]]
        resid_star = np.zeros((T_start + T, 2))

        t = 1
        start_idx = 0
        while t < T_start + T:
            if t == 1 or rng.random() < 1 / b_av:
                start_idx = rng.integers(0, T - 1)
            resid_star[t] = resid_mat[start_idx % (T - 1)]
            start_idx += 1
            t += 1

        for t in range(1, T_start + T):
            var_data[t, 0] = (coef1[0] + coef1[1] * var_data[t - 1, 0]
                              + coef1[2] * var_data[t - 1, 1] + resid_star[t, 0])
            var_data[t, 1] = (coef2[0] + coef2[1] * var_data[t - 1, 0]
                              + coef2[2] * var_data[t - 1, 1] + resid_star[t, 1])

        var_trunc = var_data[T_start : T_start + T]
        r_i_k = pd.Series(var_trunc[:, 0])
        r_n_k = pd.Series(var_trunc[:, 1])

        for j, b in enumerate(grid):
            delta_k = _delta_hat(r_i_k, r_n_k)
            se_k = _compute_se_parzen_pw(r_i_k, r_n_k)
            if se_k <= 1e-10:
                continue

            if one_sided:
                d_k = (delta_k - delta_obs) / se_k
            else:
                d_k = abs(delta_k - delta_obs) / se_k

            d_star = np.array([
                _bootstrap_statistic(r_i_k, r_n_k, delta_k, b,
                                     np.random.default_rng(rng.integers(2**32)),
                                     one_sided)
                for _ in range(M_inner)
            ])

            if one_sided:
                reject = np.sum(d_star <= d_k) / len(d_star) < alpha
            else:
                crit = np.quantile(d_star, 1 - alpha / 2)
                reject = d_k > crit
            if reject:
                emp_reject_probs[j] += 1

    emp_reject_probs /= K
    best_idx = int(np.argmin(np.abs(emp_reject_probs - alpha)))
    return grid[best_idx]


def bootstrap_sharpe_test(
    r_i: pd.Series | np.ndarray,
    r_n: pd.Series | np.ndarray,
    alpha: float | None = None,
    M: int | None = None,
    block_grid: tuple | None = None,
    K: int = 5000,
    seed: int | None = None,
    calibrate_block_length: bool | None = None,
    one_sided: bool | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Bootstrap test for Sharpe ratio difference (Ledoit-Wolf 2008).

    Tests whether SR(r_i) differs from SR(r_n) using a studentized
    circular-block bootstrap with prewhitened Parzen kernel HAC.

    Parameters
    ----------
    r_i, r_n : Return series for two strategies.
    alpha : Significance level (default from config: 0.05).
    M : Number of bootstrap samples (default from config: 5000).
    block_grid : Block sizes to test during calibration.
    K : Simulations for block length calibration.
    seed : Random seed.
    calibrate_block_length : Whether to calibrate block length
        (default from config: False — very expensive if True).
    one_sided : One-sided test (default from config: True).
        If True, tests H₀: SR(n) ≤ SR(i) vs H₁: SR(n) > SR(i).
    config : Override config.

    Returns
    -------
    dict: difference (SR_i - SR_n), p_value, se, block_length,
          test_stat, reject, one_sided, ci (two-sided only)
    """
    if config is None:
        config = _cfg
    inf = config["inference"]

    if alpha is None:
        alpha = inf["significance_level"]
    if M is None:
        M = inf["bootstrap_samples"]
    if block_grid is None:
        block_grid = tuple(inf.get("block_grid", BLOCK_GRID))
    if calibrate_block_length is None:
        calibrate_block_length = inf.get("calibrate_block_length", False)
    if one_sided is None:
        one_sided = inf["one_sided_test"]
    if seed is None:
        seed = config["project"]["seed"]

    # Convert to Series if needed
    if not isinstance(r_i, pd.Series):
        r_i = pd.Series(np.asarray(r_i, dtype=np.float64))
    if not isinstance(r_n, pd.Series):
        r_n = pd.Series(np.asarray(r_n, dtype=np.float64))

    r_i = r_i.dropna()
    r_n = r_n.dropna()

    if len(r_i) < 12 or len(r_n) < 12:
        logger.warning("bootstrap_sharpe_test: too few observations (%d, %d)",
                        len(r_i), len(r_n))
        return {"difference": np.nan, "p_value": np.nan, "se": np.nan,
                "block_length": 0, "test_stat": np.nan, "reject": False,
                "one_sided": one_sided}

    rng = np.random.default_rng(seed)

    # Observed statistics
    delta_hat = _delta_hat(r_i, r_n)
    se = _compute_se_parzen_pw(r_i, r_n)

    if se <= 1e-10:
        logger.warning("bootstrap_sharpe_test: SE ≈ 0")
        return {"difference": float(delta_hat), "p_value": 1.0, "se": 0.0,
                "block_length": 0, "test_stat": 0.0, "reject": False,
                "one_sided": one_sided}

    if one_sided:
        d_obs = delta_hat / se
    else:
        d_obs = abs(delta_hat) / se

    # Choose block length
    if calibrate_block_length:
        b = _choose_block_length(
            r_i, r_n, alpha, block_grid, K, 199,
            seed=rng.integers(2**32), one_sided=one_sided,
        )
    else:
        b = block_grid[len(block_grid) // 2]

    # Generate bootstrap distribution
    d_star = []
    for _ in range(M):
        stat = _bootstrap_statistic(r_i, r_n, delta_hat, b, rng, one_sided)
        if abs(stat) < np.inf:
            d_star.append(stat)

    d_star = np.array(d_star)

    if len(d_star) == 0:
        logger.warning("bootstrap_sharpe_test: all bootstrap statistics invalid")
        return {"difference": float(delta_hat), "p_value": np.nan, "se": float(se),
                "block_length": b, "test_stat": float(d_obs), "reject": False,
                "one_sided": one_sided}

    # P-value
    if one_sided:
        p_value = (np.sum(d_star <= d_obs) + 1) / (M + 1)
    else:
        p_value = (np.sum(d_star >= d_obs) + 1) / (M + 1)

    result: dict[str, Any] = {
        "difference": float(delta_hat),
        "p_value": float(p_value),
        "se": float(se),
        "block_length": b,
        "test_stat": float(d_obs),
        "reject": p_value < alpha,
        "one_sided": one_sided,
    }

    # CI for two-sided test
    if not one_sided and len(d_star) > 0:
        crit = float(np.quantile(d_star, 1 - alpha / 2))
        result["ci"] = (float(delta_hat - crit * se),
                        float(delta_hat + crit * se))

    return result


# ══════════════════════════════════════════════════════════════
# 6. Out-of-Sample R²
# ══════════════════════════════════════════════════════════════


def oos_r_squared(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_hist_mean: np.ndarray | pd.Series | None = None,
) -> float:
    """Out-of-sample R² (Campbell & Thompson 2008).

    R²_OOS = 1 - SS_pred / SS_hist

    Parameters
    ----------
    y_true : Realized returns.
    y_pred : Model predictions.
    y_hist_mean : Historical expanding mean at each prediction time.
        If None, uses full-sample mean (with warning).

    Returns
    -------
    float : Positive values indicate model beats historical mean.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    mask = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t, y_p = y_t[mask], y_p[mask]

    ss_pred = np.sum((y_t - y_p) ** 2)

    if y_hist_mean is not None:
        y_h = np.asarray(y_hist_mean, dtype=np.float64)[mask]
    else:
        y_h = np.full_like(y_t, np.mean(y_t))
        logger.warning(
            "oos_r_squared: using full-sample mean (not expanding). "
            "Provide y_hist_mean for proper OOS R²."
        )

    ss_hist = np.sum((y_t - y_h) ** 2)

    if ss_hist < 1e-16:
        return np.nan
    return float(1.0 - ss_pred / ss_hist)


# ══════════════════════════════════════════════════════════════
# 7. Paired t-test
# ══════════════════════════════════════════════════════════════


def paired_ttest(
    values_a: np.ndarray | pd.Series,
    values_b: np.ndarray | pd.Series,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Paired t-test for matched samples.

    For comparing feature importance scores between standard and
    weighted training across rolling windows (Hypothesis H2).

    Parameters
    ----------
    values_a, values_b : Paired observations.
    alternative : 'two-sided', 'greater', or 'less'.

    Returns
    -------
    dict: mean_diff, t_stat, p_value, n_pairs, ci_lower, ci_upper
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    diff = a - b
    diff = diff[~np.isnan(diff)]

    n = len(diff)
    if n < 2:
        return {"mean_diff": np.nan, "t_stat": np.nan, "p_value": np.nan,
                "n_pairs": n, "ci_lower": np.nan, "ci_upper": np.nan}

    mean_diff = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n))

    if se < 1e-15:
        # All differences identical (e.g. zero) → no evidence of difference
        return {"mean_diff": mean_diff, "t_stat": 0.0, "p_value": 1.0,
                "n_pairs": n, "ci_lower": mean_diff, "ci_upper": mean_diff}

    t_result = stats.ttest_1samp(diff, popmean=0.0, alternative=alternative)

    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        "mean_diff": mean_diff,
        "t_stat": float(t_result.statistic),
        "p_value": float(t_result.pvalue),
        "n_pairs": n,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


# ══════════════════════════════════════════════════════════════
# 8. 2×2 Effect Decomposition
# ══════════════════════════════════════════════════════════════


def compute_effect_decomposition(
    returns_1a: np.ndarray | pd.Series,
    returns_1b: np.ndarray | pd.Series,
    returns_2a: np.ndarray | pd.Series,
    returns_2b: np.ndarray | pd.Series,
    ff_factors: pd.DataFrame | None = None,
    yyyymm: np.ndarray | pd.Series | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Full 2×2 effect decomposition with statistical tests.

    Cells:
        1A = Standard training, standard portfolio (baseline)
        1B = Standard training, liquidity-weighted portfolio
        2A = Weighted training, standard portfolio
        2B = Weighted training, liquidity-weighted portfolio

    Parameters
    ----------
    returns_1a .. returns_2b : Monthly long-short returns for each cell.
    ff_factors : Fama-French factors (if None, loaded automatically).
    yyyymm : Date identifiers for factor alpha merge.
    config : Override config.

    Returns
    -------
    dict with nested results for Sharpe ratios, LW tests, factor alphas.
    """
    cells = {"1A": returns_1a, "1B": returns_1b,
             "2A": returns_2a, "2B": returns_2b}

    # Sharpe ratios
    sharpes = {k: sharpe_ratio(v) for k, v in cells.items()}

    # Effect decomposition
    portfolio_effect = sharpes["1B"] - sharpes["1A"]
    training_effect = sharpes["2A"] - sharpes["1A"]
    total_effect = sharpes["2B"] - sharpes["1A"]
    interaction = total_effect - portfolio_effect - training_effect

    # Ledoit-Wolf tests
    lw_portfolio = bootstrap_sharpe_test(returns_1b, returns_1a, config=config)
    lw_training = bootstrap_sharpe_test(returns_2a, returns_1a, config=config)
    lw_total = bootstrap_sharpe_test(returns_2b, returns_1a, config=config)
    # H3: Training > Portfolio ⟺ SR(2A) > SR(1B)
    lw_h3 = bootstrap_sharpe_test(returns_2a, returns_1b, one_sided=True, config=config)

    # Factor alphas (if yyyymm provided)
    alphas: dict[str, dict] = {}
    if yyyymm is not None:
        for model_name in FACTOR_MODELS:
            alphas[model_name] = {}
            for cell_name, cell_ret in cells.items():
                ret_df = pd.DataFrame({
                    "yyyymm": np.asarray(yyyymm),
                    "ret_long_short": np.asarray(cell_ret),
                })
                alphas[model_name][cell_name] = factor_alpha(
                    ret_df, model=model_name, ff_factors=ff_factors, config=config,
                )

    return {
        "sharpe_ratios": sharpes,
        "portfolio_effect": portfolio_effect,
        "training_effect": training_effect,
        "total_effect": total_effect,
        "interaction": interaction,
        "lw_portfolio": lw_portfolio,
        "lw_training": lw_training,
        "lw_total": lw_total,
        "lw_h3": lw_h3,
        "factor_alphas": alphas,
    }
