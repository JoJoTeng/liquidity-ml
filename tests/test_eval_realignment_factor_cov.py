"""Unit tests for the JKMP-style factor covariance model (script 46, factor mode)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.eval_realignment.factor_covariance import (
    FactorRiskModel,
    build_exposures,
    woodbury_solve,
)
from src.analysis.eval_realignment.implementable_mv import (
    monthly_universe_screen,
    mv_net_returns,
    prepare_mv_inputs,
    rescale_positions_unit_gross,
    run_mv_books,
)
from src.portfolio.construction import prepare_transaction_cost_context

CONFIG = {
    "project": {"output_dir": "outputs", "seed": 42},
    "data": {"target_col": "excess_ret"},
    "liquidity": {"primary": "dvol_21d", "quintile_breakpoints": "full_sample"},
    "weighting": {"primary": "dolvol"},
    "transaction_costs": {
        "spread_col": "BidAskSpread",
        "sigma_col": "excess_sigma_12m_daily",
        "adv_col": "dvol_21d",
        "lambda_market_impact": 0.1,
        "lambda_calibration": {"enabled": False},
        "weight_alpha": {"mode": "inverse_median", "scale": 3.0},
    },
    "inference": {
        "newey_west_lags": 6,
        "bootstrap_samples": 100,
        "significance_level": 0.05,
        "one_sided_test": True,
        "calibrate_block_length": False,
        "block_grid": [2, 4, 6],
    },
}

THEME_MAP = {"f1": "Value", "f2": "Value", "f3": "Momentum"}


def _yyyymm_seq(n, start=200001):
    out, y, m = [], start // 100, start % 100
    for _ in range(n):
        out.append(y * 100 + m)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def make_panel(rows):
    return pd.DataFrame(rows).rename(
        columns={
            "dvol": "liq_dvol_21d",
            "spread": "liq_BidAskSpread",
            "sigma": "liq_excess_sigma_12m_daily",
        }
    )


def _factor_panel(n_months=40, n_stocks=10, seed=0):
    """Panel with feature columns f1-f3 and churning returns."""
    rng = np.random.RandomState(seed)
    rows = []
    for month in _yyyymm_seq(n_months):
        feats = rng.uniform(0, 1, (n_stocks, 3))
        shocks = rng.normal(0.0, 0.03, n_stocks)
        for i in range(1, n_stocks + 1):
            er = 0.002 * (i - n_stocks / 2) + shocks[i - 1]
            rows.append({
                "permno": i, "yyyymm": month, "excess_ret": er, "ret": er,
                "dvol": float(i) * 1e6, "spread": 0.01, "sigma": 0.02,
                "f1": feats[i - 1, 0], "f2": feats[i - 1, 1], "f3": feats[i - 1, 2],
            })
    return make_panel(rows)


def _preds(panel, shift=0.0):
    return pd.DataFrame({
        "permno": panel["permno"].to_numpy(),
        "yyyymm": panel["yyyymm"].to_numpy(),
        "prediction": panel["excess_ret"].to_numpy(dtype=float) * 0.1 + shift,
    })


def test_exposures_hand_check():
    df = pd.DataFrame({
        "permno": [1, 2, 3],
        "f1": [0.2, 0.6, 1.0],
        "f2": [0.4, 0.6, 0.8],
        "f3": [1.0, 0.5, 0.0],
    })
    permnos, X = build_exposures(df, THEME_MAP)
    assert list(permnos) == [1, 2, 3]
    assert X.shape == (3, 3)             # intercept + Momentum + Value (alphabetical)
    assert np.allclose(X[:, 0], 1.0)     # intercept
    # Value raw means: (0.3, 0.6, 0.9) -> z-scores symmetric around 0.
    value = X[:, 2]
    assert value[1] == pytest.approx(0.0, abs=1e-12)
    assert value[0] == pytest.approx(-value[2])
    assert np.std(value) == pytest.approx(1.0)
    # Momentum raw (1.0, 0.5, 0.0) -> strictly decreasing z-scores.
    mom = X[:, 1]
    assert mom[0] > mom[1] > mom[2]


def test_factor_regression_recovers_loadings():
    rng = np.random.RandomState(1)
    frm = FactorRiskModel(THEME_MAP)
    true_f = np.array([0.01, 0.02, -0.03])  # intercept, Momentum, Value
    df = pd.DataFrame({
        "permno": np.arange(1, 201),
        "f1": rng.uniform(0, 1, 200),
        "f2": rng.uniform(0, 1, 200),
        "f3": rng.uniform(0, 1, 200),
    })
    permnos, X = frm.exposures(df)
    y = X @ true_f  # noiseless
    frm.update(X, permnos, y)
    f_hat_var = np.sqrt(frm.var_fast)
    assert np.allclose(f_hat_var, np.abs(true_f), atol=1e-10)
    # idio variance ~ 0 in the noiseless case
    assert max(frm.idio.values()) < 1e-20


def test_ewma_recombination_var_vs_corr_halflife():
    frm = FactorRiskModel(THEME_MAP)
    rng = np.random.RandomState(2)
    df = pd.DataFrame({
        "permno": np.arange(1, 101),
        "f1": rng.uniform(0, 1, 100),
        "f2": rng.uniform(0, 1, 100),
        "f3": rng.uniform(0, 1, 100),
    })
    permnos, X = frm.exposures(df)
    for _ in range(30):
        y = X @ rng.normal(0, 0.02, 3) + rng.normal(0, 0.01, 100)
        frm.update(X, permnos, y)
    omega = frm.omega()
    # Symmetric PSD with unit-diagonal correlation structure preserved
    assert np.allclose(omega, omega.T)
    assert np.all(np.linalg.eigvalsh(omega) > -1e-14)
    sig = np.sqrt(frm.var_fast)
    corr = omega / np.outer(sig, sig)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.all(np.abs(corr) <= 1.0 + 1e-9)


def test_woodbury_matches_dense_solve():
    rng = np.random.RandomState(3)
    n, k = 50, 4
    X = rng.normal(size=(n, k))
    C = np.eye(k) * 0.5 + 0.1
    C = (C + C.T) / 2
    a = rng.uniform(0.5, 2.0, n)
    rhs = rng.normal(size=n)
    dense = np.diag(a) + X @ C @ X.T
    expected = np.linalg.solve(dense, rhs)
    got = woodbury_solve(a, X, C, rhs)
    assert np.allclose(got, expected, atol=1e-10)


def test_monthly_universe_screen_top_fraction():
    panel = _factor_panel(n_months=2, n_stocks=10)
    inputs = prepare_mv_inputs(panel, CONFIG)
    months = _yyyymm_seq(2)
    universes = monthly_universe_screen(inputs, months, screen_pct=0.60)
    # 10 names, keep pct rank >= 0.40 -> permnos 4..10 (top 60% + boundary).
    assert set(universes[months[0]]) == {4, 5, 6, 7, 8, 9, 10}
    assert universes[months[0]][0] == 10  # sorted by DolVol descending


def test_run_mv_books_factor_mode_end_to_end():
    panel = _factor_panel(n_months=40, n_stocks=10)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    preds = {"standard": _preds(panel), "weighted": _preds(panel, shift=0.0005)}
    aum = 5e8
    books, diag = run_mv_books(
        preds, panel, CONFIG, tc, gammas=[10.0], aums=[aum],
        cov_model="factor", screen_pct=0.60, theme_map=THEME_MAP,
    )
    assert not diag.empty
    g_mk, p_mk = books[("standard", "markowitz", 10.0, None)]
    g_sm, p_sm = books[("standard", "static_ml", 10.0, aum)]
    assert len(g_mk) > 25 and len(g_sm) > 25
    net_mk = mv_net_returns(g_mk, p_mk, tc, aum)
    net_sm = mv_net_returns(g_sm, p_sm, tc, aum)
    assert net_sm["turnover"].iloc[1:].mean() < net_mk["turnover"].iloc[1:].mean()


def test_unit_gross_view_rescales_and_preserves_raw():
    panel = _factor_panel(n_months=40, n_stocks=10)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    preds = {"standard": _preds(panel), "weighted": _preds(panel)}
    books, _ = run_mv_books(
        preds, panel, CONFIG, tc, gammas=[10.0], aums=[5e8],
        cov_model="factor", screen_pct=0.60, theme_map=THEME_MAP,
    )
    gross_df, positions = books[("standard", "markowitz", 10.0, None)]
    raw_lev = gross_df["leverage"].copy()
    g2, p2 = rescale_positions_unit_gross(gross_df, positions)
    for ym, theta in p2.items():
        assert sum(abs(v) for v in theta.values()) == pytest.approx(1.0)
    assert (g2["leverage"] == 1.0).all()
    # ret_gross rescaled by the month's leverage
    merged = g2.merge(gross_df, on="yyyymm", suffixes=("_u", "_raw"))
    assert np.allclose(
        merged["ret_gross_u"] * merged["leverage_raw"], merged["ret_gross_raw"]
    )
    # the raw book is untouched
    assert gross_df["leverage"].equals(raw_lev)


def test_short_history_names_get_median_idio():
    frm = FactorRiskModel(THEME_MAP)
    rng = np.random.RandomState(4)
    df = pd.DataFrame({
        "permno": np.arange(1, 51),
        "f1": rng.uniform(0, 1, 50),
        "f2": rng.uniform(0, 1, 50),
        "f3": rng.uniform(0, 1, 50),
    })
    permnos, X = frm.exposures(df)
    for _ in range(15):
        y = X @ rng.normal(0, 0.02, 3) + rng.normal(0, 0.05, 50)
        frm.update(X, permnos, y)
    # A brand-new name (no residual history) gets the cross-sectional median.
    d = frm.idio_var(np.array([1, 2, 9999]))
    seasoned = [v for p, v in frm.idio.items() if frm.idio_n[p] >= 12]
    assert d[2] == pytest.approx(float(np.median(seasoned)))
    assert d[0] == pytest.approx(frm.idio[1])
    assert np.all(d > 0)
