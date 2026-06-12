"""Unit tests for the implementable MV optimizer (note section 4.5, JKMP Static tier)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.eval_realignment.capacity_portfolio import certainty_equivalent
from src.analysis.eval_realignment.implementable_mv import (
    _drift_unnormalized,
    monthly_universe,
    mv_metrics,
    mv_net_returns,
    prepare_mv_inputs,
    run_mv_books,
    solve_static_ml,
)
from src.analysis.eval_realignment.two_by_two_tables import format_two_by_two_rows
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


def _mv_panel(n_months=50, n_stocks=8, seed=0, skip=None):
    """Churning returns; dvol proportional to permno. skip = {permno: months_present}."""
    rng = np.random.RandomState(seed)
    rows = []
    months = _yyyymm_seq(n_months)
    for j, month in enumerate(months):
        shocks = rng.normal(0.0, 0.03, n_stocks)
        for i in range(1, n_stocks + 1):
            if skip and i in skip and j < n_months - skip[i]:
                continue
            er = 0.002 * (i - n_stocks / 2) + shocks[i - 1]
            rows.append({
                "permno": i, "yyyymm": month, "excess_ret": er, "ret": er,
                "dvol": float(i) * 1e6, "spread": 0.01, "sigma": 0.02,
            })
    return make_panel(rows)


def _preds(panel, shift=0.0):
    return pd.DataFrame({
        "permno": panel["permno"].to_numpy(),
        "yyyymm": panel["yyyymm"].to_numpy(),
        "prediction": panel["excess_ret"].to_numpy(dtype=float) * 0.1 + shift,
    })


def test_solve_static_ml_two_asset_hand_check():
    sigma = np.array([[0.04, 0.0], [0.0, 0.01]])
    mu = np.array([0.02, 0.01])
    lam = np.array([0.10, 0.20])
    drift = np.array([0.10, -0.05])
    theta = solve_static_ml(mu, sigma, lam, drift, gamma=2.0)
    # lhs = diag(0.18, 0.22); rhs = mu + lam*drift = (0.03, 0.0)
    assert theta[0] == pytest.approx(0.03 / 0.18)
    assert theta[1] == pytest.approx(0.0, abs=1e-15)


def test_lambda_zero_is_markowitz():
    rng = np.random.RandomState(1)
    A = rng.normal(size=(20, 4))
    sigma = A.T @ A / 20 + 0.01 * np.eye(4)
    mu = rng.normal(0.01, 0.005, 4)
    drift = rng.normal(0.0, 0.1, 4)
    gamma = 5.0
    expected = np.linalg.solve(gamma * sigma, mu)
    for lam in (None, np.zeros(4)):
        theta = solve_static_ml(mu, sigma, lam, drift, gamma)
        assert np.allclose(theta, expected, atol=1e-12)


def test_large_lambda_pins_to_drift():
    sigma = np.array([[0.04, 0.01], [0.01, 0.02]])
    mu = np.array([0.02, -0.01])
    drift = np.array([0.30, -0.10])
    theta = solve_static_ml(mu, sigma, np.full(2, 1e9), drift, gamma=5.0)
    assert np.allclose(theta, drift, atol=1e-6)


def test_drift_unnormalized_hand_check():
    idx = pd.MultiIndex.from_tuples([(1, 200001), (2, 200001)],
                                    names=["permno", "yyyymm"])
    lookup = pd.DataFrame({"ret": [0.10, 0.0]}, index=idx)
    drift = _drift_unnormalized(
        {1: 0.5, 2: -0.2}, 200001, 200002, lookup, [200001, 200002]
    )
    r_p = 0.5 * 0.10 + (-0.2) * 0.0
    assert drift[1] == pytest.approx(0.5 * 1.10 / (1 + r_p))
    assert drift[2] == pytest.approx(-0.2 * 1.00 / (1 + r_p))
    gross = abs(drift[1]) + abs(drift[2])
    assert gross != pytest.approx(0.7)  # drifted, and NOT renormalized to prior gross


def test_drift_compounds_over_skipped_months():
    idx = pd.MultiIndex.from_tuples(
        [(1, 200001), (1, 200002)], names=["permno", "yyyymm"]
    )
    lookup = pd.DataFrame({"ret": [0.10, 0.20]}, index=idx)
    theta = 0.5
    # Gap from 200001 to 200003: must compound BOTH monthly steps.
    drift = _drift_unnormalized(
        {1: theta}, 200001, 200003, lookup, [200001, 200002, 200003]
    )
    step1 = theta * 1.10 / (1 + theta * 0.10)
    step2 = step1 * 1.20 / (1 + step1 * 0.20)
    assert drift[1] == pytest.approx(step2)
    # Single-step call must NOT include the second month.
    one = _drift_unnormalized({1: theta}, 200001, 200002, lookup, [200001, 200002])
    assert one[1] == pytest.approx(step1)


def test_skipped_month_books_still_consistent():
    panel = _mv_panel(n_months=30, n_stocks=6)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    months = _yyyymm_seq(30)
    gap_month = months[20]
    preds_df = _preds(panel)
    preds_df = preds_df[preds_df["yyyymm"] != gap_month]  # no predictions that month
    preds = {"standard": preds_df, "weighted": preds_df.copy()}
    books, diag = run_mv_books(
        preds, panel, CONFIG, tc, gammas=[5.0], aums=[5e8],
        top_n=6, window=12, min_obs=8,
    )
    gross_df, positions = books[("standard", "static_ml", 5.0, 5e8)]
    assert gap_month not in positions               # the month is skipped
    assert any(m > gap_month for m in positions)    # ...but later months exist
    net = mv_net_returns(gross_df, positions, tc, 5e8)
    assert np.isfinite(net["ret_net"]).all()


def test_universe_history_filter_and_top_n():
    panel = _mv_panel(n_months=20, n_stocks=6, skip={6: 5})  # permno 6: only 5 months
    inputs = prepare_mv_inputs(panel, CONFIG)
    months = _yyyymm_seq(20)
    universes = monthly_universe(
        inputs, [months[-1]], top_n=3, window=12, min_obs=8
    )
    u = universes[months[-1]]
    assert 6 not in u            # insufficient history despite highest dvol rank
    assert u == [5, 4, 3]        # top-3 by dvol among eligible names
    assert len(u) == 3


def test_mv_net_returns_tc_formula():
    panel = _mv_panel(n_months=3, n_stocks=3)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    months = _yyyymm_seq(3)
    theta = 0.2
    gross_df = pd.DataFrame({
        "yyyymm": months[:2], "ret_gross": [0.0, 0.0],
        "n_long": [1, 1], "n_short": [0, 0], "leverage": [theta, theta],
    })
    positions = {months[0]: {1: theta}, months[1]: {1: theta}}
    aum_dollars = 500 * 1e6
    net = mv_net_returns(gross_df, positions, tc, aum_dollars)
    row = tc["lookup"].loc[(1, months[0])]
    spread = float(row[tc["spread_col"]])
    sigma = float(row[tc["sigma_col"]])
    adv = float(row[tc["adv_col"]])
    expected_m1 = theta * (
        spread / 2.0 + tc["lam"] * sigma * np.sqrt(theta * aum_dollars / adv)
    )
    assert net["transaction_cost"].iloc[0] == pytest.approx(expected_m1)
    assert net["turnover"].iloc[0] == pytest.approx(0.5 * theta)


def test_static_ml_turnover_below_markowitz():
    panel = _mv_panel(n_months=50, n_stocks=8)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    preds = {"standard": _preds(panel), "weighted": _preds(panel, shift=0.0005)}
    aum = 5e8
    books, diag = run_mv_books(
        preds, panel, CONFIG, tc, gammas=[5.0], aums=[aum],
        top_n=8, window=12, min_obs=8,
    )
    assert not diag.empty
    net_mk = mv_net_returns(*books[("standard", "markowitz", 5.0, None)], tc, aum)
    net_sm = mv_net_returns(*books[("standard", "static_ml", 5.0, aum)], tc, aum)
    assert net_sm["turnover"].iloc[1:].mean() < net_mk["turnover"].iloc[1:].mean()
    assert len(net_mk) == len(net_sm) > 30


def test_mv_metrics_schema_and_own_gamma():
    months = _yyyymm_seq(24)
    rng = np.random.RandomState(3)
    net = pd.DataFrame({
        "yyyymm": months,
        "ret_gross": rng.normal(0.01, 0.03, 24),
        "n_long": 5, "n_short": 3, "leverage": 1.4,
        "transaction_cost": 0.002, "turnover": 0.3,
    })
    net["ret_net"] = net["ret_gross"] - net["transaction_cost"]
    row, series = mv_metrics(net, gamma=5.0)
    expected_ce = certainty_equivalent(net["ret_net"].to_numpy(), 5.0) * 12
    assert row["net_ce_own_gamma"] == pytest.approx(expected_ce)
    for key in ["gamma", "net_sr_annual", "net_sd_annual",
                "leverage_mean", "turnover_mean", "n_months"]:
        assert key in row
    assert "leverage" in series.columns


def test_two_by_two_reuse_on_mv_cells():
    panel = _mv_panel(n_months=50, n_stocks=8)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    preds = {"standard": _preds(panel), "weighted": _preds(panel, shift=0.0005)}
    aum = 5e8
    books, _ = run_mv_books(
        preds, panel, CONFIG, tc, gammas=[10.0], aums=[aum],
        top_n=8, window=12, min_obs=8,
    )
    cells = {}
    for (row_type, book), cell in [
        (("standard", "markowitz"), "1A"), (("standard", "static_ml"), "1B"),
        (("weighted", "markowitz"), "2A"), (("weighted", "static_ml"), "2B"),
    ]:
        aum_key = None if book == "markowitz" else aum
        net = mv_net_returns(*books[(row_type, book, 10.0, aum_key)], tc, aum)
        cells[cell] = net.set_index("yyyymm")[
            ["ret_gross", "ret_net", "transaction_cost", "turnover",
             "n_long", "n_short"]
        ]
    common = sorted(set.intersection(*(set(s.index) for s in cells.values())))
    cells = {c: s.loc[common] for c, s in cells.items()}
    rows = {r["metric"]: r["value"] for r in
            format_two_by_two_rows(cells, CONFIG, factor_alphas=False)}
    assert "SR_net_annualized(1B)" in rows
    assert rows["Net interaction annualized"] == pytest.approx(
        rows["Net total effect annualized"]
        - rows["Net training effect annualized"]
        - rows["Net portfolio effect annualized"]
    )
