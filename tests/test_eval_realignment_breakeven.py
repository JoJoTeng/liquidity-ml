"""Unit tests for the breakeven-gated (parameter-free) capacity portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.eval_realignment.breakeven_capacity import (
    _gate_trades,
    build_breakeven_book,
    compute_alpha_maps,
    gate_diagnostics,
    metrics_from_book,
)
from src.analysis.eval_realignment.capacity_portfolio import build_capacity_book
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

DOLVOL_SPEC = {
    "model": "test", "weight_family": "dolvol", "weight_spec": "dolvol",
    "aum_label": None, "softmax_lambda": None, "tc_rank_lambda": None,
    "spec_label": "test_dolvol",
}


def make_panel(rows):
    return pd.DataFrame(rows).rename(
        columns={
            "dvol": "liq_dvol_21d",
            "spread": "liq_BidAskSpread",
            "sigma": "liq_excess_sigma_12m_daily",
        }
    )


def _yyyymm_seq(n, start=200001):
    out, y, m = [], start // 100, start % 100
    for _ in range(n):
        out.append(y * 100 + m)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _book_panel(n_months=24, n_stocks=10, spread=0.01, seed=0):
    """Time-varying (churning) signal so the book actually trades month to month."""
    rng = np.random.RandomState(seed)
    rows = []
    for month in _yyyymm_seq(n_months):
        shocks = rng.normal(0.0, 0.015, n_stocks)
        for i in range(n_stocks):
            er = 0.005 * (i - (n_stocks - 1) / 2.0) + shocks[i]
            rows.append({
                "permno": i + 1, "yyyymm": month, "excess_ret": er, "ret": er,
                "dvol": float(i + 1), "spread": spread, "sigma": 0.02,
            })
    return make_panel(rows)


def _flat_panel(n_months=12, spread=0.01):
    """Constant per-stock signal, equal weights: alpha_i = er_i exactly."""
    ers = [-0.03, -0.02, -0.001, 0.001, 0.02, 0.03]  # mean 0 -> rbar_w = 0
    rows = []
    for month in _yyyymm_seq(n_months):
        for i, er in enumerate(ers):
            rows.append({
                "permno": i + 1, "yyyymm": month, "excess_ret": er, "ret": er,
                "dvol": 1.0, "spread": spread, "sigma": 0.02,
            })
    return make_panel(rows)


def _preds(panel, values=None):
    vals = panel["excess_ret"].to_numpy() if values is None else values
    return pd.DataFrame({
        "permno": panel["permno"].to_numpy(),
        "yyyymm": panel["yyyymm"].to_numpy(),
        "prediction": np.asarray(vals, dtype=float),
    })


def test_gate_hand_check():
    idx = pd.MultiIndex.from_tuples(
        [(1, 200001), (2, 200001), (3, 200001)], names=["permno", "yyyymm"]
    )
    lookup = pd.DataFrame({"liq_BidAskSpread": [0.01, 0.01, 0.01]}, index=idx)
    target = {1: 0.30, 2: -0.30, 3: 0.05}
    drift = {3: 0.04, 4: 0.10}  # 4 has left the universe (not in target)
    alphas = {1: 0.02, 2: -0.02, 3: 0.001}  # half-spread = 0.005
    out = _gate_trades(target, drift, alphas, lookup, 200001, "liq_BidAskSpread")
    assert out[1] == pytest.approx(0.30)    # passes -> trades fully to target
    assert out[2] == pytest.approx(-0.30)   # passes (sign-symmetric)
    assert out[3] == pytest.approx(0.04)    # fails -> holds the drifted position
    assert 4 not in out                     # universe exit -> closed


def test_tiny_spread_limit_reproduces_capacity_book():
    panel = _book_panel(18, spread=1e-9)
    preds = _preds(panel)
    tc = prepare_transaction_cost_context(panel, CONFIG)

    g_gate, pos_gate = build_breakeven_book(preds, panel, CONFIG, DOLVOL_SPEC, tc)
    g_ref, pos_ref = build_capacity_book(preds, panel, CONFIG, DOLVOL_SPEC)

    assert set(pos_gate) == set(pos_ref)
    for ym in pos_ref:
        assert set(pos_gate[ym]) == set(pos_ref[ym])
        for p, v in pos_ref[ym].items():
            assert pos_gate[ym][p] == pytest.approx(v, abs=1e-12)
    assert list(g_gate["yyyymm"]) == list(g_ref["yyyymm"])


def test_huge_spread_limit_never_opens_book():
    panel = _book_panel(18, spread=10.0)  # half-spread 5.0 >> any alpha
    preds = _preds(panel)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    gross_df, positions = build_breakeven_book(preds, panel, CONFIG, DOLVOL_SPEC, tc)
    assert positions == {}
    assert gross_df.empty


def test_turnover_reduced_vs_full_rebalance():
    panel = _book_panel(30, spread=0.02)  # half-spread 1% vs alphas ~ +/-2%
    preds = _preds(panel)
    tc = prepare_transaction_cost_context(panel, CONFIG)

    targets = build_capacity_book(preds, panel, CONFIG, DOLVOL_SPEC)
    gated = build_breakeven_book(preds, panel, CONFIG, DOLVOL_SPEC, tc, targets=targets)

    row_full, _ = metrics_from_book(*targets, tc, "proportional_tc")
    row_gate, _ = metrics_from_book(*gated, tc, "proportional_tc")
    assert row_gate["turnover_mean"] < row_full["turnover_mean"]


def test_universe_exit_closes_position():
    panel = _book_panel(20, spread=1e-9)  # all names pass the gate
    preds = _preds(panel)
    cutoff = _yyyymm_seq(20)[9]
    preds = preds[~((preds["permno"] == 1) & (preds["yyyymm"] > cutoff))]
    tc = prepare_transaction_cost_context(panel, CONFIG)
    _, positions = build_breakeven_book(preds, panel, CONFIG, DOLVOL_SPEC, tc)
    assert any(1 in theta for ym, theta in positions.items() if ym <= cutoff)
    for ym, theta in positions.items():
        if ym > cutoff:
            assert 1 not in theta


def test_constraints_preserved():
    panel = _book_panel(18, spread=0.02)
    preds = _preds(panel)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    _, positions = build_breakeven_book(preds, panel, CONFIG, DOLVOL_SPEC, tc)
    assert positions
    for theta in positions.values():
        vals = np.array(list(theta.values()))
        assert vals.sum() == pytest.approx(0.0, abs=1e-9)          # dollar-neutral
        assert np.abs(vals).sum() == pytest.approx(1.0, abs=1e-9)  # unit gross


def test_failed_gate_name_never_opens():
    panel = _flat_panel(12, spread=0.01)  # alphas = er; half-spread = 0.005
    preds = _preds(panel)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    _, positions = build_breakeven_book(preds, panel, CONFIG, DOLVOL_SPEC, tc)
    assert positions
    for theta in positions.values():
        assert 3 not in theta and 4 not in theta  # |alpha|=0.001 < 0.005, drift 0
        assert {1, 2, 5, 6} <= set(theta)


def test_gate_diagnostics_pass_fractions():
    panel = _flat_panel(6, spread=0.01)
    preds = _preds(panel)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    targets = build_capacity_book(preds, panel, CONFIG, DOLVOL_SPEC)
    alpha_maps = compute_alpha_maps(preds, panel, CONFIG, DOLVOL_SPEC)
    diag = gate_diagnostics(targets[1], alpha_maps, tc)
    assert (diag["n_names"] == 6).all()
    assert (diag["n_pass"] == 4).all()
    assert diag["pass_frac"].iloc[0] == pytest.approx(4 / 6)
    # The two failing names carry tiny |theta*|, so the gross-weighted pass
    # fraction must exceed the count-based one.
    assert (diag["gross_pass_frac"] > diag["pass_frac"]).all()
    assert diag["median_half_spread"].iloc[0] == pytest.approx(0.005)
