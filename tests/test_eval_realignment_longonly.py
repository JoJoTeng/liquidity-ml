"""Unit tests for the capacity-weighted long-only Q5 book (note section 4.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.eval_realignment.breakeven_capacity import metrics_from_book
from src.analysis.eval_realignment.longonly_capacity import (
    build_longonly_book,
    compute_longonly_inputs,
    summarize_hysteresis_diag,
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

DOLVOL_SPEC = {
    "model": "test", "weight_family": "dolvol", "weight_spec": "dolvol",
    "aum_label": None, "softmax_lambda": None, "tc_rank_lambda": None,
    "spec_label": "test_dolvol",
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


def _row(month, permno, er, dvol, spread=0.01):
    return {
        "permno": permno, "yyyymm": month, "excess_ret": er, "ret": er,
        "dvol": float(dvol), "spread": spread, "sigma": 0.02,
    }


def _ladder_panel(n_months=2, spread=0.01):
    """10 stocks, dvol = permno (screen keeps permnos 4-10), er = permno/100."""
    rows = []
    for month in _yyyymm_seq(n_months):
        for i in range(1, 11):
            rows.append(_row(month, i, er=i / 100.0, dvol=i, spread=spread))
    return make_panel(rows)


def _preds_from(panel, values: dict[tuple[int, int], float] | None = None):
    """Predictions = excess_ret unless overridden per (permno, yyyymm)."""
    preds = pd.DataFrame({
        "permno": panel["permno"].to_numpy(),
        "yyyymm": panel["yyyymm"].to_numpy(),
        "prediction": panel["excess_ret"].to_numpy(dtype=float),
    })
    if values:
        for (permno, ym), val in values.items():
            preds.loc[
                (preds["permno"] == permno) & (preds["yyyymm"] == ym), "prediction"
            ] = val
    return preds


def _churn_panel(n_months=36, n_stocks=12, spread=0.02, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for month in _yyyymm_seq(n_months):
        shocks = rng.normal(0.0, 0.02, n_stocks)
        for i in range(1, n_stocks + 1):
            er = 0.002 * i + shocks[i - 1]
            rows.append(_row(month, i, er=er, dvol=i, spread=spread))
    return make_panel(rows)


def test_screen_excludes_illiquid_names():
    panel = _ladder_panel(2)
    months = _yyyymm_seq(2)
    # Give the most illiquid name (permno 1, DolVol pct 0.1) the BEST prediction.
    preds = _preds_from(panel, {(1, months[0]): 0.50, (1, months[1]): 0.50})
    tc = prepare_transaction_cost_context(panel, CONFIG)
    for hysteresis in (False, True):
        _, positions = build_longonly_book(
            preds, panel, CONFIG, DOLVOL_SPEC, tc,
            hysteresis=hysteresis, min_names=2,
        )
        for theta in positions.values():
            assert 1 not in theta


def test_q5_selection_and_capacity_weights():
    panel = _ladder_panel(1)
    preds = _preds_from(panel)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    gross_df, positions = build_longonly_book(
        preds, panel, CONFIG, DOLVOL_SPEC, tc, hysteresis=False, min_names=2,
    )
    ym = _yyyymm_seq(1)[0]
    theta = positions[ym]
    # Liquid universe = permnos 4-10; b_t = 80th pct of their predictions
    # (0.04..0.10) = 0.088 -> Q5 = {9, 10}; dolvol weights -> 9/19, 10/19.
    assert set(theta) == {9, 10}
    assert theta[9] == pytest.approx(9 / 19)
    assert theta[10] == pytest.approx(10 / 19)
    assert sum(theta.values()) == pytest.approx(1.0)
    assert all(v > 0 for v in theta.values())
    assert int(gross_df["n_short"].iloc[0]) == 0


def test_tiny_spread_hysteresis_equals_plain():
    panel = _churn_panel(18, spread=1e-12)
    preds = _preds_from(panel)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    inputs = compute_longonly_inputs(preds, panel, CONFIG, DOLVOL_SPEC)
    g_plain, p_plain = build_longonly_book(
        preds, panel, CONFIG, DOLVOL_SPEC, tc,
        hysteresis=False, min_names=2, inputs=inputs,
    )
    g_hyst, p_hyst = build_longonly_book(
        preds, panel, CONFIG, DOLVOL_SPEC, tc,
        hysteresis=True, min_names=2, inputs=inputs,
    )
    assert set(p_plain) == set(p_hyst)
    for ym in p_plain:
        assert p_hyst[ym] == pytest.approx(p_plain[ym])
    pd.testing.assert_frame_equal(
        g_plain.reset_index(drop=True), g_hyst.reset_index(drop=True)
    )


def test_hysteresis_membership_hand_check():
    m1, m2, m3 = _yyyymm_seq(3)
    rows = []
    for month in (m1, m2, m3):
        for i in range(1, 11):
            dvol = i
            if month == m3 and i == 10:
                dvol = 0.5  # permno 10 leaves the top-60% screen in month 3
            rows.append(_row(month, i, er=i / 100.0, dvol=dvol, spread=0.01))
    panel = make_panel(rows)
    # Month-2 predictions for the liquid names p4..p10:
    #   p4..p7 = .01,.02,.03,.04 ; p8 = .09 ; p9 = .085 ; p10 = .10
    #   b_t = quantile(.8) = .089 -> Q5 = {8, 10}; p9 is below by .004 < buffer .01.
    overrides = {
        (4, m2): 0.01, (5, m2): 0.02, (6, m2): 0.03, (7, m2): 0.04,
        (8, m2): 0.09, (9, m2): 0.085, (10, m2): 0.10,
        # Month 3: p9 collapses far below the boundary -> sold.
        (4, m3): 0.01, (5, m3): 0.02, (6, m3): 0.03, (7, m3): 0.04,
        (8, m3): 0.09, (9, m3): 0.001,
    }
    preds = _preds_from(panel, overrides)
    tc = prepare_transaction_cost_context(panel, CONFIG)

    diag: list[dict] = []
    _, p_hyst = build_longonly_book(
        preds, panel, CONFIG, DOLVOL_SPEC, tc,
        hysteresis=True, min_names=2, diag_out=diag,
    )
    _, p_plain = build_longonly_book(
        preds, panel, CONFIG, DOLVOL_SPEC, tc, hysteresis=False, min_names=2,
    )

    assert set(p_plain[m1]) == {9, 10}          # month 1: Q5 = {9, 10}, both books
    assert set(p_plain[m2]) == {8, 10}          # plain: p9 exits with the boundary
    assert 9 in p_hyst[m2]                      # hysteresis: p9 held in the buffer
    assert {8, 10} <= set(p_hyst[m2])
    assert 9 not in p_hyst[m3]                  # far below the buffer -> sold
    assert 10 not in p_hyst[m3]                 # left the screen -> force-sold
    d2 = next(d for d in diag if d["yyyymm"] == m2)
    assert d2["n_buffer_held"] == 1
    summary = summarize_hysteresis_diag(pd.DataFrame([
        {"row_type": "standard", **d} for d in diag
    ]))
    assert summary["mean_buffer_held"] > 0


def test_entries_ungated_despite_high_spread():
    m1, m2 = _yyyymm_seq(2)
    rows = []
    for month in (m1, m2):
        for i in range(1, 11):
            spread = 0.20 if i == 8 else 0.01  # p8 very expensive to trade
            rows.append(_row(month, i, er=i / 100.0, dvol=i, spread=spread))
    panel = make_panel(rows)
    overrides = {(8, m2): 0.50}  # p8 jumps into Q5 in month 2
    preds = _preds_from(panel, overrides)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    for hysteresis in (False, True):
        _, positions = build_longonly_book(
            preds, panel, CONFIG, DOLVOL_SPEC, tc,
            hysteresis=hysteresis, min_names=2,
        )
        assert 8 in positions[m2]


def test_hysteresis_reduces_turnover():
    panel = _churn_panel(36, spread=0.02)
    preds = _preds_from(panel)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    inputs = compute_longonly_inputs(preds, panel, CONFIG, DOLVOL_SPEC)
    plain = build_longonly_book(
        preds, panel, CONFIG, DOLVOL_SPEC, tc,
        hysteresis=False, min_names=2, inputs=inputs,
    )
    hyst = build_longonly_book(
        preds, panel, CONFIG, DOLVOL_SPEC, tc,
        hysteresis=True, min_names=2, inputs=inputs,
    )
    row_plain, _ = metrics_from_book(*plain, tc, "proportional_tc")
    row_hyst, _ = metrics_from_book(*hyst, tc, "proportional_tc")
    assert row_hyst["turnover_mean"] < row_plain["turnover_mean"]


def test_metrics_schema_matches_42():
    panel = _churn_panel(24)
    preds = _preds_from(panel)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    book = build_longonly_book(
        preds, panel, CONFIG, DOLVOL_SPEC, tc, hysteresis=False, min_names=2,
    )
    row, series = metrics_from_book(*book, tc, 500)
    for key in [
        "gross_sr_annual", "net_sr_annual", "net_ce_annual_g5",
        "turnover_mean", "tc_mean_monthly", "n_months",
    ]:
        assert key in row
    assert list(series.columns) == [
        "ret_gross", "ret_net", "transaction_cost", "turnover", "n_long", "n_short",
    ]


def test_old_format_reuse_on_longonly_cells():
    panel = _churn_panel(36)
    tc = prepare_transaction_cost_context(panel, CONFIG)
    preds = _preds_from(panel)
    inputs = compute_longonly_inputs(preds, panel, CONFIG, DOLVOL_SPEC)
    plain = build_longonly_book(
        preds, panel, CONFIG, DOLVOL_SPEC, tc,
        hysteresis=False, min_names=2, inputs=inputs,
    )
    hyst = build_longonly_book(
        preds, panel, CONFIG, DOLVOL_SPEC, tc,
        hysteresis=True, min_names=2, inputs=inputs,
    )
    _, ser_plain = metrics_from_book(*plain, tc, "proportional_tc")
    _, ser_hyst = metrics_from_book(*hyst, tc, "proportional_tc")
    common = sorted(set(ser_plain.index) & set(ser_hyst.index))
    cells = {
        "1A": ser_plain.loc[common], "1B": ser_hyst.loc[common],
        "2A": ser_plain.loc[common], "2B": ser_hyst.loc[common],
    }
    rows = {r["metric"]: r["value"] for r in
            format_two_by_two_rows(cells, CONFIG, factor_alphas=False)}
    assert "SR_net_annualized(1B)" in rows
    assert rows["Net interaction annualized"] == pytest.approx(
        rows["Net total effect annualized"]
        - rows["Net training effect annualized"]
        - rows["Net portfolio effect annualized"]
    )
    # 2A == 1A here by construction, so the training effect is exactly zero.
    assert rows["Net training effect annualized"] == pytest.approx(0.0, abs=1e-12)
