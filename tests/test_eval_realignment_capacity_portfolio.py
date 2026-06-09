"""Unit tests for the eval_realignment signal-weighted capacity portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.eval_realignment.capacity_portfolio import (
    build_capacity_book,
    capacity_metrics_for_predictions,
    capacity_net_returns,
    certainty_equivalent,
    compare_standard_vs_weighted,
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

DOLVOL_SPEC = {
    "model": "test",
    "weight_family": "dolvol",
    "weight_spec": "dolvol",
    "aum_label": None,
    "softmax_lambda": None,
    "tc_rank_lambda": None,
    "spec_label": "test_dolvol",
}


def make_panel(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).rename(
        columns={
            "dvol": "liq_dvol_21d",
            "spread": "liq_BidAskSpread",
            "sigma": "liq_excess_sigma_12m_daily",
        }
    )
    return df


def make_preds(panel: pd.DataFrame, values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "permno": panel["permno"].to_numpy(),
            "yyyymm": panel["yyyymm"].to_numpy(),
            "prediction": np.asarray(values, dtype=float),
        }
    )


def test_theta_dollar_neutral_and_unit_gross():
    rows = [
        {"permno": i + 1, "yyyymm": 200001, "excess_ret": 0.01 * (i - 2.5), "dvol": float(i + 1)}
        for i in range(6)
    ]
    panel = make_panel(rows)
    preds = make_preds(panel, [0.03, 0.02, 0.005, -0.005, -0.02, -0.03])

    gross_df, positions = build_capacity_book(preds, panel, CONFIG, DOLVOL_SPEC)
    assert list(positions.keys()) == [200001]
    theta = np.array(list(positions[200001].values()))
    assert theta.sum() == pytest.approx(0.0, abs=1e-12)        # dollar-neutral
    assert np.abs(theta).sum() == pytest.approx(1.0, abs=1e-12)  # unit gross
    row = gross_df.iloc[0]
    assert row["n_long"] >= 2 and row["n_short"] >= 2


def test_dolvol_theta_reduces_to_eq3():
    """theta = DolVol*(r_hat - rbar_w) normalized to unit gross (note Eq.3)."""
    panel = make_panel(
        [
            {"permno": 1, "yyyymm": 200001, "excess_ret": 0.0, "dvol": 1.0},
            {"permno": 2, "yyyymm": 200001, "excess_ret": 0.0, "dvol": 2.0},
            {"permno": 3, "yyyymm": 200001, "excess_ret": 0.0, "dvol": 3.0},
        ]
    )
    preds = make_preds(panel, [0.02, 0.00, -0.01])
    # rbar_w = sum(dvol*rhat)/sum(dvol) = (1*0.02 + 2*0 + 3*-0.01)/6 = -0.01/6
    dvol = np.array([1.0, 2.0, 3.0])
    rhat = np.array([0.02, 0.0, -0.01])
    rbar_w = (dvol * rhat).sum() / dvol.sum()
    theta_raw = dvol * (rhat - rbar_w)
    expected = theta_raw / np.abs(theta_raw).sum()

    _, positions = build_capacity_book(preds, panel, CONFIG, DOLVOL_SPEC, min_names=1)
    got = np.array([positions[200001][p] for p in (1, 2, 3)])
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert got.sum() == pytest.approx(0.0, abs=1e-12)


def test_unified_net_return_hand_check():
    """2-stock signed book: hand-computed drift, turnover, proportional TC."""
    idx = pd.MultiIndex.from_tuples(
        [(1, 200001), (2, 200001), (1, 200002), (2, 200002)],
        names=["permno", "yyyymm"],
    )
    lookup = pd.DataFrame(
        {
            "ret": [0.10, -0.10, 0.0, 0.0],
            "liq_BidAskSpread": [0.02, 0.02, 0.02, 0.02],
        },
        index=idx,
    )
    tc_context = {
        "lookup": lookup,
        "spread_col": "liq_BidAskSpread",
        "sigma_col": "liq_excess_sigma_12m_daily",
        "adv_col": "liq_dvol_21d",
        "lam": 0.1,
    }
    positions = {
        200001: {1: 0.5, 2: -0.5},
        200002: {1: 0.5, 2: -0.5},
    }
    gross_df = pd.DataFrame(
        {"yyyymm": [200001, 200002], "ret_gross": [0.01, 0.02],
         "n_long": [1, 1], "n_short": [1, 1]}
    )

    out = capacity_net_returns(gross_df, positions, tc_context, "proportional_tc").set_index("yyyymm")

    # Month 1 (no prior): full establishment, both legs traded 0.5 -> turnover 0.5,
    # tc = (0.5 + 0.5) * spread/2 = 1.0 * 0.01 = 0.01.
    assert out.loc[200001, "turnover"] == pytest.approx(0.5, abs=1e-12)
    assert out.loc[200001, "transaction_cost"] == pytest.approx(0.01, abs=1e-12)
    assert out.loc[200001, "ret_net"] == pytest.approx(0.0, abs=1e-12)
    # Month 2: drifted {1: 0.55, 2: -0.45} (gross-renormalized to 1), delta = 0.05 each,
    # turnover 0.05, tc = 0.10 * 0.01 = 0.001.
    assert out.loc[200002, "turnover"] == pytest.approx(0.05, abs=1e-12)
    assert out.loc[200002, "transaction_cost"] == pytest.approx(0.001, abs=1e-12)
    assert out.loc[200002, "ret_net"] == pytest.approx(0.019, abs=1e-12)


def test_certainty_equivalent_formula():
    r = [0.01, 0.02, 0.03]
    expected = np.mean(r) - 0.5 * 5.0 * np.var(r, ddof=1)
    assert certainty_equivalent(r, 5.0) == pytest.approx(expected, rel=1e-12)
    assert np.isnan(certainty_equivalent([0.01], 5.0))


def _yyyymm_sequence(n: int, start: int = 200001) -> list[int]:
    out, year, month = [], start // 100, start % 100
    for _ in range(n):
        out.append(year * 100 + month)
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return out


def _training_effect_panel(n_months: int = 36):
    rng = np.random.RandomState(0)
    rows = []
    for t, month in enumerate(_yyyymm_sequence(n_months)):
        shocks = rng.normal(0.0, 0.005, 6)
        for i in range(6):
            er = 0.01 * (i - 2.5) + shocks[i]
            rows.append(
                {
                    "permno": i + 1,
                    "yyyymm": month,
                    "excess_ret": er,
                    "ret": er,
                    "dvol": float(i + 1),
                    "spread": 0.01,
                    "sigma": 0.02,
                }
            )
    return make_panel(rows)


def test_training_effect_sign():
    """Weighted r_hat = realized, standard = anti-correlated => weighted wins."""
    panel = _training_effect_panel()
    wt = make_preds(panel, panel["excess_ret"].to_numpy())       # perfect
    std = make_preds(panel, -panel["excess_ret"].to_numpy())     # anti-correlated

    tc_context = prepare_transaction_cost_context(panel, CONFIG)

    row_std, series_std = capacity_metrics_for_predictions(
        std, panel, CONFIG, DOLVOL_SPEC, "proportional_tc", tc_context
    )
    row_wt, series_wt = capacity_metrics_for_predictions(
        wt, panel, CONFIG, DOLVOL_SPEC, "proportional_tc", tc_context
    )
    assert row_wt["net_sr_annual"] > row_std["net_sr_annual"]
    assert row_wt["net_ce_annual_g5"] > row_std["net_ce_annual_g5"]

    cmp = compare_standard_vs_weighted(series_std, series_wt, row_std, row_wt, CONFIG)
    assert set(cmp) >= {
        "net_sr_diff",
        "net_sr_diff_pval",
        "net_sr_diff_reject",
        "net_mean_diff_nw_t",
        "net_ce_diff_annual_g1",
        "net_ce_diff_annual_g5",
        "net_ce_diff_annual_g10",
    }
    assert cmp["net_sr_diff"] > 0  # SR(weighted) - SR(standard)


def test_degenerate_month_skipped():
    """A flat-signal month (r_hat constant) yields no book."""
    panel = make_panel(
        [
            {"permno": i + 1, "yyyymm": 200001, "excess_ret": 0.01 * (i - 1.5), "dvol": float(i + 1)}
            for i in range(4)
        ]
    )
    preds = make_preds(panel, [0.01, 0.01, 0.01, 0.01])  # constant -> theta = 0
    gross_df, positions = build_capacity_book(preds, panel, CONFIG, DOLVOL_SPEC)
    assert gross_df.empty
    assert positions == {}
