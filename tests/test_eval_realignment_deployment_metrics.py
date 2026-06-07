"""Unit tests for the eval_realignment deployment-weighted prediction metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.eval_realignment.deployment_weighted_metrics import (
    UNIVERSES,
    compute_deployment_weighted_error_differential,
    compute_deployment_weighted_r2,
    summarize_error_differential,
)

CONFIG = {
    "project": {"output_dir": "outputs"},
    "data": {"target_col": "excess_ret"},
    "liquidity": {"primary": "dvol_21d", "quintile_breakpoints": "full_sample"},
    "weighting": {"primary": "dolvol"},
    # dolvol weights do not use the TC inputs, but compute_weights reads this key.
    "transaction_costs": {
        "spread_col": "BidAskSpread",
        "sigma_col": "excess_sigma_12m_daily",
        "adv_col": "dvol_21d",
        "lambda_market_impact": 0.1,
        "weight_alpha": {"mode": "inverse_median", "scale": 3.0},
    },
    "inference": {"newey_west_lags": 6},
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
    """rows: dicts with permno, yyyymm, excess_ret, dvol."""
    df = pd.DataFrame(rows)
    df = df.rename(columns={"dvol": "liq_dvol_21d"})
    return df[["permno", "yyyymm", "excess_ret", "liq_dvol_21d"]]


def make_preds(panel: pd.DataFrame, values) -> pd.DataFrame:
    """Build a predictions frame aligned to panel rows."""
    return pd.DataFrame(
        {
            "permno": panel["permno"].to_numpy(),
            "yyyymm": panel["yyyymm"].to_numpy(),
            "prediction": np.asarray(values, dtype=float),
        }
    )


def _full_diff(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Return the full-cross-section rows of a stacked differential frame."""
    return diff_df[diff_df["universe"] == "full"].reset_index(drop=True)


def test_r2_weighted_hand_computed():
    """Full-universe R^2_w matches a hand computation with non-uniform w-tilde."""
    panel = make_panel(
        [
            {"permno": 1, "yyyymm": 200001, "excess_ret": 0.10, "dvol": 1.0},
            {"permno": 2, "yyyymm": 200001, "excess_ret": -0.05, "dvol": 2.0},
            {"permno": 3, "yyyymm": 200001, "excess_ret": 0.02, "dvol": 3.0},
        ]
    )
    # w_tilde = dvol / mean(dvol) = [0.5, 1.0, 1.5] (already mean-one).
    std = make_preds(panel, [0.08, -0.03, 0.00])
    wt = make_preds(panel, [0.09, -0.04, 0.01])

    table = compute_deployment_weighted_r2(std, wt, panel, CONFIG, DOLVOL_SPEC)
    full = table[table["universe"] == "full"].iloc[0]

    # ss_tot_w = 0.0081; std ss_res_w = 0.0012; wt ss_res_w = 0.0003.
    assert full["r2_weighted_std_pct"] == pytest.approx(85.18518, rel=1e-4)
    assert full["r2_weighted_wt_pct"] == pytest.approx(96.29629, rel=1e-4)
    assert full["delta_pct"] == pytest.approx(11.11111, rel=1e-4)
    assert int(full["n_obs"]) == 3


def test_constant_weights_reduce_to_unweighted_mse():
    """Equal dvol => w-tilde == 1 => weighted MSE equals the plain mean."""
    rows = []
    for month in (200001, 200002):
        for i in range(5):
            rows.append(
                {
                    "permno": i + 1,
                    "yyyymm": month,
                    "excess_ret": 0.01 * (i - 2) + (0.0 if month == 200001 else 0.005),
                    "dvol": 1.0,  # constant -> w-tilde == 1
                }
            )
    panel = make_panel(rows)
    std = make_preds(panel, np.linspace(-0.02, 0.02, len(panel)))
    wt = make_preds(panel, np.linspace(-0.01, 0.03, len(panel)))

    diff = _full_diff(
        compute_deployment_weighted_error_differential(std, wt, panel, CONFIG, DOLVOL_SPEC)
    ).set_index("yyyymm")

    # Manual unweighted per-month MSE.
    merged_std = std.merge(panel, on=["permno", "yyyymm"])
    merged_wt = wt.merge(panel, on=["permno", "yyyymm"])
    man_std = merged_std.assign(
        e=(merged_std["excess_ret"] - merged_std["prediction"]) ** 2
    ).groupby("yyyymm")["e"].mean()
    man_wt = merged_wt.assign(
        e=(merged_wt["excess_ret"] - merged_wt["prediction"]) ** 2
    ).groupby("yyyymm")["e"].mean()

    np.testing.assert_allclose(diff["wmse_std"].to_numpy(), man_std.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(diff["wmse_wt"].to_numpy(), man_wt.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(
        diff["wmse_diff"].to_numpy(),
        (man_std - man_wt).to_numpy(),
        atol=1e-12,
    )


def test_weighted_mse_uses_weight_denominator():
    """Non-uniform w-tilde: weighted MSE divides by sum(w), not the count."""
    panel = make_panel(
        [
            {"permno": 1, "yyyymm": 200001, "excess_ret": 0.10, "dvol": 1.0},
            {"permno": 2, "yyyymm": 200001, "excess_ret": -0.04, "dvol": 3.0},
        ]
    )
    # w_tilde = [0.5, 1.5]; std predicts zero -> err^2 = [0.01, 0.0016].
    std = make_preds(panel, [0.0, 0.0])
    wt = make_preds(panel, [0.10, -0.04])

    diff = _full_diff(
        compute_deployment_weighted_error_differential(std, wt, panel, CONFIG, DOLVOL_SPEC)
    )
    # weighted: (0.5*0.01 + 1.5*0.0016) / (0.5 + 1.5) = 0.0037 (not 0.0058).
    assert diff["wmse_std"].iloc[0] == pytest.approx(0.0037, rel=1e-9)
    assert diff["wmse_std"].iloc[0] != pytest.approx(0.0058, rel=1e-6)


def test_sign_convention_weighted_better():
    """Weighted strictly better => positive differential and positive R^2 delta."""
    panel = make_panel(
        [
            {"permno": i + 1, "yyyymm": 200001, "excess_ret": 0.01 * (i - 2), "dvol": float(i + 1)}
            for i in range(5)
        ]
    )
    std = make_preds(panel, np.zeros(len(panel)))           # all wrong
    wt = make_preds(panel, panel["excess_ret"].to_numpy())  # perfect

    diff = _full_diff(
        compute_deployment_weighted_error_differential(std, wt, panel, CONFIG, DOLVOL_SPEC)
    )
    assert diff["wmse_diff"].iloc[0] > 0

    table = compute_deployment_weighted_r2(std, wt, panel, CONFIG, DOLVOL_SPEC)
    full = table[table["universe"] == "full"].iloc[0]
    assert full["delta_pct"] > 0
    assert full["r2_weighted_wt_pct"] == pytest.approx(100.0, abs=1e-9)


def test_per_quintile_universes_and_counts():
    """Reports each quintile + pooled + full; Q4-Q5 = 10, Qk = 5 on 25 stocks."""
    n = 25
    panel = make_panel(
        [
            {"permno": i + 1, "yyyymm": 200001, "excess_ret": 0.001 * (i - 12), "dvol": float(i + 1)}
            for i in range(n)
        ]
    )
    rng = np.random.RandomState(0)
    std = make_preds(panel, rng.normal(0, 0.01, n))
    wt = make_preds(panel, rng.normal(0, 0.01, n))

    table = compute_deployment_weighted_r2(std, wt, panel, CONFIG, DOLVOL_SPEC)
    assert list(table["universe"]) == list(UNIVERSES)
    assert list(UNIVERSES) == ["Q1", "Q2", "Q3", "Q4", "Q5", "liquid_q4q5", "full"]

    counts = dict(zip(table["universe"], table["n_obs"]))
    # 25 stocks -> 5 per quintile -> Q4-Q5 = 10.
    for q in (1, 2, 3, 4, 5):
        assert int(counts[f"Q{q}"]) == 5
    assert int(counts["liquid_q4q5"]) == 10
    assert int(counts["full"]) == n

    # Error differential is stacked over the same universes.
    diff = compute_deployment_weighted_error_differential(std, wt, panel, CONFIG, DOLVOL_SPEC)
    assert set(diff["universe"].unique()) == set(UNIVERSES)


def test_nan_rows_dropped():
    """NaN target/prediction rows are excluded before the weighted groupby."""
    panel = make_panel(
        [
            {"permno": 1, "yyyymm": 200001, "excess_ret": 0.10, "dvol": 1.0},
            {"permno": 2, "yyyymm": 200001, "excess_ret": np.nan, "dvol": 2.0},
            {"permno": 3, "yyyymm": 200001, "excess_ret": 0.02, "dvol": 3.0},
            {"permno": 4, "yyyymm": 200001, "excess_ret": -0.01, "dvol": 4.0},
        ]
    )
    std = make_preds(panel, [0.0, 0.0, np.nan, 0.0])  # permno 3 prediction NaN
    wt = make_preds(panel, [0.10, 0.0, 0.02, -0.01])

    diff = _full_diff(
        compute_deployment_weighted_error_differential(std, wt, panel, CONFIG, DOLVOL_SPEC)
    )

    # Only permno 1 and 4 survive in the standard leg (2 and 3 dropped).
    # w_tilde for the full month = dvol/mean(dvol), mean(dvol)=2.5 -> [0.4,0.8,1.2,1.6].
    # std wMSE over {1,4}: (0.4*0.10^2 + 1.6*0.01^2)/(0.4+1.6)
    expected = (0.4 * 0.10 ** 2 + 1.6 * 0.01 ** 2) / (0.4 + 1.6)
    assert diff["wmse_std"].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_summary_is_per_universe_dataframe():
    """Newey-West summary returns one row per universe with the documented fields."""
    rows = []
    for k, month in enumerate((200001, 200002, 200003, 200004)):
        for i in range(5):
            rows.append(
                {
                    "permno": i + 1,
                    "yyyymm": month,
                    "excess_ret": 0.01 * (i - 2) + 0.001 * k,
                    "dvol": float(i + 1),
                }
            )
    panel = make_panel(rows)
    std = make_preds(panel, np.zeros(len(panel)))
    wt = make_preds(panel, panel["excess_ret"].to_numpy())

    diff = compute_deployment_weighted_error_differential(std, wt, panel, CONFIG, DOLVOL_SPEC)
    summary = summarize_error_differential(diff, CONFIG)

    assert set(summary.columns) == {
        "universe",
        "mean_wmse_diff",
        "std_err",
        "t_stat",
        "p_value",
        "n_months",
        "newey_west_lags",
    }
    assert list(summary["universe"]) == list(UNIVERSES)
    full_row = summary[summary["universe"] == "full"].iloc[0]
    assert full_row["newey_west_lags"] == 6
    assert full_row["n_months"] == 4
