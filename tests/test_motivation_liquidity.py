"""Tests for motivation-section liquidity proxy helpers."""

import copy

import numpy as np
import pandas as pd

from src.analysis.motivation import (
    _joint_gamma_f_test,
    ensure_motivation_weight_column,
    get_motivation_liquidity_choices,
    get_motivation_liquidity_key,
    interaction_fama_macbeth,
    quintile_fama_macbeth,
)
from src.config import load_config
from src.weighting.schemes import compute_weights


def test_motivation_liquidity_choices_include_tc():
    assert get_motivation_liquidity_choices() == ["dvol", "mcap", "tc"]


def test_motivation_liquidity_key_uses_aum_for_tc():
    assert get_motivation_liquidity_key("dvol", 500) == "dvol"
    assert get_motivation_liquidity_key("mcap", 500) == "mcap"
    assert get_motivation_liquidity_key("tc", 500) == "tc_500m"
    assert get_motivation_liquidity_key("tc", 12.5) == "tc_12p5m"


def test_dvol_weight_column_is_mean_one():
    df = pd.DataFrame({
        "yyyymm": [202301, 202301, 202301],
        "liq_dvol_21d": [1.0, 2.0, 3.0],
    })

    col = ensure_motivation_weight_column(df, "dvol", config=load_config())

    assert col == "w_tilde_dvol"
    np.testing.assert_allclose(df[col].values, [0.5, 1.0, 1.5])
    assert abs(df[col].mean() - 1.0) < 1e-12


def test_dvol_weight_column_matches_formal_scheme_with_missing_values():
    cfg = load_config()
    df = pd.DataFrame({
        "yyyymm": [202301, 202301, 202301, 202301],
        "liq_dvol_21d": [1.0, np.nan, 0.0, 3.0],
    })

    col = ensure_motivation_weight_column(df, "dvol", config=cfg)
    expected = compute_weights(df, scheme="dolvol", config=cfg)

    np.testing.assert_allclose(df[col].values, expected.values)


def test_tc_weight_column_reuses_formal_tc_scheme():
    cfg = copy.deepcopy(load_config())
    cfg["transaction_costs"]["lambda_calibration"]["enabled"] = False

    df = pd.DataFrame({
        "yyyymm": [202301, 202301, 202301, 202301],
        "liq_dvol_21d": [1_000_000.0, 5_000_000.0, 20_000_000.0, 50_000_000.0],
        "liq_BidAskSpread": [0.08, 0.04, 0.02, 0.01],
        "liq_excess_sigma_12m_daily": [0.04, 0.03, 0.02, 0.01],
    })

    col = ensure_motivation_weight_column(
        df, "tc", config=cfg, aum_millions=500.0
    )

    assert col == "w_tilde_tc"
    assert abs(df[col].mean() - 1.0) < 1e-12
    assert df.loc[3, col] > df.loc[0, col]


def test_quintile_fama_macbeth_fills_feature_nans_with_neutral_rank():
    months = list(range(202001, 202013))
    rows = []
    for m in months:
        for i in range(35):
            x = np.nan if i % 5 == 0 else i / 34
            rows.append({
                "yyyymm": m,
                "liq_quintile": 1,
                "excess_ret": 0.01 * (0.5 if np.isnan(x) else x),
                "x": x,
            })

    df = pd.DataFrame(rows)
    result = quintile_fama_macbeth(df, ["x"], "liq_quintile")

    assert pd.notna(result["coef_table"].loc[0, "beta_Q1"])


def test_interaction_fama_macbeth_fills_feature_nans_with_neutral_rank():
    months = list(range(202001, 202013))
    rows = []
    for m in months:
        for i in range(120):
            x = np.nan if i % 7 == 0 else i / 119
            liq_rank = i / 119
            x_filled = 0.5 if np.isnan(x) else x
            rows.append({
                "yyyymm": m,
                "liq_rank": liq_rank,
                "excess_ret": 0.01 * x_filled + 0.02 * x_filled * liq_rank,
                "x": x,
            })

    df = pd.DataFrame(rows)
    result = interaction_fama_macbeth(df, ["x"], "liq_rank")

    assert result["n_months"] == 12
    assert pd.notna(result["coef_table"].loc[0, "beta_bar"])
    assert pd.notna(result["coef_table"].loc[0, "gamma_bar"])


def test_joint_gamma_f_test_uses_hotelling_scaling():
    gamma_df = pd.DataFrame(
        {
            "x1": np.linspace(0.01, 0.03, 20),
            "x2": 0.02 + 0.01 * np.sin(np.linspace(0, 3, 20)),
        }
    )

    f_stat, p_value, df = _joint_gamma_f_test(gamma_df, ["x1", "x2"])

    T = len(gamma_df)
    k = 2
    means = gamma_df.mean().values
    cov = gamma_df.cov().values
    hotelling_t2 = T * means @ np.linalg.inv(cov) @ means
    expected_f = (T - k) / (k * (T - 1)) * hotelling_t2

    from scipy.stats import f as f_dist

    np.testing.assert_allclose(f_stat, expected_f)
    np.testing.assert_allclose(p_value, 1 - f_dist.cdf(expected_f, k, T - k))
    assert df == (k, T - k)
