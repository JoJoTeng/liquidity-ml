"""Tests for src/weighting/schemes.py (LiquidityML v3)

Tests the weighting families:
  - dolvol: DolVol/mean(DolVol)
  - softmax_rank: exp(lambda*rank(DolVol)), mean-normalized
  - tc: Eq.23 exp(-alpha*TC), with configurable alpha scale
  - tc_rank: exp(lambda*rank(-TC)), mean-normalized
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.weighting.schemes import (
    compute_weights,
    compute_tc_for_sorting,
    get_available_schemes,
    _normalize_to_mean_one,
    resolve_market_impact_lambda,
)

_cfg = load_config()
_LIQ_COL = f"liq_{_cfg['liquidity']['primary']}"


# -- Fixtures ---------------------------------------------------------

@pytest.fixture
def single_cross_section():
    """100 stocks in one month with realistic liquidity data."""
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "yyyymm": 202301,
        "permno": range(10001, 10001 + n),
        _LIQ_COL: rng.lognormal(mean=18, sigma=2, size=n),
        "liq_BidAskSpread": rng.uniform(0.001, 0.05, size=n),
        "liq_excess_sigma_12m": rng.uniform(0.04, 0.25, size=n),
        "liq_excess_sigma_12m_daily": rng.uniform(0.04, 0.25, size=n) / np.sqrt(21.0),
        "liq_dvol_21d": rng.lognormal(mean=18, sigma=2, size=n),
    })


@pytest.fixture
def multi_month_panel():
    """3 months x 50 stocks."""
    rng = np.random.RandomState(99)
    months = [202301, 202302, 202303]
    dfs = []
    for m in months:
        n = 50
        dfs.append(pd.DataFrame({
            "yyyymm": m,
            "permno": range(10001, 10001 + n),
            _LIQ_COL: rng.lognormal(mean=18, sigma=2, size=n),
            "liq_BidAskSpread": rng.uniform(0.001, 0.05, size=n),
            "liq_excess_sigma_12m": rng.uniform(0.04, 0.25, size=n),
            "liq_excess_sigma_12m_daily": rng.uniform(0.04, 0.25, size=n) / np.sqrt(21.0),
            "liq_dvol_21d": rng.lognormal(mean=18, sigma=2, size=n),
        }))
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def cross_section_with_nans(single_cross_section):
    df = single_cross_section.copy()
    rng = np.random.RandomState(7)
    mask = rng.choice(len(df), size=10, replace=False)
    df.loc[mask, _LIQ_COL] = np.nan
    df.loc[mask, "liq_BidAskSpread"] = np.nan
    return df


# -- _normalize_to_mean_one -------------------------------------------

class TestNormalizeToMeanOne:
    def test_basic(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = _normalize_to_mean_one(s)
        assert abs(result.mean() - 1.0) < 1e-10

    def test_all_zeros(self):
        s = pd.Series([0.0, 0.0, 0.0])
        result = _normalize_to_mean_one(s)
        assert (result == 1.0).all()

    def test_all_nan(self):
        s = pd.Series([np.nan, np.nan])
        result = _normalize_to_mean_one(s)
        assert (result == 1.0).all()


# -- Available schemes ------------------------------------------------

class TestRegistry:
    def test_available_schemes(self):
        assert set(get_available_schemes()) == {
            "dolvol",
            "softmax_rank",
            "tc",
            "tc_rank",
        }

    def test_unknown_scheme_raises(self, single_cross_section):
        with pytest.raises(ValueError, match="Unknown scheme"):
            compute_weights(single_cross_section, scheme="bad_scheme", config=_cfg)


# -- Mean=1 invariant -------------------------------------------------

class TestMeanOneInvariant:
    def test_dolvol_single_month(self, single_cross_section):
        w = compute_weights(single_cross_section, scheme="dolvol", config=_cfg)
        assert abs(w.mean() - 1.0) < 1e-8

    def test_tc_single_month(self, single_cross_section):
        w = compute_weights(single_cross_section, scheme="tc", config=_cfg, aum=500_000_000)
        assert abs(w.mean() - 1.0) < 1e-8

    def test_softmax_rank_single_month(self, single_cross_section):
        w = compute_weights(single_cross_section, scheme="softmax_rank", config=_cfg)
        assert abs(w.mean() - 1.0) < 1e-8

    def test_tc_rank_single_month(self, single_cross_section):
        w = compute_weights(
            single_cross_section,
            scheme="tc_rank",
            config=_cfg,
            aum=500_000_000,
        )
        assert abs(w.mean() - 1.0) < 1e-8

    def test_dolvol_multi_month(self, multi_month_panel):
        w = compute_weights(multi_month_panel, scheme="dolvol", config=_cfg)
        df = multi_month_panel[["yyyymm"]].copy()
        df["w"] = w
        per_month = df.groupby("yyyymm")["w"].mean()
        np.testing.assert_allclose(per_month.values, 1.0, atol=1e-8)

    def test_dolvol_exact_mean_denominator(self):
        data = {
            "yyyymm": [202301, 202301, 202301],
            "liq_BidAskSpread": [0.01, 0.01, 0.01],
            "liq_excess_sigma_12m": [0.10, 0.10, 0.10],
            "liq_excess_sigma_12m_daily": [0.10 / np.sqrt(21.0)] * 3,
        }
        data[_LIQ_COL] = [1.0, 2.0, 9.0]
        df = pd.DataFrame(data)
        w = compute_weights(df, scheme="dolvol", config=_cfg)
        expected = pd.Series([1.0, 2.0, 9.0]) / 4.0
        np.testing.assert_allclose(w.values, expected.values, atol=1e-12)

    def test_tc_multi_month(self, multi_month_panel):
        w = compute_weights(multi_month_panel, scheme="tc", config=_cfg, aum=500_000_000)
        df = multi_month_panel[["yyyymm"]].copy()
        df["w"] = w
        per_month = df.groupby("yyyymm")["w"].mean()
        np.testing.assert_allclose(per_month.values, 1.0, atol=1e-8)

    def test_tc_rank_multi_month(self, multi_month_panel):
        w = compute_weights(
            multi_month_panel,
            scheme="tc_rank",
            config=_cfg,
            aum=500_000_000,
        )
        df = multi_month_panel[["yyyymm"]].copy()
        df["w"] = w
        per_month = df.groupby("yyyymm")["w"].mean()
        np.testing.assert_allclose(per_month.values, 1.0, atol=1e-8)

    def test_softmax_rank_exact_lambda_two(self):
        cfg = copy.deepcopy(load_config())
        cfg["weighting"]["softmax_rank_lambda"] = 2.0
        df = pd.DataFrame({
            "yyyymm": [202301, 202301, 202301],
            _LIQ_COL: [1.0, 2.0, 9.0],
            "liq_BidAskSpread": [0.01, 0.01, 0.01],
            "liq_excess_sigma_12m": [0.10, 0.10, 0.10],
            "liq_excess_sigma_12m_daily": [0.10 / np.sqrt(21.0)] * 3,
            "liq_dvol_21d": [1.0, 2.0, 9.0],
        })

        w = compute_weights(df, scheme="softmax_rank", config=cfg)

        ranks = pd.Series([1.0, 2.0, 9.0]).rank(pct=True).to_numpy()
        raw = np.exp(2.0 * ranks)
        expected = raw / raw.mean()
        np.testing.assert_allclose(w.values, expected, atol=1e-12)

    def test_softmax_rank_exact_lambda_three(self):
        cfg = copy.deepcopy(load_config())
        cfg["weighting"]["softmax_rank_lambda"] = 3.0
        df = pd.DataFrame({
            "yyyymm": [202301, 202301, 202301],
            _LIQ_COL: [1.0, 2.0, 9.0],
            "liq_BidAskSpread": [0.01, 0.01, 0.01],
            "liq_excess_sigma_12m": [0.10, 0.10, 0.10],
            "liq_excess_sigma_12m_daily": [0.10 / np.sqrt(21.0)] * 3,
            "liq_dvol_21d": [1.0, 2.0, 9.0],
        })

        w = compute_weights(df, scheme="softmax_rank", config=cfg)

        ranks = pd.Series([1.0, 2.0, 9.0]).rank(pct=True).to_numpy()
        raw = np.exp(3.0 * ranks)
        expected = raw / raw.mean()
        np.testing.assert_allclose(w.values, expected, atol=1e-12)

    def test_tc_rank_exact_lambda_three(self):
        cfg = copy.deepcopy(load_config())
        cfg["weighting"]["tc_rank_lambda"] = 3.0
        cfg["transaction_costs"]["lambda_market_impact"] = 1.0
        cfg["transaction_costs"]["lambda_calibration"]["enabled"] = False
        cfg["transaction_costs"]["sigma_col"] = "excess_sigma_12m_daily"
        df = pd.DataFrame({
            "yyyymm": [202301, 202301, 202301],
            _LIQ_COL: [1.0, 4.0, 9.0],
            "liq_BidAskSpread": [0.0, 0.0, 0.0],
            "liq_excess_sigma_12m_daily": [1.0, 1.0, 1.0],
            "liq_dvol_21d": [1.0, 4.0, 9.0],
        })

        w = compute_weights(df, scheme="tc_rank", config=cfg, aum=3.0)

        tc = pd.Series([1.0, 0.5, 1.0 / 3.0])
        ranks = (-tc).rank(pct=True, method="average").to_numpy()
        raw = np.exp(3.0 * ranks)
        expected = raw / raw.mean()
        np.testing.assert_allclose(w.values, expected, atol=1e-12)
        assert w.iloc[2] > w.iloc[1] > w.iloc[0]


# -- NaN handling -----------------------------------------------------

class TestNaNHandling:
    def test_dolvol_no_nan_output(self, cross_section_with_nans):
        w = compute_weights(cross_section_with_nans, scheme="dolvol", config=_cfg)
        assert w.isna().sum() == 0

    def test_tc_no_nan_output(self, cross_section_with_nans):
        w = compute_weights(cross_section_with_nans, scheme="tc", config=_cfg, aum=500_000_000)
        assert w.isna().sum() == 0

    def test_softmax_rank_no_nan_output(self, cross_section_with_nans):
        w = compute_weights(cross_section_with_nans, scheme="softmax_rank", config=_cfg)
        assert w.isna().sum() == 0

    def test_tc_rank_no_nan_output(self, cross_section_with_nans):
        w = compute_weights(
            cross_section_with_nans,
            scheme="tc_rank",
            config=_cfg,
            aum=500_000_000,
        )
        assert w.isna().sum() == 0


# -- Monotonicity -----------------------------------------------------

class TestMonotonicity:
    def test_dolvol_higher_volume_higher_weight(self, single_cross_section):
        w = compute_weights(single_cross_section, scheme="dolvol", config=_cfg)
        corr = single_cross_section[_LIQ_COL].corr(w, method="spearman")
        assert corr > 0.99

    def test_tc_higher_volume_higher_weight(self, single_cross_section):
        w = compute_weights(single_cross_section, scheme="tc", config=_cfg, aum=500_000_000)
        # Higher ADV -> lower TC -> higher weight (positive corr with dvol)
        # Correlation can be modest since TC also depends on spread and sigma
        corr = single_cross_section["liq_dvol_21d"].corr(w, method="spearman")
        assert corr > 0.0

    def test_softmax_rank_higher_volume_higher_weight(self, single_cross_section):
        w = compute_weights(single_cross_section, scheme="softmax_rank", config=_cfg)
        corr = single_cross_section[_LIQ_COL].corr(w, method="spearman")
        assert corr > 0.99

    def test_tc_rank_higher_volume_higher_weight(self, single_cross_section):
        w = compute_weights(
            single_cross_section,
            scheme="tc_rank",
            config=_cfg,
            aum=500_000_000,
        )
        # Higher ADV -> lower TC rank penalty -> higher weight on average.
        corr = single_cross_section["liq_dvol_21d"].corr(w, method="spearman")
        assert corr > 0.0


# -- TC weights are AUM-dependent -------------------------------------

class TestTCAUMDependence:
    def test_different_aum_different_weights(self, single_cross_section):
        w_100m = compute_weights(single_cross_section, scheme="tc", config=_cfg, aum=100_000_000)
        w_1b = compute_weights(single_cross_section, scheme="tc", config=_cfg, aum=1_000_000_000)
        assert not np.allclose(w_100m.values, w_1b.values)

    def test_tc_requires_aum(self, single_cross_section):
        with pytest.raises(ValueError, match="aum is required"):
            compute_weights(single_cross_section, scheme="tc", config=_cfg)

    def test_tc_rank_requires_aum(self, single_cross_section):
        with pytest.raises(ValueError, match="aum is required"):
            compute_weights(single_cross_section, scheme="tc_rank", config=_cfg)

    def test_tc_rank_can_change_with_aum(self):
        cfg = copy.deepcopy(load_config())
        cfg["weighting"]["tc_rank_lambda"] = 3.0
        cfg["transaction_costs"]["lambda_market_impact"] = 1.0
        cfg["transaction_costs"]["lambda_calibration"]["enabled"] = False
        cfg["transaction_costs"]["sigma_col"] = "excess_sigma_12m_daily"
        df = pd.DataFrame({
            "yyyymm": [202301, 202301],
            _LIQ_COL: [1.0, 2.0],
            "liq_BidAskSpread": [0.001, 0.10],
            "liq_excess_sigma_12m_daily": [1.0, 1.0],
            "liq_dvol_21d": [1.0, 1e12],
        })

        w_low = compute_weights(df, scheme="tc_rank", config=cfg, aum=1e-12)
        w_high = compute_weights(df, scheme="tc_rank", config=cfg, aum=1e12)

        assert w_low.idxmax() == 0
        assert w_high.idxmax() == 1


class TestLambdaCalibration:
    def test_disabled_uses_fixed_lambda(self, single_cross_section):
        lam = resolve_market_impact_lambda(single_cross_section, _cfg)
        assert lam == pytest.approx(_cfg["transaction_costs"]["lambda_market_impact"])

    def test_anchor_calibration(self, single_cross_section):
        cfg = copy.deepcopy(load_config())
        cfg["transaction_costs"]["lambda_calibration"]["enabled"] = True
        cfg["transaction_costs"]["lambda_calibration"]["sigma_bar_stat"] = "median"
        cfg["transaction_costs"]["sigma_col"] = "excess_sigma_12m"
        df = single_cross_section.copy()
        df["liq_excess_sigma_12m"] = 0.10

        lam = resolve_market_impact_lambda(df, cfg)

        assert lam == pytest.approx(0.1)
        impact_at_anchor = lam * 0.10 * np.sqrt(0.01)
        assert impact_at_anchor == pytest.approx(0.001)


class TestTCWeightAlpha:
    def test_inverse_median_scale_two(self):
        cfg = copy.deepcopy(load_config())
        cfg["transaction_costs"]["lambda_market_impact"] = 1.0
        cfg["transaction_costs"]["lambda_calibration"]["enabled"] = False
        cfg["transaction_costs"]["sigma_col"] = "excess_sigma_12m_daily"
        cfg["transaction_costs"]["weight_alpha"] = {
            "mode": "inverse_median",
            "scale": 2.0,
        }

        df = pd.DataFrame({
            "yyyymm": [202301, 202301, 202301],
            "liq_dvol_21d": [1.0, 4.0, 9.0],
            "liq_BidAskSpread": [0.0, 0.0, 0.0],
            "liq_excess_sigma_12m_daily": [1.0, 1.0, 1.0],
        })

        w = compute_weights(df, scheme="tc", config=cfg, aum=3.0)

        tc = np.array([1.0, 0.5, 1.0 / 3.0])
        alpha = 2.0 / np.median(tc)
        raw = np.exp(-alpha * tc)
        expected = raw / raw.mean()

        np.testing.assert_allclose(w.values, expected, atol=1e-12)
        assert w.iloc[2] > w.iloc[1] > w.iloc[0]

    def test_config_default_scale_is_three(self):
        assert _cfg["transaction_costs"]["weight_alpha"]["scale"] == pytest.approx(3.0)


# -- DolVol is AUM-independent ----------------------------------------

class TestDolVolAUMIndependence:
    def test_dolvol_ignores_aum(self, single_cross_section):
        w1 = compute_weights(single_cross_section, scheme="dolvol", config=_cfg)
        w2 = compute_weights(single_cross_section, scheme="dolvol", config=_cfg, aum=500_000_000)
        pd.testing.assert_series_equal(w1, w2)

    def test_softmax_rank_ignores_aum(self, single_cross_section):
        w1 = compute_weights(single_cross_section, scheme="softmax_rank", config=_cfg)
        w2 = compute_weights(single_cross_section, scheme="softmax_rank", config=_cfg, aum=500_000_000)
        pd.testing.assert_series_equal(w1, w2)


# -- compute_tc_for_sorting -------------------------------------------

class TestTCForSorting:
    def test_returns_positive_tc(self, single_cross_section):
        tc = compute_tc_for_sorting(single_cross_section, aum=500_000_000, config=_cfg)
        assert (tc > 0).all()
        assert len(tc) == len(single_cross_section)

    def test_sorting_penalty_is_half_spread(self, single_cross_section):
        tc = compute_tc_for_sorting(single_cross_section, aum=500_000_000, config=_cfg)
        expected = single_cross_section["liq_BidAskSpread"].abs() / 2.0
        pd.testing.assert_series_equal(tc, expected, check_names=False)

    def test_sorting_penalty_is_aum_independent(self, single_cross_section):
        tc_100m = compute_tc_for_sorting(single_cross_section, aum=100_000_000, config=_cfg)
        tc_1b = compute_tc_for_sorting(single_cross_section, aum=1_000_000_000, config=_cfg)
        pd.testing.assert_series_equal(tc_100m, tc_1b)


# -- Edge cases -------------------------------------------------------

class TestEdgeCases:
    def test_single_stock_dolvol(self):
        df = pd.DataFrame({
            "yyyymm": [202301],
            _LIQ_COL: [1e8],
            "liq_BidAskSpread": [0.01],
            "liq_excess_sigma_12m": [0.10],
            "liq_excess_sigma_12m_daily": [0.10 / np.sqrt(21.0)],
            "liq_dvol_21d": [1e8],
        })
        w = compute_weights(df, scheme="dolvol", config=_cfg)
        assert len(w) == 1
        assert abs(w.iloc[0] - 1.0) < 1e-8

    def test_single_stock_tc(self):
        df = pd.DataFrame({
            "yyyymm": [202301],
            _LIQ_COL: [1e8],
            "liq_BidAskSpread": [0.01],
            "liq_excess_sigma_12m": [0.10],
            "liq_excess_sigma_12m_daily": [0.10 / np.sqrt(21.0)],
            "liq_dvol_21d": [1e8],
        })
        w = compute_weights(df, scheme="tc", config=_cfg, aum=500_000_000)
        assert len(w) == 1
        assert abs(w.iloc[0] - 1.0) < 1e-8

    def test_single_stock_tc_rank(self):
        df = pd.DataFrame({
            "yyyymm": [202301],
            _LIQ_COL: [1e8],
            "liq_BidAskSpread": [0.01],
            "liq_excess_sigma_12m": [0.10],
            "liq_excess_sigma_12m_daily": [0.10 / np.sqrt(21.0)],
            "liq_dvol_21d": [1e8],
        })
        w = compute_weights(df, scheme="tc_rank", config=_cfg, aum=500_000_000)
        assert len(w) == 1
        assert abs(w.iloc[0] - 1.0) < 1e-8
