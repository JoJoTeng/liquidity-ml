"""Tests for src/weighting/schemes.py"""

import numpy as np
import pandas as pd
import pytest

from src.weighting.schemes import (
    compute_weights,
    compute_all_weights,
    get_available_schemes,
    PRIMARY_LIQ_COL,
    PRIMARY_SCHEME,
    _normalize_to_mean_one,
    _softmax_rank,
    _linear_dolvol,
    _transaction_cost,
    _quintile_discrete,
)

# Primary liquidity column name from config (e.g., "liq_dvol_21d")
_LIQ_COL = PRIMARY_LIQ_COL


# ── Fixtures ─────────────────────────────────────────────────


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
        "liq_daily_sigma": rng.uniform(0.005, 0.04, size=n),
        "liq_dvol_21d": rng.lognormal(mean=18, sigma=2, size=n),
    })


@pytest.fixture
def multi_month_panel():
    """3 months × 50 stocks."""
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
            "liq_daily_sigma": rng.uniform(0.005, 0.04, size=n),
            "liq_dvol_21d": rng.lognormal(mean=18, sigma=2, size=n),
        }))
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def cross_section_with_nans(single_cross_section):
    """Same as single_cross_section but with 10% NaN in liquidity columns."""
    df = single_cross_section.copy()
    rng = np.random.RandomState(7)
    mask = rng.choice(len(df), size=10, replace=False)
    df.loc[mask, _LIQ_COL] = np.nan
    df.loc[mask, "liq_BidAskSpread"] = np.nan
    return df


# ── _normalize_to_mean_one ───────────────────────────────────


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


# ── Mean=1 invariant per scheme ──────────────────────────────


class TestMeanOneInvariant:
    """Every scheme must produce weights with mean=1.0 per cross-section."""

    @pytest.mark.parametrize("scheme", get_available_schemes())
    def test_single_month(self, single_cross_section, scheme):
        w = compute_weights(single_cross_section, scheme=scheme)
        assert abs(w.mean() - 1.0) < 1e-8

    @pytest.mark.parametrize("scheme", get_available_schemes())
    def test_multi_month(self, multi_month_panel, scheme):
        df = multi_month_panel
        w = compute_weights(df, scheme=scheme)
        # Check per-month mean
        df_check = df[["yyyymm"]].copy()
        df_check["w"] = w
        per_month_mean = df_check.groupby("yyyymm")["w"].mean()
        np.testing.assert_allclose(per_month_mean.values, 1.0, atol=1e-8)


# ── NaN handling ─────────────────────────────────────────────


class TestNaNHandling:
    @pytest.mark.parametrize("scheme", get_available_schemes())
    def test_no_nan_in_output(self, cross_section_with_nans, scheme):
        w = compute_weights(cross_section_with_nans, scheme=scheme)
        assert w.isna().sum() == 0

    @pytest.mark.parametrize("scheme", get_available_schemes())
    def test_nan_rows_get_near_neutral_weight(self, cross_section_with_nans, scheme):
        df = cross_section_with_nans
        w = compute_weights(df, scheme=scheme)
        nan_mask = df[_LIQ_COL].isna()
        nan_weights = w[nan_mask]
        # NaN stocks should get weight in a reasonable range (not extreme)
        assert nan_weights.min() > 0.01
        assert nan_weights.max() < 10.0


# ── Monotonicity ─────────────────────────────────────────────


class TestMonotonicity:
    """Higher liquidity → higher weight (except transaction_cost uses spread)."""

    def test_softmax_rank_monotone(self, single_cross_section):
        df = single_cross_section
        w = compute_weights(df, scheme="softmax_rank")
        # Rank correlation between dvol and weight should be positive
        corr = df[_LIQ_COL].corr(w, method="spearman")
        assert corr > 0.9

    def test_linear_dolvol_monotone(self, single_cross_section):
        df = single_cross_section
        w = compute_weights(df, scheme="linear_dolvol")
        corr = df[_LIQ_COL].corr(w, method="spearman")
        assert corr > 0.99  # perfectly monotone (linear transform)

    def test_transaction_cost_monotone(self, single_cross_section):
        df = single_cross_section
        w = compute_weights(df, scheme="transaction_cost")
        # Lower spread → higher weight (negative correlation)
        corr = df["liq_BidAskSpread"].corr(w, method="spearman")
        assert corr < -0.9

    def test_quintile_discrete_monotone(self, single_cross_section):
        df = single_cross_section
        w = compute_weights(df, scheme="quintile_discrete")
        corr = df[_LIQ_COL].corr(w, method="spearman")
        assert corr > 0.8


# ── Edge cases ───────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.parametrize("scheme", get_available_schemes())
    def test_single_stock(self, scheme):
        df = pd.DataFrame({
            "yyyymm": [202301],
            _LIQ_COL: [1e8],
            "liq_BidAskSpread": [0.01],
            "liq_daily_sigma": [0.02],
            "liq_dvol_21d": [1e8],
        })
        w = compute_weights(df, scheme=scheme)
        assert len(w) == 1
        assert abs(w.iloc[0] - 1.0) < 1e-8

    @pytest.mark.parametrize("scheme", get_available_schemes())
    def test_identical_values(self, scheme):
        n = 20
        df = pd.DataFrame({
            "yyyymm": 202301,
            _LIQ_COL: [1e8] * n,
            "liq_BidAskSpread": [0.01] * n,
            "liq_daily_sigma": [0.02] * n,
            "liq_dvol_21d": [1e8] * n,
        })
        w = compute_weights(df, scheme=scheme)
        # All identical inputs → uniform weights
        np.testing.assert_allclose(w.values, 1.0, atol=1e-8)


# ── Dispatcher ───────────────────────────────────────────────


class TestDispatcher:
    def test_unknown_scheme_raises(self, single_cross_section):
        with pytest.raises(ValueError, match="Unknown scheme"):
            compute_weights(single_cross_section, scheme="nonexistent")

    def test_default_is_primary(self, single_cross_section):
        w_default = compute_weights(single_cross_section)
        w_explicit = compute_weights(single_cross_section, scheme=PRIMARY_SCHEME)
        pd.testing.assert_series_equal(w_default, w_explicit)


# ── compute_all_weights ──────────────────────────────────────


class TestComputeAllWeights:
    def test_returns_all_schemes(self, single_cross_section):
        result = compute_all_weights(single_cross_section)
        expected_cols = {f"weight_{s}" for s in get_available_schemes()}
        assert set(result.columns) == expected_cols

    def test_index_aligned(self, multi_month_panel):
        result = compute_all_weights(multi_month_panel)
        assert (result.index == multi_month_panel.index).all()
