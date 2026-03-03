"""Tests for src/portfolio/construction.py"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.construction import (
    build_long_short_portfolio,
    build_portfolio_timeseries,
    compute_transaction_costs,
    compute_net_returns,
    _apply_position_cap,
    _assign_quantiles,
    _compute_turnover,
    PRIMARY_LIQ_COL,
    DEFAULT_WEIGHT_COL,
)

_LIQ_COL = PRIMARY_LIQ_COL
_WEIGHT_COL = DEFAULT_WEIGHT_COL


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def single_month():
    """500 stocks in one month with known data."""
    rng = np.random.RandomState(42)
    n = 500
    return pd.DataFrame({
        "permno": range(10001, 10001 + n),
        "yyyymm": 202301,
        "ret": rng.normal(0.01, 0.05, size=n),
        _LIQ_COL: rng.lognormal(mean=18, sigma=2, size=n),
        _WEIGHT_COL: rng.uniform(0.5, 2.0, size=n),
        "liq_BidAskSpread": rng.uniform(0.001, 0.05, size=n),
        "liq_daily_sigma": rng.uniform(0.01, 0.04, size=n),
        "liq_dvol_21d": rng.lognormal(mean=18, sigma=2, size=n),
    })


@pytest.fixture
def multi_month():
    """3 months × 60 stocks."""
    rng = np.random.RandomState(99)
    months = [202301, 202302, 202303]
    dfs = []
    for m in months:
        n = 60
        dfs.append(pd.DataFrame({
            "permno": range(10001, 10001 + n),
            "yyyymm": m,
            "ret": rng.normal(0.01, 0.05, size=n),
            _LIQ_COL: rng.lognormal(mean=18, sigma=2, size=n),
            _WEIGHT_COL: rng.uniform(0.5, 2.0, size=n),
            "liq_BidAskSpread": rng.uniform(0.001, 0.05, size=n),
            "liq_daily_sigma": rng.uniform(0.01, 0.04, size=n),
            "liq_dvol_21d": rng.lognormal(mean=18, sigma=2, size=n),
        }))
    return pd.concat(dfs, ignore_index=True)


# ── _apply_position_cap ──────────────────────────────────────


class TestPositionCap:
    def test_no_clipping_needed(self):
        w = pd.Series([0.25, 0.25, 0.25, 0.25])
        result = _apply_position_cap(w, cap=0.3)
        np.testing.assert_allclose(result.values, 0.25, atol=1e-10)

    def test_clipping_applied(self):
        # 10 stocks, cap=0.15 → feasible (10*0.15=1.5>1)
        w = pd.Series([0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])
        result = _apply_position_cap(w, cap=0.15)
        assert result.max() <= 0.15 + 1e-6
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_sum_to_one(self):
        w = pd.Series([0.8, 0.1, 0.05, 0.05])
        result = _apply_position_cap(w, cap=0.05)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)


# ── _assign_quantiles ────────────────────────────────────────


class TestAssignQuantiles:
    def test_decile_assignment(self):
        preds = pd.Series(range(100))
        q = _assign_quantiles(preds, 10)
        assert q.min() == 1
        assert q.max() == 10
        # Roughly 10 per decile
        counts = q.value_counts()
        assert counts.min() >= 8
        assert counts.max() <= 12

    def test_handles_ties(self):
        preds = pd.Series([1.0] * 50 + [2.0] * 50)
        q = _assign_quantiles(preds, 10)
        assert q.min() >= 1
        assert q.max() <= 10


# ── _compute_turnover ────────────────────────────────────────


class TestTurnover:
    def test_no_previous(self):
        assert _compute_turnover({}, {10001: 0.5, 10002: 0.5}) == 0.0

    def test_identical_positions(self):
        pos = {10001: 0.5, 10002: 0.5}
        assert _compute_turnover(pos, pos) == 0.0

    def test_complete_turnover(self):
        prev = {10001: 0.5, 10002: 0.5}
        curr = {10003: 0.5, 10004: 0.5}
        # All old sold (0.5+0.5) + all new bought (0.5+0.5) = 2.0
        assert abs(_compute_turnover(prev, curr) - 2.0) < 1e-10

    def test_partial_turnover(self):
        prev = {10001: 0.5, 10002: 0.5}
        curr = {10001: 0.5, 10003: 0.5}
        # 10002 sold (0.5) + 10003 bought (0.5) = 1.0
        assert abs(_compute_turnover(prev, curr) - 1.0) < 1e-10


# ── build_long_short_portfolio ───────────────────────────────


class TestBuildLongShort:
    def test_returns_expected_keys(self, single_month):
        preds = np.random.RandomState(1).randn(len(single_month))
        result = build_long_short_portfolio(single_month, preds)
        expected_keys = {
            "ret_long", "ret_short", "ret_long_short",
            "n_long", "n_short", "positions_long", "positions_short",
        }
        assert set(result.keys()) == expected_keys

    def test_long_short_decomposition(self, single_month):
        preds = np.random.RandomState(1).randn(len(single_month))
        result = build_long_short_portfolio(single_month, preds)
        np.testing.assert_allclose(
            result["ret_long_short"],
            result["ret_long"] - result["ret_short"],
            atol=1e-12,
        )

    def test_equal_weighted_positions_sum_to_one(self, single_month):
        preds = np.random.RandomState(1).randn(len(single_month))
        result = build_long_short_portfolio(single_month, preds, weighted=False)
        sum_long = sum(result["positions_long"].values())
        sum_short = sum(result["positions_short"].values())
        np.testing.assert_allclose(sum_long, 1.0, atol=1e-10)
        np.testing.assert_allclose(sum_short, 1.0, atol=1e-10)

    def test_weighted_positions_sum_to_one(self, single_month):
        preds = np.random.RandomState(1).randn(len(single_month))
        result = build_long_short_portfolio(
            single_month, preds, weighted=True, weight_col=_WEIGHT_COL,
        )
        sum_long = sum(result["positions_long"].values())
        sum_short = sum(result["positions_short"].values())
        np.testing.assert_allclose(sum_long, 1.0, atol=1e-10)
        np.testing.assert_allclose(sum_short, 1.0, atol=1e-10)

    def test_weighted_differs_from_equal(self, single_month):
        preds = np.random.RandomState(1).randn(len(single_month))
        eq = build_long_short_portfolio(single_month, preds, weighted=False)
        wt = build_long_short_portfolio(
            single_month, preds, weighted=True, weight_col=_WEIGHT_COL,
        )
        # Returns should differ (different weighting)
        assert eq["ret_long_short"] != wt["ret_long_short"]

    def test_position_cap_enforced(self, single_month):
        preds = np.random.RandomState(1).randn(len(single_month))
        result = build_long_short_portfolio(single_month, preds)
        for w in result["positions_long"].values():
            assert w <= 0.05 + 1e-10
        for w in result["positions_short"].values():
            assert w <= 0.05 + 1e-10

    def test_too_few_stocks(self):
        df = pd.DataFrame({
            "permno": [10001, 10002],
            "yyyymm": 202301,
            "ret": [0.01, -0.01],
            _LIQ_COL: [1e8, 1e8],
            _WEIGHT_COL: [1.0, 1.0],
        })
        preds = np.array([0.1, -0.1])
        result = build_long_short_portfolio(df, preds)
        assert np.isnan(result["ret_long_short"])


# ── build_portfolio_timeseries ───────────────────────────────


class TestBuildTimeseries:
    def test_returns_monthly_rows(self, multi_month):
        preds = pd.Series(
            np.random.RandomState(1).randn(len(multi_month)),
            index=multi_month.index,
        )
        results_df, pos_hist = build_portfolio_timeseries(
            multi_month, preds,
        )
        assert len(results_df) == 3  # 3 months
        assert "yyyymm" in results_df.columns
        assert "ret_long_short" in results_df.columns
        assert "turnover" in results_df.columns

    def test_first_month_zero_turnover(self, multi_month):
        preds = pd.Series(
            np.random.RandomState(1).randn(len(multi_month)),
            index=multi_month.index,
        )
        results_df, _ = build_portfolio_timeseries(multi_month, preds)
        assert results_df.iloc[0]["turnover"] == 0.0

    def test_positions_history_has_all_months(self, multi_month):
        preds = pd.Series(
            np.random.RandomState(1).randn(len(multi_month)),
            index=multi_month.index,
        )
        _, pos_hist = build_portfolio_timeseries(multi_month, preds)
        assert set(pos_hist.keys()) == {202301, 202302, 202303}


# ── Transaction costs ────────────────────────────────────────


class TestTransactionCosts:
    def test_first_month_zero_cost(self, multi_month):
        preds = pd.Series(
            np.random.RandomState(1).randn(len(multi_month)),
            index=multi_month.index,
        )
        _, pos_hist = build_portfolio_timeseries(multi_month, preds)
        tc = compute_transaction_costs(pos_hist, multi_month, aum=1e9)
        assert tc.iloc[0] == 0.0

    def test_costs_are_non_negative(self, multi_month):
        preds = pd.Series(
            np.random.RandomState(1).randn(len(multi_month)),
            index=multi_month.index,
        )
        _, pos_hist = build_portfolio_timeseries(multi_month, preds)
        tc = compute_transaction_costs(pos_hist, multi_month, aum=1e9)
        assert (tc >= 0).all()

    def test_higher_aum_higher_costs(self, multi_month):
        preds = pd.Series(
            np.random.RandomState(1).randn(len(multi_month)),
            index=multi_month.index,
        )
        _, pos_hist = build_portfolio_timeseries(multi_month, preds)
        tc_low = compute_transaction_costs(pos_hist, multi_month, aum=1e8)
        tc_high = compute_transaction_costs(pos_hist, multi_month, aum=5e9)
        # Higher AUM → higher market impact → higher total cost
        assert tc_high.sum() >= tc_low.sum()

    def test_net_returns_lower_than_gross(self, multi_month):
        preds = pd.Series(
            np.random.RandomState(1).randn(len(multi_month)),
            index=multi_month.index,
        )
        gross, pos_hist = build_portfolio_timeseries(multi_month, preds)
        net = compute_net_returns(gross, pos_hist, multi_month, aum=1e9)
        # Net should be ≤ gross (costs are non-negative)
        assert (net["ret_long_short_net"] <= net["ret_long_short"] + 1e-12).all()
