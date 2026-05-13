"""Tests for src/portfolio/construction.py (LiquidityML v3)

Tests the updated portfolio construction:
  - Standard sort (Column 1): sort on r_hat
  - TC-penalised sort (Column 2): long r_hat - spread/2, short r_hat + spread/2
  - Portfolio performance uses excess_ret, matching the model target
  - Net returns with turnover-adjusted TC
  - No liquidity filter, no position cap
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.portfolio.construction import (
    build_long_short_portfolio,
    build_portfolio_timeseries,
    compute_net_returns,
    _assign_quantiles,
    _drift_positions,
)

_cfg = load_config()


# -- Fixtures ---------------------------------------------------------

@pytest.fixture
def single_month():
    """500 stocks in one month with known data."""
    rng = np.random.RandomState(42)
    n = 500
    raw_ret = rng.normal(0.01, 0.05, size=n)
    return pd.DataFrame({
        "yyyymm": 202301,
        "permno": range(10001, 10001 + n),
        "ret": raw_ret,
        "excess_ret": raw_ret - 0.002,
        "liq_dvol_21d": rng.lognormal(mean=18, sigma=2, size=n),
        "liq_BidAskSpread": rng.uniform(0.001, 0.05, size=n),
        "liq_excess_sigma_12m": rng.uniform(0.04, 0.25, size=n),
        "liq_excess_sigma_12m_daily": rng.uniform(0.04, 0.25, size=n) / np.sqrt(21.0),
    })


@pytest.fixture
def predictions(single_month):
    """Return predictions aligned to single_month."""
    rng = np.random.RandomState(42)
    return pd.Series(rng.normal(0.01, 0.02, size=len(single_month)),
                     index=single_month.index)


@pytest.fixture
def tc_per_stock(single_month):
    """Fake TC per stock for TC-penalised sort testing."""
    rng = np.random.RandomState(99)
    return pd.Series(rng.uniform(0.001, 0.01, size=len(single_month)),
                     index=single_month.index)


# -- _assign_quantiles ------------------------------------------------

class TestAssignQuantiles:
    def test_basic(self):
        vals = pd.Series(range(100))
        quantiles = _assign_quantiles(vals, n_quantiles=5)
        assert quantiles.min() == 1
        assert quantiles.max() == 5
        assert len(quantiles) == 100

    def test_equal_sized_bins(self):
        vals = pd.Series(range(1000))
        quantiles = _assign_quantiles(vals, n_quantiles=5)
        counts = quantiles.value_counts()
        assert counts.min() == 200
        assert counts.max() == 200


# -- build_long_short_portfolio ---------------------------------------

class TestBuildLongShort:
    def test_basic_construction(self, single_month, predictions):
        result = build_long_short_portfolio(single_month, predictions)
        assert "ret_long" in result
        assert "ret_short" in result
        assert "ret_long_short" in result
        assert result["n_long"] > 0
        assert result["n_short"] > 0
        assert len(result["positions_long"]) == result["n_long"]

    def test_long_short_difference(self, single_month, predictions):
        result = build_long_short_portfolio(single_month, predictions)
        assert abs(result["ret_long_short"] -
                   (result["ret_long"] - result["ret_short"])) < 1e-10

    def test_uses_excess_return_for_performance(self):
        n = 10
        df = pd.DataFrame({
            "yyyymm": 202301,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.arange(n, dtype=float) / 100.0,
        })
        predictions = pd.Series(np.arange(n), index=df.index)

        result = build_long_short_portfolio(df, predictions)

        expected_long = np.mean([0.08, 0.09])
        expected_short = np.mean([0.00, 0.01])
        assert result["ret_long"] == pytest.approx(expected_long)
        assert result["ret_short"] == pytest.approx(expected_short)
        assert result["ret_long_short"] == pytest.approx(
            expected_long - expected_short
        )

    def test_invalid_portfolio_mode(self, single_month, predictions):
        with pytest.raises(ValueError, match="portfolio_mode"):
            build_long_short_portfolio(
                single_month,
                predictions,
                portfolio_mode="bad_mode",
            )

    def test_equal_dollar_weights(self, single_month, predictions):
        result = build_long_short_portfolio(single_month, predictions)
        long_weights = list(result["positions_long"].values())
        # Equal-dollar: all weights should be equal
        assert abs(max(long_weights) - min(long_weights)) < 1e-10

    def test_no_liquidity_filter(self, single_month, predictions):
        """All stocks should be included (no filtering)."""
        result = build_long_short_portfolio(single_month, predictions)
        total = result["n_long"] + result["n_short"]
        # Long + short = 2 quintiles out of 5 = 40% of 500 = 200 stocks
        assert total == 200

    def test_custom_long_short_quantiles(self):
        """Configured long/short quantiles should control selected legs."""
        n = 100
        df = pd.DataFrame({
            "yyyymm": 202301,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.zeros(n),
        })
        predictions = pd.Series(np.arange(n), index=df.index)

        result = build_long_short_portfolio(
            df,
            predictions,
            n_quantiles=10,
            long_quantile=9,
            short_quantile=2,
        )

        assert set(result["positions_long"]) == set(range(10081, 10091))
        assert set(result["positions_short"]) == set(range(10011, 10021))

    def test_invalid_long_short_quantiles(self, single_month, predictions):
        with pytest.raises(ValueError, match="must differ"):
            build_long_short_portfolio(
                single_month,
                predictions,
                long_quantile=1,
                short_quantile=1,
            )

    def test_tc_penalised_sort(self, single_month, predictions, tc_per_stock):
        result_std = build_long_short_portfolio(
            single_month, predictions, tc_penalised=False)
        result_tc = build_long_short_portfolio(
            single_month, predictions, tc_penalised=True,
            tc_per_stock=tc_per_stock)
        # Different sorting -> different portfolios
        assert result_std["positions_long"] != result_tc["positions_long"]

    def test_tc_penalised_short_leg_adds_tc(self):
        n = 100
        df = pd.DataFrame({
            "yyyymm": 202301,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.zeros(n),
        })
        predictions = pd.Series(np.arange(n, dtype=float), index=df.index)
        tc_per_stock = pd.Series(0.0, index=df.index)
        tc_per_stock.iloc[:10] = 1_000.0

        result = build_long_short_portfolio(
            df,
            predictions,
            tc_penalised=True,
            tc_per_stock=tc_per_stock,
            n_quantiles=10,
        )

        high_cost_low_prediction = set(range(10001, 10011))
        assert high_cost_low_prediction.isdisjoint(result["positions_short"])
        assert set(result["positions_short"]) == set(range(10011, 10021))
        assert set(result["positions_long"]) == set(range(10091, 10101))

    def test_tc_penalised_legs_are_disjoint(self):
        n = 100
        df = pd.DataFrame({
            "yyyymm": 202301,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.zeros(n),
        })
        predictions = pd.Series(0.0, index=df.index)
        tc_per_stock = pd.Series(1.0, index=df.index)
        tc_per_stock.iloc[:20] = 0.0

        result = build_long_short_portfolio(
            df,
            predictions,
            tc_penalised=True,
            tc_per_stock=tc_per_stock,
        )

        assert set(result["positions_long"]).isdisjoint(result["positions_short"])
        assert result["n_long"] == 20
        assert result["n_short"] == 20

    def test_tc_penalised_requires_tc(self, single_month, predictions):
        with pytest.raises(ValueError, match="tc_per_stock required"):
            build_long_short_portfolio(
                single_month, predictions, tc_penalised=True)

    def test_too_few_stocks(self, predictions):
        tiny = pd.DataFrame({
            "yyyymm": [202301] * 5,
            "permno": range(5),
            "ret": [0.01] * 5,
            "excess_ret": [0.008] * 5,
        })
        tiny_pred = pd.Series([0.01] * 5, index=tiny.index)
        result = build_long_short_portfolio(tiny, tiny_pred)
        assert np.isnan(result["ret_long_short"])

# -- _drift_positions -------------------------------------------------

class TestDriftPositions:
    def test_basic_drift(self):
        """Positions drift with returns and renormalise."""
        positions = {10001: 0.5, 10002: 0.5}
        lookup = pd.DataFrame({
            "permno": [10001, 10002],
            "yyyymm": [202301, 202301],
            "ret": [0.10, -0.05],
        }).set_index(["permno", "yyyymm"])

        drifted = _drift_positions(positions, 202301, lookup)
        assert drifted[10001] > drifted[10002]
        assert abs(sum(drifted.values()) - 1.0) < 1e-10

    def test_empty_positions(self):
        lookup = pd.DataFrame()
        drifted = _drift_positions({}, 202301, lookup)
        assert drifted == {}


# -- compute_net_returns ----------------------------------------------

class TestComputeNetReturns:
    def test_net_less_than_gross(self, single_month, predictions):
        result = build_long_short_portfolio(single_month, predictions)
        returns_df = pd.DataFrame([{
            "yyyymm": 202301,
            "ret_long": result["ret_long"],
            "ret_short": result["ret_short"],
            "ret_long_short": result["ret_long_short"],
        }])
        pos_hist = {202301: {
            "long": result["positions_long"],
            "short": result["positions_short"],
        }}
        net_df = compute_net_returns(
            returns_df, pos_hist, single_month, aum=500_000_000, config=_cfg)
        assert net_df["ret_long_short_net"].iloc[0] <= net_df["ret_long_short"].iloc[0]
        assert net_df["raw_trade_sum"].iloc[0] == pytest.approx(2.0)
        assert net_df["turnover"].iloc[0] == pytest.approx(1.0)

    def test_eliminated_asset_uses_previous_tc_inputs(self):
        cfg = load_config()
        cfg["transaction_costs"]["sigma_col"] = "excess_sigma_12m"
        cfg["transaction_costs"]["lambda_market_impact"] = 0.1
        cfg["transaction_costs"]["lambda_calibration"]["enabled"] = False

        panel = pd.DataFrame({
            "permno": [10001, 10002],
            "yyyymm": [202301, 202302],
            "ret": [0.0, 0.0],
            "liq_BidAskSpread": [0.02, 0.02],
            "liq_excess_sigma_12m": [0.10, 0.10],
            "liq_dvol_21d": [100_000_000.0, 100_000_000.0],
        })
        returns_df = pd.DataFrame({
            "yyyymm": [202301, 202302],
            "ret_long_short": [0.0, 0.0],
        })
        pos_hist = {
            202301: {"long": {10001: 1.0}, "short": {}},
            202302: {"long": {}, "short": {}},
        }

        net_df = compute_net_returns(
            returns_df,
            pos_hist,
            panel,
            aum=1_000_000,
            config=cfg,
        )

        expected = 0.02 / 2 + 0.1 * 0.10 * np.sqrt(500_000 / 100_000_000)
        assert net_df.loc[net_df["yyyymm"] == 202302, "transaction_cost"].iloc[0] == pytest.approx(expected)
        assert net_df.loc[net_df["yyyymm"] == 202302, "raw_trade_sum"].iloc[0] == pytest.approx(1.0)
        assert net_df.loc[net_df["yyyymm"] == 202302, "turnover"].iloc[0] == pytest.approx(0.5)
