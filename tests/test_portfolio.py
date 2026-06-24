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
from src.analysis.formal.portfolio_tables import (
    compute_two_by_three_decomposition,
    format_prediction_quantile_timeseries_rows,
)
from src.portfolio.construction import (
    build_long_short_portfolio,
    build_portfolio_timeseries,
    build_prediction_quantile_timeseries,
    compute_net_returns,
    _assign_quantiles,
    _drift_positions,
    _leg_aum,
)

_cfg = load_config()


def _portfolio_test_config():
    cfg = load_config()
    cfg.setdefault("data", {})["target_col"] = "excess_ret"
    cfg["portfolio"]["n_quantiles"] = 5
    cfg["portfolio"]["long_quantile"] = 5
    cfg["portfolio"]["short_quantile"] = 1
    cfg["transaction_costs"]["sigma_col"] = "excess_sigma_12m_daily"
    cfg["transaction_costs"]["lambda_market_impact"] = 0.1
    cfg["transaction_costs"]["lambda_calibration"]["enabled"] = False
    cfg["inference"]["bootstrap_samples"] = 10
    cfg["inference"]["calibrate_block_length"] = False
    return cfg


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
        "liq_me_raw": rng.lognormal(mean=18, sigma=1.5, size=n),
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

    def test_value_weighted_leg_weights(self):
        n = 10
        df = pd.DataFrame({
            "yyyymm": 202301,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.arange(n, dtype=float) / 100.0,
            "liq_me_raw": [1, 1, 1, 1, 1, 1, 1, 1, 2, 6],
        })
        predictions = pd.Series(np.arange(n), index=df.index)

        result = build_long_short_portfolio(
            df,
            predictions,
            portfolio_weighting="value",
        )

        assert result["positions_long"][10009] == pytest.approx(0.25)
        assert result["positions_long"][10010] == pytest.approx(0.75)
        assert result["ret_long"] == pytest.approx(0.0875)
        assert sum(result["positions_long"].values()) == pytest.approx(1.0)
        assert sum(result["positions_short"].values()) == pytest.approx(1.0)

    def test_value_weighting_requires_market_cap_column(self, single_month, predictions):
        with pytest.raises(KeyError, match="weight column"):
            build_long_short_portfolio(
                single_month.drop(columns=["liq_me_raw"]),
                predictions,
                portfolio_weighting="value",
            )

    def test_dolvol_weighted_leg_weights(self):
        n = 10
        df = pd.DataFrame({
            "yyyymm": 202301,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.arange(n, dtype=float) / 100.0,
            "liq_me_raw": np.ones(n),
            "liq_dvol_21d": [1, 1, 1, 1, 1, 1, 1, 1, 2, 6],
        })
        predictions = pd.Series(np.arange(n), index=df.index)
        result = build_long_short_portfolio(
            df, predictions, portfolio_weighting="dolvol",
        )
        # Q5 (top 2 predictions) weighted by dollar volume 2 and 6 -> 0.25 / 0.75.
        assert result["positions_long"][10009] == pytest.approx(0.25)
        assert result["positions_long"][10010] == pytest.approx(0.75)
        assert sum(result["positions_long"].values()) == pytest.approx(1.0)

    def test_signal_weighted_leg_weights(self):
        n = 10
        df = pd.DataFrame({
            "yyyymm": 202301,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.arange(n, dtype=float) / 100.0,
            "liq_me_raw": np.ones(n),
            "liq_dvol_21d": np.ones(n),
            "w_tilde": [1, 1, 1, 1, 1, 1, 1, 1, 2, 6],
        })
        predictions = pd.Series(np.arange(n), index=df.index)
        result = build_long_short_portfolio(
            df, predictions, portfolio_weighting="signal",
        )
        # Q5 (top 2 predictions) weighted by w-tilde 2 and 6 -> 0.25 / 0.75.
        assert result["positions_long"][10009] == pytest.approx(0.25)
        assert result["positions_long"][10010] == pytest.approx(0.75)
        assert sum(result["positions_long"].values()) == pytest.approx(1.0)

    def test_signal_weighting_requires_w_tilde_column(self, single_month, predictions):
        with pytest.raises(KeyError, match="w_tilde"):
            build_long_short_portfolio(
                single_month, predictions, portfolio_weighting="signal",
            )

    def test_dolvol_weighting_differs_from_value(self):
        n = 10
        df = pd.DataFrame({
            "yyyymm": 202301,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.arange(n, dtype=float) / 100.0,
            "liq_me_raw": [1, 1, 1, 1, 1, 1, 1, 1, 6, 2],
            "liq_dvol_21d": [1, 1, 1, 1, 1, 1, 1, 1, 2, 6],
        })
        predictions = pd.Series(np.arange(n), index=df.index)
        val = build_long_short_portfolio(df, predictions, portfolio_weighting="value")
        dvol = build_long_short_portfolio(df, predictions, portfolio_weighting="dolvol")
        assert val["positions_long"][10010] != pytest.approx(
            dvol["positions_long"][10010]
        )

    def test_dolvol_weighting_requires_dolvol_column(self, single_month, predictions):
        with pytest.raises(KeyError, match="liq_dvol_21d"):
            build_long_short_portfolio(
                single_month.drop(columns=["liq_dvol_21d"]),
                predictions,
                portfolio_weighting="dolvol",
            )

    def test_liquidity_screen_forms_quintiles_within_top_60(
        self, single_month, predictions
    ):
        """Top-60% DolVol screen: quintiles are formed on the screened universe,
        so the bottom 40% by dollar volume never enter a leg."""
        unscreened = build_long_short_portfolio(single_month, predictions)
        screened = build_long_short_portfolio(
            single_month, predictions,
            liquidity_screen_pct=0.40, liquidity_screen_col="liq_dvol_21d",
        )
        assert (unscreened["n_long"] + unscreened["n_short"]) == 200
        screened_total = screened["n_long"] + screened["n_short"]
        assert 110 <= screened_total <= 130  # ~2 quintiles of the top-60% (~300)
        screened_out = set(
            single_month.loc[
                single_month["liq_dvol_21d"].rank(pct=True) < 0.40, "permno"
            ]
        )
        legs = screened["permnos_long"] | screened["permnos_short"]
        assert legs.isdisjoint(screened_out)

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

    def test_tc_target_score_short_leg_adds_two_tc(self):
        n = 100
        df = pd.DataFrame({
            "yyyymm": 202301,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.zeros(n),
        })
        tc_target_scores = pd.Series(np.arange(n, dtype=float), index=df.index)
        tc_per_stock = pd.Series(0.0, index=df.index)
        tc_per_stock.iloc[:10] = 1_000.0

        result = build_long_short_portfolio(
            df,
            tc_target_scores,
            tc_target_score=True,
            tc_per_stock=tc_per_stock,
            n_quantiles=10,
        )

        high_cost_low_score = set(range(10001, 10011))
        assert high_cost_low_score.isdisjoint(result["positions_short"])
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

    def test_tc_sort_modes_are_mutually_exclusive(self, single_month, predictions):
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_long_short_portfolio(
                single_month,
                predictions,
                tc_penalised=True,
                tc_target_score=True,
                tc_per_stock=pd.Series(0.0, index=single_month.index),
            )

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


# -- build_prediction_quantile_timeseries -----------------------------

class TestBuildPredictionQuantileTimeseries:
    def test_quantiles_cover_universe_and_equal_weights(self, single_month):
        cfg = _portfolio_test_config()
        preds = pd.DataFrame({
            "permno": single_month["permno"],
            "yyyymm": single_month["yyyymm"],
            "prediction": np.arange(len(single_month), dtype=float),
        })

        out = build_prediction_quantile_timeseries(
            single_month,
            preds,
            sort_mode="prediction",
            aum=500_000_000,
            config=cfg,
            portfolio_weighting="equal",
        )

        assert set(out["prediction_quantile"]) == {1, 2, 3, 4, 5}
        assert out["n_stocks"].sum() == len(single_month)
        assert (out["n_stocks"] == len(single_month) / 5).all()
        assert np.allclose(out["weight_sum"], 1.0)

    def test_value_weighted_quantile_return(self):
        cfg = _portfolio_test_config()
        n = 10
        panel = pd.DataFrame({
            "yyyymm": [202301] * n,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.arange(n, dtype=float) / 100.0,
            "liq_me_raw": [1, 1, 1, 1, 1, 1, 1, 1, 2, 6],
            "liq_BidAskSpread": [0.01] * n,
            "liq_excess_sigma_12m_daily": [0.10] * n,
            "liq_dvol_21d": [100_000_000.0] * n,
        })
        preds = pd.DataFrame({
            "permno": panel["permno"],
            "yyyymm": panel["yyyymm"],
            "prediction": np.arange(n, dtype=float),
        })

        out = build_prediction_quantile_timeseries(
            panel,
            preds,
            sort_mode="prediction",
            aum=500_000_000,
            config=cfg,
            portfolio_weighting="value",
        )

        q5 = out.loc[out["prediction_quantile"] == 5].iloc[0]
        assert q5["gross_return"] == pytest.approx(0.0875)
        assert q5["weight_sum"] == pytest.approx(1.0)

    def test_quantile_transaction_cost_uses_full_scenario_aum(self):
        cfg = _portfolio_test_config()
        n = 10
        panel = pd.DataFrame({
            "yyyymm": [202301] * n,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.zeros(n),
            "liq_me_raw": np.ones(n),
            "liq_BidAskSpread": [1e-8] * n,
            "liq_excess_sigma_12m_daily": [0.10] * n,
            "liq_dvol_21d": [100_000_000.0] * n,
        })
        preds = pd.DataFrame({
            "permno": panel["permno"],
            "yyyymm": panel["yyyymm"],
            "prediction": np.arange(n, dtype=float),
        })

        out = build_prediction_quantile_timeseries(
            panel,
            preds,
            sort_mode="prediction",
            aum=500_000_000,
            config=cfg,
            portfolio_weighting="equal",
        )

        quantile_aum = 500_000_000
        stock_weight = 0.5
        expected = (
            1e-8 / 2.0
            + 0.1 * 0.10 * np.sqrt(stock_weight * quantile_aum / 100_000_000.0)
        )
        q1 = out.loc[out["prediction_quantile"] == 1].iloc[0]
        assert q1["transaction_cost"] == pytest.approx(expected)
        assert q1["net_return"] == pytest.approx(
            q1["gross_return"] - q1["transaction_cost"]
        )

    def test_quantile_transaction_cost_supports_proportional_only_case(self):
        cfg = _portfolio_test_config()
        n = 10
        panel = pd.DataFrame({
            "yyyymm": [202301] * n,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.zeros(n),
            "liq_me_raw": np.ones(n),
            "liq_BidAskSpread": [0.02] * n,
            "liq_excess_sigma_12m_daily": [10.0] * n,
            "liq_dvol_21d": [1_000.0] * n,
        })
        preds = pd.DataFrame({
            "permno": panel["permno"],
            "yyyymm": panel["yyyymm"],
            "prediction": np.arange(n, dtype=float),
        })

        out = build_prediction_quantile_timeseries(
            panel,
            preds,
            sort_mode="prediction",
            aum=1_000_000_000,
            config=cfg,
            portfolio_weighting="equal",
            proportional_tc_only=True,
        )

        q1 = out.loc[out["prediction_quantile"] == 1].iloc[0]
        assert q1["transaction_cost"] == pytest.approx(0.02 / 2.0)
        assert q1["net_return"] == pytest.approx(
            q1["gross_return"] - q1["transaction_cost"]
        )

    def test_tc_net_sort_uses_single_net_score_for_all_quantiles(self):
        cfg = _portfolio_test_config()
        n = 100
        panel = pd.DataFrame({
            "yyyymm": [202301] * n,
            "permno": range(10001, 10001 + n),
            "ret": np.zeros(n),
            "excess_ret": np.zeros(n),
            "liq_me_raw": np.ones(n),
            "liq_BidAskSpread": [0.0] * 50 + [10_000.0] * 50,
            "liq_excess_sigma_12m_daily": [0.10] * n,
            "liq_dvol_21d": [100_000_000.0] * n,
        })
        preds = pd.DataFrame({
            "permno": panel["permno"],
            "yyyymm": panel["yyyymm"],
            "prediction": np.arange(n, dtype=float),
        })

        out = build_prediction_quantile_timeseries(
            panel,
            preds,
            sort_mode="tc_net",
            aum=500_000_000,
            config=cfg,
            portfolio_weighting="equal",
        )

        assert out.loc[out["prediction_quantile"] == 5, "score_mean"].iloc[0] < 50.0


class TestBucketFirstFormalDecomposition:
    def test_two_by_three_exports_quantile_timeseries_and_derived_long_short(self):
        cfg = _portfolio_test_config()
        months = list(range(202001, 202013))
        rows = []
        for month_idx, yyyymm in enumerate(months):
            for i in range(10):
                rows.append({
                    "yyyymm": yyyymm,
                    "permno": 10001 + i,
                    "ret": 0.0,
                    "excess_ret": i / 100.0 + month_idx / 1000.0,
                    "liq_me_raw": 1.0 + i,
                    "liq_BidAskSpread": 1e-8,
                    "liq_excess_sigma_12m_daily": 0.10,
                    "liq_dvol_21d": 100_000_000.0,
                })
        panel = pd.DataFrame(rows)
        preds = panel[["permno", "yyyymm"]].copy()
        preds["prediction"] = panel.groupby("yyyymm").cumcount().astype(float)

        result = compute_two_by_three_decomposition(
            preds,
            preds,
            preds,
            preds,
            panel,
            aum=500_000_000,
            config=cfg,
            portfolio_weighting="equal",
        )
        quantile_rows = format_prediction_quantile_timeseries_rows(
            result,
            aum=500_000_000,
            label="500M",
        )

        assert set(result["cells"]) == {"1A", "1B", "1C", "2A", "2B", "2C"}
        assert len(quantile_rows) == 6 * 5 * len(months)

        cell_1a = result["cells"]["1A"].iloc[0]
        q = result["quantile_timeseries"]["1A"]
        q_month = q[q["yyyymm"] == cell_1a["yyyymm"]]
        q1 = q_month[q_month["prediction_quantile"] == 1].iloc[0]
        q5 = q_month[q_month["prediction_quantile"] == 5].iloc[0]

        assert cell_1a["ret_long_short"] == pytest.approx(
            q5["gross_return"] - q1["gross_return"]
        )
        assert cell_1a["transaction_cost"] == pytest.approx(
            q5["transaction_cost"] + q1["transaction_cost"]
        )
        assert cell_1a["ret_long_short_net"] == pytest.approx(
            cell_1a["ret_long_short"] - cell_1a["transaction_cost"]
        )

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

    def test_proportional_only_net_returns_exclude_market_impact(self):
        cfg = _portfolio_test_config()
        panel = pd.DataFrame({
            "permno": [10001],
            "yyyymm": [202301],
            "ret": [0.0],
            "liq_BidAskSpread": [0.02],
            "liq_excess_sigma_12m_daily": [10.0],
            "liq_dvol_21d": [1_000.0],
        })
        returns_df = pd.DataFrame({
            "yyyymm": [202301],
            "ret_long_short": [0.0],
        })
        pos_hist = {202301: {"long": {10001: 1.0}, "short": {}}}

        net_df = compute_net_returns(
            returns_df,
            pos_hist,
            panel,
            aum=1_000_000_000,
            config=cfg,
            proportional_tc_only=True,
        )

        assert net_df["transaction_cost"].iloc[0] == pytest.approx(0.02 / 2.0)
        assert net_df["ret_long_short_net"].iloc[0] == pytest.approx(-0.01)


def test_leg_aum_gross_vs_full():
    # gross (default): each leg trades $A/2 -> long-short is $A gross
    assert _leg_aum(1_000.0) == pytest.approx(500.0)
    assert _leg_aum(1_000.0, leg_capital="gross") == pytest.approx(500.0)
    # full: each leg trades the full $A -> legacy $2A gross
    assert _leg_aum(1_000.0, leg_capital="full") == pytest.approx(1_000.0)
    with pytest.raises(ValueError, match="leg_capital"):
        _leg_aum(1_000.0, leg_capital="bogus")


def test_hysteresis_retains_held_name_within_band():
    n = 10
    df = pd.DataFrame({
        "yyyymm": 202302,
        "permno": range(10001, 10001 + n),
        "ret": np.zeros(n),
        "excess_ret": np.arange(n, dtype=float) / 100.0,
    })
    preds = pd.Series(np.arange(n, dtype=float), index=df.index)
    prev_long = {10008}  # held long now just below the Q5 cutoff (pred 7 < 8)
    # wide half-spread -> within band -> retained
    res_wide = build_long_short_portfolio(
        df, preds, hysteresis=True, prev_long=prev_long, prev_short=set(),
        tc_half_spread=pd.Series(5.0, index=df.index),
    )
    assert 10008 in res_wide["positions_long"]
    assert sum(res_wide["positions_long"].values()) == pytest.approx(1.0)
    # zero half-spread -> band vanishes -> coincides with the plain book
    res_zero = build_long_short_portfolio(
        df, preds, hysteresis=True, prev_long=prev_long, prev_short=set(),
        tc_half_spread=pd.Series(0.0, index=df.index),
    )
    assert 10008 not in res_zero["positions_long"]


def test_hysteresis_excludes_tc_aware_sort():
    n = 10
    df = pd.DataFrame({
        "yyyymm": 202302, "permno": range(10001, 10001 + n),
        "ret": np.zeros(n), "excess_ret": np.arange(n, dtype=float) / 100.0,
    })
    preds = pd.Series(np.arange(n, dtype=float), index=df.index)
    with pytest.raises(ValueError, match="hysteresis"):
        build_long_short_portfolio(
            df, preds, hysteresis=True, tc_penalised=True,
            tc_per_stock=pd.Series(0.001, index=df.index),
        )


def test_apply_quantile_hysteresis_sticky():
    from src.portfolio.construction import _apply_quantile_hysteresis
    mp = pd.DataFrame({
        "permno": range(10001, 10011),
        "_quantile_score": np.arange(10, dtype=float),
    })
    mp["_q"] = _assign_quantiles(mp["_quantile_score"], 5)  # Q4={score6,7}, Q5={8,9}
    prev = {10008: 5}  # stock with score 7 (fresh Q4) was in Q5 last month
    out_wide = _apply_quantile_hysteresis(
        mp, "_quantile_score", "_q", prev, pd.Series(5.0, index=mp.index), 5
    )
    assert int(out_wide.loc[7]) == 5  # within band -> retained in Q5
    out_zero = _apply_quantile_hysteresis(
        mp, "_quantile_score", "_q", prev, pd.Series(0.0, index=mp.index), 5
    )
    assert int(out_zero.loc[7]) == 4  # band vanishes -> fresh Q4
