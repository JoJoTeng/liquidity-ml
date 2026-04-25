"""Tests for src/evaluation/statistics.py"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistics import (
    bootstrap_sharpe_test,
    compute_effect_decomposition,
    factor_alpha,
    grs_test,
    load_ff_factors,
    newey_west_tstat,
    oos_r_squared,
    paired_ttest,
    sharpe_ratio,
)


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def monthly_returns():
    """100 months of simulated returns with known properties."""
    rng = np.random.RandomState(42)
    return rng.normal(0.005, 0.04, size=100)


@pytest.fixture
def two_return_series():
    """Two correlated return series with different Sharpe ratios."""
    rng = np.random.RandomState(42)
    T = 200
    factor = rng.normal(0, 0.03, size=T)
    ra = 0.008 + 0.8 * factor + rng.normal(0, 0.02, T)  # higher SR
    rb = 0.002 + 0.8 * factor + rng.normal(0, 0.02, T)  # lower SR
    return pd.Series(ra), pd.Series(rb)


@pytest.fixture
def ff_factors_df():
    """Simulated FF factors for testing factor_alpha."""
    rng = np.random.RandomState(42)
    T = 100
    # Sequential yyyymm values
    dates = [200001 + i for i in range(T)]
    return pd.DataFrame({
        "yyyymm": dates,
        "Mkt-RF": rng.normal(0.005, 0.04, T),
        "SMB": rng.normal(0.002, 0.03, T),
        "HML": rng.normal(0.003, 0.03, T),
        "RMW": rng.normal(0.003, 0.02, T),
        "CMA": rng.normal(0.002, 0.02, T),
    })


# ── TestSharpeRatio ──────────────────────────────────────────


class TestSharpeRatio:
    def test_known_value(self):
        # Constant returns: mean=0.01, std~0.04
        rng = np.random.RandomState(1)
        r = rng.normal(0.01, 0.04, size=10000)
        sr = sharpe_ratio(r)
        # Expected monthly Sharpe: 0.01/0.04 ≈ 0.25
        assert abs(sr - 0.25) < 0.02

    def test_zero_volatility(self):
        r = np.array([0.01] * 50)
        assert sharpe_ratio(r) == 0.0

    def test_nan_handling(self):
        r = np.array([0.01, 0.02, np.nan, 0.03, 0.04])
        sr = sharpe_ratio(r)
        assert not np.isnan(sr)

    def test_too_few_obs(self):
        assert np.isnan(sharpe_ratio(np.array([0.01])))

    def test_no_annualization(self):
        r = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        sr_ann = sharpe_ratio(r, annualize=True)
        sr_raw = sharpe_ratio(r)
        np.testing.assert_allclose(sr_ann, sr_raw * np.sqrt(12), atol=1e-10)

    def test_negative_sharpe(self):
        rng = np.random.RandomState(1)
        r = rng.normal(-0.01, 0.04, size=10000)
        assert sharpe_ratio(r) < 0


# ── TestNeweyWestTstat ───────────────────────────────────────


class TestNeweyWestTstat:
    def test_significant_positive_mean(self):
        rng = np.random.RandomState(42)
        r = rng.normal(0.02, 0.01, size=200)
        result = newey_west_tstat(r)
        assert result["t_stat"] > 2.0
        assert result["p_value"] < 0.05

    def test_noise_only(self):
        rng = np.random.RandomState(42)
        r = rng.normal(0.0, 0.05, size=100)
        result = newey_west_tstat(r)
        # With zero true mean, p-value should be large (most of the time)
        assert result["mean"] == pytest.approx(0.0, abs=0.015)

    def test_returns_correct_keys(self, monthly_returns):
        result = newey_west_tstat(monthly_returns)
        expected = {"mean", "std_err", "t_stat", "p_value", "n_obs"}
        assert set(result.keys()) == expected

    def test_custom_lags(self, monthly_returns):
        r1 = newey_west_tstat(monthly_returns, lags=3)
        r2 = newey_west_tstat(monthly_returns, lags=12)
        # Different lags → different SEs (generally)
        assert r1["std_err"] != r2["std_err"]


# ── TestFactorAlpha ──────────────────────────────────────────


class TestFactorAlpha:
    def test_no_factor_exposure(self, ff_factors_df):
        """Alpha ≈ mean return when returns are uncorrelated with factors."""
        rng = np.random.RandomState(99)
        T = len(ff_factors_df)
        ret_df = pd.DataFrame({
            "yyyymm": ff_factors_df["yyyymm"],
            "ret_long_short": rng.normal(0.005, 0.02, T),  # independent
        })
        result = factor_alpha(ret_df, model="ff3", ff_factors=ff_factors_df)
        # Alpha should be close to mean return
        np.testing.assert_allclose(
            result["alpha"], ret_df["ret_long_short"].mean(), atol=0.003
        )

    def test_correct_beta_sign(self, ff_factors_df):
        """Returns that load on Mkt-RF should have positive beta."""
        ret_df = pd.DataFrame({
            "yyyymm": ff_factors_df["yyyymm"],
            "ret_long_short": 0.001 + 1.2 * ff_factors_df["Mkt-RF"].values
                              + np.random.RandomState(1).normal(0, 0.01, len(ff_factors_df)),
        })
        result = factor_alpha(ret_df, model="capm", ff_factors=ff_factors_df)
        assert result["betas"]["Mkt-RF"] > 0.5

    def test_all_models(self, ff_factors_df):
        """CAPM, FF3, FF5 all run without error."""
        rng = np.random.RandomState(42)
        T = len(ff_factors_df)
        ret_df = pd.DataFrame({
            "yyyymm": ff_factors_df["yyyymm"],
            "ret_long_short": rng.normal(0.003, 0.02, T),
        })
        for model in ["capm", "ff3", "ff5"]:
            result = factor_alpha(ret_df, model=model, ff_factors=ff_factors_df)
            assert "alpha" in result
            assert "residuals" in result
            assert result["n_obs"] == T

    def test_returns_all_keys(self, ff_factors_df):
        rng = np.random.RandomState(42)
        T = len(ff_factors_df)
        ret_df = pd.DataFrame({
            "yyyymm": ff_factors_df["yyyymm"],
            "ret_long_short": rng.normal(0.003, 0.02, T),
        })
        result = factor_alpha(ret_df, model="ff3", ff_factors=ff_factors_df)
        expected = {
            "alpha", "alpha_annual", "alpha_se", "alpha_tstat", "alpha_pvalue",
            "betas", "beta_tstats", "beta_pvalues", "r_squared", "n_obs", "residuals",
        }
        assert set(result.keys()) == expected

    def test_invalid_model(self, ff_factors_df):
        ret_df = pd.DataFrame({
            "yyyymm": ff_factors_df["yyyymm"][:10],
            "ret_long_short": np.zeros(10),
        })
        with pytest.raises(ValueError, match="Unknown model"):
            factor_alpha(ret_df, model="ff99", ff_factors=ff_factors_df)


# ── TestGRSTest ──────────────────────────────────────────────


class TestGRSTest:
    def test_zero_alphas(self):
        """All alphas = 0 → large p-value."""
        rng = np.random.RandomState(42)
        N, T, K = 5, 200, 3
        alpha_vec = np.zeros(N)
        resid_matrix = rng.normal(0, 0.01, (T, N))
        factor_data = rng.normal(0.005, 0.03, (T, K))
        result = grs_test(alpha_vec, resid_matrix, factor_data, T)
        assert result["p_value"] > 0.5  # fail to reject

    def test_nonzero_alphas(self):
        """Large alphas → small p-value."""
        rng = np.random.RandomState(42)
        N, T, K = 5, 200, 3
        alpha_vec = np.array([0.05, 0.04, 0.03, 0.06, 0.05])
        resid_matrix = rng.normal(0, 0.01, (T, N))
        factor_data = rng.normal(0.005, 0.03, (T, K))
        result = grs_test(alpha_vec, resid_matrix, factor_data, T)
        assert result["p_value"] < 0.05

    def test_returns_correct_keys(self):
        rng = np.random.RandomState(42)
        result = grs_test(
            np.zeros(3), rng.normal(0, 1, (100, 3)),
            rng.normal(0, 1, (100, 2)), 100,
        )
        assert set(result.keys()) == {"f_stat", "p_value", "n_portfolios", "n_factors"}

    def test_capm_single_factor(self):
        """GRS works with K=1 (CAPM)."""
        rng = np.random.RandomState(42)
        N, T = 5, 200
        alpha_vec = np.zeros(N)
        resid_matrix = rng.normal(0, 0.01, (T, N))
        factor_data = rng.normal(0.005, 0.04, (T, 1))
        result = grs_test(alpha_vec, resid_matrix, factor_data, T)
        assert not np.isnan(result["f_stat"])


# ── TestBootstrapSharpeTest ──────────────────────────────────


class TestBootstrapSharpeTest:
    """Use small M for speed. calibrate_block_length=False always."""

    def test_identical_series_not_reject(self):
        rng = np.random.RandomState(42)
        r = pd.Series(rng.normal(0.005, 0.04, 100))
        result = bootstrap_sharpe_test(
            r, r, M=199, calibrate_block_length=False,
            one_sided=False, seed=42,
        )
        assert not result["reject"]
        assert abs(result["difference"]) < 1e-10

    def test_different_sharpes_reject(self, two_return_series):
        ra, rb = two_return_series
        result = bootstrap_sharpe_test(
            ra, rb, M=999, calibrate_block_length=False,
            one_sided=False, seed=42,
        )
        # ra has higher SR than rb
        assert result["difference"] > 0

    def test_one_sided_vs_two_sided(self, two_return_series):
        ra, rb = two_return_series
        r_one = bootstrap_sharpe_test(
            ra, rb, M=499, calibrate_block_length=False,
            one_sided=True, seed=42,
        )
        r_two = bootstrap_sharpe_test(
            ra, rb, M=499, calibrate_block_length=False,
            one_sided=False, seed=42,
        )
        assert r_one["one_sided"] is True
        assert r_two["one_sided"] is False

    def test_returns_correct_keys(self, two_return_series):
        ra, rb = two_return_series
        result = bootstrap_sharpe_test(
            ra, rb, M=99, calibrate_block_length=False, seed=42,
        )
        expected = {"difference", "p_value", "se", "block_length",
                    "test_stat", "reject", "one_sided"}
        assert expected.issubset(set(result.keys()))

    def test_numpy_input(self):
        """Works with numpy arrays, not just Series."""
        rng = np.random.RandomState(42)
        ra = rng.normal(0.01, 0.04, 100)
        rb = rng.normal(0.005, 0.04, 100)
        result = bootstrap_sharpe_test(
            ra, rb, M=99, calibrate_block_length=False, seed=42,
        )
        assert "p_value" in result


# ── TestOosRSquared ──────────────────────────────────────────


class TestOosRSquared:
    def test_perfect_prediction(self):
        y = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        assert oos_r_squared(y, y) == pytest.approx(1.0)

    def test_mean_prediction(self):
        rng = np.random.RandomState(42)
        y = rng.normal(0.01, 0.05, 100)
        y_pred = np.full_like(y, y.mean())
        r2 = oos_r_squared(y, y_pred)
        # Predicting the mean → R² ≈ 0
        assert abs(r2) < 0.02

    def test_bad_prediction(self):
        y = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        y_pred = np.array([0.10, -0.10, 0.20, -0.20, 0.30])
        r2 = oos_r_squared(y, y_pred)
        assert r2 < 0  # worse than mean

    def test_with_expanding_mean(self):
        rng = np.random.RandomState(42)
        y = rng.normal(0.01, 0.05, 50)
        # Expanding mean as benchmark
        y_hist = np.array([np.mean(y[:i+1]) for i in range(len(y))])
        r2 = oos_r_squared(y, y, y_hist_mean=y_hist)
        assert r2 > 0.9  # perfect prediction beats expanding mean

    def test_nan_handling(self):
        y = np.array([0.01, np.nan, 0.03, 0.04])
        y_pred = np.array([0.01, 0.02, 0.03, 0.04])
        r2 = oos_r_squared(y, y_pred)
        assert not np.isnan(r2)


# ── TestPairedTtest ──────────────────────────────────────────


class TestPairedTtest:
    def test_identical_series(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paired_ttest(a, a)
        assert result["mean_diff"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0)

    def test_significant_difference(self):
        rng = np.random.RandomState(42)
        a = rng.normal(5.0, 0.5, 50)
        b = rng.normal(3.0, 0.5, 50)
        result = paired_ttest(a, b)
        assert result["p_value"] < 0.001
        assert result["mean_diff"] > 1.0

    def test_one_sided(self):
        rng = np.random.RandomState(42)
        a = rng.normal(5.0, 1.0, 30)
        b = rng.normal(4.0, 1.0, 30)
        r_greater = paired_ttest(a, b, alternative="greater")
        r_two = paired_ttest(a, b, alternative="two-sided")
        # One-sided p ≈ two-sided p / 2 when in the right direction
        assert r_greater["p_value"] < r_two["p_value"]

    def test_returns_correct_keys(self):
        result = paired_ttest(np.array([1, 2, 3]), np.array([1, 2, 3]))
        expected = {"mean_diff", "t_stat", "p_value", "n_pairs",
                    "ci_lower", "ci_upper"}
        assert set(result.keys()) == expected


# ── TestEffectDecomposition ──────────────────────────────────


class TestEffectDecomposition:
    def test_effects_add_up(self):
        """Total = Portfolio + Training + Interaction."""
        rng = np.random.RandomState(42)
        r1a = rng.normal(0.003, 0.04, 100)
        r1b = rng.normal(0.004, 0.04, 100)
        r2a = rng.normal(0.005, 0.04, 100)
        r2b = rng.normal(0.006, 0.04, 100)

        result = compute_effect_decomposition(r1a, r1b, r2a, r2b)

        total = result["total_effect"]
        portfolio = result["portfolio_effect"]
        training = result["training_effect"]
        interaction = result["interaction"]

        np.testing.assert_allclose(
            total, portfolio + training + interaction, atol=1e-10
        )

    def test_identical_cells_zero_effects(self):
        """All cells same → all effects ≈ 0."""
        rng = np.random.RandomState(42)
        r = rng.normal(0.005, 0.04, 100)
        result = compute_effect_decomposition(r, r, r, r)

        assert abs(result["portfolio_effect"]) < 1e-10
        assert abs(result["training_effect"]) < 1e-10
        assert abs(result["total_effect"]) < 1e-10
        assert abs(result["interaction"]) < 1e-10


# ── TestLoadFFFactors ────────────────────────────────────────


class TestLoadFFFactors:
    def test_ff3_columns(self):
        ff = load_ff_factors(n_factors=3)
        assert "yyyymm" in ff.columns
        assert "Mkt-RF" in ff.columns
        assert "SMB" in ff.columns
        assert "HML" in ff.columns

    def test_ff3_values_in_decimal(self):
        ff = load_ff_factors(n_factors=3)
        # Monthly factor returns should be small decimals, not percentages
        for col in ["Mkt-RF", "SMB", "HML"]:
            assert ff[col].abs().max() < 0.5, (
                f"{col} values look like percentages, not decimals"
            )
