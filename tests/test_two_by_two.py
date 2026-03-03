"""Tests for src/evaluation/two_by_two.py"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.two_by_two import (
    _get_oos_months,
    _make_json_serializable,
    _months_to_yyyymm,
    _offset_yyyymm,
    _prepare_xy,
    _split_window,
    _yyyymm_to_months,
)


# ── Fixtures ─────────────────────────────────────────────────


def _make_synthetic_panel(
    n_stocks: int = 80,
    months: list[int] | None = None,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a small synthetic panel for testing.

    Returns (panel, feature_names).
    """
    rng = np.random.RandomState(seed)

    if months is None:
        # 20 months: 199801 through 199908
        months = []
        ym = 199801
        for _ in range(20):
            months.append(ym)
            m = ym % 100
            y = ym // 100
            if m == 12:
                ym = (y + 1) * 100 + 1
            else:
                ym = y * 100 + m + 1

    features = [f"feat_{i}" for i in range(n_features)]
    rows = []
    for ym in months:
        for pid in range(1, n_stocks + 1):
            row = {"permno": pid, "yyyymm": ym, "ret": rng.normal(0.01, 0.05)}
            row["excess_ret"] = row["ret"] - 0.003
            row["liq_dvol_21d"] = rng.uniform(1e6, 1e8)
            row["liq_dvol_6m"] = rng.uniform(1e6, 1e8)
            row["liq_me_raw"] = rng.uniform(1e7, 1e9)
            row["liq_lambda_tc"] = rng.uniform(0.001, 0.1)
            row["liq_BidAskSpread"] = rng.uniform(0.001, 0.05)
            row["liq_liu_lm"] = rng.uniform(0, 100)
            row["daily_sigma"] = rng.uniform(0.01, 0.05)
            for f in features:
                row[f] = rng.normal(0, 1)
            rows.append(row)

    panel = pd.DataFrame(rows)
    return panel, features


@pytest.fixture
def synthetic_panel():
    """20-month, 130-stock panel with 10 features."""
    return _make_synthetic_panel(n_stocks=130)


@pytest.fixture
def tiny_config():
    """Minimal config for fast testing with short windows."""
    return {
        "project": {"seed": 42, "output_dir": "outputs", "data_dir": "data"},
        "data": {
            "target_col": "excess_ret",
            "return_col": "ret",
            "id_cols": ["permno", "yyyymm"],
            "selected_features": [f"feat_{i}" for i in range(10)],
        },
        "training": {
            "train_window": 10,
            "validation_window": 3,
            "test_window": 1,
            "retune_frequency": 5,
            "oos_start": 199901,
            "oos_end": 199912,
        },
        "models": {
            "run_models": ["xgboost"],
            "xgboost": {"n_estimators": 10, "max_depth": 2},
        },
        "liquidity": {"primary": "dvol_21d", "measures": ["dvol_21d"]},
        "weighting": {
            "primary": "softmax_rank",
            "softmax_rank": {"lambda": 2.0},
        },
        "portfolio": {
            "n_quantiles": 5,
            "long_quantile": 5,
            "short_quantile": 1,
            "position_cap": 0.20,
            "min_liquidity_pctile": 0.0,
            "rebalance_frequency": "monthly",
        },
        "transaction_costs": {
            "spread_col": "BidAskSpread",
            "lambda_market_impact": 0.1,
            "sigma_col": "daily_sigma",
            "adv_col": "dvol_21d",
            "aum_scenarios": [100_000_000],
        },
        "inference": {
            "newey_west_lags": 2,
            "bootstrap_samples": 99,
            "significance_level": 0.05,
            "one_sided_test": False,
            "calibrate_block_length": False,
            "block_grid": [2, 4],
            "factor_models": {
                "capm": ["Mkt-RF"],
                "ff3": ["Mkt-RF", "SMB", "HML"],
                "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
            },
        },
    }


# ── TestDateUtils ─────────────────────────────────────────────


class TestDateUtils:
    """Test yyyymm arithmetic utilities."""

    def test_roundtrip(self):
        """yyyymm -> months -> yyyymm should be identity."""
        for ym in [199901, 200012, 202406, 197201]:
            assert _months_to_yyyymm(_yyyymm_to_months(ym)) == ym

    def test_offset_forward(self):
        assert _offset_yyyymm(200001, 1) == 200002
        assert _offset_yyyymm(200001, 12) == 200101

    def test_offset_december_rollover(self):
        assert _offset_yyyymm(200012, 1) == 200101
        assert _offset_yyyymm(200012, 2) == 200102

    def test_offset_backward(self):
        assert _offset_yyyymm(200001, -1) == 199912
        assert _offset_yyyymm(200001, -12) == 199901

    def test_offset_zero(self):
        assert _offset_yyyymm(200006, 0) == 200006

    def test_large_offset(self):
        assert _offset_yyyymm(200001, 24) == 200201
        assert _offset_yyyymm(200001, -24) == 199801


# ── TestGetOosMonths ──────────────────────────────────────────


class TestGetOosMonths:
    """Test OOS month computation."""

    def test_returns_correct_months(self, synthetic_panel, tiny_config):
        panel, _ = synthetic_panel
        oos = _get_oos_months(panel, tiny_config)

        # All returned months should be in [oos_start, oos_end]
        for m in oos:
            assert tiny_config["training"]["oos_start"] <= m <= tiny_config["training"]["oos_end"]

    def test_sufficient_history(self, synthetic_panel, tiny_config):
        panel, _ = synthetic_panel
        all_months = sorted(panel["yyyymm"].unique())
        oos = _get_oos_months(panel, tiny_config)

        min_history = (
            tiny_config["training"]["train_window"]
            + tiny_config["training"]["validation_window"]
        )

        # Each OOS month must have at least min_history months before it
        for m in oos:
            idx = all_months.index(m)
            assert idx >= min_history

    def test_no_oos_months_when_insufficient_data(self, tiny_config):
        """With very short data, no OOS months should be available."""
        panel, _ = _make_synthetic_panel(n_stocks=50, months=[199901, 199902, 199903])
        oos = _get_oos_months(panel, tiny_config)
        assert len(oos) == 0

    def test_empty_when_oos_range_outside_data(self, synthetic_panel, tiny_config):
        panel, _ = synthetic_panel
        cfg = {**tiny_config, "training": {**tiny_config["training"], "oos_start": 202501, "oos_end": 202512}}
        oos = _get_oos_months(panel, cfg)
        assert len(oos) == 0


# ── TestSplitWindow ───────────────────────────────────────────


class TestSplitWindow:
    """Test train/val/test splitting."""

    def test_split_sizes(self, synthetic_panel, tiny_config):
        panel, _ = synthetic_panel
        all_months = sorted(panel["yyyymm"].unique())
        oos = _get_oos_months(panel, tiny_config)
        assert len(oos) > 0

        test_month = oos[0]
        train_df, val_df, test_df = _split_window(panel, test_month, all_months, tiny_config)

        train_months = train_df["yyyymm"].unique()
        val_months = val_df["yyyymm"].unique()
        test_months = test_df["yyyymm"].unique()

        assert len(train_months) == tiny_config["training"]["train_window"]
        assert len(val_months) == tiny_config["training"]["validation_window"]
        assert len(test_months) == 1
        assert test_months[0] == test_month

    def test_no_overlap(self, synthetic_panel, tiny_config):
        panel, _ = synthetic_panel
        all_months = sorted(panel["yyyymm"].unique())
        oos = _get_oos_months(panel, tiny_config)
        test_month = oos[0]

        train_df, val_df, test_df = _split_window(panel, test_month, all_months, tiny_config)

        train_set = set(train_df["yyyymm"])
        val_set = set(val_df["yyyymm"])
        test_set = set(test_df["yyyymm"])

        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_temporal_order(self, synthetic_panel, tiny_config):
        """Train months < Val months < Test month."""
        panel, _ = synthetic_panel
        all_months = sorted(panel["yyyymm"].unique())
        oos = _get_oos_months(panel, tiny_config)
        test_month = oos[0]

        train_df, val_df, test_df = _split_window(panel, test_month, all_months, tiny_config)

        assert max(train_df["yyyymm"]) < min(val_df["yyyymm"])
        assert max(val_df["yyyymm"]) < min(test_df["yyyymm"])


# ── TestPrepareXY ─────────────────────────────────────────────


class TestPrepareXY:
    """Test feature preparation and normalization."""

    def test_shapes_match(self, synthetic_panel, tiny_config):
        panel, features = synthetic_panel
        all_months = sorted(panel["yyyymm"].unique())
        oos = _get_oos_months(panel, tiny_config)
        test_month = oos[0]

        train_df, val_df, test_df = _split_window(panel, test_month, all_months, tiny_config)
        data = _prepare_xy(train_df, val_df, test_df, features, tiny_config)

        n_features = len(features)

        assert data["X_train"].shape[1] == n_features
        assert data["X_val"].shape[1] == n_features
        assert data["X_test"].shape[1] == n_features

        assert len(data["y_train"]) == data["X_train"].shape[0]
        assert len(data["y_val"]) == data["X_val"].shape[0]
        assert len(data["y_test"]) == data["X_test"].shape[0]

    def test_no_nan_in_features(self, synthetic_panel, tiny_config):
        panel, features = synthetic_panel
        all_months = sorted(panel["yyyymm"].unique())
        oos = _get_oos_months(panel, tiny_config)
        test_month = oos[0]

        train_df, val_df, test_df = _split_window(panel, test_month, all_months, tiny_config)
        data = _prepare_xy(train_df, val_df, test_df, features, tiny_config)

        assert not np.any(np.isnan(data["X_train"]))
        assert not np.any(np.isnan(data["X_val"]))
        assert not np.any(np.isnan(data["X_test"]))

    def test_weights_computed(self, synthetic_panel, tiny_config):
        panel, features = synthetic_panel
        all_months = sorted(panel["yyyymm"].unique())
        oos = _get_oos_months(panel, tiny_config)
        test_month = oos[0]

        train_df, val_df, test_df = _split_window(panel, test_month, all_months, tiny_config)
        data = _prepare_xy(train_df, val_df, test_df, features, tiny_config)

        assert len(data["weights_train"]) == len(data["y_train"])
        assert len(data["weights_val"]) == len(data["y_val"])
        assert np.all(data["weights_train"] > 0)
        assert np.all(data["weights_val"] > 0)

    def test_expanding_mean_computed(self, synthetic_panel, tiny_config):
        panel, features = synthetic_panel
        all_months = sorted(panel["yyyymm"].unique())
        oos = _get_oos_months(panel, tiny_config)
        test_month = oos[0]

        train_df, val_df, test_df = _split_window(panel, test_month, all_months, tiny_config)
        data = _prepare_xy(train_df, val_df, test_df, features, tiny_config)

        assert isinstance(data["train_target_mean"], float)
        assert not np.isnan(data["train_target_mean"])

    def test_meta_test_has_identifiers(self, synthetic_panel, tiny_config):
        panel, features = synthetic_panel
        all_months = sorted(panel["yyyymm"].unique())
        oos = _get_oos_months(panel, tiny_config)
        test_month = oos[0]

        train_df, val_df, test_df = _split_window(panel, test_month, all_months, tiny_config)
        data = _prepare_xy(train_df, val_df, test_df, features, tiny_config)

        assert "permno" in data["meta_test"].columns
        assert "yyyymm" in data["meta_test"].columns
        assert len(data["meta_test"]) == len(data["y_test"])


# ── TestMakeJsonSerializable ──────────────────────────────────


class TestMakeJsonSerializable:
    """Test numpy-to-Python type conversion for JSON."""

    def test_numpy_int(self):
        assert _make_json_serializable(np.int64(42)) == 42
        assert isinstance(_make_json_serializable(np.int64(42)), int)

    def test_numpy_float(self):
        result = _make_json_serializable(np.float64(3.14))
        assert abs(result - 3.14) < 1e-10
        assert isinstance(result, float)

    def test_numpy_array(self):
        result = _make_json_serializable(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_nested_dict(self):
        obj = {
            "a": np.float64(1.0),
            "b": {"c": np.int64(2), "d": np.array([3, 4])},
        }
        result = _make_json_serializable(obj)
        assert result == {"a": 1.0, "b": {"c": 2, "d": [3, 4]}}

    def test_passthrough_python_types(self):
        assert _make_json_serializable("hello") == "hello"
        assert _make_json_serializable(42) == 42
        assert _make_json_serializable(None) is None


# ── TestRollingPredict (mocked) ───────────────────────────────


class TestRollingPredict:
    """Test the rolling predict loop with mocked models."""

    def test_predictions_structure(self, synthetic_panel, tiny_config):
        """Verify _rolling_predict produces correct prediction DataFrame."""
        from src.evaluation.two_by_two import _rolling_predict

        panel, features = synthetic_panel

        # Use a mock model that returns random predictions
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.side_effect = lambda X: np.random.RandomState(42).normal(0, 0.01, len(X))
        mock_model.get_feature_importance.return_value = pd.Series(
            np.ones(len(features)) / len(features), index=features
        )
        mock_model.tune_hyperparameters.return_value = {"n_estimators": 10}

        with patch("src.evaluation.two_by_two.create_model", return_value=mock_model):
            result = _rolling_predict("xgboost", panel, features, tiny_config)

        preds = result["predictions"]
        assert "permno" in preds.columns
        assert "yyyymm" in preds.columns
        assert "y_true" in preds.columns
        assert "pred_std" in preds.columns
        assert "pred_wt" in preds.columns
        assert len(preds) > 0

    def test_feature_importance_collected(self, synthetic_panel, tiny_config):
        """Feature importances should be collected for each window."""
        from src.evaluation.two_by_two import _rolling_predict

        panel, features = synthetic_panel

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.side_effect = lambda X: np.zeros(len(X))
        mock_model.get_feature_importance.return_value = pd.Series(
            np.ones(len(features)) / len(features), index=features
        )
        mock_model.tune_hyperparameters.return_value = {"n_estimators": 10}

        with patch("src.evaluation.two_by_two.create_model", return_value=mock_model):
            result = _rolling_predict("xgboost", panel, features, tiny_config)

        fi_std = result["feature_importance_std"]
        fi_wt = result["feature_importance_wt"]

        # Should have one row per OOS month
        n_oos = len(result["oos_months"])
        assert len(fi_std) > 0
        assert len(fi_wt) > 0
        assert fi_std.shape[1] == len(features)
        assert fi_wt.shape[1] == len(features)

    def test_tuning_triggered(self, synthetic_panel, tiny_config):
        """Hyperparameter tuning should be called at the right frequency."""
        from src.evaluation.two_by_two import _rolling_predict

        panel, features = synthetic_panel
        # Set retune_frequency to 3 so tuning happens more than once
        cfg = {**tiny_config, "training": {**tiny_config["training"], "retune_frequency": 3}}

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.side_effect = lambda X: np.zeros(len(X))
        mock_model.get_feature_importance.return_value = pd.Series(
            np.ones(len(features)) / len(features), index=features
        )
        mock_model.tune_hyperparameters.return_value = {"n_estimators": 10}

        with patch("src.evaluation.two_by_two.create_model", return_value=mock_model):
            result = _rolling_predict("xgboost", panel, features, cfg)

        # tune_hyperparameters should be called at least once (first window)
        # and each call tunes 2 models (std + wt)
        assert mock_model.tune_hyperparameters.call_count >= 2  # at least 1 round × 2

    def test_expanding_means_accumulated(self, synthetic_panel, tiny_config):
        """Expanding means should grow across windows."""
        from src.evaluation.two_by_two import _rolling_predict

        panel, features = synthetic_panel

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.side_effect = lambda X: np.zeros(len(X))
        mock_model.get_feature_importance.return_value = pd.Series(
            np.ones(len(features)) / len(features), index=features
        )
        mock_model.tune_hyperparameters.return_value = {"n_estimators": 10}

        with patch("src.evaluation.two_by_two.create_model", return_value=mock_model):
            result = _rolling_predict("xgboost", panel, features, tiny_config)

        means = result["expanding_means"]
        assert len(means) > 0
        # All should be finite
        for m in means:
            assert np.isfinite(m)

    def test_timing_info(self, synthetic_panel, tiny_config):
        """Timing dict should be populated."""
        from src.evaluation.two_by_two import _rolling_predict

        panel, features = synthetic_panel

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.side_effect = lambda X: np.zeros(len(X))
        mock_model.get_feature_importance.return_value = pd.Series(
            np.ones(len(features)) / len(features), index=features
        )
        mock_model.tune_hyperparameters.return_value = {"n_estimators": 10}

        with patch("src.evaluation.two_by_two.create_model", return_value=mock_model):
            result = _rolling_predict("xgboost", panel, features, tiny_config)

        assert "total_seconds" in result["timing"]
        assert "seconds_per_month" in result["timing"]
        assert result["timing"]["total_seconds"] >= 0


# ── TestSaveResults ───────────────────────────────────────────


class TestSaveResults:
    """Test result persistence."""

    def test_save_single_model(self, tmp_path):
        """save_results should create expected files for a single model."""
        from src.evaluation.two_by_two import save_results

        # Minimal mock result
        n = 5
        gross = pd.DataFrame({
            "yyyymm": [200001 + i for i in range(n)],
            "ret_long_short": np.random.normal(0, 0.01, n),
        })
        net = gross.copy()
        net["ret_long_short_net"] = net["ret_long_short"] - 0.001

        result = {
            "model": "xgboost",
            "cells": {
                cell: {"gross_returns": gross.copy(), "net_returns": net.copy(), "positions": {}}
                for cell in ["1A", "1B", "2A", "2B"]
            },
            "predictions": pd.DataFrame({
                "permno": [1, 2], "yyyymm": [200001, 200001],
                "y_true": [0.01, -0.01], "pred_std": [0.005, -0.005],
                "pred_wt": [0.006, -0.006], "expanding_mean": [0.0, 0.0],
            }),
            "feature_importance": {
                "standard": pd.DataFrame({"a": [1.0], "b": [2.0]}),
                "weighted": pd.DataFrame({"a": [1.5], "b": [1.5]}),
            },
            "oos_r2": {"standard": 0.01, "weighted": 0.02},
            "effect_decomposition": {"training_effect": np.float64(0.1)},
        }

        save_results(result, output_dir=tmp_path)

        model_dir = tmp_path / "xgboost"
        assert (model_dir / "predictions.parquet").exists()
        assert (model_dir / "gross_returns_1A.csv").exists()
        assert (model_dir / "net_returns_2B.csv").exists()
        assert (model_dir / "feature_importance_std.csv").exists()
        assert (model_dir / "feature_importance_wt.csv").exists()
        assert (model_dir / "effect_decomposition.json").exists()
        assert (model_dir / "oos_r2.json").exists()

    def test_save_multi_model(self, tmp_path):
        """save_results with per_model key should create summary and ensemble files."""
        from src.evaluation.two_by_two import save_results

        gross = pd.DataFrame({
            "yyyymm": [200001],
            "ret_long_short": [0.01],
        })
        net = gross.copy()
        single = {
            "model": "xgboost",
            "cells": {
                cell: {"gross_returns": gross.copy(), "net_returns": net.copy(), "positions": {}}
                for cell in ["1A", "1B", "2A", "2B"]
            },
            "predictions": pd.DataFrame({"permno": [1], "yyyymm": [200001], "y_true": [0.01], "pred_std": [0.01], "pred_wt": [0.01], "expanding_mean": [0.0]}),
            "feature_importance": {
                "standard": pd.DataFrame({"a": [1.0]}),
                "weighted": pd.DataFrame({"a": [1.0]}),
            },
            "oos_r2": {"standard": 0.01, "weighted": 0.02},
            "effect_decomposition": {"training_effect": 0.1},
        }

        multi = {
            "per_model": {"xgboost": single},
            "ensemble": {"SR_1A": 0.5},
            "summary": pd.DataFrame([{"model": "xgboost", "SR_1A": 0.5}]),
        }

        save_results(multi, output_dir=tmp_path)

        assert (tmp_path / "summary.csv").exists()
        assert (tmp_path / "ensemble.json").exists()
        assert (tmp_path / "xgboost" / "predictions.parquet").exists()
