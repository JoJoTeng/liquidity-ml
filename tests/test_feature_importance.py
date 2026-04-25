"""Tests for src/analysis/feature_importance.py"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import feature_importance as fi
from src.analysis.feature_importance import (
    aggregate_importance,
    compute_group_importance,
    compute_importance_ratio,
    compute_mean_abs_shap,
    cross_model_h2_summary,
    load_importance,
    _categorize_feature,
)
from src.data.loader import (
    ILLIQUIDITY_FEATURES,
    TRADABLE_FEATURES,
)


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def tiny_importance_df():
    """10 windows × 20 features including tradable and illiquidity groups."""
    rng = np.random.RandomState(42)
    n_windows = 10

    # Use real feature names from both groups + some "other" features
    tradable = TRADABLE_FEATURES[:4]  # first 4 tradable
    illiq = ILLIQUIDITY_FEATURES[:4]  # first 4 illiquidity
    other = [f"OtherFeat_{i}" for i in range(12)]
    features = tradable + illiq + other

    data = rng.exponential(0.01, size=(n_windows, len(features)))
    # Make tradable features somewhat more important
    data[:, :4] *= 2.0

    months = list(range(200001, 200001 + n_windows))
    return pd.DataFrame(data, columns=features, index=months)


@pytest.fixture
def shifted_importance_pair(tiny_importance_df):
    """Two DataFrames where 'weighted' clearly shifts toward tradable."""
    fi_std = tiny_importance_df.copy()

    fi_wt = tiny_importance_df.copy()
    tradable = TRADABLE_FEATURES[:4]
    illiq = ILLIQUIDITY_FEATURES[:4]
    # Boost tradable and reduce illiquidity in weighted version
    fi_wt[tradable] *= 3.0
    fi_wt[illiq] *= 0.3

    return fi_std, fi_wt


# ── TestComputeMeanAbsShap ───────────────────────────────────


class TestComputeMeanAbsShap:
    def test_basic_computation(self):
        df = pd.DataFrame({
            "a": [1.0, -2.0, 3.0],
            "b": [0.5, 0.5, 0.5],
        })
        result = compute_mean_abs_shap(df)
        assert result["a"] == pytest.approx(2.0)
        assert result["b"] == pytest.approx(0.5)

    def test_handles_negative_values(self):
        df = pd.DataFrame({"x": [-1.0, -2.0, -3.0]})
        result = compute_mean_abs_shap(df)
        assert result["x"] == pytest.approx(2.0)

    def test_returns_series(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        result = compute_mean_abs_shap(df)
        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b"]


# ── TestAggregateImportance ──────────────────────────────────


class TestAggregateImportance:
    def test_mean_aggregation(self):
        df = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
        result = aggregate_importance(df, method="mean")
        assert result["a"] == pytest.approx(2.0)
        assert result["b"] == pytest.approx(3.0)

    def test_median_aggregation(self):
        df = pd.DataFrame({"a": [1, 3, 100], "b": [2, 4, 5]})
        result = aggregate_importance(df, method="median")
        assert result["a"] == pytest.approx(3.0)
        assert result["b"] == pytest.approx(4.0)

    def test_sorted_descending(self):
        df = pd.DataFrame({"low": [1, 1], "high": [10, 10]})
        result = aggregate_importance(df)
        assert result.index[0] == "high"

    def test_unknown_method_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_importance(df, method="mode")


# ── TestComputeGroupImportance ───────────────────────────────


class TestComputeGroupImportance:
    def test_tradable_group(self, tiny_importance_df):
        result = compute_group_importance(tiny_importance_df, "tradable")
        assert len(result) == len(tiny_importance_df)
        # All values should be positive (exponential data)
        assert (result > 0).all()

    def test_illiquidity_group(self, tiny_importance_df):
        result = compute_group_importance(tiny_importance_df, "illiquidity")
        assert len(result) == len(tiny_importance_df)
        assert (result > 0).all()

    def test_missing_features_handled(self):
        """DataFrame with no matching group features returns zeros."""
        df = pd.DataFrame({"UnknownFeat1": [1, 2], "UnknownFeat2": [3, 4]})
        result = compute_group_importance(df, "tradable")
        assert (result == 0.0).all()

    def test_unknown_group_raises(self, tiny_importance_df):
        with pytest.raises(ValueError, match="Unknown group"):
            compute_group_importance(tiny_importance_df, "momentum")

    def test_returns_series_per_window(self, tiny_importance_df):
        result = compute_group_importance(tiny_importance_df, "tradable")
        assert isinstance(result, pd.Series)
        assert len(result) == len(tiny_importance_df)


# ── TestComputeImportanceRatio ───────────────────────────────


class TestComputeImportanceRatio:
    def test_ratio_bounds(self, tiny_importance_df):
        ratio = compute_importance_ratio(tiny_importance_df)
        assert (ratio >= 0).all()
        assert (ratio <= 1).all()

    def test_all_tradable(self):
        """If only tradable features have importance, ratio → 1."""
        tradable = TRADABLE_FEATURES[:2]
        illiq = ILLIQUIDITY_FEATURES[:2]
        df = pd.DataFrame({
            tradable[0]: [10, 10],
            tradable[1]: [10, 10],
            illiq[0]: [0, 0],
            illiq[1]: [0, 0],
        })
        ratio = compute_importance_ratio(df)
        np.testing.assert_allclose(ratio.values, 1.0)

    def test_equal_groups(self):
        """If tradable = illiquidity importance, ratio = 0.5."""
        tradable = TRADABLE_FEATURES[:2]
        illiq = ILLIQUIDITY_FEATURES[:2]
        df = pd.DataFrame({
            tradable[0]: [5, 5],
            tradable[1]: [5, 5],
            illiq[0]: [5, 5],
            illiq[1]: [5, 5],
        })
        ratio = compute_importance_ratio(df)
        np.testing.assert_allclose(ratio.values, 0.5)


# ── TestH2GroupShift ─────────────────────────────────────────


class TestH2GroupShift:
    def test_structure(self, shifted_importance_pair):
        fi_std, fi_wt = shifted_importance_pair
        result = fi.test_h2_group_shift(fi_std, fi_wt)
        expected_keys = {
            "ratio_std_mean", "ratio_wt_mean", "ratio_diff",
            "ratio_test", "tradable_test", "illiquidity_test", "n_windows",
        }
        assert set(result.keys()) == expected_keys

    def test_significant_shift(self, shifted_importance_pair):
        """Weighted clearly favors tradable → p should be small."""
        fi_std, fi_wt = shifted_importance_pair
        result = fi.test_h2_group_shift(fi_std, fi_wt)
        assert result["ratio_wt_mean"] > result["ratio_std_mean"]
        assert result["ratio_diff"] > 0
        assert result["ratio_test"]["p_value"] < 0.05

    def test_no_shift(self, tiny_importance_df):
        """Identical DataFrames → high p-value."""
        result = fi.test_h2_group_shift(
            tiny_importance_df, tiny_importance_df.copy()
        )
        assert result["ratio_diff"] == pytest.approx(0.0, abs=1e-10)
        assert result["ratio_test"]["p_value"] > 0.5

    def test_extended_features(self, shifted_importance_pair):
        fi_std, fi_wt = shifted_importance_pair
        result = fi.test_h2_group_shift(fi_std, fi_wt, extended=True)
        assert "ratio_test" in result


# ── TestH2PerFeature ─────────────────────────────────────────


class TestH2PerFeature:
    def test_output_columns(self, shifted_importance_pair):
        fi_std, fi_wt = shifted_importance_pair
        result = fi.test_h2_per_feature(fi_std, fi_wt)
        expected_cols = {
            "feature", "mean_std", "mean_wt", "mean_diff",
            "t_stat", "p_value", "significant", "category",
        }
        assert set(result.columns) == expected_cols

    def test_category_assignment(self, shifted_importance_pair):
        fi_std, fi_wt = shifted_importance_pair
        result = fi.test_h2_per_feature(fi_std, fi_wt)
        categories = set(result["category"].unique())
        assert categories <= {"tradable", "illiquidity", "other"}

    def test_sorted_by_pvalue(self, shifted_importance_pair):
        fi_std, fi_wt = shifted_importance_pair
        result = fi.test_h2_per_feature(fi_std, fi_wt)
        p_vals = result["p_value"].values
        assert (p_vals[:-1] <= p_vals[1:] + 1e-15).all()

    def test_one_row_per_feature(self, shifted_importance_pair):
        fi_std, fi_wt = shifted_importance_pair
        result = fi.test_h2_per_feature(fi_std, fi_wt)
        assert len(result) == len(fi_std.columns)


# ── TestCategorizeFeature ────────────────────────────────────


class TestCategorizeFeature:
    def test_tradable(self):
        assert _categorize_feature("CBOperProf") == "tradable"

    def test_illiquidity(self):
        assert _categorize_feature("AM") == "illiquidity"

    def test_other(self):
        assert _categorize_feature("IdioVol3F") == "other"


# ── TestLoadImportance ───────────────────────────────────────


class TestLoadImportance:
    def test_load_shap_from_disk(self, tmp_path):
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir()

        df = pd.DataFrame(
            {"feat1": [0.1, 0.2], "feat2": [0.3, 0.4]},
            index=[200001, 200002],
        )
        df.to_csv(model_dir / "shap_importance_std.csv")
        df.to_csv(model_dir / "shap_importance_wt.csv")

        result = load_importance("xgboost", "shap", output_dir=tmp_path)
        assert "standard" in result and "weighted" in result
        assert list(result["standard"].columns) == ["feat1", "feat2"]
        assert len(result["standard"]) == 2

    def test_load_gain_from_disk(self, tmp_path):
        model_dir = tmp_path / "xgboost"
        model_dir.mkdir()

        df = pd.DataFrame(
            {"feat1": [0.5, 0.6]},
            index=[200001, 200002],
        )
        df.to_csv(model_dir / "feature_importance_std.csv")
        df.to_csv(model_dir / "feature_importance_wt.csv")

        result = load_importance("xgboost", "gain", output_dir=tmp_path)
        assert result["standard"]["feat1"].iloc[0] == pytest.approx(0.5)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_importance("xgboost", "shap", output_dir=tmp_path)


# ── TestCrossModelSummary ────────────────────────────────────


class TestCrossModelSummary:
    def test_structure(self, tmp_path):
        """Create mock data for 2 models, verify output DataFrame."""
        for name in ["xgboost", "random_forest"]:
            model_dir = tmp_path / name
            model_dir.mkdir()
            rng = np.random.RandomState(42)

            tradable = TRADABLE_FEATURES[:2]
            illiq = ILLIQUIDITY_FEATURES[:2]
            features = tradable + illiq

            df = pd.DataFrame(
                rng.exponential(0.01, size=(10, len(features))),
                columns=features,
                index=range(200001, 200011),
            )
            df.to_csv(model_dir / "shap_importance_std.csv")
            df.to_csv(model_dir / "shap_importance_wt.csv")

        result = cross_model_h2_summary(
            model_names=["xgboost", "random_forest"],
            output_dir=tmp_path,
        )
        assert len(result) == 2
        assert "model" in result.columns
        assert "t_stat" in result.columns

    def test_handles_missing_model(self, tmp_path):
        result = cross_model_h2_summary(
            model_names=["nonexistent_model"],
            output_dir=tmp_path,
        )
        assert len(result) == 0


# ── TestShapComputation (integration) ────────────────────────


class TestShapComputation:
    """Test get_shap_values() on actual models with tiny data."""

    @pytest.fixture
    def tiny_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5).astype(np.float32)
        y = X @ rng.randn(5) + rng.randn(50) * 0.1
        return X, y

    @pytest.fixture
    def feature_names(self):
        return [f"feat_{i}" for i in range(5)]

    def test_xgboost_shap_values(self, tiny_data, feature_names):
        from src.models.xgboost_model import XGBoostPredictor

        X, y = tiny_data
        model = XGBoostPredictor(
            config={"n_estimators": 10, "max_depth": 3, "min_child_weight": 1},
            seed=42,
        )
        model.fit(X[:40], y[:40])

        sv = model.get_shap_values(X[40:], feature_names=feature_names)
        assert isinstance(sv, pd.DataFrame)
        assert sv.shape == (10, 5)
        assert list(sv.columns) == feature_names

    @pytest.mark.skip(reason="RandomForest removed in LiquidityML v3")
    def test_random_forest_shap_values(self, tiny_data, feature_names):
        pass

    def test_xgboost_shap_additivity(self, tiny_data):
        """SHAP values + expected value should approximate predictions."""
        from src.models.xgboost_model import XGBoostPredictor

        X, y = tiny_data
        model = XGBoostPredictor(
            config={"n_estimators": 50, "max_depth": 3, "min_child_weight": 1},
            seed=42,
        )
        model.fit(X[:40], y[:40])

        sv = model.get_shap_values(X[40:])
        preds = model.predict(X[40:])

        # SHAP sum + base value ≈ predictions
        import shap
        explainer = shap.TreeExplainer(model.model)
        base = explainer.expected_value
        shap_sum = sv.values.sum(axis=1) + base

        np.testing.assert_allclose(shap_sum, preds, atol=0.05)

    @pytest.mark.slow
    def test_nn_requires_background(self, tiny_data, feature_names):
        """NeuralNet should raise ValueError without X_background."""
        from src.models.neural_network_model import NeuralNetPredictor

        X, y = tiny_data
        model = NeuralNetPredictor(
            config={
                "hidden_layers": [8],
                "dropout": 0.0,
                "learning_rate": 0.01,
                "batch_size": 32,
                "epochs": 2,
                "patience": 2,
            },
            seed=42,
        )
        model.fit(X[:40], y[:40])

        with pytest.raises(ValueError, match="X_background is required"):
            model.get_shap_values(X[40:], feature_names=feature_names)
