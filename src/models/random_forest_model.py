"""
Random Forest Return Predictor
===============================
Ensemble of decision trees for stock return prediction.
Supports sample_weight natively via sklearn's RandomForestRegressor.

RF provides a natural benchmark against XGBoost:
- No boosting → less prone to overfitting on noisy return data
- Bagging + random feature subsets → built-in regularization
- sample_weight enters the splitting criterion at each node

Key implementation note for weighted training:
  sklearn RF passes sample_weight to the impurity calculation,
  so liquid stocks contribute more to each split decision.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseReturnPredictor

logger = logging.getLogger(__name__)


class RandomForestPredictor(BaseReturnPredictor):
    """Random Forest regression model with weighted training support."""

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42):
        from src.config import load_config
        default_cfg = load_config()["models"]["random_forest"]
        cfg = {**default_cfg, **(config or {})}
        super().__init__(name="random_forest", config=cfg, seed=seed)
        self.model: RandomForestRegressor | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> "RandomForestPredictor":
        """Train Random Forest with optional implementability weights.

        sample_weight is passed to sklearn's fit(), which incorporates
        it into the weighted impurity criterion at each tree split.

        X_val / y_val / sample_weight_val are accepted for API consistency
        but not used (RF has no early stopping; OOB score serves as
        internal validation).
        """
        self.model = RandomForestRegressor(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            max_features=self.config["max_features"],
            min_samples_leaf=self.config["min_samples_leaf"],
            n_jobs=self.config.get("n_jobs", -1),
            random_state=self.seed,
            oob_score=True,  # use OOB for internal validation
        )

        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        self.is_fitted = True

        weighted_str = "weighted" if sample_weight is not None else "standard"
        logger.info(
            "RandomForest %s training: %d trees, OOB R²=%.4f, train shape %s",
            weighted_str,
            self.config["n_estimators"],
            self.model.oob_score_,
            X_train.shape,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self.model.predict(X)

    def get_feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.Series:
        """Return impurity-based (Gini) feature importance."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        importance = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importance))]
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

    def get_shap_values(
        self,
        X_test: np.ndarray,
        X_background: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Compute SHAP values using TreeExplainer (exact Tree SHAP)."""
        import shap

        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        cfg = config or {}
        check_additivity = cfg.get("tree_check_additivity", False)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(
            X_test, check_additivity=check_additivity
        )

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_test.shape[1])]

        return pd.DataFrame(shap_values, columns=feature_names)

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Grid search over configured search space using validation MSE.

        When sample_weight_val is provided, the validation MSE is weighted
        so that tuning optimizes the same objective as weighted training.
        """
        search_space = self.config.get("search_space", {})
        if not search_space:
            logger.info("No search space configured; using defaults.")
            return self.config

        keys = list(search_space.keys())
        values = list(search_space.values())
        param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

        logger.info("Tuning RandomForest over %d combinations...", len(param_grid))

        best_mse = np.inf
        best_params = {}

        for params in param_grid:
            trial_config = {**self.config, **params}
            model = RandomForestPredictor(config=trial_config, seed=self.seed)
            model.fit(X_train, y_train, X_val, y_val, sample_weight)
            preds = model.predict(X_val)
            residuals = (y_val - preds) ** 2
            mse = np.average(residuals, weights=sample_weight_val) if sample_weight_val is not None else np.mean(residuals)

            if mse < best_mse:
                best_mse = mse
                best_params = trial_config

        logger.info(
            "Best MSE: %.6f | Best params: %s",
            best_mse, {k: best_params.get(k) for k in keys},
        )
        self.config = best_params
        self.best_params = best_params
        return best_params
