"""
XGBoost Return Predictor
========================
Gradient boosted trees for stock return prediction.
Supports sample_weight natively for implementability-weighted training.

XGBoost passes sample_weight directly into the gradient computation,
so weighted training genuinely changes the model (not just resampling).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import BaseReturnPredictor

logger = logging.getLogger(__name__)


class XGBoostPredictor(BaseReturnPredictor):
    """XGBoost regression model with weighted training support."""

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42):
        from src.config import load_config
        default_cfg = load_config()["models"]["xgboost"]
        cfg = {**default_cfg, **(config or {})}
        super().__init__(name="xgboost", config=cfg, seed=seed)
        self.model: xgb.XGBRegressor | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> "XGBoostPredictor":
        """Train XGBoost with optional implementability weights.

        The sample_weight parameter directly enters the gradient computation,
        making the model focus on accurately predicting liquid stocks.
        sample_weight_val weights the early-stopping validation metric
        so that tuning and training optimize the same objective.
        """
        params = {
            "n_estimators": self.config["n_estimators"],
            "max_depth": self.config["max_depth"],
            "learning_rate": self.config["learning_rate"],
            "subsample": self.config["subsample"],
            "colsample_bytree": self.config["colsample_bytree"],
            "min_child_weight": self.config["min_child_weight"],
            "reg_alpha": self.config["reg_alpha"],
            "reg_lambda": self.config["reg_lambda"],
            "objective": "reg:squarederror",
            "random_state": self.seed,
            "n_jobs": -1,
            "verbosity": 0,
        }

        # Early stopping setup
        early_stopping = self.config.get("early_stopping_rounds", 20)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            params["early_stopping_rounds"] = early_stopping

        self.model = xgb.XGBRegressor(**params)

        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False
            if sample_weight_val is not None:
                fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]

        self.model.fit(X_train, y_train, **fit_kwargs)
        self.is_fitted = True

        n_trees = (
            self.model.best_iteration
            if hasattr(self.model, "best_iteration") and self.model.best_iteration
            else params["n_estimators"]
        )
        weighted_str = "weighted" if sample_weight is not None else "standard"
        logger.info(
            "XGBoost %s training: %d trees, train shape %s",
            weighted_str, n_trees, X_train.shape,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self.model.predict(X)

    def get_feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.Series:
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
        """Grid search over the configured search space.

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

        logger.info("Tuning XGBoost over %d combinations...", len(param_grid))

        best_mse = np.inf
        best_params = {}

        for params in param_grid:
            trial_config = {**self.config, **params}
            model = XGBoostPredictor(config=trial_config, seed=self.seed)
            model.fit(X_train, y_train, X_val, y_val, sample_weight, sample_weight_val)
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
