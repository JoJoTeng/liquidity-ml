"""
Elastic-Net Return Predictor
=============================
Linear model with combined L1 + L2 regularisation.
Replaces Random Forest in the LiquidityML v3 model set.

Supports sample_weight natively via sklearn's ElasticNet.
Hyperparameter tuning: grid/random search (same pattern as XGBoost).
Feature importance: permutation importance (no native gain measure).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.inspection import permutation_importance as sklearn_perm_importance
from sklearn.metrics import mean_squared_error

from src.models.base import BaseReturnPredictor

logger = logging.getLogger(__name__)


class ElasticNetPredictor(BaseReturnPredictor):
    """Elastic-Net regression for stock return prediction.

    The model is a simple penalised linear regression:
        min_beta  (1/2N) sum w_i (r_i - x_i'beta)^2
                  + alpha * (rho ||beta||_1 + (1-rho)/2 ||beta||_2^2)

    where alpha controls total penalty strength and rho (l1_ratio) controls
    the L1/L2 mix (rho=1 -> Lasso, rho=0 -> Ridge).

    Parameters
    ----------
    config : dict with keys 'alpha', 'l1_ratio', 'max_iter', 'search_space'.
    seed : random seed for reproducibility.
    """

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42):
        from src.config import load_config

        full_cfg = load_config()
        model_cfg = full_cfg.get("models", {}).get("elastic_net", {})
        if config:
            model_cfg.update(config)

        super().__init__(name="elastic_net", config=model_cfg, seed=seed)

        self.alpha = model_cfg.get("alpha", 0.001)
        self.l1_ratio = model_cfg.get("l1_ratio", 0.5)
        self.max_iter = model_cfg.get("max_iter", 10000)

        self.model: ElasticNet | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> "ElasticNetPredictor":
        """Train the Elastic-Net model.

        sample_weight is passed directly to sklearn's .fit() method,
        which supports it natively for weighted least squares.
        """
        self.model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=self.seed,
            warm_start=False,
        )
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        self.is_fitted = True

        # Log fit quality
        train_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)
        n_nonzero = np.sum(np.abs(self.model.coef_) > 1e-10)
        logger.info(
            "ElasticNet fit: alpha=%.4f, l1_ratio=%.2f, "
            "train_MSE=%.6f, nonzero_coefs=%d/%d",
            self.alpha,
            self.l1_ratio,
            train_mse,
            n_nonzero,
            len(self.model.coef_),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate return predictions."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self.model.predict(X)

    def get_feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.Series:
        """Return absolute coefficient values as importance proxy."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted.")
        coefs = np.abs(self.model.coef_)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(coefs))]
        return pd.Series(coefs, index=feature_names, name="importance")

    def get_permutation_importance(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str] | None = None,
        n_repeats: int = 10,
        seed: int | None = None,
    ) -> pd.Series:
        """Compute permutation importance on test data."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted.")
        if seed is None:
            seed = self.seed

        result = sklearn_perm_importance(
            self.model,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=seed,
            scoring="neg_mean_squared_error",
        )
        importances = result.importances_mean
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]
        return pd.Series(importances, index=feature_names, name="perm_importance")

    def get_shap_values(
        self,
        X_test: np.ndarray,
        X_background: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Compute SHAP values using LinearExplainer (exact for linear models).

        Parameters
        ----------
        X_test : (N, P) feature matrix to explain.
        X_background : (M, P) background data for the masker.
        feature_names : optional feature names for column labels.

        Returns
        -------
        pd.DataFrame of shape (N, P) with SHAP values.
        """
        import shap

        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted.")
        if X_background is None:
            raise ValueError("X_background is required for LinearExplainer.")

        masker = shap.maskers.Independent(X_background)
        explainer = shap.LinearExplainer(self.model, masker)
        sv = explainer.shap_values(X_test)

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_test.shape[1])]

        return pd.DataFrame(sv, columns=feature_names)

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Grid/random search over the configured search space."""
        search_space = self.config.get("search_space", {})
        if not search_space:
            logger.info("No search space configured; using defaults.")
            return self.config

        n_trials = self.config.get("n_random_search", 50)
        keys = list(search_space.keys())
        values = list(search_space.values())

        # Full grid size
        full_grid_size = 1
        for v in values:
            full_grid_size *= len(v)

        # Build parameter grid
        if full_grid_size <= n_trials:
            param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
            logger.info(
                "Tuning ElasticNet: full grid %d combinations (<= %d)",
                full_grid_size, n_trials,
            )
        else:
            rng = np.random.RandomState(self.seed)
            param_grid = []
            for _ in range(n_trials):
                combo = {k: rng.choice(v) for k, v in zip(keys, values)}
                param_grid.append(combo)
            logger.info(
                "Tuning ElasticNet: random %d of %d combinations",
                n_trials, full_grid_size,
            )

        best_mse = float("inf")
        best_params = {}

        for params in param_grid:
            model = ElasticNet(
                alpha=params.get("alpha", self.alpha),
                l1_ratio=params.get("l1_ratio", self.l1_ratio),
                max_iter=self.max_iter,
                random_state=self.seed,
            )
            model.fit(X_train, y_train, sample_weight=sample_weight)
            pred_val = model.predict(X_val)

            if sample_weight_val is not None:
                mse = np.average((y_val - pred_val) ** 2, weights=sample_weight_val)
            else:
                mse = mean_squared_error(y_val, pred_val)

            if mse < best_mse:
                best_mse = mse
                best_params = params.copy()

        # Apply best params
        if "alpha" in best_params:
            self.alpha = best_params["alpha"]
        if "l1_ratio" in best_params:
            self.l1_ratio = best_params["l1_ratio"]
        self.best_params = best_params

        logger.info(
            "ElasticNet tuned: alpha=%.6f, l1_ratio=%.3f (val_MSE=%.6f)",
            self.alpha,
            self.l1_ratio,
            best_mse,
        )
        return best_params
