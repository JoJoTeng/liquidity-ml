"""
Elastic-Net Return Predictor
=============================
Linear model with combined L1 + L2 regularisation.

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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler

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
        default_cfg = load_config()["models"]["elastic_net"]
        cfg = {**default_cfg, **(config or {})}
        super().__init__(name="elastic_net", config=cfg, seed=seed)
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
            alpha=self.config["alpha"],
            l1_ratio=self.config["l1_ratio"],
            max_iter=self.config["max_iter"],
            random_state=self.seed,
            warm_start=False,
        )
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        self.is_fitted = True

        # Log fit quality. train_mse is weighted to match the training
        # objective when sample_weight is set (sklearn falls back to
        # unweighted when sample_weight is None).
        train_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(
            y_train, train_pred, sample_weight=sample_weight,
        )
        n_nonzero = np.sum(np.abs(self.model.coef_) > 1e-10)
        logger.info(
            "ElasticNet fit: alpha=%.6f, l1_ratio=%.2f, "
            "train_MSE=%.6f, nonzero_coefs=%d/%d",
            self.config["alpha"],
            self.config["l1_ratio"],
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
        return pd.Series(
            coefs, index=feature_names, name="importance",
        ).sort_values(ascending=False)

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
            # ParameterSampler samples without replacement when all params
            # are given as lists, so we get n_trials distinct combinations.
            param_grid = list(
                ParameterSampler(
                    search_space, n_iter=n_trials, random_state=self.seed,
                )
            )
            logger.info(
                "Tuning ElasticNet: random %d of %d combinations",
                n_trials, full_grid_size,
            )

        best_mse = np.inf
        best_params = {}

        for params in param_grid:
            trial_config = {**self.config, **params}
            model = ElasticNetPredictor(config=trial_config, seed=self.seed)
            model.fit(X_train, y_train, sample_weight=sample_weight)
            pred_val = model.predict(X_val)

            if sample_weight_val is not None:
                mse = np.average((y_val - pred_val) ** 2, weights=sample_weight_val)
            else:
                mse = mean_squared_error(y_val, pred_val)

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
