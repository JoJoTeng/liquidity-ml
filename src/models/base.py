"""
Base Model Interface
====================
Abstract base class for all ML models in the framework.
Ensures consistent API for standard and weighted training.

All 3 models (XGBoost, Random Forest, Neural Network) inherit from this.
Each must support sample_weight in .fit() for liquidity-weighted training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseReturnPredictor(ABC):
    """Abstract base class for stock return prediction models.

    All models support two training modes:
    1. Standard training (uniform weights)
    2. Implementability-weighted training (liquidity-based sample_weight)
    """

    def __init__(self, name: str, config: dict[str, Any], seed: int = 42):
        self.name = name
        self.config = config
        self.seed = seed
        self.is_fitted = False
        self.best_params: dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> "BaseReturnPredictor":
        """Train the model, optionally with sample weights.

        Parameters
        ----------
        X_train : (N, P) feature matrix
        y_train : (N,) target returns
        X_val : optional validation features (for early stopping)
        y_val : optional validation targets
        sample_weight : (N,) implementability weights (None = uniform)
        sample_weight_val : (M,) validation weights for early stopping
            metric (None = uniform). Should match the training objective.

        Returns self for chaining.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate return predictions.

        Parameters
        ----------
        X : (N, P) feature matrix

        Returns
        -------
        (N,) predicted returns.
        """
        ...

    @abstractmethod
    def get_feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.Series:
        """Return feature importance scores.

        Returns
        -------
        pd.Series indexed by feature name, values = importance scores.
        """
        ...

    def get_shap_values(
        self,
        X_test: np.ndarray,
        X_background: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Compute SHAP values for test observations.

        Parameters
        ----------
        X_test : (N, P) feature matrix to explain.
        X_background : (M, P) background data for non-tree explainers.
            Required for neural networks. Ignored by TreeExplainer.
        feature_names : Optional feature names for column labels.
        config : SHAP config dict (from config["shap"]).

        Returns
        -------
        pd.DataFrame of shape (N, P) with SHAP values per observation
        per feature. Columns = feature names.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_shap_values(). "
            "Override in subclass."
        )

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Hyperparameter search (override in subclass if needed).

        Default: return current config (no tuning).
        """
        return self.config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={self.is_fitted})"
