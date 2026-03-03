"""
Feedforward Neural Network Return Predictor (TensorFlow/Keras)
===============================================================
Multi-layer feedforward network for stock return prediction.
Implements weighted MSE loss for implementability-weighted training.

Architecture follows Gu, Kelly, Xiu (2020):
  Input → [Dense → BatchNorm → ReLU → Dropout] × L → Output

Key implementation for weighted training:
  Standard MSE: L = (1/N) Σ (y_i - ŷ_i)²
  Weighted MSE: L = (1/N) Σ w_i · (y_i - ŷ_i)²

  where w_i = liquidity weight (normalized to mean 1).
  Keras natively supports this via model.fit(sample_weight=...).

TensorFlow/Keras is used for:
  - Native sample_weight support in model.fit()
  - Built-in callbacks (EarlyStopping, ReduceLROnPlateau)
  - BatchNormalization layer
  - Easy GPU acceleration
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.models.base import BaseReturnPredictor

# Suppress TF info/warning logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import tensorflow as tf


def _build_model(
    input_dim: int,
    hidden_layers: list[int],
    dropout: float = 0.5,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
) -> "tf.keras.Model":
    """Build a Keras Sequential feedforward network.

    Architecture per hidden layer:
        Dense(L2) → BatchNormalization → ReLU → Dropout
    """
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(
            tf.keras.layers.Dense(
                units,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(dropout))

    # Output: single scalar prediction
    model.add(tf.keras.layers.Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


class NeuralNetPredictor(BaseReturnPredictor):
    """Feedforward neural network with weighted MSE training (TensorFlow)."""

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42):
        from src.config import load_config

        default_cfg = load_config()["models"]["neural_network"]
        cfg = {**default_cfg, **(config or {})}
        super().__init__(name="neural_network", config=cfg, seed=seed)
        self.model = None  # tf.keras.Model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> "NeuralNetPredictor":
        """Train feedforward NN with optional implementability weights.

        Keras natively supports sample_weight in model.fit():
            loss_i = w_i * (y_i - ŷ_i)²

        When sample_weight_val is provided, the validation loss (used by
        EarlyStopping) is also weighted, ensuring training and early
        stopping optimize the same objective.

        Uses EarlyStopping + ReduceLROnPlateau callbacks.
        """
        import tensorflow as tf

        # Reproducibility
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        cfg = self.config
        input_dim = X_train.shape[1]

        # Build model
        self.model = _build_model(
            input_dim=input_dim,
            hidden_layers=cfg["hidden_layers"],
            dropout=cfg["dropout"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg.get("weight_decay", 1e-4),
        )

        # Callbacks
        callbacks = []

        if X_val is not None and y_val is not None:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=cfg.get("patience", 10),
                    restore_best_weights=True,
                    verbose=0,
                )
            )
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0,
                )
            )
            # Keras accepts (X, y, sample_weight) 3-tuple for weighted val_loss
            if sample_weight_val is not None:
                validation_data = (X_val, y_val, sample_weight_val)
            else:
                validation_data = (X_val, y_val)
        else:
            validation_data = None

        # Train
        history = self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            validation_data=validation_data,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )

        self.is_fitted = True

        # Log summary
        n_epochs = len(history.history["loss"])
        weighted_str = "weighted" if sample_weight is not None else "standard"
        final_train_loss = history.history["loss"][-1]
        val_str = ""
        if validation_data is not None and "val_loss" in history.history:
            val_str = f", val_loss={history.history['val_loss'][-1]:.6f}"

        logger.info(
            "NeuralNet %s training: %d epochs, hidden=%s, "
            "train_loss=%.6f%s, shape %s",
            weighted_str,
            n_epochs,
            cfg["hidden_layers"],
            final_train_loss,
            val_str,
            X_train.shape,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        preds = self.model.predict(X, verbose=0)
        return preds.squeeze(-1)

    def get_feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.Series:
        """First-layer weight magnitude as a fast importance proxy.

        For full SHAP analysis, use src/analysis/feature_importance.py.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        # Find first Dense layer with weights
        first_dense = None
        for layer in self.model.layers:
            if hasattr(layer, "kernel"):
                first_dense = layer
                break

        if first_dense is None:
            raise RuntimeError("No Dense layer found.")

        # Sum absolute weights across output neurons → (input_dim,)
        weights = first_dense.kernel.numpy()  # shape: (input_dim, units)
        importance = np.abs(weights).sum(axis=1)

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
        """Compute SHAP values using DeepExplainer (fallback: KernelExplainer).

        Requires X_background (subsample of training data) as the reference
        distribution for the SHAP explainer.
        """
        import shap

        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        if X_background is None:
            raise ValueError(
                "X_background is required for NeuralNet SHAP. "
                "Pass a subsample of training data."
            )

        cfg = config or {}
        n_bg = cfg.get("background_samples", 100)
        max_test = cfg.get("max_test_samples", None)

        # Subsample background
        if len(X_background) > n_bg:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(len(X_background), n_bg, replace=False)
            X_bg = X_background[idx].astype(np.float32)
        else:
            X_bg = X_background.astype(np.float32)

        # Optionally cap test samples for speed
        explain_idx = None
        if max_test is not None and len(X_test) > max_test:
            rng = np.random.RandomState(self.seed)
            explain_idx = rng.choice(len(X_test), max_test, replace=False)
            X_explain = X_test[explain_idx].astype(np.float32)
        else:
            X_explain = X_test.astype(np.float32)

        # Try DeepExplainer first, fall back to KernelExplainer
        try:
            explainer = shap.DeepExplainer(self.model, X_bg)
            sv = explainer.shap_values(X_explain)
        except Exception:
            logger.info("DeepExplainer failed, falling back to KernelExplainer")
            predict_fn = lambda x: self.model.predict(  # noqa: E731
                x.astype(np.float32), verbose=0
            ).flatten()
            explainer = shap.KernelExplainer(predict_fn, X_bg)
            sv = explainer.shap_values(X_explain, nsamples=100)

        if isinstance(sv, list):
            sv = sv[0]

        # Pad back to full test size if subsampled
        if explain_idx is not None:
            full_sv = np.zeros((len(X_test), X_test.shape[1]))
            full_sv[explain_idx] = sv
            sv = full_sv

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

        logger.info("Tuning NeuralNet over %d combinations...", len(param_grid))

        best_mse = np.inf
        best_params = {}

        for params in param_grid:
            trial_config = {**self.config, **params}
            model = NeuralNetPredictor(config=trial_config, seed=self.seed)
            model.fit(X_train, y_train, X_val, y_val, sample_weight, sample_weight_val)
            preds = model.predict(X_val)
            residuals = (y_val - preds) ** 2
            mse = (
                np.average(residuals, weights=sample_weight_val)
                if sample_weight_val is not None
                else np.mean(residuals)
            )

            if mse < best_mse:
                best_mse = mse
                best_params = trial_config

        logger.info(
            "Best MSE: %.6f | Best params: %s",
            best_mse,
            {k: best_params.get(k) for k in keys},
        )
        self.config = best_params
        self.best_params = best_params
        return best_params
