"""
Feedforward Neural Network Return Predictor (TensorFlow/Keras)
===============================================================
Multi-layer feedforward network for stock return prediction.
Implements weighted MSE loss for implementability-weighted training.

Architecture follows Gu, Kelly, Xiu (2020) NN3:
  Input -> [Dense -> BatchNorm -> ReLU] x 3 -> Output
  Hidden layers: 32, 16, 8 (geometric pyramid rule)

Regularisation stack (GKX 2020):
  1. L1 penalty on weights (sparsity)
  2. Learning rate shrinkage via Adam
  3. Early stopping (halt when validation error rises)
  4. Batch normalisation (stabilise activations)

Key implementation for weighted training:
  Standard MSE: L = (1/N) sum (y_i - y_hat_i)^2
  Weighted MSE: L = (1/N) sum w_i * (y_i - y_hat_i)^2
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.models.base import BaseReturnPredictor

logger = logging.getLogger(__name__)

# Suppress TF logs unless explicitly wanted
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _build_model(
    input_dim: int,
    hidden_layers: list[int],
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = True,
    l1_penalty: float = 0.0,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
):
    """Build a Keras Sequential feedforward network.

    Architecture per hidden layer (GKX 2020 default):
        Dense -> BatchNormalization -> activation [-> Dropout if dropout > 0]

    weight_decay is passed to Adam as decoupled L2 weight decay (AdamW
    style). Requires TF >= 2.11; if 0.0 (the project default) the
    optimizer behaves as plain Adam.
    """
    import tensorflow as tf

    regularizer = tf.keras.regularizers.l1(l1_penalty) if l1_penalty > 0 else None

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(
            units,
            kernel_regularizer=regularizer,
            use_bias=True,
        ))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))

    # Output: single scalar prediction (no activation)
    model.add(tf.keras.layers.Dense(1, kernel_regularizer=regularizer))

    optimizer_kwargs = {"learning_rate": learning_rate}
    if weight_decay > 0:
        optimizer_kwargs["weight_decay"] = weight_decay
    model.compile(
        optimizer=tf.keras.optimizers.Adam(**optimizer_kwargs),
        loss="mse",
    )
    return model


class NeuralNetPredictor(BaseReturnPredictor):
    """Feedforward neural network (TensorFlow/Keras) with weighted MSE.

    Architecture (Gu, Kelly, Xiu 2020 NN3):
        Input -> [Dense -> BatchNorm -> ReLU] x len(hidden_layers) -> Dense(1)
    Loss:
        L = (1/N) sum w_i (r_i - r-hat_i)^2 + lambda_L1 * ||W||_1
    Trained with Adam; early stopping monitors weighted val MSE with
    patience 'patience' and restores best weights. An ensemble of
    n_ensemble_seeds independent initialisations is averaged at predict
    time when n_ensemble_seeds > 1.

    Parameters
    ----------
    config : dict with keys 'hidden_layers', 'activation', 'batch_norm',
        'dropout', 'batch_size', 'epochs', 'patience', 'learning_rate',
        'l1_penalty', 'weight_decay', 'n_ensemble_seeds', 'search_space'.
    seed : random seed for reproducibility.
    """

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42):
        from src.config import load_config

        default_cfg = load_config()["models"]["neural_network"]
        cfg = {**default_cfg, **(config or {})}
        super().__init__(name="neural_network", config=cfg, seed=seed)
        self.model = None  # tf.keras.Sequential

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

        Uses early stopping + ReduceLROnPlateau scheduling.
        """
        import tensorflow as tf

        cfg = self.config
        n_ensemble = cfg.get("n_ensemble_seeds", 1)

        self._ensemble_models = []
        all_val_losses = []

        for seed_i in range(n_ensemble):
            model_seed = self.seed + seed_i
            tf.keras.utils.set_random_seed(model_seed)

            model_i, val_loss_i, n_epochs_i = self._train_single(
                X_train, y_train, X_val, y_val,
                sample_weight, sample_weight_val, cfg, model_seed,
            )
            self._ensemble_models.append(model_i)
            all_val_losses.append(val_loss_i)

            if n_ensemble > 1:
                logger.info(
                    "  Ensemble member %d/%d: val_loss=%.6f, epochs=%d",
                    seed_i + 1, n_ensemble, val_loss_i, n_epochs_i,
                )

        self.model = self._ensemble_models[0]
        self.is_fitted = True

        weighted_str = "weighted" if sample_weight is not None else "standard"
        logger.info(
            "NeuralNet %s training: %d ensemble members, hidden=%s, "
            "mean_val_loss=%.6f, shape %s",
            weighted_str,
            n_ensemble,
            cfg["hidden_layers"],
            np.mean(all_val_losses),
            X_train.shape,
        )
        return self

    def _train_single(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        sample_weight: np.ndarray | None,
        sample_weight_val: np.ndarray | None,
        cfg: dict,
        seed: int,
    ) -> tuple:
        """Train a single network. Returns (model, best_val_loss, n_epochs)."""
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)

        input_dim = X_train.shape[1]
        batch_size = cfg["batch_size"]
        epochs = cfg["epochs"]
        patience = cfg.get("patience", 5)

        model = _build_model(
            input_dim=input_dim,
            hidden_layers=cfg["hidden_layers"],
            activation=cfg.get("activation", "relu"),
            dropout=cfg.get("dropout", 0.0),
            batch_norm=cfg.get("batch_norm", True),
            l1_penalty=cfg.get("l1_penalty", 0.0),
            learning_rate=cfg.get("learning_rate", 0.001),
            weight_decay=cfg.get("weight_decay", 0.0),
        )

        # Callbacks
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6,
            ),
        ]

        has_val = X_val is not None and y_val is not None
        if has_val:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                )
            )
            validation_data = (X_val.astype(np.float32), y_val.astype(np.float32))
            if sample_weight_val is not None:
                validation_data = (
                    X_val.astype(np.float32),
                    y_val.astype(np.float32),
                    sample_weight_val.astype(np.float32),
                )
        else:
            validation_data = None

        history = model.fit(
            X_train.astype(np.float32),
            y_train.astype(np.float32),
            sample_weight=sample_weight.astype(np.float32) if sample_weight is not None else None,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=min(batch_size, len(X_train)),
            callbacks=callbacks,
            verbose=0,
        )

        n_epochs_run = len(history.history["loss"])
        best_val_loss = (
            min(history.history["val_loss"]) if has_val else history.history["loss"][-1]
        )

        return model, best_val_loss, n_epochs_run

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble average."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        ensemble = getattr(self, "_ensemble_models", [self.model])
        all_preds = []
        for m in ensemble:
            p = m.predict(X.astype(np.float32), verbose=0).flatten()
            all_preds.append(p)

        return np.mean(all_preds, axis=0)

    def get_permutation_importance(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str] | None = None,
        n_repeats: int = 10,
        seed: int | None = None,
    ) -> pd.Series:
        """Compute permutation importance for the ensemble's predictions.

        Wraps NeuralNetPredictor.predict() (which averages over all
        ensemble members) as the sklearn estimator, so the resulting
        importance reflects the deployed prediction function rather than
        a single ensemble member. With n_ensemble_seeds=1 the ensemble
        is one model and behavior is identical to single-model
        permutation importance.

        Permutation importance is non-linear in the model (MSE is
        quadratic in predictions), so per-member-then-average would
        give the wrong semantic. The single-call-on-ensemble-mean form
        used here is correct.
        """
        from sklearn.base import BaseEstimator
        from sklearn.inspection import permutation_importance as sklearn_perm_importance

        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        if seed is None:
            seed = self.seed

        # Wrap for sklearn API. Inheriting from BaseEstimator gives us
        # __sklearn_tags__ (required by sklearn >=1.6) plus default
        # implementations of get_params/set_params; we override the
        # latter two as no-ops because we don't expose tunable params.
        # The wrapper delegates predict() to the outer NeuralNetPredictor,
        # which averages over self._ensemble_models.
        class _SklearnWrapper(BaseEstimator):
            _estimator_type = "regressor"

            def __init__(self, predictor=None):
                self._predictor = predictor

            def fit(self, X, y, **kwargs):
                return self  # no-op: model is already trained

            def predict(self, X):
                return self._predictor.predict(X)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        wrapper = _SklearnWrapper(self)
        result = sklearn_perm_importance(
            wrapper,
            X_test.astype(np.float32),
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
        """Compute SHAP values using DeepExplainer with optional Kernel fallback.

        If ``config["max_test_samples"]`` is set, only the sampled test rows are
        returned. This keeps downstream mean(|SHAP|) aggregates from being
        diluted by zero-padded unexplained observations.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        if X_background is None:
            raise ValueError(
                "X_background is required for NeuralNet SHAP. "
                "Pass a subsample of training data."
            )

        import shap

        cfg = config or {}
        n_bg = cfg.get("background_samples", 100)
        max_test = cfg.get("max_test_samples", None)
        use_kernel = cfg.get("use_kernel_fallback", True)

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

        # Compute SHAP values per ensemble member, then average to match
        # predict()'s ensemble-mean. SHAP is linear in the model
        # (Lundberg & Lee 2017, additivity axiom): SHAP(mean of f_k)
        # equals mean of SHAP(f_k). With n_ensemble_seeds=1 the loop
        # runs once and the result is identical to single-model SHAP.
        ensemble = getattr(self, "_ensemble_models", [self.model])

        per_member_sv = []
        for m in ensemble:
            sv = None
            try:
                explainer = shap.DeepExplainer(m, X_bg)
                sv = explainer.shap_values(X_explain)
            except Exception as e:
                if use_kernel:
                    logger.info("DeepExplainer failed (%s), using KernelExplainer", e)

                    def predict_fn(x, model=m):  # default-arg binds current m
                        return model.predict(
                            x.astype(np.float32), verbose=0,
                        ).flatten()

                    explainer = shap.KernelExplainer(predict_fn, X_bg)
                    sv = explainer.shap_values(X_explain, nsamples=100)
                else:
                    raise RuntimeError(
                        f"DeepExplainer failed and KernelExplainer fallback is disabled: {e}"
                    )

            # Normalise SHAP shape per member
            if isinstance(sv, list):
                sv = sv[0]
            sv = np.asarray(sv)
            # TF DeepExplainer in shap >=0.44 returns shape (N, P, 1) for
            # single-output regression; squeeze the trailing dim.
            if sv.ndim == 3 and sv.shape[-1] == 1:
                sv = sv.squeeze(-1)

            per_member_sv.append(sv)

        sv = np.mean(per_member_sv, axis=0)

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_test.shape[1])]

        index = explain_idx if explain_idx is not None else None
        return pd.DataFrame(sv, columns=feature_names, index=index)

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Grid search over configured search space using validation MSE."""
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

        # tune_n_ensemble_seeds controls how many ensemble seeds are used
        # for each candidate during the search (defaults to 1 for speed).
        # Set this to match n_ensemble_seeds if you want tuning to score
        # candidates the same way they will be deployed.
        tune_seeds = self.config.get("tune_n_ensemble_seeds", 1)

        for i, params in enumerate(param_grid):
            trial_config = {
                **self.config, **params, "n_ensemble_seeds": tune_seeds,
            }
            model = NeuralNetPredictor(config=trial_config, seed=self.seed)
            model.fit(X_train, y_train, X_val, y_val, sample_weight, sample_weight_val)
            preds = model.predict(X_val)
            mse = mean_squared_error(
                y_val, preds, sample_weight=sample_weight_val,
            )

            logger.info(
                "  Combo %d/%d: %s -> MSE=%.6f",
                i + 1, len(param_grid),
                {k: params[k] for k in keys}, mse,
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
