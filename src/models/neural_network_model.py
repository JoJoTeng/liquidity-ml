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


def _make_epoch_loss_logger(tf, logger: logging.Logger, max_epochs: int, frequency: int):
    """Create a lazy TensorFlow callback that logs train/validation loss."""

    class _EpochLossLogger(tf.keras.callbacks.Callback):
        def __init__(self, logger: logging.Logger, max_epochs: int, frequency: int):
            super().__init__()
            self.logger = logger
            self.max_epochs = max_epochs
            self.frequency = max(1, frequency)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            epoch_num = epoch + 1
            if epoch_num % self.frequency != 0 and epoch_num != self.max_epochs:
                return

            train_loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

            self.logger.info(
                "    epoch %03d/%03d: train_loss=%.8f val_loss=%.8f lr=%.6g",
                epoch_num,
                self.max_epochs,
                train_loss if train_loss is not None else np.nan,
                val_loss if val_loss is not None else np.nan,
                lr,
            )

    return _EpochLossLogger(logger, max_epochs, frequency)


def _build_model(
    input_dim: int,
    hidden_layers: list[int],
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = True,
    l1_penalty: float = 0.0,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    loss: str = "mse",
    huber_delta: float = 0.1,
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

    loss_name = str(loss).lower()
    if loss_name in {"mse", "mean_squared_error"}:
        loss_obj = "mse"
    elif loss_name == "huber":
        loss_obj = tf.keras.losses.Huber(delta=float(huber_delta))
    else:
        raise ValueError(f"Unsupported neural-network loss: {loss!r}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(**optimizer_kwargs),
        loss=loss_obj,
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
    )
    return model


class NeuralNetPredictor(BaseReturnPredictor):
    """Feedforward neural network (TensorFlow/Keras) with optional robust loss.

    Architecture (Gu, Kelly, Xiu 2020 NN3):
        Input -> [Dense -> BatchNorm -> ReLU] x len(hidden_layers) -> Dense(1)
    Loss:
        MSE or Huber prediction loss, optionally sample-weighted, plus
        lambda_L1 * ||W||_1 when configured.
    Trained with Adam; early stopping monitors validation loss with
    patience 'patience' and restores best weights. An ensemble of
    n_ensemble_seeds independent initialisations is averaged at predict
    time when n_ensemble_seeds > 1.

    Parameters
    ----------
    config : dict with keys 'hidden_layers', 'activation', 'batch_norm',
        'dropout', 'batch_size', 'epochs', 'patience', 'learning_rate',
        'loss', 'huber_delta', 'l1_penalty', 'weight_decay',
        'n_ensemble_seeds', 'search_space'.
    seed : random seed for reproducibility.
    """

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42):
        from src.config import load_config

        default_cfg = load_config()["models"]["neural_network"]
        cfg = {**default_cfg, **(config or {})}
        super().__init__(name="neural_network", config=cfg, seed=seed)
        self.model = None  # tf.keras.Sequential
        self.fit_diagnostics: dict[str, Any] = {}
        self.tuning_diagnostics: list[dict[str, Any]] = []

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

        Uses early stopping and, when configured, ReduceLROnPlateau scheduling.
        """
        import tensorflow as tf

        cfg = self.config
        n_ensemble = cfg.get("n_ensemble_seeds", 1)

        self._ensemble_models = []
        member_diagnostics = []

        for seed_i in range(n_ensemble):
            model_seed = self.seed + seed_i
            tf.keras.utils.set_random_seed(model_seed)

            model_i, diagnostics_i = self._train_single(
                X_train, y_train, X_val, y_val,
                sample_weight, sample_weight_val, cfg, model_seed,
            )
            self._ensemble_models.append(model_i)
            member_diagnostics.append(diagnostics_i)

            if n_ensemble > 1:
                logger.info(
                    "  Ensemble member %d/%d: val_loss=%.6f, epochs=%d",
                    seed_i + 1,
                    n_ensemble,
                    diagnostics_i["best_val_loss"],
                    diagnostics_i["epochs_run"],
                )

        self.model = self._ensemble_models[0]
        self.is_fitted = True
        self.fit_diagnostics = self._summarize_fit_diagnostics(
            member_diagnostics,
            cfg,
            weighted=sample_weight is not None,
            n_train=len(X_train),
            n_val=len(X_val) if X_val is not None else 0,
        )

        weighted_str = "weighted" if sample_weight is not None else "standard"
        logger.info(
            "NeuralNet %s training: %d ensemble members, hidden=%s, "
            "mean_val_loss=%.6f, mean_epochs=%.1f, shape %s",
            weighted_str,
            n_ensemble,
            cfg["hidden_layers"],
            self.fit_diagnostics["best_val_loss_mean"],
            self.fit_diagnostics["epochs_run_mean"],
            X_train.shape,
        )
        return self

    @staticmethod
    def _summarize_fit_diagnostics(
        member_diagnostics: list[dict[str, Any]],
        cfg: dict,
        weighted: bool,
        n_train: int,
        n_val: int,
    ) -> dict[str, Any]:
        """Aggregate per-seed Keras history summaries for saved diagnostics."""
        if not member_diagnostics:
            return {}

        def _mean(key: str) -> float:
            vals = [d[key] for d in member_diagnostics if pd.notna(d.get(key))]
            return float(np.mean(vals)) if vals else np.nan

        def _min(key: str) -> float:
            vals = [d[key] for d in member_diagnostics if pd.notna(d.get(key))]
            return float(np.min(vals)) if vals else np.nan

        def _max(key: str) -> float:
            vals = [d[key] for d in member_diagnostics if pd.notna(d.get(key))]
            return float(np.max(vals)) if vals else np.nan

        return {
            "weighted": bool(weighted),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "hidden_layers": str(cfg.get("hidden_layers")),
            "activation": cfg.get("activation", "relu"),
            "batch_norm": bool(cfg.get("batch_norm", True)),
            "dropout": float(cfg.get("dropout", 0.0)),
            "batch_size": int(cfg["batch_size"]),
            "epochs_config": int(cfg["epochs"]),
            "patience_config": int(cfg.get("patience", 5)),
            "learning_rate": float(cfg.get("learning_rate", 0.001)),
            "loss": cfg.get("loss", "mse"),
            "huber_delta": float(cfg.get("huber_delta", 0.1)),
            "reduce_lr_on_plateau": bool(cfg.get("reduce_lr_on_plateau", True)),
            "reduce_lr_factor": float(cfg.get("reduce_lr_factor", 0.5)),
            "reduce_lr_patience": int(cfg.get("reduce_lr_patience", 3)),
            "min_learning_rate": float(cfg.get("min_learning_rate", 1e-6)),
            "l1_penalty": float(cfg.get("l1_penalty", 0.0)),
            "weight_decay": float(cfg.get("weight_decay", 0.0)),
            "n_ensemble_seeds": int(cfg.get("n_ensemble_seeds", 1)),
            "epochs_run_mean": _mean("epochs_run"),
            "epochs_run_min": _min("epochs_run"),
            "epochs_run_max": _max("epochs_run"),
            "best_epoch_mean": _mean("best_epoch"),
            "best_epoch_min": _min("best_epoch"),
            "best_epoch_max": _max("best_epoch"),
            "best_train_mse_mean": _mean("best_train_mse"),
            "best_val_mse_mean": _mean("best_val_mse"),
            "best_train_loss_mean": _mean("best_train_loss"),
            "best_val_loss_mean": _mean("best_val_loss"),
            "final_train_mse_mean": _mean("final_train_mse"),
            "final_val_mse_mean": _mean("final_val_mse"),
            "final_train_loss_mean": _mean("final_train_loss"),
            "final_val_loss_mean": _mean("final_val_loss"),
            "stopped_early_any": any(
                bool(d.get("stopped_early", False)) for d in member_diagnostics
            ),
            "hit_epoch_cap_any": any(
                bool(d.get("hit_epoch_cap", False)) for d in member_diagnostics
            ),
        }

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
        """Train a single network and summarize its Keras loss history."""
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)

        input_dim = X_train.shape[1]
        batch_size = cfg["batch_size"]
        epochs = cfg["epochs"]
        patience = cfg.get("patience", 5)
        log_epoch_metrics = bool(cfg.get("log_epoch_metrics", False))
        epoch_log_frequency = int(cfg.get("epoch_log_frequency", 1))
        has_val = X_val is not None and y_val is not None

        model = _build_model(
            input_dim=input_dim,
            hidden_layers=cfg["hidden_layers"],
            activation=cfg.get("activation", "relu"),
            dropout=cfg.get("dropout", 0.0),
            batch_norm=cfg.get("batch_norm", True),
            l1_penalty=cfg.get("l1_penalty", 0.0),
            learning_rate=cfg.get("learning_rate", 0.001),
            weight_decay=cfg.get("weight_decay", 0.0),
            loss=cfg.get("loss", "mse"),
            huber_delta=cfg.get("huber_delta", 0.1),
        )

        # Callbacks
        callbacks = []
        if has_val and bool(cfg.get("reduce_lr_on_plateau", True)):
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=float(cfg.get("reduce_lr_factor", 0.5)),
                    patience=int(cfg.get("reduce_lr_patience", 3)),
                    min_lr=float(cfg.get("min_learning_rate", 1e-6)),
                )
            )

        if log_epoch_metrics:
            callbacks.append(
                _make_epoch_loss_logger(
                    tf,
                    logger=logger,
                    max_epochs=epochs,
                    frequency=epoch_log_frequency,
                )
            )

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

        train_loss = np.asarray(history.history["loss"], dtype=float)
        train_mse = np.asarray(history.history.get("mse", train_loss), dtype=float)
        val_loss = (
            np.asarray(history.history["val_loss"], dtype=float)
            if has_val and "val_loss" in history.history
            else None
        )
        val_mse = (
            np.asarray(history.history["val_mse"], dtype=float)
            if has_val and "val_mse" in history.history
            else val_loss
        )
        n_epochs_run = len(train_loss)

        if val_loss is not None and len(val_loss) > 0:
            best_idx = int(np.argmin(val_loss))
            best_val_loss = float(val_loss[best_idx])
            final_val_loss = float(val_loss[-1])
        else:
            best_idx = int(np.argmin(train_loss))
            best_val_loss = np.nan
            final_val_loss = np.nan

        diagnostics = {
            "seed": int(seed),
            "epochs_run": int(n_epochs_run),
            "best_epoch": int(best_idx + 1),
            "best_train_mse": float(train_mse[best_idx]),
            "best_val_mse": (
                float(val_mse[best_idx])
                if val_mse is not None and len(val_mse) > 0
                else np.nan
            ),
            "best_train_loss": float(train_loss[best_idx]),
            "best_val_loss": best_val_loss,
            "final_train_mse": float(train_mse[-1]),
            "final_val_mse": (
                float(val_mse[-1])
                if val_mse is not None and len(val_mse) > 0
                else np.nan
            ),
            "final_train_loss": float(train_loss[-1]),
            "final_val_loss": final_val_loss,
            "stopped_early": bool(has_val and n_epochs_run < epochs),
            "hit_epoch_cap": bool(n_epochs_run >= epochs),
        }

        return model, diagnostics

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
        self.tuning_diagnostics = []

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
            trial_diagnostics = {
                "candidate": i + 1,
                "n_candidates": len(param_grid),
                "selection_mse": float(mse),
                **{k: params[k] for k in keys},
                **model.fit_diagnostics,
            }
            self.tuning_diagnostics.append(trial_diagnostics)

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
        # The tuning loop forced n_ensemble_seeds=tune_seeds for speed; restore
        # the configured deployment ensemble size so the final fit and downstream
        # rolling refits use the intended seed count.
        best_params["n_ensemble_seeds"] = self.config.get("n_ensemble_seeds", 1)
        self.config = best_params
        self.best_params = best_params
        return best_params
