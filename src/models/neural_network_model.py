"""
Feedforward Neural Network Return Predictor (PyTorch)
=====================================================
Multi-layer feedforward network for stock return prediction.
Implements weighted MSE loss for implementability-weighted training.

Architecture follows Gu, Kelly, Xiu (2020):
  Input → [Linear → BatchNorm → ReLU → Dropout] × L → Output

Key implementation for weighted training:
  Standard MSE: L = (1/N) Σ (y_i - ŷ_i)²
  Weighted MSE: L = (1/N) Σ w_i · (y_i - ŷ_i)²

  where w_i = liquidity weight (normalized to mean 1).
"""

from __future__ import annotations

import copy
import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseReturnPredictor

logger = logging.getLogger(__name__)


def _build_model(
    input_dim: int,
    hidden_layers: list[int],
    dropout: float = 0.5,
    weight_decay: float = 1e-4,
) -> nn.Sequential:
    """Build a PyTorch Sequential feedforward network.

    Architecture per hidden layer:
        Linear → BatchNorm → ReLU → Dropout
    """
    layers: list[nn.Module] = []
    in_features = input_dim

    for units in hidden_layers:
        layers.append(nn.Linear(in_features, units))
        layers.append(nn.BatchNorm1d(units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_features = units

    # Output: single scalar prediction
    layers.append(nn.Linear(in_features, 1))

    return nn.Sequential(*layers)


class NeuralNetPredictor(BaseReturnPredictor):
    """Feedforward neural network with weighted MSE training (PyTorch)."""

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42):
        from src.config import load_config

        default_cfg = load_config()["models"]["neural_network"]
        cfg = {**default_cfg, **(config or {})}
        super().__init__(name="neural_network", config=cfg, seed=seed)
        self.model: nn.Sequential | None = None
        self.device = torch.device("cpu")

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

        Weighted MSE: loss_i = w_i * (y_i - ŷ_i)²

        When sample_weight_val is provided, the validation loss is also
        weighted, ensuring training and early stopping optimize the same
        objective.

        Uses early stopping + ReduceLROnPlateau scheduling.
        """
        # Reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        cfg = self.config
        input_dim = X_train.shape[1]
        batch_size = cfg["batch_size"]
        epochs = cfg["epochs"]
        patience = cfg.get("patience", 10)
        lr_patience = 5

        # Build model
        self.model = _build_model(
            input_dim=input_dim,
            hidden_layers=cfg["hidden_layers"],
            dropout=cfg["dropout"],
            weight_decay=cfg.get("weight_decay", 1e-4),
        ).to(self.device)

        # Optimizer with L2 regularization (weight_decay)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=1e-6
        )

        # Prepare tensors
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        w_t = (
            torch.tensor(sample_weight, dtype=torch.float32, device=self.device)
            if sample_weight is not None
            else None
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_v = torch.tensor(y_val, dtype=torch.float32, device=self.device)
            w_v = (
                torch.tensor(sample_weight_val, dtype=torch.float32, device=self.device)
                if sample_weight_val is not None
                else None
            )

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        n_epochs_run = 0

        for epoch in range(epochs):
            self.model.train()
            # Shuffle training data
            perm = torch.randperm(len(X_t))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_t), batch_size):
                idx = perm[start : start + batch_size]
                X_batch = X_t[idx]
                y_batch = y_t[idx]

                optimizer.zero_grad()
                pred = self.model(X_batch).squeeze(-1)
                residuals = (pred - y_batch) ** 2

                if w_t is not None:
                    loss = (w_t[idx] * residuals).mean()
                else:
                    loss = residuals.mean()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            n_epochs_run = epoch + 1

            # Validation
            if has_val:
                val_loss = self._compute_val_loss(X_v, y_v, w_v)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

            # Progress logging every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_str = f", val_loss={val_loss:.6f}" if has_val else ""
                logger.debug(
                    "Epoch %d/%d: train_loss=%.6f%s",
                    epoch + 1, epochs, avg_train_loss, val_str,
                )

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True

        # Log summary
        weighted_str = "weighted" if sample_weight is not None else "standard"
        val_str = f", val_loss={best_val_loss:.6f}" if has_val else ""
        logger.info(
            "NeuralNet %s training: %d epochs, hidden=%s, "
            "train_loss=%.6f%s, shape %s",
            weighted_str,
            n_epochs_run,
            cfg["hidden_layers"],
            avg_train_loss,
            val_str,
            X_train.shape,
        )
        return self

    @torch.no_grad()
    def _compute_val_loss(
        self,
        X_v: torch.Tensor,
        y_v: torch.Tensor,
        w_v: torch.Tensor | None,
    ) -> float:
        """Compute (optionally weighted) MSE on validation set."""
        self.model.eval()
        pred = self.model(X_v).squeeze(-1)
        residuals = (pred - y_v) ** 2
        if w_v is not None:
            return (w_v * residuals).mean().item()
        return residuals.mean().item()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        preds = self.model(X_t).squeeze(-1)
        return preds.cpu().numpy()

    def get_feature_importance(
        self, feature_names: list[str] | None = None
    ) -> pd.Series:
        """First-layer weight magnitude as a fast importance proxy."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        # First Linear layer
        first_linear = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                first_linear = module
                break

        if first_linear is None:
            raise RuntimeError("No Linear layer found.")

        # Sum absolute weights across output neurons → (input_dim,)
        weights = first_linear.weight.detach().cpu().numpy()  # (out, in)
        importance = np.abs(weights).sum(axis=0)

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
        """Compute SHAP values using DeepExplainer (fallback: KernelExplainer)."""
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
        self.model.eval()
        try:
            bg_tensor = torch.tensor(X_bg, dtype=torch.float32, device=self.device)
            explain_tensor = torch.tensor(X_explain, dtype=torch.float32, device=self.device)
            explainer = shap.DeepExplainer(self.model, bg_tensor)
            sv = explainer.shap_values(explain_tensor)
        except Exception:
            logger.info("DeepExplainer failed, falling back to KernelExplainer")

            def predict_fn(x):
                with torch.no_grad():
                    t = torch.tensor(x, dtype=torch.float32, device=self.device)
                    return self.model(t).squeeze(-1).cpu().numpy()

            explainer = shap.KernelExplainer(predict_fn, X_bg)
            sv = explainer.shap_values(X_explain, nsamples=100)

        if isinstance(sv, list):
            sv = sv[0]
        if isinstance(sv, torch.Tensor):
            sv = sv.cpu().numpy()

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

        for i, params in enumerate(param_grid):
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

            logger.info(
                "  Combo %d/%d: %s → MSE=%.6f",
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
