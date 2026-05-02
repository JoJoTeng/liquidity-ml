"""Shared rolling-window model training engine.

This module is the canonical training path for both the motivation Step 3
standard models and the formal 2x2 experiment. Keeping the engine here avoids
quiet drift between "motivation" and "formal" standard-model results.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models import create_model

logger = logging.getLogger(__name__)


def _mean_abs_shap_for_window(
    model,
    model_name: str,
    X_test: np.ndarray,
    X_train: np.ndarray,
    features: list[str],
    shap_cfg: dict,
) -> pd.Series:
    """Compute mean absolute SHAP for one rolling test window."""
    cfg = dict(shap_cfg)

    if model_name == "xgboost":
        shap_df = model.get_shap_values(
            X_test,
            feature_names=features,
            config=cfg,
        )
    elif model_name == "neural_network":
        cfg.setdefault("use_kernel_fallback", False)
        shap_df = model.get_shap_values(
            X_test,
            X_background=X_train,
            feature_names=features,
            config=cfg,
        )
    elif model_name == "elastic_net":
        shap_df = model.get_shap_values(
            X_test,
            X_background=X_train,
            feature_names=features,
            config=cfg,
        )
    else:
        raise ValueError(f"Unsupported model for SHAP: {model_name}")

    return shap_df.abs().mean()


def run_rolling_training(
    panel: pd.DataFrame,
    features: list[str],
    model_name: str,
    weights: pd.Series | None,
    config: dict,
    seed: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run rolling-window ML training on the processed panel.

    ``panel`` is expected to come from ``load_processed_panel()``. The selected
    model features are already rank-normalized by ``scripts/01_process_data.py``;
    this engine only applies the neutral missing-value fill of 0.5.

    Returns
    -------
    predictions_df, importance_df, native_importance_df, params_df
        ``importance_df`` is SHAP mean absolute importance by feature/window.
        ``native_importance_df`` is gain/coef/permutation importance by
        feature/window.
    """
    train_cfg = config["training"]
    train_win = train_cfg["train_window"]
    val_win = train_cfg["validation_window"]
    retune_freq = train_cfg["retune_frequency"]
    oos_start = train_cfg["oos_start"]
    oos_end = train_cfg["oos_end"]
    target = config["data"]["target_col"]
    shap_cfg = config.get("shap", {})
    compute_shap = bool(shap_cfg.get("compute_shap", True))

    all_months = sorted(panel["yyyymm"].unique())
    oos_months = [m for m in all_months if oos_start <= m <= oos_end]

    if not oos_months:
        raise ValueError(f"No OOS months in range [{oos_start}, {oos_end}]")

    logger.info(
        "Rolling training [%s]: model=%s, months=%d, train=%d, val=%d, retune=%d",
        label, model_name, len(oos_months), train_win, val_win, retune_freq,
    )

    predictions_all = []
    importance_all = []
    native_importance_all = []
    params_all = []
    current_params: dict | None = None
    windows_since_tune = retune_freq

    for i, test_month in enumerate(oos_months):
        test_idx = all_months.index(test_month)

        train_start_idx = test_idx - val_win - train_win
        val_start_idx = test_idx - val_win

        if train_start_idx < 0:
            continue

        train_months = all_months[train_start_idx:val_start_idx]
        val_months = all_months[val_start_idx:test_idx]

        train_df = panel[panel["yyyymm"].isin(train_months)].copy()
        val_df = panel[panel["yyyymm"].isin(val_months)].copy()
        test_df = panel[panel["yyyymm"] == test_month].copy()

        if len(train_df) < 100 or len(val_df) < 10 or len(test_df) < 50:
            continue

        train_df[features] = train_df[features].fillna(0.5)
        val_df[features] = val_df[features].fillna(0.5)
        test_df[features] = test_df[features].fillna(0.5)

        X_train = train_df[features].values
        y_train = train_df[target].values
        X_val = val_df[features].values
        y_val = val_df[target].values
        X_test = test_df[features].values
        y_test = test_df[target].values

        valid_train = ~np.isnan(y_train)
        valid_val = ~np.isnan(y_val)
        valid_test = ~np.isnan(y_test)

        if valid_train.sum() < 100 or valid_test.sum() < 30:
            continue

        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_val, y_val = X_val[valid_val], y_val[valid_val]
        X_test = X_test[valid_test]
        test_df = test_df[valid_test]
        y_test = y_test[valid_test]

        w_train = None
        w_val = None
        if weights is not None:
            w_train = weights.reindex(train_df.index).fillna(1.0).values[valid_train]
            w_val = weights.reindex(val_df.index).fillna(1.0).values[valid_val]

        windows_since_tune += 1
        if windows_since_tune >= retune_freq:
            logger.info(
                "  [%s] Tuning at month %d (%d/%d)",
                label, test_month, i + 1, len(oos_months),
            )
            tuning_model = create_model(model_name, seed=seed)
            best = tuning_model.tune_hyperparameters(
                X_train, y_train, X_val, y_val,
                sample_weight=w_train, sample_weight_val=w_val,
            )
            current_params = best
            windows_since_tune = 0
            params_all.append({"yyyymm": test_month, **best})

        model = create_model(model_name, config=current_params, seed=seed)
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            sample_weight=w_train,
            sample_weight_val=w_val,
        )

        preds = model.predict(X_test)
        pred_df = pd.DataFrame({
            "permno": test_df["permno"].values,
            "yyyymm": test_month,
            "prediction": preds,
        }, index=test_df.index)
        predictions_all.append(pred_df)

        shap_row = {"yyyymm": test_month}
        if compute_shap:
            try:
                mean_abs_shap = _mean_abs_shap_for_window(
                    model, model_name, X_test, X_train, features, shap_cfg,
                )
                shap_row.update(mean_abs_shap.to_dict())
            except Exception as exc:
                logger.warning(
                    "  [%s] SHAP failed at month %d: %s",
                    label, test_month, exc,
                )
        elif i == 0:
            logger.info("  [%s] SHAP disabled by config", label)
        importance_all.append(shap_row)

        native_row = {"yyyymm": test_month}
        if model_name == "neural_network":
            try:
                perm_imp = model.get_permutation_importance(
                    X_test, y_test, feature_names=features, seed=seed,
                )
                native_row.update(perm_imp.to_dict())
            except Exception as exc:
                logger.warning("  [%s] Permutation importance failed: %s", label, exc)
        else:
            imp = model.get_feature_importance(feature_names=features)
            native_row.update(imp.to_dict())
        native_importance_all.append(native_row)

        if (i + 1) % 24 == 0:
            logger.info("  [%s] Progress: %d/%d months", label, i + 1, len(oos_months))

    predictions_df = pd.concat(predictions_all) if predictions_all else pd.DataFrame()
    importance_df = pd.DataFrame(importance_all) if importance_all else pd.DataFrame()
    native_importance_df = (
        pd.DataFrame(native_importance_all) if native_importance_all else pd.DataFrame()
    )
    params_df = pd.DataFrame(params_all) if params_all else pd.DataFrame()

    return predictions_df, importance_df, native_importance_df, params_df
