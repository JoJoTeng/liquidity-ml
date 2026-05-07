"""Shared rolling-window model training engine.

This module is the canonical training path for both the motivation Step 3
standard models and the formal 2x2 experiment. Keeping the engine here avoids
quiet drift between "motivation" and "formal" standard-model results.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

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


def _param_grid_from_config(model_config: dict, seed: int) -> list[dict]:
    """Build the configured hyperparameter grid/random draw."""
    search_space = model_config.get("search_space", {})
    if not search_space:
        return []

    n_trials = model_config.get("n_random_search", 50)
    keys = list(search_space.keys())
    values = list(search_space.values())

    full_grid_size = 1
    for value_grid in values:
        full_grid_size *= len(value_grid)

    if full_grid_size <= n_trials:
        logger.info(
            "Tuning grid: full grid %d combinations (<= %d)",
            full_grid_size, n_trials,
        )
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    logger.info("Tuning grid: random search %d / %d combinations", n_trials, full_grid_size)
    return list(ParameterSampler(search_space, n_iter=n_trials, random_state=seed))


def _month_time_series_cv_masks(
    month_values: np.ndarray,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray, list[int], list[int]]]:
    """Return expanding-window CV masks that never split a calendar month."""
    unique_months = np.array(sorted(pd.unique(month_values)))
    if len(unique_months) <= n_splits:
        raise ValueError(
            f"Need more than {n_splits} train months for TimeSeriesSplit; "
            f"got {len(unique_months)}"
        )

    folds = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_month_idx, val_month_idx in splitter.split(unique_months):
        train_months = unique_months[train_month_idx]
        val_months = unique_months[val_month_idx]
        train_mask = np.isin(month_values, train_months)
        val_mask = np.isin(month_values, val_months)
        folds.append((
            train_mask,
            val_mask,
            train_months.tolist(),
            val_months.tolist(),
        ))
    return folds


def _tune_hyperparameters_timeseries_cv(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_month_values: np.ndarray,
    sample_weight: np.ndarray | None,
    config: dict,
    seed: int,
    n_splits: int,
) -> tuple[dict, list[dict]]:
    """Tune model hyperparameters with month-blocked expanding TimeSeriesSplit."""
    model_config = config["models"][model_name]
    param_grid = _param_grid_from_config(model_config, seed)
    if not param_grid:
        logger.info("No search space configured; using defaults.")
        return model_config, []

    keys = list(model_config.get("search_space", {}).keys())
    folds = _month_time_series_cv_masks(train_month_values, n_splits)
    logger.info(
        "TimeSeriesSplit tuning: model=%s, candidates=%d, folds=%d",
        model_name, len(param_grid), len(folds),
    )

    best_mse = np.inf
    best_params: dict = {}
    diagnostics = []

    for candidate_idx, params in enumerate(param_grid, start=1):
        trial_config = {**model_config, **params}
        if model_name == "neural_network":
            trial_config = {
                **trial_config,
                "n_ensemble_seeds": model_config.get("tune_n_ensemble_seeds", 1),
            }

        fold_mses = []
        for fold_idx, (tr_mask, va_mask, _, val_months) in enumerate(folds, start=1):
            w_tr = sample_weight[tr_mask] if sample_weight is not None else None
            w_va = sample_weight[va_mask] if sample_weight is not None else None

            model = create_model(model_name, config=trial_config, seed=seed)
            model.fit(
                X_train[tr_mask],
                y_train[tr_mask],
                X_val=X_train[va_mask],
                y_val=y_train[va_mask],
                sample_weight=w_tr,
                sample_weight_val=w_va,
            )
            preds = model.predict(X_train[va_mask])
            fold_mse = mean_squared_error(
                y_train[va_mask],
                preds,
                sample_weight=w_va,
            )
            fold_mses.append(float(fold_mse))
            diagnostics.append({
                "candidate": candidate_idx,
                "n_candidates": len(param_grid),
                "fold": fold_idx,
                "n_folds": len(folds),
                "selection_mse": float(fold_mse),
                "val_start": int(val_months[0]),
                "val_end": int(val_months[-1]),
                **{k: params[k] for k in keys},
                **getattr(model, "fit_diagnostics", {}),
            })

        avg_mse = float(np.mean(fold_mses))
        logger.info(
            "  Candidate %d/%d: %s -> CV MSE=%.6f",
            candidate_idx, len(param_grid), {k: params[k] for k in keys}, avg_mse,
        )

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_params = trial_config

    if model_name == "neural_network":
        best_params["n_ensemble_seeds"] = model_config.get("n_ensemble_seeds", 1)

    logger.info(
        "Best TimeSeriesSplit MSE: %.6f | Best params: %s",
        best_mse,
        {k: best_params.get(k) for k in keys},
    )
    return best_params, diagnostics


def run_rolling_training(
    panel: pd.DataFrame,
    features: list[str],
    model_name: str,
    weights: pd.Series | None,
    config: dict,
    seed: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run rolling-window ML training on the processed panel.

    ``panel`` is expected to come from ``load_processed_panel()``. The selected
    model features are already rank-normalized by ``scripts/01_process_data.py``;
    this engine only applies the neutral missing-value fill of 0.5.

    Returns
    -------
    predictions_df, importance_df, native_importance_df, params_df, diagnostics_df
        ``importance_df`` is SHAP mean absolute importance by feature/window.
        ``native_importance_df`` is gain/coef/permutation importance by
        feature/window.
        ``diagnostics_df`` records model fit diagnostics when exposed by the
        model implementation. For neural networks it includes early-stopping
        and tuning-candidate loss summaries.
    """
    train_cfg = config["training"]
    train_win = train_cfg["train_window"]
    val_win = train_cfg["validation_window"]
    retune_freq = train_cfg["retune_frequency"]
    tuning_method = train_cfg.get("tuning_method", "validation")
    cv_n_splits = train_cfg.get("cv_n_splits", 5)
    oos_start = train_cfg["oos_start"]
    oos_end = train_cfg["oos_end"]
    target = config["data"]["target_col"]
    shap_cfg = config.get("shap", {})
    skip_importance = bool(train_cfg.get("skip_importance", False))
    compute_shap = bool(shap_cfg.get("compute_shap", True)) and not skip_importance
    compute_native_importance = not skip_importance

    all_months = sorted(panel["yyyymm"].unique())
    oos_months = [m for m in all_months if oos_start <= m <= oos_end]

    if not oos_months:
        raise ValueError(f"No OOS months in range [{oos_start}, {oos_end}]")

    logger.info(
        "Rolling training [%s]: model=%s, months=%d, train=%d, val=%d, "
        "retune=%d, tuning=%s",
        label, model_name, len(oos_months), train_win, val_win,
        retune_freq, tuning_method,
    )

    predictions_all = []
    importance_all = []
    native_importance_all = []
    params_all = []
    diagnostics_all = []
    current_params: dict | None = None
    windows_since_tune = retune_freq

    for i, test_month in enumerate(oos_months):
        test_idx = all_months.index(test_month)

        if val_win > 0:
            train_start_idx = test_idx - val_win - train_win
            val_start_idx = test_idx - val_win
        else:
            train_start_idx = test_idx - train_win
            val_start_idx = test_idx

        if train_start_idx < 0:
            continue

        train_months = all_months[train_start_idx:val_start_idx]
        val_months = all_months[val_start_idx:test_idx]

        train_df = panel[panel["yyyymm"].isin(train_months)].copy()
        val_df = panel[panel["yyyymm"].isin(val_months)].copy()
        test_df = panel[panel["yyyymm"] == test_month].copy()

        if len(train_df) < 100 or len(test_df) < 50:
            continue
        if val_win > 0 and len(val_df) < 10:
            continue

        train_df[features] = train_df[features].fillna(0.5)
        if val_win > 0:
            val_df[features] = val_df[features].fillna(0.5)
        test_df[features] = test_df[features].fillna(0.5)

        X_train = train_df[features].values
        y_train = train_df[target].values
        X_val = val_df[features].values if val_win > 0 else None
        y_val = val_df[target].values if val_win > 0 else None
        X_test = test_df[features].values
        y_test = test_df[target].values

        valid_train = ~np.isnan(y_train)
        valid_val = ~np.isnan(y_val) if val_win > 0 else None
        valid_test = ~np.isnan(y_test)

        if valid_train.sum() < 100 or valid_test.sum() < 30:
            continue

        train_month_values = train_df["yyyymm"].values[valid_train]
        X_train, y_train = X_train[valid_train], y_train[valid_train]
        if val_win > 0:
            X_val, y_val = X_val[valid_val], y_val[valid_val]
        X_test = X_test[valid_test]
        test_df = test_df[valid_test]
        y_test = y_test[valid_test]

        w_train = None
        w_val = None
        if weights is not None:
            w_train = weights.reindex(train_df.index).fillna(1.0).values[valid_train]
            if val_win > 0:
                w_val = weights.reindex(val_df.index).fillna(1.0).values[valid_val]

        windows_since_tune += 1
        if windows_since_tune >= retune_freq:
            logger.info(
                "  [%s] Tuning at month %d (%d/%d)",
                label, test_month, i + 1, len(oos_months),
            )
            if tuning_method == "timeseries_cv":
                best, tuning_diagnostics = _tune_hyperparameters_timeseries_cv(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    train_month_values=train_month_values,
                    sample_weight=w_train,
                    config=config,
                    seed=seed,
                    n_splits=cv_n_splits,
                )
            elif tuning_method == "validation":
                if X_val is None or y_val is None:
                    raise ValueError(
                        "validation tuning requires training.validation_window > 0"
                    )
                tuning_model = create_model(model_name, seed=seed)
                best = tuning_model.tune_hyperparameters(
                    X_train, y_train, X_val, y_val,
                    sample_weight=w_train, sample_weight_val=w_val,
                )
                tuning_diagnostics = getattr(tuning_model, "tuning_diagnostics", [])
            else:
                raise ValueError(
                    "training.tuning_method must be 'validation' or 'timeseries_cv'; "
                    f"got {tuning_method!r}"
                )

            for row in tuning_diagnostics:
                diagnostics_all.append({
                    "yyyymm": test_month,
                    "label": label,
                    "stage": "tuning_candidate",
                    **row,
                })
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
        fit_diagnostics = getattr(model, "fit_diagnostics", {})
        if fit_diagnostics:
            diagnostics_all.append({
                "yyyymm": test_month,
                "label": label,
                "stage": "fit",
                **fit_diagnostics,
            })

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
            reason = "skip_importance" if skip_importance else "config"
            logger.info("  [%s] SHAP disabled by %s", label, reason)
        importance_all.append(shap_row)

        native_row = {"yyyymm": test_month}
        if not compute_native_importance:
            if i == 0:
                logger.info("  [%s] Native/permutation importance disabled", label)
        elif model_name == "neural_network":
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
    diagnostics_df = (
        pd.DataFrame(diagnostics_all) if diagnostics_all else pd.DataFrame()
    )

    return predictions_df, importance_df, native_importance_df, params_df, diagnostics_df
