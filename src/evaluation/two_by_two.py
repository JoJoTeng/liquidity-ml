"""
Rolling-Window 2×2 Experiment Framework
========================================
Orchestrates the full experiment: for each model type, runs a rolling
window over the OOS period, trains standard and weighted models,
generates predictions, builds portfolios for the 2×2 design matrix,
and computes the effect decomposition.

2×2 Design Matrix (run independently per model):
                        Standard Portfolio    Liquidity-Weighted Portfolio
Standard Training       1A (Baseline)         1B
Weighted Training       2A                    2B (Combined)

Effect Decomposition:
  Portfolio Effect = SR(1B) − SR(1A)
  Training Effect  = SR(2A) − SR(1A)   ← central hypothesis
  Total Effect     = SR(2B) − SR(1A)
  Interaction      = Total − Portfolio − Training
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, get_output_dir
from src.data.loader import load_panel, get_feature_names, normalize_features
from src.evaluation.statistics import (
    compute_effect_decomposition,
    load_ff_factors,
    oos_r_squared,
    oos_r_squared_monthly,
    sharpe_ratio,
)
from src.models import create_model
from src.portfolio.construction import (
    DEFAULT_WEIGHT_COL,
    build_portfolio_timeseries,
    compute_net_returns_all_aum,
)
from src.weighting import PRIMARY_SCHEME, compute_all_weights, compute_weights

logger = logging.getLogger(__name__)


# ── Date utilities ────────────────────────────────────────────


def _yyyymm_to_months(yyyymm: int) -> int:
    """Convert yyyymm integer to absolute month count for arithmetic."""
    return (yyyymm // 100) * 12 + (yyyymm % 100)


def _months_to_yyyymm(months: int) -> int:
    """Convert absolute month count back to yyyymm integer."""
    y, m = divmod(months - 1, 12)
    return y * 100 + m + 1


def _offset_yyyymm(yyyymm: int, offset: int) -> int:
    """Add *offset* months to a yyyymm integer.

    Examples
    --------
    >>> _offset_yyyymm(200012, 1)
    200101
    >>> _offset_yyyymm(200001, -1)
    199912
    """
    return _months_to_yyyymm(_yyyymm_to_months(yyyymm) + offset)


# ── Window helpers ────────────────────────────────────────────


def _get_oos_months(panel: pd.DataFrame, config: dict) -> list[int]:
    """Return sorted list of yyyymm test months in the OOS period.

    Only includes months that (a) fall within [oos_start, oos_end] and
    (b) have enough prior history for the full train + val window.
    """
    train_cfg = config["training"]
    oos_start = train_cfg["oos_start"]
    oos_end = train_cfg["oos_end"]
    min_history = train_cfg["train_window"] + train_cfg["validation_window"]

    all_months = sorted(panel["yyyymm"].unique())

    oos_months = []
    for idx, m in enumerate(all_months):
        if m < oos_start or m > oos_end:
            continue
        # Need at least min_history months strictly before this test month
        if idx >= min_history:
            oos_months.append(m)

    if oos_months:
        logger.info(
            "OOS months: %d (from %d to %d)",
            len(oos_months),
            oos_months[0],
            oos_months[-1],
        )
    else:
        logger.warning("No OOS months available — check config and data range.")

    return oos_months


def _split_window(
    panel: pd.DataFrame,
    test_yyyymm: int,
    all_months: list[int],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the panel into train / val / test sets for a given test month.

    Window layout::

        [--- train_window months ---|--- val_window months ---|-- test (1 mo) --]

    Val = the ``val_window`` months immediately before ``test_yyyymm``.
    Train = the ``train_window`` months immediately before val.
    """
    train_cfg = config["training"]
    train_window = train_cfg["train_window"]
    val_window = train_cfg["validation_window"]

    # Months strictly before test
    months_before = [m for m in all_months if m < test_yyyymm]

    val_months = set(months_before[-val_window:])
    train_months = set(months_before[-(train_window + val_window) : -val_window])
    test_months = {test_yyyymm}

    train_df = panel[panel["yyyymm"].isin(train_months)].copy()
    val_df = panel[panel["yyyymm"].isin(val_months)].copy()
    test_df = panel[panel["yyyymm"].isin(test_months)].copy()

    return train_df, val_df, test_df


def _prepare_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    config: dict,
) -> dict[str, Any]:
    """Normalize features and extract arrays for model training.

    Normalization strategy (avoids look-ahead bias):
    - Train + Val are normalized together (both known at prediction time).
    - Test is normalized independently (its own cross-section).

    After normalization, remaining NaN in features are filled with 0.0
    (neutral rank) and rows with NaN target are dropped.
    """
    target_col = config["data"]["target_col"]  # "excess_ret"

    # ── Normalize: train+val together, test independently ──
    train_val = pd.concat([train_df, val_df], ignore_index=False)
    train_val_norm = normalize_features(train_val, features)
    test_norm = normalize_features(test_df, features)

    # Split train+val back
    train_norm = train_val_norm.loc[train_df.index]
    val_norm = train_val_norm.loc[val_df.index]

    # ── Fill NaN features with 0.0 (neutral rank) ──
    for df in [train_norm, val_norm, test_norm]:
        df[features] = df[features].fillna(0.0)

    # ── Drop rows with NaN target ──
    train_norm = train_norm.dropna(subset=[target_col])
    val_norm = val_norm.dropna(subset=[target_col])
    test_norm = test_norm.dropna(subset=[target_col])

    # ── Compute liquidity weights (uses raw liq_* columns, not features) ──
    weights_train = compute_weights(train_norm, scheme=PRIMARY_SCHEME)
    weights_val = compute_weights(val_norm, scheme=PRIMARY_SCHEME)

    # ── Extract numpy arrays ──
    X_train = train_norm[features].values
    y_train = train_norm[target_col].values
    X_val = val_norm[features].values
    y_val = val_norm[target_col].values
    X_test = test_norm[features].values
    y_test = test_norm[target_col].values

    meta_test = test_norm[["permno", "yyyymm"]].copy()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
        "weights_train": weights_train.values,
        "weights_val": weights_val.values,
        "train_target_mean": float(np.mean(y_train)),
    }


# ── Core rolling-window loop ─────────────────────────────────


def _rolling_predict(
    model_name: str,
    panel: pd.DataFrame,
    features: list[str],
    config: dict,
) -> dict[str, Any]:
    """Run the rolling-window prediction loop for ONE model type.

    For each OOS test month:
      1. Split into train / val / test
      2. Normalize features (train+val together, test independently)
      3. Retune hyper-parameters if due (every ``retune_frequency`` months)
      4. Train standard model (sample_weight=None) and weighted model
      5. Predict on test set with both models
      6. Collect feature importances
    """
    train_cfg = config["training"]
    retune_freq = train_cfg["retune_frequency"]
    seed = config["project"]["seed"]

    all_months = sorted(panel["yyyymm"].unique())
    oos_months = _get_oos_months(panel, config)

    if not oos_months:
        raise ValueError(
            "No OOS months available. Check config dates and panel coverage."
        )

    # Accumulate results
    predictions_list: list[pd.DataFrame] = []
    fi_std_list: list[pd.Series] = []
    fi_wt_list: list[pd.Series] = []
    expanding_means: list[float] = []

    # SHAP importance accumulators
    compute_shap = config.get("shap", {}).get("compute_shap", False)
    shap_cfg = config.get("shap", {})
    shap_std_list: list[pd.Series] = []
    shap_wt_list: list[pd.Series] = []

    # Hyper-parameter state (persists between windows, updated on retune)
    best_params_std: dict[str, Any] | None = None
    best_params_wt: dict[str, Any] | None = None
    months_since_retune = retune_freq  # forces retune on first window

    # Cumulative training targets for expanding OOS R² benchmark
    all_train_targets: list[float] = []

    t_start = time.time()

    for i, test_month in enumerate(oos_months):
        t_month = time.time()

        # ── 1. Split ──
        train_df, val_df, test_df = _split_window(
            panel, test_month, all_months, config
        )

        if len(test_df) < 50:
            logger.warning(
                "Month %d: only %d test stocks — skipping.", test_month, len(test_df)
            )
            continue
        if len(train_df) < 1000:
            logger.warning(
                "Month %d: only %d train rows — skipping.", test_month, len(train_df)
            )
            continue

        # ── 2. Prepare arrays ──
        data = _prepare_xy(train_df, val_df, test_df, features, config)

        # Track expanding target mean
        all_train_targets.extend(data["y_train"].tolist())
        expanding_means.append(float(np.mean(all_train_targets)))

        # ── 3. Retune if due ──
        needs_retune = months_since_retune >= retune_freq

        if needs_retune:
            logger.info(
                "Month %d (%d/%d): RETUNING %s",
                test_month,
                i + 1,
                len(oos_months),
                model_name,
            )

            # Tune standard model
            tuner_std = create_model(model_name, config=None, seed=seed)
            best_params_std = tuner_std.tune_hyperparameters(
                data["X_train"],
                data["y_train"],
                data["X_val"],
                data["y_val"],
                sample_weight=None,
                sample_weight_val=None,
            )

            # Tune weighted model (different optimal params)
            tuner_wt = create_model(model_name, config=None, seed=seed)
            best_params_wt = tuner_wt.tune_hyperparameters(
                data["X_train"],
                data["y_train"],
                data["X_val"],
                data["y_val"],
                sample_weight=data["weights_train"],
                sample_weight_val=data["weights_val"],
            )

            months_since_retune = 0

        # ── 4. Train standard model ──
        model_std = create_model(model_name, config=best_params_std, seed=seed)
        model_std.fit(
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            sample_weight=None,
            sample_weight_val=None,
        )

        # ── 5. Train weighted model ──
        model_wt = create_model(model_name, config=best_params_wt, seed=seed)
        model_wt.fit(
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            sample_weight=data["weights_train"],
            sample_weight_val=data["weights_val"],
        )

        # ── 6. Predict ──
        pred_std = model_std.predict(data["X_test"])
        pred_wt = model_wt.predict(data["X_test"])

        month_preds = data["meta_test"].copy()
        month_preds["y_true"] = data["y_test"]
        month_preds["pred_std"] = pred_std
        month_preds["pred_wt"] = pred_wt
        predictions_list.append(month_preds)

        # ── 7. Feature importances ──
        fi_std = model_std.get_feature_importance(features)
        fi_wt = model_wt.get_feature_importance(features)
        fi_std.name = test_month
        fi_wt.name = test_month
        fi_std_list.append(fi_std)
        fi_wt_list.append(fi_wt)

        # ── 8. SHAP values (optional) ──
        if compute_shap:
            try:
                sv_std = model_std.get_shap_values(
                    data["X_test"],
                    X_background=data["X_train"],
                    feature_names=features,
                    config=shap_cfg,
                )
                s_std = sv_std.abs().mean(axis=0)
                s_std.name = test_month
                shap_std_list.append(s_std)

                sv_wt = model_wt.get_shap_values(
                    data["X_test"],
                    X_background=data["X_train"],
                    feature_names=features,
                    config=shap_cfg,
                )
                s_wt = sv_wt.abs().mean(axis=0)
                s_wt.name = test_month
                shap_wt_list.append(s_wt)
            except Exception as e:
                logger.warning(
                    "SHAP computation failed for month %d: %s", test_month, e
                )

        months_since_retune += 1

        # ── Progress logging ──
        elapsed = time.time() - t_month
        if (i + 1) % 12 == 0 or i == 0:
            total_elapsed = time.time() - t_start
            eta = total_elapsed / (i + 1) * (len(oos_months) - i - 1)
            logger.info(
                "Month %d (%d/%d): train=%d val=%d test=%d | "
                "%.1fs this | %.1fs total | ETA %.0fs",
                test_month,
                i + 1,
                len(oos_months),
                len(data["X_train"]),
                len(data["X_val"]),
                len(data["X_test"]),
                elapsed,
                total_elapsed,
                eta,
            )

    # ── Assemble results ──
    if not predictions_list:
        raise ValueError("No predictions generated — all months were skipped.")

    predictions = pd.concat(predictions_list, ignore_index=True)
    fi_std_df = pd.DataFrame(fi_std_list)  # rows = months, cols = features
    fi_wt_df = pd.DataFrame(fi_wt_list)
    shap_std_df = pd.DataFrame(shap_std_list) if shap_std_list else pd.DataFrame()
    shap_wt_df = pd.DataFrame(shap_wt_list) if shap_wt_list else pd.DataFrame()

    total_time = time.time() - t_start
    logger.info(
        "%s rolling prediction complete: %d months, %.1f min",
        model_name,
        len(oos_months),
        total_time / 60,
    )

    return {
        "predictions": predictions,
        "feature_importance_std": fi_std_df,
        "feature_importance_wt": fi_wt_df,
        "shap_importance_std": shap_std_df,
        "shap_importance_wt": shap_wt_df,
        "oos_months": oos_months,
        "expanding_means": expanding_means,
        "timing": {
            "total_seconds": total_time,
            "seconds_per_month": total_time / max(len(oos_months), 1),
        },
    }


# ── Public API ────────────────────────────────────────────────


def run_two_by_two(
    model_name: str,
    panel: pd.DataFrame | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Run the full 2×2 experiment for one model type.

    Parameters
    ----------
    model_name : ``"xgboost"``, ``"random_forest"``, or ``"neural_network"``
    panel : Pre-loaded data panel.  If *None*, calls ``load_panel()``.
    config : Override config dict.  If *None*, loads from ``config.yaml``.

    Returns
    -------
    dict
        Comprehensive results including per-cell portfolio returns,
        predictions, feature importances, OOS R², and the full
        effect decomposition with Ledoit-Wolf tests and factor alphas.
    """
    if config is None:
        config = load_config()

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("2×2 Experiment: %s", model_name)
    logger.info("=" * 60)

    # ── 1. Load data ──
    if panel is None:
        logger.info("Loading panel …")
        panel = load_panel(config)

    features = get_feature_names(panel)
    logger.info("Features: %d", len(features))

    # ── 2. Rolling predictions ──
    rolling = _rolling_predict(model_name, panel, features, config)
    predictions = rolling["predictions"]

    logger.info(
        "Predictions: %d rows, %d months, %d unique permnos",
        len(predictions),
        predictions["yyyymm"].nunique(),
        predictions["permno"].nunique(),
    )

    # ── 3. Prepare test panel for portfolio construction ──
    # Subset the full panel to only (permno, yyyymm) pairs that have predictions
    oos_keys = predictions[["permno", "yyyymm"]].drop_duplicates()
    test_panel = panel.merge(oos_keys, on=["permno", "yyyymm"], how="inner")

    # Ensure portfolio weight columns exist
    if DEFAULT_WEIGHT_COL not in test_panel.columns:
        logger.info("Computing portfolio weights for test panel …")
        weight_df = compute_all_weights(test_panel)
        test_panel = pd.concat([test_panel, weight_df], axis=1)

    # Merge predictions
    pred_merged = test_panel.merge(
        predictions[["permno", "yyyymm", "pred_std", "pred_wt"]],
        on=["permno", "yyyymm"],
        how="inner",
    )

    # ── 4. Build 4 portfolio time series ──
    pred_std = pd.Series(pred_merged["pred_std"].values, index=pred_merged.index)
    pred_wt = pd.Series(pred_merged["pred_wt"].values, index=pred_merged.index)

    # Cell 1A: standard training, equal-weight portfolio
    results_1a, pos_1a = build_portfolio_timeseries(
        pred_merged, pred_std, weighted=False, config=config
    )
    # Cell 1B: standard training, liquidity-weighted portfolio
    results_1b, pos_1b = build_portfolio_timeseries(
        pred_merged, pred_std, weighted=True, config=config
    )
    # Cell 2A: weighted training, equal-weight portfolio
    results_2a, pos_2a = build_portfolio_timeseries(
        pred_merged, pred_wt, weighted=False, config=config
    )
    # Cell 2B: weighted training, liquidity-weighted portfolio
    results_2b, pos_2b = build_portfolio_timeseries(
        pred_merged, pred_wt, weighted=True, config=config
    )

    # ── 5. Transaction costs & net returns ──
    cells: dict[str, dict[str, Any]] = {}
    for label, gross, pos_hist in [
        ("1A", results_1a, pos_1a),
        ("1B", results_1b, pos_1b),
        ("2A", results_2a, pos_2a),
        ("2B", results_2b, pos_2b),
    ]:
        net = compute_net_returns_all_aum(gross, pos_hist, panel, config=config)
        cells[label] = {
            "gross_returns": gross,
            "net_returns": net,
            "positions": pos_hist,
        }

    # ── 6. Effect decomposition ──
    # Align all 4 return series to common months
    common_months = sorted(
        set(results_1a["yyyymm"])
        & set(results_1b["yyyymm"])
        & set(results_2a["yyyymm"])
        & set(results_2b["yyyymm"])
    )

    def _aligned(df: pd.DataFrame) -> np.ndarray:
        return (
            df[df["yyyymm"].isin(common_months)]
            .sort_values("yyyymm")["ret_long_short"]
            .values
        )

    r_1a = _aligned(results_1a)
    r_1b = _aligned(results_1b)
    r_2a = _aligned(results_2a)
    r_2b = _aligned(results_2b)
    yyyymm_arr = np.array(common_months)

    ff_factors = load_ff_factors(n_factors=5, config=config)

    effect_decomp = compute_effect_decomposition(
        r_1a, r_1b, r_2a, r_2b,
        ff_factors=ff_factors,
        yyyymm=yyyymm_arr,
        config=config,
    )

    # ── 7. OOS R² ──
    month_to_mean = dict(zip(rolling["oos_months"], rolling["expanding_means"]))
    predictions["expanding_mean"] = predictions["yyyymm"].map(month_to_mean)

    oos_r2_std = oos_r_squared(
        predictions["y_true"].values,
        predictions["pred_std"].values,
        predictions["expanding_mean"].values,
    )
    oos_r2_wt = oos_r_squared(
        predictions["y_true"].values,
        predictions["pred_wt"].values,
        predictions["expanding_mean"].values,
    )

    # Monthly OOS R²
    oos_r2_monthly_std = oos_r_squared_monthly(predictions, pred_col="pred_std")
    oos_r2_monthly_wt = oos_r_squared_monthly(predictions, pred_col="pred_wt")
    oos_r2_monthly = oos_r2_monthly_std.rename(columns={"oos_r2": "R2_std"}).merge(
        oos_r2_monthly_wt.rename(columns={"oos_r2": "R2_wt"}),
        on="yyyymm",
    )

    # ── Summary log ──
    total_time = time.time() - t0
    logger.info("-" * 60)
    logger.info("Results for %s (%.1f min):", model_name, total_time / 60)
    for cell_name in ["1A", "1B", "2A", "2B"]:
        sr = sharpe_ratio(
            cells[cell_name]["gross_returns"]["ret_long_short"].dropna()
        )
        logger.info("  Cell %s: SR = %.3f", cell_name, sr)
    logger.info("  OOS R² (std): %.4f%%", oos_r2_std * 100)
    logger.info("  OOS R² (wt):  %.4f%%", oos_r2_wt * 100)
    logger.info("  Training effect:  %.3f", effect_decomp["training_effect"])
    logger.info("  Portfolio effect:  %.3f", effect_decomp["portfolio_effect"])
    logger.info("-" * 60)

    result = {
        "model": model_name,
        "cells": cells,
        "predictions": predictions,
        "feature_importance": {
            "standard": rolling["feature_importance_std"],
            "weighted": rolling["feature_importance_wt"],
        },
        "oos_r2": {"standard": oos_r2_std, "weighted": oos_r2_wt},
        "oos_r2_monthly": oos_r2_monthly,
        "effect_decomposition": effect_decomp,
        "oos_months": rolling["oos_months"],
        "timing": rolling["timing"],
        "config": copy.deepcopy(config),
    }

    # Include SHAP importance if computed
    if not rolling["shap_importance_std"].empty:
        result["shap_importance"] = {
            "standard": rolling["shap_importance_std"],
            "weighted": rolling["shap_importance_wt"],
        }

    return result


def run_all_models(
    panel: pd.DataFrame | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Run the 2×2 experiment for all models and compute ensemble results.

    Loads the panel once and shares it across models.

    Returns
    -------
    dict
        ``per_model``: mapping model_name → ``run_two_by_two`` result.
        ``ensemble``: averaged Sharpe ratios and effects across models.
        ``summary``: DataFrame with one row per model and key metrics.
    """
    if config is None:
        config = load_config()
    if panel is None:
        logger.info("Loading panel (shared across all models) …")
        panel = load_panel(config)

    model_names = config["models"]["run_models"]
    logger.info(
        "Running 2×2 experiment for %d models: %s", len(model_names), model_names
    )

    per_model: dict[str, dict[str, Any]] = {}
    for name in model_names:
        per_model[name] = run_two_by_two(name, panel=panel, config=config)

    # ── Ensemble summary ──
    rows = []
    for name, result in per_model.items():
        row: dict[str, Any] = {"model": name}
        for cell in ["1A", "1B", "2A", "2B"]:
            sr = sharpe_ratio(
                result["cells"][cell]["gross_returns"]["ret_long_short"].dropna()
            )
            row[f"SR_{cell}"] = sr
        row["OOS_R2_std"] = result["oos_r2"]["standard"]
        row["OOS_R2_wt"] = result["oos_r2"]["weighted"]
        row["training_effect"] = result["effect_decomposition"]["training_effect"]
        row["portfolio_effect"] = result["effect_decomposition"]["portfolio_effect"]
        row["total_effect"] = result["effect_decomposition"]["total_effect"]
        row["interaction"] = result["effect_decomposition"]["interaction"]
        rows.append(row)

    summary = pd.DataFrame(rows)

    ensemble = {
        "SR_1A": summary["SR_1A"].mean(),
        "SR_1B": summary["SR_1B"].mean(),
        "SR_2A": summary["SR_2A"].mean(),
        "SR_2B": summary["SR_2B"].mean(),
        "training_effect": summary["training_effect"].mean(),
        "portfolio_effect": summary["portfolio_effect"].mean(),
        "total_effect": summary["total_effect"].mean(),
        "interaction": summary["interaction"].mean(),
    }

    logger.info("=" * 60)
    logger.info("ENSEMBLE SUMMARY")
    logger.info("=" * 60)
    logger.info("\n%s", summary.to_string(index=False, float_format="%.4f"))
    logger.info("\nEnsemble averages:")
    for k, v in ensemble.items():
        logger.info("  %s: %.4f", k, v)

    return {
        "per_model": per_model,
        "ensemble": ensemble,
        "summary": summary,
    }


# ── Persistence ───────────────────────────────────────────────


def save_results(
    results: dict[str, Any],
    output_dir: str | Path | None = None,
) -> None:
    """Save experiment results to disk.

    Saves under ``outputs/experiment/{model_name}/``.

    Handles output from both ``run_two_by_two`` (single model) and
    ``run_all_models`` (multiple models + ensemble).
    """
    if output_dir is None:
        out_base = get_output_dir() / "experiment"
    else:
        out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    if "per_model" in results:
        # Multi-model result
        for name, model_result in results["per_model"].items():
            _save_single_model(model_result, out_base / name)
        results["summary"].to_csv(out_base / "summary.csv", index=False)
        with open(out_base / "ensemble.json", "w") as f:
            json.dump(results["ensemble"], f, indent=2, default=str)
    else:
        # Single-model result
        _save_single_model(results, out_base / results["model"])

    logger.info("Results saved to %s", out_base)


def _save_single_model(result: dict[str, Any], model_dir: Path) -> None:
    """Persist a single model's results."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Predictions
    result["predictions"].to_parquet(model_dir / "predictions.parquet", index=False)

    # Portfolio returns per cell
    for cell_name in ["1A", "1B", "2A", "2B"]:
        cell = result["cells"][cell_name]
        cell["gross_returns"].to_csv(
            model_dir / f"gross_returns_{cell_name}.csv", index=False
        )
        cell["net_returns"].to_csv(
            model_dir / f"net_returns_{cell_name}.csv", index=False
        )

    # Feature importances
    result["feature_importance"]["standard"].to_csv(
        model_dir / "feature_importance_std.csv"
    )
    result["feature_importance"]["weighted"].to_csv(
        model_dir / "feature_importance_wt.csv"
    )

    # SHAP importances (if computed)
    if "shap_importance" in result and not result["shap_importance"]["standard"].empty:
        result["shap_importance"]["standard"].to_csv(
            model_dir / "shap_importance_std.csv"
        )
        result["shap_importance"]["weighted"].to_csv(
            model_dir / "shap_importance_wt.csv"
        )

    # Effect decomposition
    decomp = _make_json_serializable(result["effect_decomposition"])
    with open(model_dir / "effect_decomposition.json", "w") as f:
        json.dump(decomp, f, indent=2, default=str)

    # OOS R²
    with open(model_dir / "oos_r2.json", "w") as f:
        json.dump(result["oos_r2"], f, indent=2)

    # Monthly OOS R²
    if "oos_r2_monthly" in result:
        result["oos_r2_monthly"].to_csv(model_dir / "oos_r2_monthly.csv", index=False)


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
