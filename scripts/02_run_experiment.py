"""
02 - Run 2x2 Experiment (LiquidityML v3)
==========================================
Rolling-window ML training for the formal analysis.

Each invocation trains ONE specification:
  - One model (elastic_net / xgboost / neural_network)
  - One weight family + AUM/lambda when needed
    (dolvol, softmax_rank at a chosen lambda, or tc at a given AUM)

Produces:
  - M_std (standard training) predictions + importance
  - M_w  (weighted training)  predictions + importance
  - Tuned hyperparameters per retune window

M_std is shared across weight families - if predictions already exist
in outputs/formalanalysis/experiment/{model}/standard/, training is
skipped.  This avoids redundant computation when running multiple
weighted jobs per model.

HPC usage:
  python scripts/02_run_experiment.py --model xgboost --weights dolvol
  python scripts/02_run_experiment.py --model xgboost --weights softmax_rank --softmax-lambda 2
  python scripts/02_run_experiment.py --model xgboost --weights softmax_rank --softmax-lambda 3
  python scripts/02_run_experiment.py --model xgboost --weights tc --aum 10
  python scripts/02_run_experiment.py --model xgboost --weights tc --aum 500
  python scripts/02_run_experiment.py --model elastic_net --weights dolvol
  ...

Quick test (2020-2024, ~60 months):
  python scripts/02_run_experiment.py --model xgboost --weights dolvol --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# -- Project imports ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_data_dir
from src.data.loader import load_panel
from src.models import create_model
from src.weighting.schemes import compute_weights

# -- Logging -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("02_experiment")


# -- CLI ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run 2x2 experiment (one specification)")
    p.add_argument("--model", required=True,
                    choices=["elastic_net", "xgboost", "neural_network"])
    p.add_argument("--weights", required=True,
                    choices=["dolvol", "softmax_rank", "tc"])
    p.add_argument("--aum", type=int, default=None,
                    help="AUM in $M (e.g. 500). Required if --weights tc.")
    p.add_argument("--softmax-lambda", type=float, default=None,
                    help=(
                        "Override weighting.softmax_rank_lambda. "
                        "Only valid when --weights softmax_rank."
                    ))
    p.add_argument("--quick", action="store_true",
                    help="Quick test: 2020-2024 only")
    return p.parse_args()


def _lambda_label(lam: float) -> str:
    """Compact filesystem-safe label, e.g. 3 -> lam3 and 2.5 -> lam2p5."""
    token = f"{lam:g}".replace("-", "m").replace(".", "p")
    return f"lam{token}"


# -- Rolling Window Engine ---------------------------------------------

def run_rolling_training(
    panel: pd.DataFrame,
    features: list[str],
    model_name: str,
    weights: pd.Series | None,
    config: dict,
    seed: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run rolling-window ML training and collect OOS predictions.

    Returns
    -------
    (predictions_df, importance_df, native_importance_df, params_df)
    importance_df: SHAP mean(|SHAP|) per feature per window (primary).
    native_importance_df: gain/coef/permutation per window (secondary).
    """
    train_cfg = config["training"]
    train_win = train_cfg["train_window"]
    val_win = train_cfg["validation_window"]
    retune_freq = train_cfg["retune_frequency"]
    oos_start = train_cfg["oos_start"]
    oos_end = train_cfg["oos_end"]
    target = config["data"]["target_col"]

    all_months = sorted(panel["yyyymm"].unique())
    oos_months = [m for m in all_months if oos_start <= m <= oos_end]

    if not oos_months:
        raise ValueError(f"No OOS months in range [{oos_start}, {oos_end}]")

    logger.info(
        "Rolling training [%s]: model=%s, months=%d, train=%d, val=%d, retune=%d",
        label, model_name, len(oos_months), train_win, val_win, retune_freq,
    )

    predictions_all = []
    importance_all = []       # SHAP (primary)
    native_importance_all = [] # gain / coef / permutation (secondary)
    params_all = []
    current_params: dict | None = None
    windows_since_tune = retune_freq  # Force tune on first window

    for i, test_month in enumerate(oos_months):
        test_idx = all_months.index(test_month)

        # Define window boundaries
        train_start_idx = test_idx - val_win - train_win
        val_start_idx = test_idx - val_win

        if train_start_idx < 0:
            continue

        train_months = all_months[train_start_idx:val_start_idx]
        val_months = all_months[val_start_idx:test_idx]

        # Split data
        train_mask = panel["yyyymm"].isin(train_months)
        val_mask = panel["yyyymm"].isin(val_months)
        test_mask = panel["yyyymm"] == test_month

        train_df = panel[train_mask].copy()
        val_df = panel[val_mask].copy()
        test_df = panel[test_mask].copy()

        if len(train_df) < 100 or len(val_df) < 10 or len(test_df) < 50:
            continue

        # Per-window rank normalization to [0,1] — same as motivation pipeline
        # (groupby yyyymm rank per column, no look-ahead)
        for col in features:
            train_df[col] = train_df.groupby("yyyymm")[col].rank(pct=True)
            val_df[col] = val_df.groupby("yyyymm")[col].rank(pct=True)
            test_df[col] = test_df.groupby("yyyymm")[col].rank(pct=True)

        # Fill NaN with 0.5 (neutral rank) — same as motivation pipeline
        train_df[features] = train_df[features].fillna(0.5)
        val_df[features] = val_df[features].fillna(0.5)
        test_df[features] = test_df[features].fillna(0.5)

        X_train = train_df[features].values
        y_train = train_df[target].values
        X_val = val_df[features].values
        y_val = val_df[target].values
        X_test = test_df[features].values

        # Drop NaN targets — same as motivation pipeline
        valid_train = ~np.isnan(y_train)
        valid_val = ~np.isnan(y_val)
        valid_test = ~np.isnan(y_test := test_df[target].values)

        if valid_train.sum() < 100 or valid_test.sum() < 30:
            continue

        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_val, y_val = X_val[valid_val], y_val[valid_val]
        X_test = X_test[valid_test]
        test_df = test_df[valid_test]
        y_test = y_test[valid_test]

        # Weights for this training window (filter by valid mask)
        w_train = None
        w_val = None
        if weights is not None:
            w_train = weights.reindex(train_df.index).fillna(1.0).values[valid_train]
            w_val = weights.reindex(val_df.index).fillna(1.0).values[valid_val]

        # Tune hyperparameters if needed (same as motivation pipeline)
        windows_since_tune += 1
        if windows_since_tune >= retune_freq:
            logger.info("  [%s] Tuning at month %d (%d/%d)", label, test_month, i+1, len(oos_months))
            tuning_model = create_model(model_name, seed=seed)
            best = tuning_model.tune_hyperparameters(
                X_train, y_train, X_val, y_val,
                sample_weight=w_train, sample_weight_val=w_val,
            )
            current_params = best
            windows_since_tune = 0
            params_all.append({"yyyymm": test_month, **best})

        # Create fresh model with tuned config (same as motivation: create_model each month)
        model = create_model(model_name, config=current_params, seed=seed)

        # Fit on training data (same as motivation: fit on train only)
        model.fit(X_train, y_train, sample_weight=w_train)

        # Predict
        preds = model.predict(X_test)

        # Store predictions
        pred_df = pd.DataFrame({
            "permno": test_df["permno"].values,
            "yyyymm": test_month,
            "prediction": preds,
        }, index=test_df.index)
        predictions_all.append(pred_df)

        # -- Feature importance: SHAP (primary) --
        shap_row = {"yyyymm": test_month}
        try:
            if model_name == "xgboost":
                shap_df = model.get_shap_values(
                    X_test, feature_names=features,
                )
                mean_abs_shap = shap_df.abs().mean()
                shap_row.update(mean_abs_shap.to_dict())
            elif model_name == "neural_network":
                shap_df = model.get_shap_values(
                    X_test,
                    X_background=X_train[:min(100, len(X_train))],
                    feature_names=features,
                    config={"use_kernel_fallback": False},
                )
                mean_abs_shap = shap_df.abs().mean()
                shap_row.update(mean_abs_shap.to_dict())
            elif model_name == "elastic_net":
                shap_df = model.get_shap_values(
                    X_test,
                    X_background=X_train,
                    feature_names=features,
                )
                mean_abs_shap = shap_df.abs().mean()
                shap_row.update(mean_abs_shap.to_dict())
        except Exception as e:
            logger.warning(
                "  [%s] SHAP failed at month %d: %s",
                label, test_month, e,
            )
        importance_all.append(shap_row)

        # -- Native importance (secondary: gain / coef / perm) --
        native_row = {"yyyymm": test_month}
        if model_name == "neural_network":
            try:
                perm_imp = model.get_permutation_importance(
                    X_test, y_test,
                    feature_names=features, seed=seed,
                )
                native_row.update(perm_imp.to_dict())
            except Exception as e:
                logger.warning("  [%s] Permutation importance failed: %s", label, e)
        else:
            imp = model.get_feature_importance(feature_names=features)
            native_row.update(imp.to_dict())
        native_importance_all.append(native_row)

        if (i + 1) % 24 == 0:
            logger.info("  [%s] Progress: %d/%d months", label, i+1, len(oos_months))

    predictions_df = pd.concat(predictions_all) if predictions_all else pd.DataFrame()
    importance_df = pd.DataFrame(importance_all) if importance_all else pd.DataFrame()
    native_importance_df = pd.DataFrame(native_importance_all) if native_importance_all else pd.DataFrame()
    params_df = pd.DataFrame(params_all) if params_all else pd.DataFrame()

    return predictions_df, importance_df, native_importance_df, params_df


# -- Main --------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config()

    # Reproducibility
    SEED = config["project"]["seed"]
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = "0"

    # Try to set TF seed if neural network
    if args.model == "neural_network":
        try:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(SEED)
        except ImportError:
            pass

    # Validate args
    if args.weights == "tc" and args.aum is None:
        print("ERROR: --aum required when --weights tc", file=sys.stderr)
        sys.exit(1)
    if args.weights != "softmax_rank" and args.softmax_lambda is not None:
        print(
            "ERROR: --softmax-lambda is only valid when --weights softmax_rank",
            file=sys.stderr,
        )
        sys.exit(1)
    aum_dollars = args.aum * 1_000_000 if args.aum else None

    softmax_lam = None
    if args.weights == "softmax_rank":
        softmax_lam = (
            args.softmax_lambda
            if args.softmax_lambda is not None
            else config.get("weighting", {}).get("softmax_rank_lambda", 2.0)
        )
        softmax_lam = float(softmax_lam)
        if not np.isfinite(softmax_lam) or softmax_lam < 0:
            print("ERROR: --softmax-lambda must be finite and non-negative", file=sys.stderr)
            sys.exit(1)
        config.setdefault("weighting", {})["softmax_rank_lambda"] = softmax_lam

    # Override OOS period for quick test
    if args.quick:
        config["training"]["oos_start"] = 202001
        logger.info("QUICK MODE: OOS 2020-01 to 2024-12")

    # -- Output paths --
    base_dir = Path(config["project"]["output_dir"]) / "formalanalysis" / "experiment"
    model_dir = base_dir / args.model
    std_dir = model_dir / "standard"

    if args.weights == "tc":
        wt_dir = model_dir / f"tc_{args.aum}m"
    elif args.weights == "softmax_rank" and args.softmax_lambda is not None:
        wt_dir = model_dir / f"softmax_rank_{_lambda_label(softmax_lam)}"
    else:
        wt_dir = model_dir / args.weights

    std_dir.mkdir(parents=True, exist_ok=True)
    wt_dir.mkdir(parents=True, exist_ok=True)

    # -- Load data (same as motivation Step 3 pipeline) --
    logger.info("Loading panel data...")
    panel = load_panel(config)

    # Load 113 features from feature_list.json (same as 07_step3_ml_diagnostics.py)
    # These are Clear Predictors that survived the 70% coverage filter in 01_process_data.py
    data_dir = get_data_dir()
    feature_list_path = data_dir / "feature_list.json"
    if not feature_list_path.exists():
        logger.error("feature_list.json not found! Run 01_process_data.py first.")
        sys.exit(1)
    with open(feature_list_path) as f:
        feature_meta = json.load(f)
    features = feature_meta["features"]
    # Verify features exist in panel
    missing = [f for f in features if f not in panel.columns]
    if missing:
        logger.warning("Features in feature_list.json but not in panel: %s", missing)
        features = [f for f in features if f in panel.columns]
    logger.info("Panel: %d rows, %d features, %d months",
                len(panel), len(features), panel["yyyymm"].nunique())

    # -- Compute weights for weighted training --
    weight_label = args.weights
    if args.weights == "softmax_rank":
        weight_label = f"softmax_rank(lambda={softmax_lam:g})"
    logger.info("Computing %s weights (aum=%s)...", weight_label,
                f"${args.aum}M" if args.aum else "N/A")
    w = compute_weights(panel, scheme=args.weights, config=config, aum=aum_dollars)

    # -- Phase 1: Standard training (M_std) --
    std_pred_path = std_dir / "predictions.parquet"
    if std_pred_path.exists():
        logger.info("M_std predictions already exist at %s - skipping", std_pred_path)
        preds_std = pd.read_parquet(std_pred_path)
    else:
        logger.info("=== Training M_std (standard) ===")
        t0 = time.time()
        preds_std, imp_std, native_std, params_std = run_rolling_training(
            panel, features, args.model,
            weights=None,  # Standard: no weights
            config=config, seed=SEED, label="std",
        )
        elapsed = time.time() - t0
        logger.info("M_std complete: %d predictions in %.1f min", len(preds_std), elapsed/60)

        # Save
        preds_std.to_parquet(std_pred_path, index=False)
        imp_std.to_csv(std_dir / "importance_shap.csv", index=False)
        if len(native_std) > 0:
            native_std.to_csv(std_dir / "importance_native.csv", index=False)
        if len(params_std) > 0:
            params_std.to_csv(std_dir / "tuned_params.csv", index=False)
        logger.info("M_std saved to %s", std_dir)

    # -- Phase 2: Weighted training (M_w) --
    logger.info("=== Training M_w (weighted: %s, aum=%s) ===",
                weight_label, f"${args.aum}M" if args.aum else "N/A")
    t0 = time.time()
    preds_wt, imp_wt, native_wt, params_wt = run_rolling_training(
        panel, features, args.model,
        weights=w,  # Weighted
        config=config, seed=SEED, label="wt",
    )
    elapsed = time.time() - t0
    logger.info("M_w complete: %d predictions in %.1f min", len(preds_wt), elapsed/60)

    # Save
    preds_wt.to_parquet(wt_dir / "predictions.parquet", index=False)
    imp_wt.to_csv(wt_dir / "importance_shap.csv", index=False)
    if len(native_wt) > 0:
        native_wt.to_csv(wt_dir / "importance_native.csv", index=False)
    if len(params_wt) > 0:
        params_wt.to_csv(wt_dir / "tuned_params.csv", index=False)

    # Symlink standard predictions into weighted dir for convenience
    std_link = wt_dir / "predictions_std.parquet"
    if not std_link.exists():
        try:
            std_link.symlink_to(std_pred_path.resolve())
        except OSError:
            pass  # Symlinks may not work on all systems

    # -- Save environment metadata --
    meta = {
        "model": args.model,
        "weights": args.weights,
        "weight_spec": wt_dir.name,
        "aum": args.aum,
        "softmax_rank_lambda": softmax_lam,
        "seed": SEED,
        "n_threads": os.environ.get("OMP_NUM_THREADS", "unknown"),
        "n_predictions_std": len(preds_std),
        "n_predictions_wt": len(preds_wt),
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat(),
        "quick": args.quick,
    }
    try:
        import xgboost; meta["xgboost_version"] = xgboost.__version__
    except ImportError:
        pass
    try:
        import sklearn; meta["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass

    with open(wt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("All done. Outputs in:\n  std: %s\n  wt:  %s", std_dir, wt_dir)


if __name__ == "__main__":
    main()
