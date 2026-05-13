"""
20 - Formal Model Training (LiquidityML v3)
===========================================
Rolling-window ML training for the formal analysis.

Each invocation trains ONE specification:
  - One model (elastic_net / xgboost / neural_network)
  - One weight family + AUM/lambda when needed
    (dolvol, softmax_rank at a chosen lambda, tc at a given AUM,
     or tc_rank at a chosen lambda and AUM)

Produces:
  - M_std (standard training) predictions + importance
  - M_w  (weighted training)  predictions + importance
  - Tuned hyperparameters per retune window

M_std is shared across weight families. If the complete standard artifact set
already exists in outputs/formalanalysis/experiment/{model}/standard/, training
is skipped. This avoids redundant computation when running multiple weighted
jobs per model, and lets motivation Step 3 pre-populate the same standard cache.

Training reads data/processed_panel.parquet. The feature columns are already
rank-normalized by scripts/01_process_data.py; the shared rolling trainer only
fills missing feature values with the neutral value 0.5.

HPC usage:
  python scripts/20_formal_run_experiment.py --model neural_network --standard-only --force-standard --skip-importance
  python scripts/20_formal_run_experiment.py --model xgboost --weights dolvol
  python scripts/20_formal_run_experiment.py --model xgboost --weights softmax_rank --softmax-lambda 2
  python scripts/20_formal_run_experiment.py --model xgboost --weights softmax_rank --softmax-lambda 3
  python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 10
  python scripts/20_formal_run_experiment.py --model xgboost --weights tc --aum 500
  python scripts/20_formal_run_experiment.py --model xgboost --weights tc_rank --tc-rank-lambda 3 --aum 500
  python scripts/20_formal_run_experiment.py --model elastic_net --weights dolvol
  ...

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
from src.data.loader import load_processed_panel
from src.training import run_rolling_training
from src.weighting.schemes import compute_weights

# -- Logging -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("20_formal_experiment")


# -- CLI ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train standard and weighted formal models for one specification"
    )
    p.add_argument("--model", required=True,
                    choices=["elastic_net", "xgboost", "neural_network"])
    p.add_argument("--weights", default=None,
                    choices=["dolvol", "softmax_rank", "tc", "tc_rank"])
    p.add_argument("--aum", type=int, default=None,
                    choices=[10, 100, 500, 1000],
                    help=(
                        "AUM in $M for TC or TC-rank weights. Choices match "
                        "transaction_costs.aum_scenarios in config."
                    ))
    p.add_argument("--softmax-lambda", type=float, default=None,
                    help=(
                        "Softmax-rank lambda. Required when "
                        "--weights softmax_rank; must be in "
                        "weighting.softmax_rank_lambdas."
                    ))
    p.add_argument("--tc-rank-lambda", type=float, default=None,
                    help=(
                        "TC-rank softmax lambda. Required when "
                        "--weights tc_rank; must be in weighting.tc_rank_lambdas."
                    ))
    p.add_argument(
        "--standard-only",
        action="store_true",
        help="Train/load only M_std and skip weighted training.",
    )
    p.add_argument(
        "--force-standard",
        action="store_true",
        help="Retrain M_std even if cached standard artifacts already exist.",
    )
    p.add_argument(
        "--skip-importance",
        action="store_true",
        help=(
            "Skip SHAP and native/permutation importance. Use this for fast "
            "training diagnostics when only predictions and loss logs are needed."
        ),
    )
    return p.parse_args()


def _lambda_label(lam: float) -> str:
    """Compact filesystem-safe label, e.g. 3 -> lam3 and 2.5 -> lam2p5."""
    token = f"{lam:g}".replace("-", "m").replace(".", "p")
    return f"lam{token}"


# -- Main --------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config()
    if args.skip_importance:
        config.setdefault("training", {})["skip_importance"] = True

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
    if not args.standard_only and args.weights is None:
        print("ERROR: --weights is required unless --standard-only is set", file=sys.stderr)
        sys.exit(1)
    if args.weights in {"tc", "tc_rank"} and args.aum is None:
        print(f"ERROR: --aum required when --weights {args.weights}", file=sys.stderr)
        sys.exit(1)
    if args.weights == "softmax_rank" and args.softmax_lambda is None:
        print(
            "ERROR: --softmax-lambda required when --weights softmax_rank",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.weights != "softmax_rank" and args.softmax_lambda is not None:
        print(
            "ERROR: --softmax-lambda is only valid when --weights softmax_rank",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.weights == "tc_rank" and args.tc_rank_lambda is None:
        print(
            "ERROR: --tc-rank-lambda required when --weights tc_rank",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.weights != "tc_rank" and args.tc_rank_lambda is not None:
        print(
            "ERROR: --tc-rank-lambda is only valid when --weights tc_rank",
            file=sys.stderr,
        )
        sys.exit(1)
    aum_dollars = args.aum * 1_000_000 if args.aum else None

    softmax_lam = None
    tc_rank_lam = None
    if args.weights == "softmax_rank":
        softmax_lam = float(args.softmax_lambda)
        if not np.isfinite(softmax_lam) or softmax_lam < 0:
            print("ERROR: --softmax-lambda must be finite and non-negative", file=sys.stderr)
            sys.exit(1)
        allowed_lams = config.get("weighting", {}).get("softmax_rank_lambdas", [])
        if allowed_lams and softmax_lam not in {float(lam) for lam in allowed_lams}:
            choices = ", ".join(f"{float(lam):g}" for lam in allowed_lams)
            print(
                f"ERROR: --softmax-lambda must be one of: {choices}",
                file=sys.stderr,
            )
            sys.exit(1)
        config.setdefault("weighting", {})["softmax_rank_lambda"] = softmax_lam
    elif args.weights == "tc_rank":
        tc_rank_lam = float(args.tc_rank_lambda)
        if not np.isfinite(tc_rank_lam) or tc_rank_lam < 0:
            print("ERROR: --tc-rank-lambda must be finite and non-negative", file=sys.stderr)
            sys.exit(1)
        allowed_lams = config.get("weighting", {}).get("tc_rank_lambdas", [])
        if allowed_lams and tc_rank_lam not in {float(lam) for lam in allowed_lams}:
            choices = ", ".join(f"{float(lam):g}" for lam in allowed_lams)
            print(
                f"ERROR: --tc-rank-lambda must be one of: {choices}",
                file=sys.stderr,
            )
            sys.exit(1)
        config.setdefault("weighting", {})["tc_rank_lambda"] = tc_rank_lam

    # -- Output paths --
    base_dir = Path(config["project"]["output_dir"]) / "formalanalysis" / "experiment"
    model_dir = base_dir / args.model
    std_dir = model_dir / "standard"

    if args.weights == "tc":
        wt_dir = model_dir / f"tc_{args.aum}m"
    elif args.weights == "tc_rank":
        wt_dir = model_dir / f"tc_rank_{_lambda_label(tc_rank_lam)}_{args.aum}m"
    elif args.weights == "softmax_rank":
        wt_dir = model_dir / f"softmax_rank_{_lambda_label(softmax_lam)}"
    elif args.weights is not None:
        wt_dir = model_dir / args.weights
    else:
        wt_dir = None

    std_dir.mkdir(parents=True, exist_ok=True)
    if wt_dir is not None:
        wt_dir.mkdir(parents=True, exist_ok=True)

    data_dir = get_data_dir()

    # -- Load data (same as motivation Step 3 pipeline) --
    logger.info("Loading processed panel data...")
    panel = load_processed_panel(data_dir)

    # Load 113 features from feature_list.json (same as 04_motivation_step3_ml_diagnostics.py)
    # These are Clear Predictors that survived the 70% coverage filter in 01_process_data.py
    feature_list_path = data_dir / "feature_list.json"
    if not feature_list_path.exists():
        logger.error("feature_list.json not found! Run scripts/01_process_data.py first.")
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

    # -- Phase 1: Standard training (M_std) --
    std_pred_path = std_dir / "predictions.parquet"
    std_diagnostics_path = std_dir / "training_diagnostics.csv"
    std_required = [
        std_pred_path,
        std_dir / "importance_shap.csv",
        std_dir / "importance_native.csv",
        std_dir / "tuned_params.csv",
        std_dir / "training_meta.json",
    ]
    if all(p.exists() for p in std_required) and not args.force_standard:
        logger.info("Complete M_std artifacts already exist at %s - skipping", std_dir)
        preds_std = pd.read_parquet(std_pred_path)
    else:
        missing_std = [p.name for p in std_required if not p.exists()]
        if args.force_standard:
            logger.info("--force-standard requested; retraining standard model")
        elif std_pred_path.exists():
            logger.info(
                "M_std cache is incomplete (%s missing); retraining standard model",
                ", ".join(missing_std),
            )
        logger.info("=== Training M_std (standard) ===")
        t0 = time.time()
        preds_std, imp_std, native_std, params_std, diag_std = run_rolling_training(
            panel, features, args.model,
            weights=None,  # Standard: no weights
            config=config, seed=SEED, label="std",
        )
        elapsed = time.time() - t0
        logger.info("M_std complete: %d predictions in %.1f min", len(preds_std), elapsed/60)

        # Save
        preds_std.to_parquet(std_pred_path, index=False)
        imp_std.to_csv(std_dir / "importance_shap.csv", index=False)
        native_std.to_csv(std_dir / "importance_native.csv", index=False)
        params_std.to_csv(std_dir / "tuned_params.csv", index=False)
        diag_std.to_csv(std_diagnostics_path, index=False)
        with open(std_dir / "training_meta.json", "w") as f:
            json.dump({
                "model": args.model,
                "data_source": "processed_panel.parquet",
                "features_pre_normalized": True,
                "feature_missing_fill": 0.5,
                "training_engine": "src.training.run_rolling_training",
                "tuning_method": config["training"].get("tuning_method", "validation"),
                "cv_n_splits": config["training"].get("cv_n_splits"),
                "validation_window": config["training"].get("validation_window"),
                "training_diagnostics": "training_diagnostics.csv",
                "skip_importance": bool(args.skip_importance),
            }, f, indent=2)
        logger.info("M_std saved to %s", std_dir)

    if args.standard_only:
        logger.info("--standard-only requested; skipping weighted training")
        logger.info("All done. Outputs in:\n  std: %s", std_dir)
        return

    # -- Compute weights for weighted training --
    assert args.weights is not None
    assert wt_dir is not None
    weight_label = args.weights
    if args.weights == "softmax_rank":
        weight_label = f"softmax_rank(lambda={softmax_lam:g})"
    elif args.weights == "tc_rank":
        weight_label = f"tc_rank(lambda={tc_rank_lam:g})"
    logger.info("Computing %s weights (aum=%s)...", weight_label,
                f"${args.aum}M" if args.aum else "N/A")
    w = compute_weights(panel, scheme=args.weights, config=config, aum=aum_dollars)

    # -- Phase 2: Weighted training (M_w) --
    logger.info("=== Training M_w (weighted: %s, aum=%s) ===",
                weight_label, f"${args.aum}M" if args.aum else "N/A")
    t0 = time.time()
    preds_wt, imp_wt, native_wt, params_wt, diag_wt = run_rolling_training(
        panel, features, args.model,
        weights=w,  # Weighted
        config=config, seed=SEED, label="wt",
    )
    elapsed = time.time() - t0
    logger.info("M_w complete: %d predictions in %.1f min", len(preds_wt), elapsed/60)

    # Save
    preds_wt.to_parquet(wt_dir / "predictions.parquet", index=False)
    imp_wt.to_csv(wt_dir / "importance_shap.csv", index=False)
    native_wt.to_csv(wt_dir / "importance_native.csv", index=False)
    params_wt.to_csv(wt_dir / "tuned_params.csv", index=False)
    diag_wt.to_csv(wt_dir / "training_diagnostics.csv", index=False)

    # -- Save environment metadata --
    meta = {
        "model": args.model,
        "weights": args.weights,
        "weight_spec": wt_dir.name,
        "aum": args.aum,
        "softmax_rank_lambda": softmax_lam,
        "tc_rank_lambda": tc_rank_lam,
        "data_source": "processed_panel.parquet",
        "features_pre_normalized": True,
        "feature_missing_fill": 0.5,
        "tuning_method": config["training"].get("tuning_method", "validation"),
        "cv_n_splits": config["training"].get("cv_n_splits"),
        "validation_window": config["training"].get("validation_window"),
        "seed": SEED,
        "n_threads": os.environ.get("OMP_NUM_THREADS", "unknown"),
        "n_predictions_std": len(preds_std),
        "n_predictions_wt": len(preds_wt),
        "skip_importance": bool(args.skip_importance),
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat(),
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
