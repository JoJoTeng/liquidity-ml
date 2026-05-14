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
  - Optional TC-target variants, where the target is
    excess_ret - BidAskSpread/2:
      * M_std_tc_target (no sample weights)
      * M_w_tc_target (same sample weights as M_w)
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
import copy
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
from src.weighting.schemes import compute_tc_for_sorting, compute_weights

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
    p.add_argument(
        "--include-tc-target",
        action="store_true",
        help=(
            "Also train target-adjusted models with target equal to "
            "excess_ret - BidAskSpread/2. Produces both no-weight and weighted "
            "versions under experiment/{model}/tc_target/."
        ),
    )
    p.add_argument(
        "--force-tc-target",
        action="store_true",
        help="Retrain TC-target artifacts even if complete cached outputs exist.",
    )
    return p.parse_args()


def _lambda_label(lam: float) -> str:
    """Compact filesystem-safe label, e.g. 3 -> lam3 and 2.5 -> lam2p5."""
    token = f"{lam:g}".replace("-", "m").replace(".", "p")
    return f"lam{token}"


def _required_training_artifacts(out_dir: Path, meta_name: str) -> list[Path]:
    """Return the complete artifact set for one training run."""
    return [
        out_dir / "predictions.parquet",
        out_dir / "importance_shap.csv",
        out_dir / "importance_native.csv",
        out_dir / "tuned_params.csv",
        out_dir / "training_diagnostics.csv",
        out_dir / meta_name,
    ]


def _train_and_save(
    *,
    panel: pd.DataFrame,
    features: list[str],
    model_name: str,
    weights: pd.Series | None,
    config: dict,
    seed: int,
    label: str,
    out_dir: Path,
    meta_name: str,
    meta: dict,
    force: bool,
    description: str,
) -> pd.DataFrame:
    """Run one training block unless a complete cached artifact set exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.parquet"
    required = _required_training_artifacts(out_dir, meta_name)

    if all(path.exists() for path in required) and not force:
        logger.info("Complete %s artifacts already exist at %s - skipping", description, out_dir)
        return pd.read_parquet(pred_path)

    missing = [path.name for path in required if not path.exists()]
    if force:
        logger.info("Retraining %s because force was requested", description)
    elif pred_path.exists():
        logger.info(
            "%s cache is incomplete (%s missing); retraining",
            description,
            ", ".join(missing),
        )

    logger.info("=== Training %s ===", description)
    t0 = time.time()
    preds, imp, native, params, diag = run_rolling_training(
        panel,
        features,
        model_name,
        weights=weights,
        config=config,
        seed=seed,
        label=label,
    )
    elapsed = time.time() - t0
    logger.info("%s complete: %d predictions in %.1f min", description, len(preds), elapsed / 60)

    preds.to_parquet(pred_path, index=False)
    imp.to_csv(out_dir / "importance_shap.csv", index=False)
    native.to_csv(out_dir / "importance_native.csv", index=False)
    params.to_csv(out_dir / "tuned_params.csv", index=False)
    diag.to_csv(out_dir / "training_diagnostics.csv", index=False)

    meta = {
        **meta,
        "n_predictions": len(preds),
        "training_diagnostics": "training_diagnostics.csv",
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / meta_name, "w") as handle:
        json.dump(meta, handle, indent=2)

    logger.info("%s saved to %s", description, out_dir)
    return preds


def _build_tc_target_panel(
    panel: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict, str]:
    """Create a panel/config pair whose target is excess return minus spread/2."""
    base_target = config["data"]["target_col"]
    target_col = f"{base_target}_minus_tc_prop"
    tc_prop = compute_tc_for_sorting(panel, aum=0.0, config=config)

    panel_tc = panel.copy()
    panel_tc[target_col] = panel_tc[base_target] - tc_prop

    config_tc = copy.deepcopy(config)
    config_tc.setdefault("data", {})["target_col"] = target_col
    return panel_tc, config_tc, target_col


def _common_meta(
    *,
    args: argparse.Namespace,
    config: dict,
    seed: int,
    target_col: str,
    target_adjustment: str | None = None,
) -> dict:
    """Metadata shared by formal training artifacts."""
    return {
        "model": args.model,
        "data_source": "processed_panel.parquet",
        "features_pre_normalized": True,
        "feature_missing_fill": 0.5,
        "training_engine": "src.training.run_rolling_training",
        "target_col": target_col,
        "target_adjustment": target_adjustment,
        "tuning_method": config["training"].get("tuning_method", "validation"),
        "cv_n_splits": config["training"].get("cv_n_splits"),
        "validation_window": config["training"].get("validation_window"),
        "skip_importance": bool(args.skip_importance),
        "seed": seed,
        "n_threads": os.environ.get("OMP_NUM_THREADS", "unknown"),
        "python_version": sys.version,
    }


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

    panel_tc_target = None
    config_tc_target = None
    tc_target_col = None
    tc_target_std_dir = None
    tc_target_wt_dir = None
    if args.include_tc_target:
        panel_tc_target, config_tc_target, tc_target_col = _build_tc_target_panel(
            panel,
            config,
        )
        tc_target_root = model_dir / "tc_target"
        tc_target_std_dir = tc_target_root / "standard"
        if wt_dir is not None:
            tc_target_wt_dir = tc_target_root / wt_dir.name
        logger.info(
            "Prepared TC-target training panel: %s = %s - BidAskSpread/2",
            tc_target_col,
            config["data"]["target_col"],
        )

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
                "target_col": config["data"]["target_col"],
                "target_adjustment": None,
                "tuning_method": config["training"].get("tuning_method", "validation"),
                "cv_n_splits": config["training"].get("cv_n_splits"),
                "validation_window": config["training"].get("validation_window"),
                "training_diagnostics": "training_diagnostics.csv",
                "skip_importance": bool(args.skip_importance),
            }, f, indent=2)
        logger.info("M_std saved to %s", std_dir)

    preds_tc_std = None
    if args.include_tc_target:
        assert panel_tc_target is not None
        assert config_tc_target is not None
        assert tc_target_col is not None
        assert tc_target_std_dir is not None
        tc_std_meta = _common_meta(
            args=args,
            config=config_tc_target,
            seed=SEED,
            target_col=tc_target_col,
            target_adjustment="minus_bid_ask_half_spread",
        )
        tc_std_meta.update({
            "weights": None,
            "weight_spec": "standard",
            "sample_weighted": False,
            "base_target_col": config["data"]["target_col"],
            "tc_target_component": "BidAskSpread/2",
        })
        preds_tc_std = _train_and_save(
            panel=panel_tc_target,
            features=features,
            model_name=args.model,
            weights=None,
            config=config_tc_target,
            seed=SEED,
            label="std_tc_target",
            out_dir=tc_target_std_dir,
            meta_name="training_meta.json",
            meta=tc_std_meta,
            force=args.force_tc_target or args.force_standard,
            description="M_std_tc_target",
        )

    if args.standard_only:
        logger.info("--standard-only requested; skipping weighted training")
        msg = f"All done. Outputs in:\n  std: {std_dir}"
        if tc_target_std_dir is not None:
            msg += f"\n  std_tc_target: {tc_target_std_dir}"
        logger.info(msg)
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
        "target_col": config["data"]["target_col"],
        "target_adjustment": None,
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

    if args.include_tc_target:
        assert panel_tc_target is not None
        assert config_tc_target is not None
        assert tc_target_col is not None
        assert tc_target_wt_dir is not None
        tc_wt_meta = _common_meta(
            args=args,
            config=config_tc_target,
            seed=SEED,
            target_col=tc_target_col,
            target_adjustment="minus_bid_ask_half_spread",
        )
        tc_wt_meta.update({
            "weights": args.weights,
            "weight_spec": wt_dir.name,
            "aum": args.aum,
            "softmax_rank_lambda": softmax_lam,
            "tc_rank_lambda": tc_rank_lam,
            "sample_weighted": True,
            "base_target_col": config["data"]["target_col"],
            "tc_target_component": "BidAskSpread/2",
            "n_predictions_std_tc_target": (
                len(preds_tc_std) if preds_tc_std is not None else None
            ),
        })
        _train_and_save(
            panel=panel_tc_target,
            features=features,
            model_name=args.model,
            weights=w,
            config=config_tc_target,
            seed=SEED,
            label="wt_tc_target",
            out_dir=tc_target_wt_dir,
            meta_name="meta.json",
            meta=tc_wt_meta,
            force=args.force_tc_target,
            description=f"M_w_tc_target ({weight_label})",
        )

    msg = f"All done. Outputs in:\n  std: {std_dir}\n  wt:  {wt_dir}"
    if tc_target_std_dir is not None:
        msg += f"\n  std_tc_target: {tc_target_std_dir}"
    if tc_target_wt_dir is not None:
        msg += f"\n  wt_tc_target:  {tc_target_wt_dir}"
    logger.info(msg)


if __name__ == "__main__":
    main()
