"""
Step 3: Standard ML Is Affected
=================================
Shows that standard ML trained under equal weights allocates
capacity toward illiquid-stock patterns and is least accurate for liquid stocks.

Uses the same canonical rolling-window training engine as the formal 2x2
experiment. Standard-model artifacts are stored under
outputs/formalanalysis/experiment/{model}/standard/, so running this script can
also populate the M_std cache used by scripts/20_formal_run_experiment.py.

The feature list (113 features after 70% missing filter) is read from
feature_list.json, produced by 01_process_data.py.
Model features are read from processed_panel.parquet and are already
rank-normalized; the rolling trainer only fills missing feature values with 0.5.

Prerequisite: Run scripts/00_fetch_data.py + scripts/01_process_data.py first.

Outputs are saved to outputs/motivation/step3/{model}/{liquidity_key}/:
  3.1  importance_vs_illiquidity.png     Feature importance vs illiquidity-relatedness
  3.2  importance_vs_liquid_r2.png       Feature importance vs liquid-stock R²
  3.3  r2_by_quintile_{cs,zero,hist}.png  OOS R² by liquidity quintile
  3.4  r2_by_quintile.csv                 OOS R² table
  3.5  utility_weighted_r2.json            Unweighted vs utility-weighted R²

  Plus intermediate files:
       predictions.parquet              OOS predictions (permno, yyyymm, y_true, y_pred)
       feature_importance.csv           Selected importance source per rolling window
       training_diagnostics.csv         NN early-stopping/tuning diagnostics when available

Usage:
  python scripts/04_motivation_step3_ml_diagnostics.py
  python scripts/04_motivation_step3_ml_diagnostics.py --model elastic_net
  python scripts/04_motivation_step3_ml_diagnostics.py --liquidity mcap
  python scripts/04_motivation_step3_ml_diagnostics.py --force-train
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Keep matplotlib/fontconfig quiet on systems where the default user cache
# directories are not writable (common in sandboxed runs and cluster jobs).
_runtime_cache = Path(tempfile.gettempdir()) / "liquidity_ml_cache"
_runtime_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_runtime_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_runtime_cache / "xdg"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_data_dir, get_output_dir
from src.data.loader import load_processed_panel
from src.training import run_rolling_training
from src.analysis.motivation import (
    assign_nyse_quintiles,
    compute_illiquidity_relatedness,
    attach_quintile_benchmarks,
    compute_quintile_oos_r2,
    compute_utility_weighted_r2,
    compute_univariate_liquid_r2,
    plot_importance_vs_illiquidity,
    plot_importance_vs_liquid_r2,
    plot_r2_by_quintile,
    ensure_motivation_weight_column,
    get_motivation_liquidity_choices,
    get_motivation_liquidity_config,
    get_motivation_liquidity_key,
    FOCAL_CHARACTERISTICS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _step3_output_dir(base_output: Path, model_name: str, liquidity_key: str) -> Path:
    """Return the model-namespaced Step 3 diagnostics path."""
    step3_root = base_output / "motivation" / "step3"
    return step3_root / model_name / liquidity_key


def _standard_artifact_paths(std_dir: Path) -> dict[str, Path]:
    return {
        "predictions": std_dir / "predictions.parquet",
        "importance_shap": std_dir / "importance_shap.csv",
        "importance_native": std_dir / "importance_native.csv",
        "tuned_params": std_dir / "tuned_params.csv",
        "training_meta": std_dir / "training_meta.json",
        "training_diagnostics": std_dir / "training_diagnostics.csv",
    }


def _standard_artifacts_complete(paths: dict[str, Path]) -> bool:
    required = {
        name: path
        for name, path in paths.items()
        if name != "training_diagnostics"
    }
    return all(path.exists() for path in required.values())


def _to_step3_predictions(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Convert formal prediction output to Step 3's y_true/y_pred schema."""
    if {"y_true", "y_pred"}.issubset(preds.columns):
        return preds[["permno", "yyyymm", "y_true", "y_pred"]].copy()

    pred = preds[["permno", "yyyymm", "prediction"]].rename(
        columns={"prediction": "y_pred"}
    )
    target = panel[["permno", "yyyymm", target_col]].rename(
        columns={target_col: "y_true"}
    )
    pred = pred.merge(target, on=["permno", "yyyymm"], how="left")
    return pred[["permno", "yyyymm", "y_true", "y_pred"]].dropna(
        subset=["y_true", "y_pred"]
    )


def _importance_matrix(
    importance_df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Return a yyyymm-indexed feature-importance matrix."""
    out = importance_df.copy()
    if "yyyymm" in out.columns:
        out = out.set_index("yyyymm")
    keep = [f for f in features if f in out.columns]
    return out[keep]


def main():
    parser = argparse.ArgumentParser(description="Step 3: Standard ML Diagnostics")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["elastic_net", "xgboost", "neural_network"],
        help="Standard model to diagnose (default: xgboost)",
    )
    parser.add_argument(
        "--liquidity",
        type=str,
        default="dvol",
        choices=get_motivation_liquidity_choices(),
        help="Primary liquidity measure (default: dvol)",
    )
    parser.add_argument(
        "--aum",
        type=float,
        default=500.0,
        help="AUM in $M for --liquidity tc (default: 500)",
    )
    parser.add_argument(
        "--importance",
        type=str,
        default="shap",
        choices=["native", "shap"],
        help="Importance source for scatter diagnostics (default: shap)",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain the formal standard model even if cached artifacts exist",
    )
    args = parser.parse_args()

    liq = get_motivation_liquidity_config(args.liquidity)

    config = load_config()
    data_dir = get_data_dir()
    base_output = Path(get_output_dir())
    liquidity_key = get_motivation_liquidity_key(args.liquidity, args.aum)
    output_dir = _step3_output_dir(base_output, args.model, liquidity_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Raw LaTeX tables are data artifacts; the curated paper tables live in
    # paper/TablesNew and are built by scripts/build_paper_tables.py.
    tex_dir = output_dir / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model: %s", args.model)
    logger.info("Liquidity: %s (%s)", args.liquidity, liq["label"])
    if args.liquidity == "tc":
        logger.info("TC AUM: $%.0fM", args.aum)
    logger.info("Output: %s", output_dir)

    # Load feature list (113 features that survived 70% missing filter)
    feature_list_path = data_dir / "feature_list.json"
    if not feature_list_path.exists():
        logger.error("feature_list.json not found! Run scripts/01_process_data.py first.")
        sys.exit(1)

    with open(feature_list_path) as f:
        feature_meta = json.load(f)
    features = feature_meta["features"]

    # ── Load processed panel through the same path used by the formal experiment ──
    logger.info("Loading processed panel...")
    panel = load_processed_panel(data_dir)
    logger.info(
        "Panel: %d rows, dates %d–%d",
        len(panel), panel["yyyymm"].min(), panel["yyyymm"].max(),
    )

    if "exchcd" not in panel.columns:
        logger.error("exchcd not in panel! Re-run scripts/00_fetch_data.py.")
        sys.exit(1)

    # Verify features exist in panel
    missing_feats = [f for f in features if f not in panel.columns]
    if missing_feats:
        logger.warning("Features in feature_list.json but not in panel: %s", missing_feats)
        features = [f for f in features if f in panel.columns]
    logger.info("Features: %d", len(features))

    # processed_panel.parquet already has selected model features normalized.
    # Raw liq_* columns remain unchanged for weights, TC, and quintiles.
    panel_for_diagnostics = panel.copy()
    focal = list(FOCAL_CHARACTERISTICS.keys())

    # Assign quintiles + weights (using raw liquidity columns)
    logger.info("Assigning NYSE quintiles...")
    selected_weight_col = ensure_motivation_weight_column(
        panel_for_diagnostics, args.liquidity, config=config, aum_millions=args.aum
    )
    panel_for_diagnostics["liq_quintile"] = assign_nyse_quintiles(
        panel_for_diagnostics, liq["quintile_col"], ascending=liq["ascending"]
    )
    panel_for_diagnostics["w_tilde"] = panel_for_diagnostics[selected_weight_col]

    # Save quintile lookup for reproducibility (used by step3d/3e as well)
    panel_for_diagnostics[["permno", "yyyymm", "liq_quintile", "w_tilde"]].to_csv(
        output_dir / "quintile_lookup.csv", index=False
    )
    logger.info("Saved quintile_lookup.csv (%d rows)", len(panel_for_diagnostics))

    # ══════════════════════════════════════════════════════
    # Phase 1: Train/load the canonical formal standard model
    # ══════════════════════════════════════════════════════
    pred_path = output_dir / "predictions.parquet"
    imp_path = output_dir / "feature_importance.csv"
    tuned_params_path = output_dir / "tuned_params.csv"
    training_diagnostics_path = output_dir / "training_diagnostics.csv"

    std_dir = base_output / "formalanalysis" / "experiment" / args.model / "standard"
    std_dir.mkdir(parents=True, exist_ok=True)
    std_paths = _standard_artifact_paths(std_dir)

    if not args.force_train and _standard_artifacts_complete(std_paths):
        logger.info("Loading canonical formal M_std artifacts from %s", std_dir)
        preds_std = pd.read_parquet(std_paths["predictions"])
        imp_std = pd.read_csv(std_paths["importance_shap"])
        native_std = pd.read_csv(std_paths["importance_native"])
        params_std = pd.read_csv(std_paths["tuned_params"])
        try:
            diag_std = (
                pd.read_csv(std_paths["training_diagnostics"])
                if std_paths["training_diagnostics"].exists()
                else pd.DataFrame()
            )
        except pd.errors.EmptyDataError:
            # Tolerate a zero-byte/header-less diagnostics file (older cache runs
            # wrote an empty frame); diagnostics are optional for this script.
            logger.warning(
                "training_diagnostics.csv is empty; continuing without diagnostics"
            )
            diag_std = pd.DataFrame()
    else:
        if args.force_train:
            logger.info("--force-train requested; retraining formal M_std")
        else:
            missing_std = [name for name, path in std_paths.items() if not path.exists()]
            logger.info("Formal M_std cache incomplete (%s); training", ", ".join(missing_std))
        logger.info("=" * 60)
        logger.info("Phase 1: Rolling %s standard training (this may take a while)...", args.model)
        seed = config["project"]["seed"]
        preds_std, imp_std, native_std, params_std, diag_std = run_rolling_training(
            panel, features, args.model,
            weights=None,
            config=config, seed=seed, label="std",
        )
        preds_std.to_parquet(std_paths["predictions"], index=False)
        imp_std.to_csv(std_paths["importance_shap"], index=False)
        native_std.to_csv(std_paths["importance_native"], index=False)
        params_std.to_csv(std_paths["tuned_params"], index=False)
        diag_std.to_csv(std_paths["training_diagnostics"], index=False)
        with open(std_paths["training_meta"], "w") as f:
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
            }, f, indent=2)
        logger.info("Saved canonical formal M_std artifacts to %s", std_dir)

    importance_source = imp_std if args.importance == "shap" else native_std
    predictions = _to_step3_predictions(
        preds_std, panel_for_diagnostics, config["data"]["target_col"]
    )
    importances = _importance_matrix(importance_source, features)
    if importances.empty:
        logger.warning("Selected %s importance matrix is empty", args.importance)

    # Save Step 3-local diagnostic copies in the schema used by this script.
    # Step 3d/3e should read this model-namespaced path after their cleanup.
    predictions.to_parquet(pred_path, index=False)
    importances.to_csv(imp_path)
    params_std.to_csv(tuned_params_path, index=False)
    if not diag_std.empty:
        diag_std.to_csv(training_diagnostics_path, index=False)
    logger.info(
        "Prepared Step 3 artifacts: predictions=%d rows, importances=%d months, "
        "params=%d windows, diagnostics=%d rows",
        len(predictions), len(importances), len(params_std), len(diag_std),
    )

    # ══════════════════════════════════════════════════════
    # Phase 2: Compute diagnostics
    # ══════════════════════════════════════════════════════
    avg_importance = importances.mean()

    # ── Output 3.2: Importance vs illiquidity-relatedness ──
    logger.info("=" * 60)
    logger.info("Output 3.2: Feature importance vs illiquidity-relatedness")
    illiq_rho = compute_illiquidity_relatedness(
        panel_for_diagnostics, features, liq_col=liq["quintile_col"]
    )
    illiq_rho.to_csv(output_dir / "illiquidity_relatedness.csv")

    rho_31 = plot_importance_vs_illiquidity(
        avg_importance, illiq_rho, focal,
        output_dir / "importance_vs_illiquidity.png",
        importance_label=f"average {args.importance} importance ({args.model})",
    )
    logger.info("Spearman ρ (mean importance vs illiquidity relatedness): %.3f", rho_31)

    # ── Output 3.1: Importance vs liquid-stock R² ──
    logger.info("=" * 60)
    logger.info("Output 3.1: Feature importance vs liquid-stock R²")
    liquid_r2 = compute_univariate_liquid_r2(panel_for_diagnostics, features, "liq_quintile")
    liquid_r2.to_csv(output_dir / "univariate_liquid_r2.csv")

    plot_importance_vs_liquid_r2(
        avg_importance, liquid_r2, focal,
        output_dir / "importance_vs_liquid_r2.png",
        importance_label=f"average {args.importance} importance ({args.model})",
    )

    # ── Output 3.3: Illiquidity cluster importance aggregate ──
    logger.info("=" * 60)
    logger.info("Output 3.3: Illiquidity cluster importance aggregate")
    illiq_threshold = 0.5
    illiq_mask = illiq_rho.abs() > illiq_threshold
    total_importance = avg_importance.sum()
    illiq_importance = avg_importance[illiq_mask].sum()
    illiq_imp_share = illiq_importance / total_importance if total_importance > 0 else 0

    total_liquid_r2 = liquid_r2.abs().sum()
    illiq_liquid_r2 = liquid_r2[illiq_mask].abs().sum()
    illiq_r2_share = illiq_liquid_r2 / total_liquid_r2 if total_liquid_r2 > 0 else 0

    cluster_summary = pd.DataFrame([{
        "threshold": illiq_threshold,
        "n_illiq_features": int(illiq_mask.sum()),
        "n_total_features": len(illiq_mask),
        "illiq_importance_share": illiq_imp_share,
        "illiq_liquid_r2_share": illiq_r2_share,
        "ratio": illiq_imp_share / illiq_r2_share if illiq_r2_share > 0 else np.nan,
    }])
    cluster_summary.to_csv(output_dir / "illiquidity_cluster_summary.csv", index=False)
    logger.info(
        "Illiquidity cluster (|rho|>%.1f): %d features, %.1f%% of importance, %.1f%% of liquid R2",
        illiq_threshold, illiq_mask.sum(),
        illiq_imp_share * 100, illiq_r2_share * 100,
    )

    # ── Output 3.4 + 3.5: R² by quintile ──
    logger.info("=" * 60)
    logger.info("Output 3.4/3.5: OOS R² by liquidity quintile")
    # Historical mean benchmark: for each OOS month t, use the mean of
    # stock i's returns from the start of training window to t-1.
    # Window length = train_window + val_window (132 months), matching
    # the information set available to the ML model at prediction time.
    train_window = config["training"]["train_window"]       # 120
    val_window = config["training"]["validation_window"]    # 12
    hist_window = train_window + val_window                 # 132
    logger.info("Computing historical mean benchmark (%d-month rolling window per stock)...", hist_window)

    q_r2 = compute_quintile_oos_r2(
        predictions,
        panel_for_diagnostics,
        "liq_quintile",
        return_col=config["data"]["target_col"],
        hist_window=hist_window,
    )
    pred_with_q = attach_quintile_benchmarks(
        predictions,
        panel_for_diagnostics,
        "liq_quintile",
        return_col=config["data"]["target_col"],
        hist_window=hist_window,
        extra_cols=["w_tilde"],
    )

    # Save predictions with quintile + weights + all benchmarks for reproducibility
    pred_with_q.to_csv(output_dir / "predictions_with_quintile.csv", index=False)
    logger.info("Saved predictions_with_quintile.csv (%d rows)", len(pred_with_q))

    # Reorder columns: zero, CS, hist (matching paper Table 3)
    col_order = ["quintile", "pooled_r2_zero", "pooled_r2_cs", "pooled_r2_hist",
                 "avg_monthly_r2_zero", "avg_monthly_r2_cs", "avg_monthly_r2_hist", "avg_n_month"]
    q_r2 = q_r2[[c for c in col_order if c in q_r2.columns]]
    q_r2.to_csv(output_dir / "r2_by_quintile.csv", index=False)
    logger.info("R² by quintile (all 3 benchmarks):\n%s",
                q_r2[["quintile", "pooled_r2_zero", "pooled_r2_cs", "pooled_r2_hist"]].to_string(index=False))

    # Save both benchmark versions
    plot_r2_by_quintile(q_r2, output_dir / "r2_by_quintile_cs.png", r2_col="pooled_r2_cs")
    plot_r2_by_quintile(q_r2, output_dir / "r2_by_quintile_zero.png", r2_col="pooled_r2_zero")
    plot_r2_by_quintile(q_r2, output_dir / "r2_by_quintile_hist.png", r2_col="pooled_r2_hist")

    # ── Table 3: R² by quintile ──
    quintile_labels = {
        1: "Q1 (Illiquid)", 2: "Q2", 3: "Q3", 4: "Q4",
        5: "Q5 (Liquid)", "Full": "Full sample",
    }
    q_r2_table = q_r2[["quintile", "pooled_r2_zero", "pooled_r2_cs", "pooled_r2_hist", "avg_n_month"]].copy()
    q_r2_table["quintile"] = q_r2_table["quintile"].map(quintile_labels)
    q_r2_table["pooled_r2_zero"] = (q_r2_table["pooled_r2_zero"] * 100).round(3)
    q_r2_table["pooled_r2_cs"] = (q_r2_table["pooled_r2_cs"] * 100).round(3)
    q_r2_table["pooled_r2_hist"] = (q_r2_table["pooled_r2_hist"] * 100).round(3)
    q_r2_table["avg_n_month"] = q_r2_table["avg_n_month"].round(0).astype("Int64")
    # CSV: human-readable column names
    q_r2_csv = q_r2_table.rename(columns={
        "quintile": "Quintile",
        "pooled_r2_zero": "R2_zero (%)",
        "pooled_r2_cs": "R2_CS (%)",
        "pooled_r2_hist": "R2_hist (%)",
        "avg_n_month": "Avg N/month",
    })
    q_r2_csv.to_csv(output_dir / "table3_r2_by_quintile.csv", index=False)
    logger.info("Table 3 (Output 3.5):\n%s", q_r2_csv.to_string(index=False))

    # LaTeX: math notation
    q_r2_tex = q_r2_table.rename(columns={
        "quintile": "Quintile",
        "pooled_r2_zero": r"$R^2_{\mathrm{zero}}$ (\%)",
        "pooled_r2_cs": r"$R^2_{\mathrm{CS}}$ (\%)",
        "pooled_r2_hist": r"$R^2_{\mathrm{hist}}$ (\%)",
        "avg_n_month": r"Avg.\ $N$/month",
    })
    q_r2_tex.to_latex(
        tex_dir / "R2ByQuintileML.tex",
        index=False, escape=False,
        caption="OOS $R^2$ by Liquidity Quintile",
        label="tab:r2_by_quintile",
    )

    # ── Utility-weighted evaluation R² ──
    logger.info("=" * 60)
    logger.info("Utility-weighted evaluation R²")
    r2_results = compute_utility_weighted_r2(predictions, panel_for_diagnostics)
    logger.info(
        "Zero benchmark:     Unweighted R²: %.4f%%  Utility-weighted R²: %.4f%%  Gap: %.4f pp",
        r2_results["r2_standard_zero"] * 100,
        r2_results["r2_weighted_zero"] * 100,
        r2_results["gap_zero"] * 100,
    )

    if args.liquidity != "mcap":
        # Market-cap weighted R² robustness check from the motivation document.
        logger.info("Computing market-cap weighted R² robustness...")
        panel_mcap = panel_for_diagnostics.copy()
        ensure_motivation_weight_column(panel_mcap, "mcap", config=config)
        r2_mcap = compute_utility_weighted_r2(predictions, panel_mcap, w_col="w_tilde_mcap")
        r2_results["r2_weighted_mcap_zero"] = r2_mcap["r2_weighted_zero"]
        r2_results["gap_mcap_zero"] = (
            r2_results["r2_standard_zero"] - r2_mcap["r2_weighted_zero"]
        )
        logger.info(
            "Market-cap weighted robustness: Zero R²=%.4f%%, Gap=%.4f pp",
            r2_mcap["r2_weighted_zero"] * 100,
            r2_results["gap_mcap_zero"] * 100,
        )

    with open(output_dir / "utility_weighted_r2.json", "w") as f:
        json.dump(r2_results, f, indent=2)

    # ── Table 4: Unweighted vs utility-weighted evaluation R² ──
    primary_weight_label = liq["label"]
    table4_rows = [
        {
            "Metric": "Unweighted evaluation",
            "Pooled R² (%)": round(r2_results["r2_standard_zero"] * 100, 3),
        },
        {
            "Metric": f"Utility-weighted evaluation ({primary_weight_label})",
            "Pooled R² (%)": round(r2_results["r2_weighted_zero"] * 100, 3),
        },
    ]
    if args.liquidity != "mcap":
        table4_rows.append({
            "Metric": "Utility-weighted evaluation (market-cap)",
            "Pooled R² (%)": round(r2_results["r2_weighted_mcap_zero"] * 100, 3),
        })
    table4_rows.append({
        "Metric": f"Gap (unweighted - {primary_weight_label})",
        "Pooled R² (%)": round(r2_results["gap_zero"] * 100, 3),
    })
    table4 = pd.DataFrame(table4_rows)
    table4.to_csv(output_dir / "table4_utility_weighted_r2.csv", index=False)
    table4.to_latex(
        tex_dir / "UtilityWeightedR2.tex",
        index=False, escape=False,
        caption="Unweighted vs.\\ Utility-Weighted Evaluation OOS $R^2$",
        label="tab:utility_weighted_r2",
    )
    logger.info("Table 4:\n%s", table4.to_string(index=False))

    # ── Save summary metadata ──
    meta = {
        "model": args.model,
        "importance": args.importance,
        "liquidity": args.liquidity,
        "liquidity_key": liquidity_key,
        "selected_weight_col": selected_weight_col,
        "formal_standard_dir": str(std_dir),
        "data_source": "processed_panel.parquet",
        "features_pre_normalized": True,
        "feature_missing_fill": 0.5,
        "tuning_method": config["training"].get("tuning_method", "validation"),
        "cv_n_splits": config["training"].get("cv_n_splits"),
        "validation_window": config["training"].get("validation_window"),
        "spearman_rho_importance_illiq": rho_31,
        "r2_standard_zero": r2_results["r2_standard_zero"],
        "r2_weighted_zero": r2_results["r2_weighted_zero"],
        "n_oos_months": len(importances),
        "n_predictions": len(predictions),
        "n_features": len(features),
    }
    with open(output_dir / "step3_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 3 complete. Outputs saved to %s", output_dir)
    logger.info("  3.1  importance_vs_illiquidity.png (ρ=%.3f)", rho_31)
    logger.info("  3.2  importance_vs_liquid_r2.png")
    logger.info("  3.3  r2_by_quintile_{cs,zero,hist}.png")
    logger.info("  3.4  r2_by_quintile.csv (zero, CS-mean, and historical benchmarks)")
    logger.info(
        "  3.5  Zero utility R²: unweighted=%.4f%%, utility-weighted=%.4f%%, Gap=%.4f pp",
        r2_results["r2_standard_zero"] * 100,
        r2_results["r2_weighted_zero"] * 100,
        r2_results["gap_zero"] * 100,
    )


if __name__ == "__main__":
    main()
