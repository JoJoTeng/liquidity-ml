"""
Step 3: Standard ML Is Affected
=================================
Shows that standard ML (XGBoost) trained under equal weights allocates
capacity toward illiquid-stock patterns and is least accurate for liquid stocks.

Self-contained: trains XGBoost with rolling windows from the RAW panel
(signed_predictors_all_wide.csv via load_panel()). Per-window rank
normalization to [0,1] is done inside rolling_xgboost_predict(),
matching the protocol in 02_run_experiment.py — no look-ahead bias.

The feature list (113 features after 70% missing filter) is read from
feature_list.json, produced by 01_process_data.py.

Prerequisite: Run scripts/00_fetch_data.py + scripts/01_process_data.py first.

Outputs (saved to outputs/motivation/step3/{liquidity}/):
  3.1  importance_vs_illiquidity.png     Feature importance vs illiquidity-relatedness
  3.2  importance_vs_liquid_r2.png       Feature importance vs liquid-stock R²
  3.3  r2_by_quintile.png               OOS R² by liquidity quintile
  3.4  r2_by_quintile.csv               OOS R² table
  3.5  utility_weighted_r2.json          Standard vs utility-weighted R²

  Plus intermediate files:
       predictions.parquet              OOS predictions (permno, yyyymm, y_true, y_pred)
       feature_importance.csv           Gain importance per rolling window

Usage:
  python scripts/07_step3_ml_diagnostics.py
  python scripts/07_step3_ml_diagnostics.py --liquidity mcap
  python scripts/07_step3_ml_diagnostics.py --recompute  # skip training, use saved predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_data_dir, get_output_dir
from src.data.loader import load_panel
from src.analysis.motivation import (
    assign_nyse_quintiles,
    compute_implementability_weights,
    rolling_xgboost_predict,
    compute_illiquidity_relatedness,
    compute_quintile_oos_r2,
    compute_utility_weighted_r2,
    compute_univariate_liquid_r2,
    plot_importance_vs_illiquidity,
    plot_importance_vs_liquid_r2,
    plot_r2_by_quintile,
    FOCAL_CHARACTERISTICS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Step 3: Standard ML Diagnostics")
    parser.add_argument(
        "--liquidity",
        type=str,
        default="dvol",
        choices=["dvol", "mcap", "amihud", "spread"],
        help="Primary liquidity measure (default: dvol)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Skip training; use saved predictions.parquet and feature_importance.csv",
    )
    parser.add_argument(
        "--benchmark",
        default="cs",
        choices=["cs", "zero"],
        help="R² benchmark for bar chart: cs (cross-sectional mean) or zero (default: cs)",
    )
    args = parser.parse_args()

    LIQ_CONFIG = {
        "dvol": {"quintile_col": "liq_dvol_21d", "ascending": True, "label": "Dollar Volume"},
        "mcap": {"quintile_col": "liq_me_raw", "ascending": True, "label": "Market Cap"},
        "amihud": {"quintile_col": "raw_Illiquidity", "ascending": False, "label": "Amihud"},
        "spread": {"quintile_col": "raw_BidAskSpread", "ascending": False, "label": "Bid-Ask Spread"},
    }
    liq = LIQ_CONFIG[args.liquidity]

    config = load_config()
    data_dir = get_data_dir()
    output_dir = Path(get_output_dir()) / "motivation_raw" / "step3" / args.liquidity
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_dir = Path("paper/TablesNew")
    tex_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Liquidity: %s (%s)", args.liquidity, liq["label"])
    logger.info("Output: %s", output_dir)

    # ── Load RAW panel (not processed) ──────────────────
    #   Step 3 trains its own XGBoost with per-window normalization,
    #   matching the protocol in 02_run_experiment.py. Loading the raw
    #   panel avoids double rank-transform (processed_panel already has
    #   features in [0,1]). rolling_xgboost_predict() does per-window
    #   rank → [0,1] → fillna(0.5) independently for each window.
    #
    #   The diagnostic analyses (3.1, 3.2) use Spearman correlations or
    #   FM t-stats, which are rank-invariant — so raw features are fine.

    logger.info("Loading raw panel via load_panel()...")
    panel = load_panel(config)
    logger.info("Panel: %d rows, dates %d–%d", len(panel), panel["yyyymm"].min(), panel["yyyymm"].max())

    if "exchcd" not in panel.columns:
        logger.error("exchcd not in panel! Re-run 00_fetch_data.py.")
        sys.exit(1)

    # Load feature list (113 features that survived 70% missing filter)
    feature_list_path = data_dir / "feature_list.json"
    if not feature_list_path.exists():
        logger.error("feature_list.json not found! Run 01_process_data.py first.")
        sys.exit(1)

    with open(feature_list_path) as f:
        feature_meta = json.load(f)
    features = feature_meta["features"]

    # Verify features exist in raw panel
    missing_feats = [f for f in features if f not in panel.columns]
    if missing_feats:
        logger.warning("Features in feature_list.json but not in panel: %s", missing_feats)
        features = [f for f in features if f in panel.columns]
    logger.info("Features: %d", len(features))
    focal = list(FOCAL_CHARACTERISTICS.keys())

    # Assign quintiles + weights (using raw liquidity columns)
    logger.info("Assigning NYSE quintiles...")
    panel["liq_quintile"] = assign_nyse_quintiles(
        panel, liq["quintile_col"], ascending=liq["ascending"]
    )
    panel["w_tilde"] = compute_implementability_weights(
        panel, liq_col=liq["quintile_col"]
    )

    # ══════════════════════════════════════════════════════
    # Phase 1: Train XGBoost (or load saved predictions)
    # ══════════════════════════════════════════════════════
    pred_path = output_dir / "predictions.parquet"
    imp_path = output_dir / "feature_importance.csv"

    if args.recompute and pred_path.exists() and imp_path.exists():
        logger.info("Loading saved predictions and importances...")
        predictions = pd.read_parquet(pred_path)
        importances = pd.read_csv(imp_path, index_col="yyyymm")
    else:
        logger.info("=" * 60)
        logger.info("Phase 1: Rolling XGBoost training (this may take a while)...")
        predictions, importances = rolling_xgboost_predict(panel, features, config)

        # Save intermediate results
        predictions.to_parquet(pred_path, index=False)
        importances.to_csv(imp_path)
        logger.info("Saved predictions (%d rows) and importances (%d months)",
                     len(predictions), len(importances))

    # ══════════════════════════════════════════════════════
    # Phase 2: Compute diagnostics
    # ══════════════════════════════════════════════════════
    avg_importance = importances.mean()

    # ── Output 3.1: Importance vs illiquidity-relatedness ──
    logger.info("=" * 60)
    logger.info("Output 3.1: Feature importance vs illiquidity-relatedness")
    illiq_rho = compute_illiquidity_relatedness(panel, features)
    illiq_rho.to_csv(output_dir / "illiquidity_relatedness.csv")

    rho_31 = plot_importance_vs_illiquidity(
        avg_importance, illiq_rho, focal,
        output_dir / "importance_vs_illiquidity.png",
    )
    logger.info("Spearman ρ (importance vs illiquidity): %.3f", rho_31)

    # ── Output 3.2: Importance vs liquid-stock R² ──
    logger.info("=" * 60)
    logger.info("Output 3.2: Feature importance vs liquid-stock R²")
    liquid_r2 = compute_univariate_liquid_r2(panel, features, "liq_quintile")
    liquid_r2.to_csv(output_dir / "univariate_liquid_r2.csv")

    plot_importance_vs_liquid_r2(
        avg_importance, liquid_r2, focal,
        output_dir / "importance_vs_liquid_r2.png",
    )

    # ── Output 3.3 + 3.4: R² by quintile ──
    logger.info("=" * 60)
    logger.info("Output 3.3/3.4: OOS R² by liquidity quintile")
    q_r2 = compute_quintile_oos_r2(predictions, panel, "liq_quintile")
    q_r2.to_csv(output_dir / "r2_by_quintile.csv", index=False)
    logger.info("\n%s", q_r2.to_string(index=False))

    r2_col = "pooled_r2_cs" if args.benchmark == "cs" else "pooled_r2_zero"
    plot_r2_by_quintile(q_r2, output_dir / "r2_by_quintile.png", r2_col=r2_col)

    # LaTeX table — format to match document Table 3 template
    q_r2_tex = q_r2.copy()
    q_r2_tex["pooled_r2_cs"] = (q_r2_tex["pooled_r2_cs"] * 100).round(3)
    q_r2_tex["pooled_r2_zero"] = (q_r2_tex["pooled_r2_zero"] * 100).round(3)
    q_r2_tex["avg_monthly_r2_cs"] = (q_r2_tex["avg_monthly_r2_cs"] * 100).round(3)
    q_r2_tex["avg_monthly_r2_zero"] = (q_r2_tex["avg_monthly_r2_zero"] * 100).round(3)
    q_r2_tex["avg_n_month"] = q_r2_tex["avg_n_month"].round(0)

    # Format quintile labels to match document
    quintile_labels = {
        1: "Q1 (Illiquid)", 2: "Q2", 3: "Q3", 4: "Q4",
        5: "Q5 (Liquid)", "Full": "Full sample",
    }
    q_r2_tex["quintile"] = q_r2_tex["quintile"].map(quintile_labels)

    q_r2_tex = q_r2_tex.rename(columns={
        "quintile": "Quintile",
        "pooled_r2_cs": r"Pooled $R^2_{CS}$ (\%)",
        "pooled_r2_zero": r"Pooled $R^2_{0}$ (\%)",
        "avg_monthly_r2_cs": r"Avg.\ Monthly $R^2_{CS}$ (\%)",
        "avg_monthly_r2_zero": r"Avg.\ Monthly $R^2_{0}$ (\%)",
        "avg_n_month": r"Avg.\ $N$/month",
    })
    q_r2_tex.to_latex(
        tex_dir / "R2ByQuintileML.tex",
        index=False,
        escape=False,
        caption="OOS $R^2$ by Liquidity Quintile (Baseline XGBoost)",
        label="tab:r2_by_quintile_ml",
    )

    # ── Output 3.5: Utility-weighted R² ──
    logger.info("=" * 60)
    logger.info("Output 3.5: Utility-weighted R²")
    r2_results = compute_utility_weighted_r2(predictions, panel)
    logger.info(
        "CS-mean benchmark:  Standard R²: %.4f%%  Weighted R²: %.4f%%  Gap: %.4f pp",
        r2_results["r2_standard_cs"] * 100,
        r2_results["r2_weighted_cs"] * 100,
        r2_results["gap_cs"] * 100,
    )
    logger.info(
        "Zero benchmark:     Standard R²: %.4f%%  Weighted R²: %.4f%%  Gap: %.4f pp",
        r2_results["r2_standard_zero"] * 100,
        r2_results["r2_weighted_zero"] * 100,
        r2_results["gap_zero"] * 100,
    )

    with open(output_dir / "utility_weighted_r2.json", "w") as f:
        json.dump(r2_results, f, indent=2)

    # ── Save summary metadata ──
    meta = {
        "spearman_rho_importance_illiq": rho_31,
        "r2_standard_cs": r2_results["r2_standard_cs"],
        "r2_weighted_cs": r2_results["r2_weighted_cs"],
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
    logger.info("  3.3  r2_by_quintile.png")
    logger.info("  3.4  r2_by_quintile.csv (both CS-mean and zero benchmarks)")
    logger.info(
        "  3.5  CS-mean: Std R²=%.4f%%, Wtd R²=%.4f%%, Gap=%.4f pp",
        r2_results["r2_standard_cs"] * 100,
        r2_results["r2_weighted_cs"] * 100,
        r2_results["gap_cs"] * 100,
    )
    logger.info(
        "       Zero:    Std R²=%.4f%%, Wtd R²=%.4f%%, Gap=%.4f pp",
        r2_results["r2_standard_zero"] * 100,
        r2_results["r2_weighted_zero"] * 100,
        r2_results["gap_zero"] * 100,
    )


if __name__ == "__main__":
    main()
