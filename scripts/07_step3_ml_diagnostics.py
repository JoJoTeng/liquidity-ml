"""
Step 3: Standard ML Is Affected
=================================
Shows that standard ML (XGBoost) trained under equal weights allocates
capacity toward illiquid-stock patterns and is least accurate for liquid stocks.

Self-contained: trains XGBoost with rolling windows from processed_panel.parquet.
Independent of 02_run_experiment.py.

Prerequisite: Run scripts/01_process_data.py first.

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
    output_dir = Path(get_output_dir()) / "motivation" / "step3" / args.liquidity
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_dir = Path("paper/TablesNew")
    tex_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Liquidity: %s (%s)", args.liquidity, liq["label"])
    logger.info("Output: %s", output_dir)

    # ── Load processed panel ──────────────────────────────
    panel_path = data_dir / "processed_panel.parquet"
    if not panel_path.exists():
        logger.error("processed_panel.parquet not found! Run 01_process_data.py first.")
        sys.exit(1)

    logger.info("Loading processed panel...")
    panel = pd.read_parquet(panel_path)
    logger.info("Panel: %d rows, dates %d–%d", len(panel), panel["yyyymm"].min(), panel["yyyymm"].max())

    # Load feature list
    with open(data_dir / "feature_list.json") as f:
        feature_meta = json.load(f)
    features = feature_meta["features"]
    logger.info("Features: %d", len(features))
    focal = list(FOCAL_CHARACTERISTICS.keys())

    # Assign quintiles + weights
    logger.info("Assigning NYSE quintiles...")
    panel["liq_quintile"] = assign_nyse_quintiles(
        panel, liq["quintile_col"], ascending=liq["ascending"]
    )
    panel["w_tilde"] = compute_implementability_weights(
        panel, liq_col=liq["quintile_col"] if liq["ascending"] else liq["quintile_col"]
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

    plot_r2_by_quintile(q_r2, output_dir / "r2_by_quintile.png")

    # LaTeX table — format to match document Table 3 template
    q_r2_tex = q_r2.copy()
    q_r2_tex["pooled_r2"] = (q_r2_tex["pooled_r2"] * 100).round(3)
    q_r2_tex["avg_monthly_r2"] = (q_r2_tex["avg_monthly_r2"] * 100).round(3)
    q_r2_tex["avg_n_month"] = q_r2_tex["avg_n_month"].round(0)

    # Format quintile labels to match document
    quintile_labels = {
        1: "Q1 (Illiquid)", 2: "Q2", 3: "Q3", 4: "Q4",
        5: "Q5 (Liquid)", "Full": "Full sample",
    }
    q_r2_tex["quintile"] = q_r2_tex["quintile"].map(quintile_labels)

    q_r2_tex = q_r2_tex.rename(columns={
        "quintile": "Quintile",
        "pooled_r2": r"Pooled $R^2$ (\%)",
        "avg_monthly_r2": r"Avg.\ Monthly $R^2$ (\%)",
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
        "Standard R²: %.4f%%  Weighted R²: %.4f%%  Gap: %.4f pp",
        r2_results["r2_standard"] * 100,
        r2_results["r2_weighted"] * 100,
        r2_results["gap"] * 100,
    )

    with open(output_dir / "utility_weighted_r2.json", "w") as f:
        json.dump(r2_results, f, indent=2)

    # ── Save summary metadata ──
    meta = {
        "spearman_rho_importance_illiq": rho_31,
        "r2_standard": r2_results["r2_standard"],
        "r2_weighted": r2_results["r2_weighted"],
        "r2_gap": r2_results["gap"],
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
    logger.info("  3.4  r2_by_quintile.csv")
    logger.info(
        "  3.5  Standard R²=%.4f%%, Weighted R²=%.4f%%, Gap=%.4f pp",
        r2_results["r2_standard"] * 100,
        r2_results["r2_weighted"] * 100,
        r2_results["gap"] * 100,
    )


if __name__ == "__main__":
    main()
