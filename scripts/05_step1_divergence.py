"""
Step 1: Distributional Divergence Analysis
============================================
Demonstrates that P_train(x) != P_deploy(x) across all characteristics,
not just mechanical liquidity dimensions.

Prerequisite: Run scripts/01_process_data.py first to generate
              data/processed_panel.parquet with 142 features in [0, 1].

Outputs (saved to outputs/motivation/step1/):
  1.1  divergence_bar_chart.png       Ranked bar chart of marginal divergences
  1.2  divergence_by_category.csv     Summary table by economic category
  1.3  weight_regression_top15.csv    Fama-MacBeth regression: R̄² + top 15 coefficients
  1.4  density_comparison.png         KDE plots for 6 focal characteristics
  1.5  weight_distribution.png        Histogram of log₁₀(w̃) + percentiles

Usage:
  python scripts/05_step1_divergence.py
  python scripts/05_step1_divergence.py --skip-regression
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_data_dir, get_output_dir
from src.analysis.motivation import (
    load_feature_categories,
    compute_implementability_weights,
    assign_nyse_quintiles,
    compute_marginal_divergence,
    compute_divergence_stats,
    summarize_divergence_by_category,
    fama_macbeth_weight_regression,
    plot_divergence_bar_chart,
    plot_divergence_by_category,
    plot_density_comparison,
    plot_weight_distribution,
    DENSITY_PLOT_FEATURES,
    FOCAL_CHARACTERISTICS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Step 1: Distributional Divergence")
    parser.add_argument(
        "--liquidity",
        type=str,
        default="dvol",
        choices=["dvol", "mcap", "amihud", "spread"],
        help="Primary liquidity measure (default: dvol). "
             "dvol=dollar volume, mcap=market cap, "
             "amihud=Amihud illiquidity, spread=bid-ask spread",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip Fama-MacBeth regression (Output 1.3) for faster iteration",
    )
    args = parser.parse_args()

    # Map liquidity choice to column names
    LIQ_CONFIG = {
        "dvol":   {"weight_col": "liq_dvol_21d",     "quintile_col": "liq_dvol_21d",     "ascending": True,  "label": "Dollar Volume (21-day)"},
        "mcap":   {"weight_col": "liq_me_raw",        "quintile_col": "liq_me_raw",        "ascending": True,  "label": "Market Capitalization"},
        "amihud": {"weight_col": "raw_Illiquidity",   "quintile_col": "raw_Illiquidity",   "ascending": False, "label": "Amihud Illiquidity"},
        "spread": {"weight_col": "raw_BidAskSpread",  "quintile_col": "raw_BidAskSpread",  "ascending": False, "label": "Bid-Ask Spread"},
    }
    liq = LIQ_CONFIG[args.liquidity]

    config = load_config()
    data_dir = get_data_dir()
    output_dir = Path(get_output_dir()) / "motivation_raw" / "step1" / args.liquidity
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Liquidity measure: %s (%s)", args.liquidity, liq["label"])
    logger.info("Output directory: %s", output_dir)

    # ── Load processed panel from 01_process_data.py ──────
    panel_path = data_dir / "processed_panel.parquet"
    if not panel_path.exists():
        logger.error(
            "processed_panel.parquet not found! Run scripts/01_process_data.py first."
        )
        sys.exit(1)

    logger.info("Loading processed panel from %s...", panel_path)
    panel = pd.read_parquet(panel_path)
    logger.info("Panel shape: %s", panel.shape)
    logger.info("Date range: %d to %d", panel["yyyymm"].min(), panel["yyyymm"].max())
    logger.info(
        "Avg stocks per month: %.0f",
        panel.groupby("yyyymm").size().mean(),
    )

    # Verify exchcd is present
    if "exchcd" not in panel.columns:
        logger.error("exchcd not in panel! Re-run 00 + 01.")
        sys.exit(1)

    logger.info(
        "Exchange distribution:\n%s",
        panel["exchcd"].value_counts().sort_index().to_string(),
    )

    # ── Load feature list from 01_process_data.py ─────────
    feature_list_path = data_dir / "feature_list.json"
    if not feature_list_path.exists():
        logger.error("feature_list.json not found! Run scripts/01_process_data.py first.")
        sys.exit(1)

    with open(feature_list_path) as f:
        feature_meta = json.load(f)
    features = feature_meta["features"]
    logger.info("Using %d features (from 01_process_data.py)", len(features))

    # ── Load categories ───────────────────────────────────
    categories = load_feature_categories("config/feature_categories.json")

    # Verify focal characteristics are present
    focal_missing = [f for f in FOCAL_CHARACTERISTICS if f not in features]
    if focal_missing:
        logger.warning("Focal characteristics missing from feature set: %s", focal_missing)

    # ── Implementability weights ──────────────────────────
    logger.info("Computing implementability weights using %s...", liq["weight_col"])
    panel["w_tilde"] = compute_implementability_weights(panel, liq_col=liq["weight_col"])

    # ── NYSE breakpoint quintiles ─────────────────────────
    logger.info("Assigning NYSE breakpoint quintiles using %s...", liq["quintile_col"])
    panel["liq_quintile"] = assign_nyse_quintiles(
        panel, liq["quintile_col"], ascending=liq["ascending"]
    )

    q_counts = panel.groupby(["yyyymm", "liq_quintile"]).size().unstack()
    logger.info(
        "Avg stocks per quintile per month:\n%s",
        q_counts.mean().to_string(),
    )

    # ══════════════════════════════════════════════════════
    # Output 1.5: Weight Distribution (do first — validates weights)
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Output 1.5: Weight distribution histogram")
    plot_weight_distribution(
        panel, "w_tilde", output_dir / "weight_distribution.png",
        vw_col="liq_me_raw",
    )

    # Sanity check
    w_median = panel["w_tilde"].median()
    w_95 = panel["w_tilde"].quantile(0.95)
    logger.info("Weight sanity: median=%.4f, 95th=%.1f", w_median, w_95)
    if w_median > 0.5:
        logger.warning(
            "Median w̃ = %.3f seems high (expected 0.05-0.2). "
            "Check if weights are computed correctly.",
            w_median,
        )

    # ══════════════════════════════════════════════════════
    # Output 1.1: Marginal Divergence Bar Chart
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Output 1.1: Marginal divergence computation")
    div_df = compute_marginal_divergence(panel, features, "w_tilde")
    div_df.to_parquet(output_dir / "divergence_monthly.parquet")
    logger.info("Monthly divergences: %d months × %d features", *div_df.shape)

    stats = compute_divergence_stats(div_df)
    stats.to_csv(output_dir / "divergence_stats.csv", index=False)

    n_sig = (stats["t_stat"].abs() > 2).sum()
    n_total = len(stats)
    logger.info(
        "Divergence: %d/%d features significant (|t| > 2)", n_sig, n_total
    )

    # Count non-liquidity significant
    liq_features = {
        "Illiquidity", "BidAskSpread", "DolVol", "Size", "Price",
        "zerotrade1M", "zerotrade6M", "zerotrade12M",
        "Herf", "PriceDelayRsq", "BetaLiquidityPS",
    }
    non_liq_sig = stats[
        (stats["t_stat"].abs() > 2)
        & (~stats["feature"].isin(liq_features))
    ]
    logger.info(
        "Non-liquidity significant: %d features", len(non_liq_sig)
    )

    plot_divergence_bar_chart(
        stats, categories["broad"], output_dir / "divergence_bar_chart_appendix.png"
    )

    # ══════════════════════════════════════════════════════
    # Output 1.2: Category Summary Table
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Output 1.2: Divergence by category")
    cat_summary = summarize_divergence_by_category(stats, categories["broad"])
    cat_summary.to_csv(output_dir / "divergence_by_category.csv", index=False)
    logger.info("\n%s", cat_summary.to_string(index=False))

    # Aggregated category bar chart (main paper version of Output 1.1)
    plot_divergence_by_category(
        cat_summary, output_dir / "divergence_bar_chart.png"
    )

    # Save LaTeX version for paper
    tex_dir = Path("paper/TablesNew")
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / "DivergenceByCategory.tex"
    cat_tex = cat_summary.rename(columns={
        "Avg. |d_bar|": r"Avg.\ $|\bar{d}|$",
        "# Significant (|t| > 2)": r"\# Significant ($|t|>2$)",
        "# Characteristics": r"\# Characteristics",
    })
    cat_tex.to_latex(
        tex_path,
        index=False,
        float_format="%.4f",
        escape=False,
        caption="Distributional Divergence by Category",
        label="tab:divergence_by_category",
    )
    logger.info("Saved LaTeX table: %s", tex_path)

    # ══════════════════════════════════════════════════════
    # Output 1.3: Fama-MacBeth Weight Regression
    # ══════════════════════════════════════════════════════
    if not args.skip_regression:
        logger.info("=" * 60)
        logger.info("Output 1.3: Fama-MacBeth weight regression")
        fm = fama_macbeth_weight_regression(panel, features, "w_tilde")
        logger.info(
            "Fama-MacBeth: R̄² = %.3f (median %.3f) over %d months",
            fm["r2_mean"],
            fm["r2_median"],
            fm["n_months"],
        )

        # Coefficient tables
        fm["coef_stats"].to_csv(
            output_dir / "weight_regression_all.csv", index=False
        )
        fm["coef_stats"].head(15).to_csv(
            output_dir / "weight_regression_top15.csv", index=False
        )

        # Monthly R² time series
        fm["r2_df"].to_csv(
            output_dir / "weight_regression_r2_monthly.csv", index=False
        )

        # Summary metadata (JSON)
        fm_meta = {
            "r2_mean": fm["r2_mean"],
            "r2_median": fm["r2_median"],
            "n_months": fm["n_months"],
        }
        with open(output_dir / "weight_regression_meta.json", "w") as f:
            json.dump(fm_meta, f, indent=2)

        logger.info("Top 15 coefficients:")
        logger.info("\n%s", fm["coef_stats"].head(15).to_string(index=False))

        # Save LaTeX version for paper
        top15 = fm["coef_stats"].head(15)[
            ["feature", "delta_bar", "t_stat"]
        ].copy()
        top15 = top15.rename(columns={
            "feature": "Characteristic",
            "delta_bar": r"$\bar{\delta}_j$",
            "t_stat": "$t$-stat",
        })
        top15[r"$\bar{\delta}_j$"] = top15[r"$\bar{\delta}_j$"].round(3)
        top15["$t$-stat"] = top15["$t$-stat"].round(2)
        top15.insert(0, "Rank", range(1, 16))

        tex_path = tex_dir / "WeightRegression.tex"
        top15.to_latex(
            tex_path,
            index=False,
            escape=False,
            caption=(
                f"Fama-MacBeth Weight Regression: Top 15 Characteristics "
                f"($\\bar{{R}}^2 = {fm['r2_mean']:.3f}$, $T = {fm['n_months']}$)"
            ),
            label="tab:weight_regression",
        )
        logger.info("Saved LaTeX table: %s", tex_path)
    else:
        logger.info("Skipping Fama-MacBeth regression (--skip-regression)")

    # ══════════════════════════════════════════════════════
    # Output 1.4: Density Comparison Plots
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Output 1.4: Density comparison plots")

    # Use density plot features that are actually in our feature set
    density_features = [f for f in DENSITY_PLOT_FEATURES if f in features]
    if len(density_features) < len(DENSITY_PLOT_FEATURES):
        logger.warning(
            "Using %d/%d density features (missing: %s)",
            len(density_features),
            len(DENSITY_PLOT_FEATURES),
            set(DENSITY_PLOT_FEATURES) - set(density_features),
        )

    plot_density_comparison(
        panel, density_features, "w_tilde", output_dir / "density_comparison.png"
    )

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1 complete. Outputs saved to %s", output_dir)
    logger.info("  1.1  divergence_bar_chart.png")
    logger.info("  1.2  divergence_by_category.csv")
    if not args.skip_regression:
        logger.info("  1.3  weight_regression_top15.csv (R̄² = %.3f)", fm["r2_mean"])
    logger.info("  1.4  density_comparison.png")
    logger.info("  1.5  weight_distribution.png")


if __name__ == "__main__":
    main()
