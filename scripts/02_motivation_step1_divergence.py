"""
Step 1: Distributional Divergence Analysis
============================================
Demonstrates that P_train(x) != P_deploy(x) across all characteristics,
not just mechanical liquidity dimensions.

Prerequisite: Run scripts/01_process_data.py first to generate
              data/processed_panel.parquet and data/feature_list.json.

Outputs (saved to outputs/motivation/step1/):
  pre  weight_distribution.png        Diagnostic histogram of log10(w_tilde)
  1.1  divergence_bar_chart.png       Aggregate category divergence chart
       divergence_bar_chart_appendix.png
                                       Full feature-level divergence chart
  1.2  divergence_by_category.csv     Summary table by economic category
  1.3  weight_regression_top15.csv    Fama-MacBeth regression: R̄² + top 15 coefficients
  1.4  density_comparison.png         KDE plots for 6 focal characteristics

Usage:
  python scripts/02_motivation_step1_divergence.py
  python scripts/02_motivation_step1_divergence.py --skip-regression
  python scripts/02_motivation_step1_divergence.py --liquidity tc --aum 500
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
    assign_nyse_quintiles,
    compute_marginal_divergence,
    compute_divergence_stats,
    summarize_divergence_by_category,
    fama_macbeth_weight_regression,
    plot_divergence_bar_chart,
    plot_divergence_by_category,
    plot_density_comparison,
    plot_weight_distribution,
    ensure_motivation_weight_column,
    get_motivation_liquidity_choices,
    get_motivation_liquidity_config,
    get_motivation_liquidity_key,
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
        choices=get_motivation_liquidity_choices(),
        help="Primary liquidity measure (default: dvol). "
             "dvol=dollar volume, mcap=market cap, tc=transaction-cost weight",
    )
    parser.add_argument(
        "--aum",
        type=float,
        default=500.0,
        help="AUM in $M for --liquidity tc (default: 500)",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip Fama-MacBeth regression (Output 1.3) for faster iteration",
    )
    parser.add_argument(
        "--vw",
        action="store_true",
        help="Add value-weight (market cap) overlay to density comparison plots",
    )
    args = parser.parse_args()

    liq = get_motivation_liquidity_config(args.liquidity)

    config = load_config()
    data_dir = get_data_dir()
    liquidity_key = get_motivation_liquidity_key(args.liquidity, args.aum)
    output_dir = Path(get_output_dir()) / "motivation" / "step1" / liquidity_key
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Liquidity measure: %s (%s)", args.liquidity, liq["label"])
    if args.liquidity == "tc":
        logger.info("TC AUM: $%.0fM", args.aum)
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
        logger.error("exchcd not in panel! Re-run scripts/00_fetch_data.py and scripts/01_process_data.py.")
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
    logger.info("Computing implementability weights using %s...", liq["label"])
    selected_weight_col = ensure_motivation_weight_column(
        panel, args.liquidity, config=config, aum_millions=args.aum
    )
    ensure_motivation_weight_column(panel, "dvol", config=config)
    ensure_motivation_weight_column(panel, "mcap", config=config)
    # Keep w_tilde as alias for backward compatibility
    panel["w_tilde"] = panel[selected_weight_col]

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
    # Preliminary diagnostic: Weight Distribution
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Precheck: weight distribution histogram")
    plot_weight_distribution(
        panel, "w_tilde", output_dir / "weight_distribution.png",
        vw_col="liq_me_raw",
    )

    # Sanity check
    w_median = panel["w_tilde"].median()
    w_95 = panel["w_tilde"].quantile(0.95)
    logger.info("Weight sanity: median=%.4f, 95th=%.1f", w_median, w_95)
    if args.liquidity == "dvol" and w_median > 0.5:
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
    div_df.to_csv(output_dir / "divergence_monthly.csv")
    logger.info("Monthly divergences: %d months × %d features", *div_df.shape)

    stats = compute_divergence_stats(div_df)
    stats.to_csv(output_dir / "divergence_stats.csv", index=False)

    n_sig = (stats["t_stat"].abs() > 2).sum()
    n_total = len(stats)
    logger.info(
        "Divergence: %d/%d features significant (|t| > 2)", n_sig, n_total
    )

    # Count significant features outside the broad Liquidity taxonomy.
    # The taxonomy is built from SignalDoc and constrained here to the final
    # 113-feature universe used in this run.
    liq_features = {
        f for f in features if categories["broad"].get(f) == "Liquidity"
    }
    uncategorized = sorted(f for f in features if f not in categories["broad"])
    if uncategorized:
        logger.warning("Features missing from category map: %s", uncategorized)
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

    # Save raw LaTeX table as a data artifact under the script's output dir.
    # The curated, reference-styled paper table is built separately by
    # scripts/build_paper_tables.py, which owns paper/TablesNew.
    tex_dir = output_dir / "tables"
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

        # Ex-liquidity spanning: repeat the regression on the non-liquidity
        # characteristics only. Because w̃ is a function of dollar volume and
        # the full regressor set includes dollar volume and its transforms,
        # part of the full R² is mechanical; dropping the Liquidity category
        # shows the spanning does not rest on those variables (reported in the
        # paper alongside the full R²).
        non_liq_features = [f for f in features if f not in liq_features]
        fm_exliq = fama_macbeth_weight_regression(panel, non_liq_features, "w_tilde")
        logger.info(
            "Ex-liquidity (%d non-liquidity chars): R̄² = %.3f (median %.3f) over %d months",
            len(non_liq_features), fm_exliq["r2_mean"], fm_exliq["r2_median"],
            fm_exliq["n_months"],
        )

        # Summary metadata (JSON)
        fm_meta = {
            "r2_mean": fm["r2_mean"],
            "r2_median": fm["r2_median"],
            "n_months": fm["n_months"],
            "r2_mean_ex_liquidity": fm_exliq["r2_mean"],
            "r2_median_ex_liquidity": fm_exliq["r2_median"],
            "n_liquidity_excluded": len(liq_features),
            "n_features_ex_liquidity": len(non_liq_features),
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
        panel, density_features, "w_tilde", output_dir / "density_comparison.png",
        vw_col="liq_me_raw" if args.vw else None,
    )

    # Save raw data underlying the density plots
    density_cols = ["permno", "yyyymm", "w_tilde"] + density_features
    density_cols = [c for c in density_cols if c in panel.columns]
    panel[density_cols].to_csv(output_dir / "density_panel_data.csv", index=False)
    logger.info("Saved density panel data: %d rows × %d cols", len(panel), len(density_cols))

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 1 complete. Outputs saved to %s", output_dir)
    logger.info("  pre  weight_distribution.png")
    logger.info("  1.1  divergence_bar_chart.png (aggregate categories)")
    logger.info("       divergence_bar_chart_appendix.png (full feature level)")
    logger.info("  1.2  divergence_by_category.csv")
    if not args.skip_regression:
        logger.info("  1.3  weight_regression_top15.csv (R̄² = %.3f)", fm["r2_mean"])
    logger.info("  1.4  density_comparison.png")


if __name__ == "__main__":
    main()
