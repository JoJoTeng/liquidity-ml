"""
Step 2: Heterogeneous Predictability
======================================
Tests whether return-predictor relationships differ between liquid and
illiquid stocks. If E[r|x] is the same everywhere, the covariate shift
from Step 1 is benign.

Prerequisite: Run scripts/01_process_data.py first.

Outputs (saved to outputs/motivation/step2/{liquidity}/):
  2.1  quintile_fm_coefficients.csv     FM coefficients × 5 quintiles
  2.2  coefficient_plots.png            β̄_j,q by quintile with 95% CI
  2.3  interaction_regression.csv       Main effects + interactions + F-test
       interaction_regression_dummy.csv Robustness: above-median dummy
  2.4  divergence_vs_heterogeneity.png  Scatter of |d̄_j| vs |γ̄_j|

Usage:
  python scripts/06_step2_heterogeneity.py
  python scripts/06_step2_heterogeneity.py --liquidity mcap
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
    quintile_fama_macbeth,
    format_quintile_table,
    interaction_fama_macbeth,
    plot_quintile_coefficients,
    plot_divergence_vs_heterogeneity,
    FOCAL_CHARACTERISTICS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Step 2: Heterogeneous Predictability")
    parser.add_argument(
        "--liquidity",
        type=str,
        default="dvol",
        choices=["dvol", "mcap", "amihud", "spread"],
        help="Primary liquidity measure (default: dvol)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run interaction regression on all ~113 characteristics (not just 15 focal)",
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
    output_dir = Path(get_output_dir()) / "motivation_raw" / "step2" / args.liquidity
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

    # ── Assign quintiles ──────────────────────────────────
    logger.info("Assigning NYSE quintiles using %s...", liq["quintile_col"])
    panel["liq_quintile"] = assign_nyse_quintiles(
        panel, liq["quintile_col"], ascending=liq["ascending"]
    )

    # Liquidity rank [0,1] for interaction regression
    panel["liq_rank"] = panel.groupby("yyyymm")[liq["quintile_col"]].rank(pct=True)
    if not liq["ascending"]:
        # Invert so higher rank = more liquid
        panel["liq_rank"] = 1 - panel["liq_rank"]

    focal = list(FOCAL_CHARACTERISTICS.keys())
    logger.info("Focal characteristics: %d", len(focal))

    # Verify focal chars exist
    missing = [f for f in focal if f not in panel.columns]
    if missing:
        logger.error("Focal characteristics missing from panel: %s", missing)
        sys.exit(1)

    # ══════════════════════════════════════════════════════
    # Output 2.1: Quintile Fama-MacBeth Regressions
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Output 2.1: Quintile-specific Fama-MacBeth regressions")
    q_results = quintile_fama_macbeth(panel, focal, "liq_quintile")

    # Raw coefficients (for programmatic use)
    q_results["coef_table"].to_csv(
        output_dir / "quintile_fm_coefficients_raw.csv", index=False
    )

    # Formatted table: "β̄ (t-stat)" per cell
    formatted = format_quintile_table(q_results["coef_table"])
    formatted.to_csv(
        output_dir / "quintile_fm_coefficients.csv", index=False
    )
    logger.info("Quintile FM table (formatted):\n%s", formatted.to_string(index=False))

    # Count features with significant Q5-Q1 difference
    ct = q_results["coef_table"]
    n_sig_diff = (ct["t_Q5-Q1"].abs() > 2).sum()
    logger.info("Features with significant Q5-Q1 difference: %d/%d", n_sig_diff, len(ct))

    # LaTeX table (formatted with β (t) cells)
    formatted.to_latex(
        tex_dir / "QuintileFM.tex",
        index=False,
        caption="Fama-MacBeth Coefficients by Liquidity Quintile",
        label="tab:quintile_fm",
    )
    logger.info("Saved LaTeX: %s", tex_dir / "QuintileFM.tex")

    # ══════════════════════════════════════════════════════
    # Output 2.2: Coefficient Plots
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Output 2.2: Quintile coefficient plots")
    plot_quintile_coefficients(
        q_results, focal, output_dir / "coefficient_plots.png"
    )

    # ══════════════════════════════════════════════════════
    # Output 2.3: Interaction Regression
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Output 2.3: Interaction Fama-MacBeth regression")

    # Primary: continuous liquidity rank
    int_results = interaction_fama_macbeth(panel, focal, "liq_rank")
    int_results["coef_table"].to_csv(
        output_dir / "interaction_regression.csv", index=False
    )
    logger.info(
        "Joint F-test p-value: %.4f (H0: all γ = 0)", int_results["f_test_pvalue"]
    )
    n_sig_gamma = (int_results["coef_table"]["gamma_t"].abs() > 2).sum()
    logger.info("Individual γ with |t| > 2: %d/%d", n_sig_gamma, len(focal))
    logger.info("\n%s", int_results["coef_table"].to_string(index=False))

    # Robustness: above-median dummy
    logger.info("Robustness: above-median dummy interaction")
    int_dummy = interaction_fama_macbeth(panel, focal, "liq_rank", use_dummy=True)
    int_dummy["coef_table"].to_csv(
        output_dir / "interaction_regression_dummy.csv", index=False
    )
    logger.info(
        "Dummy F-test p-value: %.4f", int_dummy["f_test_pvalue"]
    )

    # LaTeX table (primary)
    tex_int = int_results["coef_table"].copy()
    tex_int = tex_int.rename(columns={
        "feature": "Characteristic",
        "beta_bar": r"$\bar{\beta}_j$",
        "beta_t": "$t(\\beta)$",
        "gamma_bar": r"$\bar{\gamma}_j$",
        "gamma_t": "$t(\\gamma)$",
    })
    for c in tex_int.columns:
        if c != "Characteristic":
            tex_int[c] = tex_int[c].round(3)
    tex_int.to_latex(
        tex_dir / "InteractionRegression.tex",
        index=False,
        escape=False,
        caption=(
            "Interaction Regression: Main Effects and Liquidity Interactions "
            f"(Joint F-test $p = {int_results['f_test_pvalue']:.4f}$)"
        ),
        label="tab:interaction_regression",
    )
    logger.info("Saved LaTeX: %s", tex_dir / "InteractionRegression.tex")

    # Save metadata
    meta = {
        "f_test_stat_continuous": int_results["f_test_stat"],
        "f_test_pvalue_continuous": int_results["f_test_pvalue"],
        "f_test_df_continuous": int_results["f_test_df"],
        "f_test_stat_dummy": int_dummy["f_test_stat"],
        "f_test_pvalue_dummy": int_dummy["f_test_pvalue"],
        "f_test_df_dummy": int_dummy["f_test_df"],
        "n_months": int_results["n_months"],
        "n_sig_gamma_continuous": int(n_sig_gamma),
        "n_sig_gamma_dummy": int((int_dummy["coef_table"]["gamma_t"].abs() > 2).sum()),
    }
    with open(output_dir / "interaction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ══════════════════════════════════════════════════════
    # Output 2.4: Divergence vs Heterogeneity Scatter
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Output 2.4: Divergence vs heterogeneity scatter")

    step1_dir = Path(get_output_dir()) / "motivation_raw" / "step1" / args.liquidity
    div_stats_path = step1_dir / "divergence_stats.csv"
    if not div_stats_path.exists():
        logger.warning(
            "divergence_stats.csv not found at %s. "
            "Run 05_step1_divergence.py first. Skipping Output 2.4.",
            div_stats_path,
        )
    else:
        div_stats = pd.read_csv(div_stats_path)
        rho = plot_divergence_vs_heterogeneity(
            div_stats, int_results, focal,
            output_dir / "divergence_vs_heterogeneity.png",
        )
        meta["spearman_rho"] = rho
        with open(output_dir / "interaction_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ══════════════════════════════════════════════════════
    # Output 2.3b/2.1b: Full Interaction Regression (all ~113 chars)
    # ══════════════════════════════════════════════════════
    if args.full:
        from src.analysis.motivation import (
            get_motivation_features, load_signaldoc, build_feature_categories,
        )

        logger.info("=" * 60)
        logger.info("Loading full feature set for --full mode...")
        signaldoc = load_signaldoc()
        all_features = get_motivation_features(signaldoc, panel)
        missing_full = [f for f in all_features if f not in panel.columns]
        if missing_full:
            logger.warning("Missing from panel (%d): %s", len(missing_full), missing_full)
            all_features = [f for f in all_features if f in panel.columns]
        logger.info("Full feature set: %d characteristics", len(all_features))

        # 2.3b: Full interaction regression
        logger.info("Output 2.3b: Full interaction regression (%d features)", len(all_features))
        int_full = interaction_fama_macbeth(panel, all_features, "liq_rank")
        int_full["coef_table"].to_csv(output_dir / "interaction_regression_full.csv", index=False)
        n_sig_full = int((int_full["coef_table"]["gamma_t"].abs() > 2).sum())
        logger.info("Full: %d/%d sig γ, F-test p=%.6f", n_sig_full, len(all_features), int_full["f_test_pvalue"])

        # Category summary
        cat_info = build_feature_categories(signaldoc)
        broad_map = cat_info["broad"]
        ct_full = int_full["coef_table"].copy()
        ct_full["category"] = ct_full["feature"].map(broad_map).fillna("Other")
        cat_summary = ct_full.groupby("category").agg(
            n_features=("feature", "count"),
            n_sig_gamma=("gamma_t", lambda x: int((x.abs() > 2).sum())),
            avg_abs_gamma=("gamma_bar", lambda x: x.abs().mean()),
        ).reset_index().sort_values("n_sig_gamma", ascending=False)
        cat_summary.to_csv(output_dir / "interaction_by_category.csv", index=False)
        logger.info("Category summary:\n%s", cat_summary.to_string(index=False))

        # Robustness: dummy
        int_full_dummy = interaction_fama_macbeth(panel, all_features, "liq_rank", use_dummy=True)
        int_full_dummy["coef_table"].to_csv(output_dir / "interaction_regression_full_dummy.csv", index=False)

        # 2.1b: Full quintile FM
        logger.info("Output 2.1b: Full quintile FM (%d features)", len(all_features))
        q_full = quintile_fama_macbeth(panel, all_features, "liq_quintile")
        q_full["coef_table"].to_csv(output_dir / "quintile_fm_coefficients_full_raw.csv", index=False)

        # Update metadata
        meta["f_test_stat_full"] = int_full["f_test_stat"]
        meta["f_test_pvalue_full"] = int_full["f_test_pvalue"]
        meta["n_sig_gamma_full"] = n_sig_full
        meta["n_features_full"] = len(all_features)
        meta["f_test_stat_full_dummy"] = int_full_dummy["f_test_stat"]
        meta["f_test_pvalue_full_dummy"] = int_full_dummy["f_test_pvalue"]
        with open(output_dir / "interaction_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Full-feature outputs saved.")

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 2 complete. Outputs saved to %s", output_dir)
    logger.info("  2.1  quintile_fm_coefficients.csv")
    logger.info("  2.2  coefficient_plots.png")
    logger.info("  2.3  interaction_regression.csv (F-test p=%.4f)", int_results["f_test_pvalue"])
    logger.info("       interaction_regression_dummy.csv (F-test p=%.4f)", int_dummy["f_test_pvalue"])
    if "spearman_rho" in meta:
        logger.info("  2.4  divergence_vs_heterogeneity.png (ρ=%.3f)", meta["spearman_rho"])


if __name__ == "__main__":
    main()
