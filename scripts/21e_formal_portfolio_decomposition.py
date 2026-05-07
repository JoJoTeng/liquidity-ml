"""
21e - Formal Portfolio Decomposition
====================================

Generate the formal Section 9 portfolio tables:
    outputs/formalanalysis/analysis/{model}/{weight_spec}/two_by_two_{aum}.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/within_quintile_portfolio.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/within_quintile_portfolio_tc_sort_{aum}.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/within_quintile_portfolio_scaled_aum_{aum}.csv
    outputs/formalanalysis/analysis/formal_hypothesis_tests.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

_runtime_cache = Path(tempfile.gettempdir()) / "liquidity_ml_cache"
_runtime_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_runtime_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_runtime_cache / "xdg"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.formal.common import aum_label, load_predictions  # noqa: E402
from src.analysis.formal.portfolio_tables import (  # noqa: E402
    compute_liquidity_distribution_table,
    compute_quintile_aum_scaling,
    compute_quintile_sr_scissors_tables,
    compute_two_by_two_decomposition,
    format_within_quintile_from_scissors,
    format_two_by_two_decomposition_rows,
    plot_sr_scissors_comparison,
    plot_sr_scissors_table,
)
from src.analysis.formal.script_utils import (  # noqa: E402
    add_experiment_filters,
    formal_output_dirs,
    formal_spec_dir,
    load_filtered_specs,
    parse_aum_millions,
)
from src.config import load_config  # noqa: E402
from src.data.loader import load_processed_panel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("21e_formal_portfolio_decomposition")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate formal portfolio decomposition tables"
    )
    add_experiment_filters(parser)
    parser.add_argument(
        "--aum",
        type=int,
        action="append",
        default=None,
        help=(
            "AUM in $M for a subset run. Repeat for multiple values. "
            "Default runs all transaction_costs.aum_scenarios."
        ),
    )
    parser.add_argument(
        "--primary-aum",
        type=int,
        default=500,
        help="Primary AUM in $M for within-quintile tables and hypothesis summary.",
    )
    parser.add_argument(
        "--skip-within-quintile",
        action="store_true",
        help="Skip within-quintile portfolio tables.",
    )
    parser.add_argument(
        "--skip-two-by-two",
        action="store_true",
        help=(
            "Skip the 2x2 decomposition and only generate within-quintile "
            "portfolio/scissors outputs."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write tables only and skip PNG figures.",
    )
    parser.add_argument(
        "--skip-tc-aware-scissors",
        action="store_true",
        help=(
            "Skip extra within-quintile/scissors outputs for weighted "
            "predictions with TC-aware sorting."
        ),
    )
    parser.add_argument(
        "--scaled-quintile-aum",
        action="store_true",
        help=(
            "Also write within-quintile/scissors robustness outputs that scale "
            "each quintile's AUM by its average selected-leg stock count "
            "relative to Q1."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    _, experiment_dir, analysis_dir = formal_output_dirs(config)
    specs = load_filtered_specs(experiment_dir, args, logger)

    logger.info("Loading processed panel")
    panel = load_processed_panel()

    aum_scenarios = parse_aum_millions(args.aum, config)
    primary_aum = int(args.primary_aum) * 1_000_000
    primary_decompositions = {}
    distribution_table = compute_liquidity_distribution_table(
        panel,
        aum_scenarios,
        config,
    )

    for spec in specs:
        spec_label = spec["spec_label"]
        logger.info("=== %s ===", spec_label)
        out_dir = formal_spec_dir(analysis_dir, spec)
        preds_standard, preds_weighted = load_predictions(spec)

        if not args.skip_two_by_two:
            for aum in aum_scenarios:
                label = aum_label(aum)
                logger.info("2x2 decomposition at AUM=%s", label)
                result = compute_two_by_two_decomposition(
                    preds_standard,
                    preds_weighted,
                    panel,
                    aum=aum,
                    config=config,
                )
                rows = format_two_by_two_decomposition_rows(result)
                pd.DataFrame(rows).to_csv(
                    out_dir / f"two_by_two_{label}.csv",
                    index=False,
                )
                if aum == primary_aum:
                    primary_decompositions[spec_label] = result

        if not args.skip_within_quintile:
            logger.info("Within-quintile/scissors portfolio tables")
            distribution_table.to_csv(
                out_dir / "table1_distribution.csv",
                index=False,
            )
            scissors_tables = compute_quintile_sr_scissors_tables(
                preds_standard,
                preds_weighted,
                panel,
                aum_scenarios=aum_scenarios,
                config=config,
                tc_sort_aum=(
                    None if args.skip_tc_aware_scissors else primary_aum
                ),
            )
            within_quintile = format_within_quintile_from_scissors(
                scissors_tables,
                primary_aum,
            )
            if len(within_quintile) > 0:
                within_quintile.to_csv(
                    out_dir / "within_quintile_portfolio.csv",
                    index=False,
                )
            scissors_tables["std"].to_csv(
                out_dir / "table3_sr_quintile_std.csv",
                index=False,
            )
            scissors_tables["weighted"].to_csv(
                out_dir / "table3_sr_quintile_weighted.csv",
                index=False,
            )
            tc_sort_table = scissors_tables.get("weighted_tc_sort")
            tc_sort_label = aum_label(primary_aum)
            if tc_sort_table is not None and not tc_sort_table.empty:
                tc_sort_table.to_csv(
                    out_dir / f"table3_sr_quintile_weighted_tc_sort_{tc_sort_label}.csv",
                    index=False,
                )
                within_quintile_tc_sort = format_within_quintile_from_scissors(
                    {
                        "std": scissors_tables["std"],
                        "weighted": tc_sort_table,
                    },
                    primary_aum,
                )
                if len(within_quintile_tc_sort) > 0:
                    within_quintile_tc_sort.to_csv(
                        out_dir / f"within_quintile_portfolio_tc_sort_{tc_sort_label}.csv",
                        index=False,
                    )
            if not args.no_figures:
                plot_sr_scissors_table(
                    scissors_tables["std"],
                    out_dir / "figure_sr_scissors_std.png",
                    spec["model"],
                    "Standard Training",
                    aum_scenarios,
                )
                plot_sr_scissors_table(
                    scissors_tables["weighted"],
                    out_dir / "figure_sr_scissors_weighted.png",
                    spec["model"],
                    "Weighted Training",
                    aum_scenarios,
                )
                plot_sr_scissors_comparison(
                    scissors_tables["std"],
                    scissors_tables["weighted"],
                    out_dir / "figure_sr_scissors_comparison.png",
                    spec["model"],
                    aum_scenarios,
                )
                if tc_sort_table is not None and not tc_sort_table.empty:
                    plot_sr_scissors_table(
                        tc_sort_table,
                        out_dir / f"figure_sr_scissors_weighted_tc_sort_{tc_sort_label}.png",
                        spec["model"],
                        f"Weighted Training + TC-Aware Sort ({tc_sort_label})",
                        aum_scenarios,
                    )
                    plot_sr_scissors_comparison(
                        scissors_tables["std"],
                        tc_sort_table,
                        out_dir / f"figure_sr_scissors_comparison_tc_sort_{tc_sort_label}.png",
                        spec["model"],
                        aum_scenarios,
                        right_title=f"Weighted + TC-Aware Sort ({tc_sort_label})",
                    )

            if args.scaled_quintile_aum:
                logger.info(
                    "Within-quintile/scissors scaled-AUM robustness tables"
                )
                scale_table = compute_quintile_aum_scaling(
                    preds_standard,
                    panel,
                    base_aum=primary_aum,
                    config=config,
                    reference_quintile=1,
                )
                scale_table.to_csv(
                    out_dir / f"quintile_aum_scaling_{tc_sort_label}.csv",
                    index=False,
                )
                scale_map = dict(
                    zip(scale_table["quintile"], scale_table["AUM_Scale"])
                )
                scaled_tables = compute_quintile_sr_scissors_tables(
                    preds_standard,
                    preds_weighted,
                    panel,
                    aum_scenarios=aum_scenarios,
                    config=config,
                    tc_sort_aum=(
                        None if args.skip_tc_aware_scissors else primary_aum
                    ),
                    quintile_aum_scale=scale_map,
                )
                scaled_tables["std"].to_csv(
                    out_dir / f"table3_sr_quintile_std_scaled_aum_{tc_sort_label}.csv",
                    index=False,
                )
                scaled_tables["weighted"].to_csv(
                    out_dir / f"table3_sr_quintile_weighted_scaled_aum_{tc_sort_label}.csv",
                    index=False,
                )
                within_scaled = format_within_quintile_from_scissors(
                    scaled_tables,
                    primary_aum,
                )
                if len(within_scaled) > 0:
                    within_scaled.to_csv(
                        out_dir / f"within_quintile_portfolio_scaled_aum_{tc_sort_label}.csv",
                        index=False,
                    )

                scaled_tc_sort_table = scaled_tables.get("weighted_tc_sort")
                if (
                    scaled_tc_sort_table is not None
                    and not scaled_tc_sort_table.empty
                ):
                    scaled_tc_sort_table.to_csv(
                        out_dir / (
                            "table3_sr_quintile_weighted_tc_sort_scaled_aum_"
                            f"{tc_sort_label}.csv"
                        ),
                        index=False,
                    )
                    within_scaled_tc_sort = format_within_quintile_from_scissors(
                        {
                            "std": scaled_tables["std"],
                            "weighted": scaled_tc_sort_table,
                        },
                        primary_aum,
                    )
                    if len(within_scaled_tc_sort) > 0:
                        within_scaled_tc_sort.to_csv(
                            out_dir / (
                                "within_quintile_portfolio_tc_sort_scaled_aum_"
                                f"{tc_sort_label}.csv"
                            ),
                            index=False,
                        )

                if not args.no_figures:
                    plot_sr_scissors_comparison(
                        scaled_tables["std"],
                        scaled_tables["weighted"],
                        out_dir / f"figure_sr_scissors_comparison_scaled_aum_{tc_sort_label}.png",
                        spec["model"],
                        aum_scenarios,
                        left_title=f"Standard Training (Scaled AUM base {tc_sort_label})",
                        right_title=f"Weighted Training (Scaled AUM base {tc_sort_label})",
                    )
                    if (
                        scaled_tc_sort_table is not None
                        and not scaled_tc_sort_table.empty
                    ):
                        plot_sr_scissors_comparison(
                            scaled_tables["std"],
                            scaled_tc_sort_table,
                            out_dir / (
                                "figure_sr_scissors_comparison_tc_sort_scaled_aum_"
                                f"{tc_sort_label}.png"
                            ),
                            spec["model"],
                            aum_scenarios,
                            left_title=(
                                f"Standard Training (Scaled AUM base {tc_sort_label})"
                            ),
                            right_title=(
                                "Weighted + TC-Aware Sort "
                                f"(Scaled AUM base {tc_sort_label})"
                            ),
                        )

    if primary_decompositions:
        _write_hypothesis_summary(primary_decompositions, analysis_dir)
    logger.info("Done")


def _write_hypothesis_summary(results: dict, out_dir: Path) -> None:
    """Save consolidated hypothesis-test statistics from primary-AUM decompositions."""
    summary = {}
    for spec_label, result in results.items():
        decomp = result["decomposition"]
        summary[spec_label] = {
            "H1_training_share_pct": result["training_share"],
            "H1_lw_pvalue": decomp["lw_h3"].get("p_value"),
            "H3_total_effect": decomp["total_effect"],
            "H3_lw_pvalue": decomp["lw_total"].get("p_value"),
            "sharpe_ratios": decomp["sharpe_ratios"],
            "training_effect": decomp["training_effect"],
            "portfolio_effect": decomp["portfolio_effect"],
            "interaction": decomp["interaction"],
        }

    with open(out_dir / "formal_hypothesis_tests.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
