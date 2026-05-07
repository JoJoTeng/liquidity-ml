"""
21a - Formal Liquidity-Sorted R2
================================

Generate the formal liquid-stock prediction tables:
    outputs/formalanalysis/analysis/{model}/{weight_spec}/r2_by_quintile.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/utility_weighted_r2.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

_runtime_cache = Path(tempfile.gettempdir()) / "liquidity_ml_cache"
_runtime_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_runtime_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_runtime_cache / "xdg"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.formal.common import load_predictions  # noqa: E402
from src.analysis.formal.liquidity_sorted_r2 import (  # noqa: E402
    evaluate_liquid_stock_r2,
    format_legacy_quintile_r2_table,
    plot_quintile_r2_comparison,
)
from src.analysis.formal.script_utils import (  # noqa: E402
    add_experiment_filters,
    formal_output_dirs,
    formal_spec_dir,
    load_filtered_specs,
)
from src.config import load_config  # noqa: E402
from src.data.loader import load_processed_panel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("21a_formal_liquid_r2")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate formal liquid-stock R2 tables")
    add_experiment_filters(parser)
    parser.add_argument(
        "--extra-benchmarks",
        action="store_true",
        help=(
            "Also report cross-sectional-mean and historical-mean R2 columns. "
            "Default reports zero-benchmark R2 only."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write tables only and skip PNG figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    _, experiment_dir, analysis_dir = formal_output_dirs(config)
    specs = load_filtered_specs(experiment_dir, args, logger)

    logger.info("Loading processed panel")
    panel = load_processed_panel()

    for spec in specs:
        spec_label = spec["spec_label"]
        logger.info("=== %s ===", spec_label)
        out_dir = formal_spec_dir(analysis_dir, spec)
        preds_standard, preds_weighted = load_predictions(spec)
        result = evaluate_liquid_stock_r2(
            preds_standard,
            preds_weighted,
            panel,
            config,
            spec,
            include_extra_benchmarks=args.extra_benchmarks,
        )
        result["quintile_r2"].to_csv(
            out_dir / "r2_by_quintile.csv",
            index=False,
        )
        format_legacy_quintile_r2_table(result["quintile_r2"]).to_csv(
            out_dir / "table2_oos_r2_quintile.csv",
            index=False,
        )
        result["utility_weighted_r2"].to_csv(
            out_dir / "utility_weighted_r2.csv",
            index=False,
        )
        if not args.no_figures:
            plot_quintile_r2_comparison(
                result["quintile_r2"],
                out_dir / "figure_r2_by_quintile.png",
                spec["model"],
            )

    logger.info("Done")


if __name__ == "__main__":
    main()
