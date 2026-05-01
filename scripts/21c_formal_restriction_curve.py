"""
21c - Formal Restriction-Curve Comparison
=========================================

Compare each weighted formal model with the Step 3d hard-restriction curve:
    outputs/formalanalysis/analysis/{model}/{weighting}/restriction_curve_comparison.csv
    outputs/formalanalysis/analysis/{model}/{weighting}/restriction_curve_comparison.png
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
from src.analysis.formal.restriction_curve import (  # noqa: E402
    compare_weighted_model_to_restriction_curve,
    plot_restriction_curve_comparison,
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
logger = logging.getLogger("21c_formal_restriction_curve")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate formal restriction-curve comparison outputs"
    )
    add_experiment_filters(parser)
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
        _, preds_weighted = load_predictions(spec)
        restriction_csv = (
            Path(config["project"]["output_dir"])
            / "motivation"
            / "step3_restriction"
            / spec["model"]
            / "dvol"
            / "global"
            / "baseline"
            / "restriction_comparison.csv"
        )
        result = compare_weighted_model_to_restriction_curve(
            preds_weighted,
            panel,
            config,
            restriction_csv,
        )
        if result["restriction_df"] is None:
            continue

        result["restriction_df"].to_csv(
            out_dir / "restriction_curve_comparison.csv",
            index=False,
        )
        if not args.no_figures:
            plot_restriction_curve_comparison(
                result,
                out_dir / "restriction_curve_comparison.png",
                spec["model"],
                spec["weight_family"],
            )

    logger.info("Done")


if __name__ == "__main__":
    main()
