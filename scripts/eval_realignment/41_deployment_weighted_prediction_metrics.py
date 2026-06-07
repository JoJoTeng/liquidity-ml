"""
41 - Deployment-Weighted Prediction Metrics (note section 4.1)
=============================================================

Realigned evaluation track. For each formal experiment spec, reports the
deployment-weighted prediction metrics the note's section 4.1 prescribes, scoring
the standard and weighted predictions under the SAME w-tilde (the spec's own
training weight family), on the full cross-section and the Q4-Q5 liquid subset:

  * deployment_weighted_r2.csv                 -- w-tilde-weighted OOS R^2 (Eq. 2)
  * deployment_weighted_error_diff_full.csv    -- monthly w-tilde-weighted MSE diff
  * deployment_weighted_error_diff_liquid_q4q5.csv
  * deployment_weighted_error_diff_stats.csv   -- Newey-West summary per universe

Predictions are READ from outputs/formalanalysis/experiment/...; outputs are
written under outputs/eval_realignment/analysis/{model}/{weight_spec}/, with
breakpoint-dependent files under liquidity_breakpoints/{mode}/. No formalanalysis
output, module, or config is modified.
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

# This script lives one directory deeper than scripts/*.py, so the repo root is
# three parents up rather than two.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.analysis.eval_realignment.deployment_weighted_metrics import (  # noqa: E402
    FIGURE_UNIVERSES,
    compute_deployment_weighted_error_differential,
    compute_deployment_weighted_r2,
    plot_deployment_weighted_error_differential,
    summarize_error_differential,
)
from src.analysis.eval_realignment.script_utils import (  # noqa: E402
    add_experiment_filters,
    add_liquidity_breakpoint_option,
    config_for_liquidity_breakpoint,
    eval_realignment_output_dirs,
    eval_realignment_spec_dir,
    liquidity_breakpoint_dir,
    load_filtered_specs,
    namespace_liquidity_breakpoints,
    resolve_liquidity_breakpoints,
)
from src.analysis.formal.common import load_predictions  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.loader import load_processed_panel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("41_deployment_weighted_prediction_metrics")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate deployment-weighted prediction metrics (note section 4.1)"
    )
    add_experiment_filters(parser)
    add_liquidity_breakpoint_option(parser)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write tables only and skip PNG figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    _, experiment_dir, analysis_dir = eval_realignment_output_dirs(config)
    specs = load_filtered_specs(experiment_dir, args, logger)

    logger.info("Loading processed panel")
    panel = load_processed_panel()
    breakpoint_modes = resolve_liquidity_breakpoints(
        config,
        args.liquidity_breakpoints,
    )
    use_breakpoint_namespace = namespace_liquidity_breakpoints(
        args.liquidity_breakpoints,
    )

    for spec in specs:
        logger.info("=== %s ===", spec["spec_label"])
        base_out_dir = eval_realignment_spec_dir(analysis_dir, spec)
        preds_standard, preds_weighted = load_predictions(spec)

        for breakpoint_mode in breakpoint_modes:
            logger.info("Liquidity breakpoints: %s", breakpoint_mode)
            breakpoint_config = config_for_liquidity_breakpoint(
                config,
                breakpoint_mode,
            )
            out_dir = liquidity_breakpoint_dir(
                base_out_dir,
                breakpoint_mode,
                use_breakpoint_namespace,
            )

            # Metric 1: w-tilde-weighted OOS R^2 per quintile + pooled Q4-Q5 + full.
            r2_table = compute_deployment_weighted_r2(
                preds_standard,
                preds_weighted,
                panel,
                breakpoint_config,
                spec,
            )
            r2_table.to_csv(out_dir / "deployment_weighted_r2.csv", index=False)

            # Metric 2: w-tilde-weighted monthly squared-error differential,
            # stacked across universes (each quintile, pooled Q4-Q5, full).
            diff = compute_deployment_weighted_error_differential(
                preds_standard,
                preds_weighted,
                panel,
                breakpoint_config,
                spec,
            )
            diff.to_csv(
                out_dir / "deployment_weighted_error_diff.csv",
                index=False,
            )

            stats = summarize_error_differential(diff, breakpoint_config)
            stats.insert(1, "weight_label", r2_table["weight_label"].iloc[0])
            stats.to_csv(
                out_dir / "deployment_weighted_error_diff_stats.csv",
                index=False,
            )

            if not args.no_figures:
                for universe in FIGURE_UNIVERSES:
                    u_diff = diff[diff["universe"] == universe]
                    if not u_diff.empty:
                        plot_deployment_weighted_error_differential(
                            u_diff,
                            out_dir / f"deployment_weighted_error_diff_{universe}.png",
                            spec["model"],
                            universe,
                        )

    logger.info("Done")


if __name__ == "__main__":
    main()
