"""
21e - Formal Portfolio Decomposition
====================================

Generate the formal Section 9 portfolio tables:
    outputs/formalanalysis/analysis/{model}/{weight_spec}/two_by_two_{aum}.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/within_quintile_portfolio.csv
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
    compute_two_by_two_decomposition,
    compute_within_quintile_portfolio_table,
    format_two_by_two_decomposition_rows,
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

    for spec in specs:
        spec_label = spec["spec_label"]
        logger.info("=== %s ===", spec_label)
        out_dir = formal_spec_dir(analysis_dir, spec)
        preds_standard, preds_weighted = load_predictions(spec)

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
            logger.info("Within-quintile portfolio table at AUM=%s", aum_label(primary_aum))
            within_quintile = compute_within_quintile_portfolio_table(
                preds_standard,
                preds_weighted,
                panel,
                aum=primary_aum,
                config=config,
            )
            if len(within_quintile) > 0:
                within_quintile.to_csv(
                    out_dir / "within_quintile_portfolio.csv",
                    index=False,
                )

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
