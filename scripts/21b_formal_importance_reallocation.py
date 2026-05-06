"""
21b - Formal Importance Reallocation
====================================

Generate the formal feature-importance reallocation tables and figure.
Mean absolute SHAP importance is the default source; pass ``--importance native``
or ``--importance naive`` to use model-native importance:
    outputs/formalanalysis/analysis/{model}/{weight_spec}/importance_shift.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/gamma_regression.json
    outputs/formalanalysis/analysis/{model}/{weight_spec}/group_shares.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/importance_reallocation.png
"""

from __future__ import annotations

import argparse
import json
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

from src.analysis.formal.common import load_importance  # noqa: E402
from src.analysis.formal.importance_reallocation import (  # noqa: E402
    analyze_importance_reallocation,
    plot_importance_reallocation,
)
from src.analysis.formal.script_utils import (  # noqa: E402
    add_experiment_filters,
    formal_output_dirs,
    formal_spec_dir,
    load_filtered_specs,
)
from src.config import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("21b_formal_importance_reallocation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate formal feature-importance reallocation outputs"
    )
    add_experiment_filters(parser)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write tables only and skip PNG figures.",
    )
    parser.add_argument(
        "--importance",
        default="shap",
        choices=["native", "naive", "shap"],
        help=(
            "Importance source for reallocation analysis. 'shap' reads "
            "importance_shap.csv and is the default. 'native' reads "
            "importance_native.csv; 'naive' is accepted as an alias."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    _, experiment_dir, analysis_dir = formal_output_dirs(config)
    specs = load_filtered_specs(experiment_dir, args, logger)

    step2_dir = Path(config["project"]["output_dir"]) / "motivation" / "step2" / "dvol"
    for spec in specs:
        spec_label = spec["spec_label"]
        logger.info("=== %s ===", spec_label)
        out_dir = formal_spec_dir(analysis_dir, spec)
        try:
            importance_standard, importance_weighted = load_importance(
                spec,
                source=args.importance,
            )
        except FileNotFoundError as exc:
            logger.warning("Importance files missing for %s: %s", spec_label, exc)
            continue

        result = analyze_importance_reallocation(
            importance_standard,
            importance_weighted,
            interaction_csv_path=step2_dir / "interaction_regression_full.csv",
            quintile_raw_csv_path=step2_dir / "quintile_fm_coefficients_full_raw.csv",
        )
        result["importance_shift"].to_csv(
            out_dir / "importance_shift.csv",
            index=False,
        )

        if "delta_vs_gamma_regression" in result:
            with open(out_dir / "gamma_regression.json", "w") as handle:
                json.dump(result["delta_vs_gamma_regression"], handle, indent=2)
        (out_dir / "delta_gamma.csv").unlink(missing_ok=True)
        if "group_shares" in result:
            result["group_shares"].to_csv(
                out_dir / "group_shares.csv",
                index=False,
            )
        if not args.no_figures:
            plot_importance_reallocation(
                result["importance_shift"],
                out_dir / "importance_reallocation.png",
                spec["model"],
                importance_source="shap" if args.importance == "shap" else "native",
            )

    logger.info("Done")


if __name__ == "__main__":
    main()
