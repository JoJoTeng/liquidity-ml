"""
42 - Signal-Weighted Capacity Portfolio (note section 4.2)
==========================================================

Realigned evaluation track. For each formal experiment spec, build the note's
section 4.2 capacity portfolio: a continuous, dollar-neutral book holding each
stock in proportion to its capacity (the spec's own w-tilde) and its centered
signal, theta_it ∝ w̃_it·(r̂_it − rbar_w_t), sized to unit gross exposure. The book
is built from BOTH the standard and weighted r̂ (same w-tilde) and compared, to
test whether the prediction-level training improvement survives at the portfolio
level. Reports gross/net Sharpe and certainty-equivalent (gamma in {1,5,10}) across
the AUM grid, with Newey-West and Ledoit-Wolf bootstrap inference.

The capacity weight is selected by --capacity-weighting (default: all three):
'signal' = the spec's w-tilde (the section 4.2 book), 'equal' = 1/N, 'value' =
market cap. The equal/value books are deployment-stage benchmarks that isolate the
capacity-weighting effect (signal vs equal/value) from the training effect
(standard vs weighted). Per spec / AUM / weighting (the 'signal' weighting keeps
the legacy names; 'equal'/'value' add an infix):

  * capacity_portfolio[_equal|_value]_metrics_{aum}.csv  -- standard/weighted/difference
  * capacity_portfolio[_equal|_value]_monthly_{aum}.csv  -- monthly gross/net series
  * capacity_portfolio[_equal|_value]_net_cumret_{aum}.png (unless --no-figures)

The book uses the full cross-section (no quantile cut, no portfolio-stage cost
sort). Predictions are READ from outputs/formalanalysis/experiment/...; outputs are
written under outputs/eval_realignment/analysis/{model}/{weight_spec}/. No
formalanalysis output, module, or config is modified.
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

# One directory deeper than scripts/*.py, so the repo root is three parents up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402

from src.analysis.eval_realignment.capacity_portfolio import (  # noqa: E402
    build_capacity_book,
    capacity_metrics_from_book,
    compare_standard_vs_weighted,
    plot_capacity_net_cumret,
)
from src.analysis.eval_realignment.script_utils import (  # noqa: E402
    add_experiment_filters,
    eval_realignment_output_dirs,
    eval_realignment_spec_dir,
    load_filtered_specs,
)
from src.analysis.formal.common import (  # noqa: E402
    aum_label,
    formal_weight_label,
    load_predictions,
)
from src.analysis.formal.script_utils import parse_aum_millions  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.loader import load_processed_panel  # noqa: E402
from src.portfolio.construction import prepare_transaction_cost_context  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("42_signal_weighted_capacity_portfolio")

METRICS_COLS = [
    "row_type",
    "weight_label",
    "aum_label",
    "n_months",
    "gross_sr_annual",
    "net_sr_annual",
    "gross_sr_monthly",
    "net_sr_monthly",
    "gross_mean_annual",
    "net_mean_annual",
    "gross_ce_annual_g1",
    "gross_ce_annual_g5",
    "gross_ce_annual_g10",
    "net_ce_annual_g1",
    "net_ce_annual_g5",
    "net_ce_annual_g10",
    "turnover_mean",
    "tc_mean_monthly",
    "net_sr_diff",
    "net_sr_diff_pval",
    "net_sr_diff_reject",
    "net_mean_diff_nw_t",
    "net_ce_diff_annual_g1",
    "net_ce_diff_annual_g5",
    "net_ce_diff_annual_g10",
]

MONTHLY_COLS = [
    "yyyymm",
    "row_type",
    "ret_gross",
    "ret_net",
    "transaction_cost",
    "turnover",
    "n_long",
    "n_short",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the section 4.2 signal-weighted capacity portfolio"
    )
    add_experiment_filters(parser)
    parser.add_argument(
        "--aum",
        nargs="*",
        default=None,
        help=(
            "AUM scenarios in $M, 'proportional_tc', or 'all' (default). "
            "Sizes the realized market-impact term; AUM is the gross capital."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write tables only and skip PNG figures.",
    )
    parser.add_argument(
        "--capacity-weighting",
        nargs="*",
        choices=["signal", "equal", "value", "all"],
        default=None,
        help=(
            "Capacity weighting(s) for the book: 'signal' (the spec's w-tilde, the "
            "section 4.2 default), 'equal' (1/N benchmark), 'value' (market-cap "
            "benchmark), or 'all'. Default runs all three."
        ),
    )
    return parser.parse_args()


def _resolve_capacity_weightings(arg: list[str] | None) -> list[str]:
    """Capacity weightings to run; default (or 'all') is signal+equal+value."""
    allw = ["signal", "equal", "value"]
    if not arg or "all" in arg:
        return allw
    return [w for w in allw if w in arg]


def _cap_token(capacity_weighting: str) -> str:
    """Filename infix per weighting; signal keeps the legacy (empty) token."""
    return "" if capacity_weighting == "signal" else f"_{capacity_weighting}"


def main():
    args = parse_args()
    config = load_config()
    _, experiment_dir, analysis_dir = eval_realignment_output_dirs(config)
    specs = load_filtered_specs(experiment_dir, args, logger)

    logger.info("Loading processed panel")
    panel = load_processed_panel()
    aum_scenarios = parse_aum_millions(args.aum, config)
    # TC inputs are spec-independent; build the lookup once.
    tc_context = prepare_transaction_cost_context(panel, config)
    capacity_weightings = _resolve_capacity_weightings(args.capacity_weighting)

    for spec in specs:
        logger.info("=== %s ===", spec["spec_label"])
        out_dir = eval_realignment_spec_dir(analysis_dir, spec)
        preds_standard, preds_weighted = load_predictions(spec)
        weight_label = formal_weight_label(spec, config)

        for cap_w in capacity_weightings:
            tok = _cap_token(cap_w)
            logger.info("--- capacity weighting: %s ---", cap_w)
            # Positions are AUM-independent; build each book once, re-cost per AUM.
            book_std = build_capacity_book(
                preds_standard, panel, config, spec, capacity_weighting=cap_w
            )
            book_wt = build_capacity_book(
                preds_weighted, panel, config, spec, capacity_weighting=cap_w
            )

            for aum in aum_scenarios:
                label = aum_label(aum)
                row_std, series_std = capacity_metrics_from_book(
                    *book_std, tc_context, aum
                )
                row_wt, series_wt = capacity_metrics_from_book(
                    *book_wt, tc_context, aum
                )
                if row_std is None or row_wt is None:
                    logger.warning(
                        "No valid capacity book for %s [%s] @ %s; skipping",
                        spec["spec_label"],
                        cap_w,
                        label,
                    )
                    continue

                cmp = compare_standard_vs_weighted(
                    series_std, series_wt, row_std, row_wt, config
                )
                base = {"weight_label": weight_label, "aum_label": label}
                rows = [
                    {"row_type": "standard", **base, **row_std},
                    {"row_type": "weighted", **base, **row_wt},
                    {"row_type": "difference", **base, **cmp},
                ]
                pd.DataFrame(rows).reindex(columns=METRICS_COLS).to_csv(
                    out_dir / f"capacity_portfolio{tok}_metrics_{label}.csv",
                    index=False,
                )

                monthly = pd.concat(
                    [
                        series_std.assign(row_type="standard").reset_index(),
                        series_wt.assign(row_type="weighted").reset_index(),
                    ],
                    ignore_index=True,
                )
                monthly[MONTHLY_COLS].to_csv(
                    out_dir / f"capacity_portfolio{tok}_monthly_{label}.csv",
                    index=False,
                )

                if not args.no_figures:
                    plot_capacity_net_cumret(
                        series_std,
                        series_wt,
                        out_dir / f"capacity_portfolio{tok}_net_cumret_{label}.png",
                        spec["model"],
                        label,
                    )

    logger.info("Done")


if __name__ == "__main__":
    main()
