"""
43 - Breakeven-Gated Capacity Portfolio (parameter-free no-trade gate)
======================================================================

Cost-aware execution layer on the section-4.2 capacity book (script 42). Instead of
fully rebalancing to the target theta*_t = w̃·(r̂ − rbar_w) each month, every name is
gated by its own breakeven: trade fully to target iff the deployment alpha clears the
one-way proportional cost,

    |r̂_i − rbar_w_t| >= ½·spread_i,

otherwise hold the drifted position. Both sides are monthly returns, so there is no
tuning parameter and the gate is AUM-independent; market impact stays in the realized
net-of-cost accounting. Names that leave the prediction universe are closed.

For each spec it builds the full-rebalance (42) and breakeven-gated books from BOTH
standard and weighted predictions, and writes:
  * capacity_breakeven_metrics_{aum}.csv  -- book (full_rebalance/breakeven_gate)
                                             x row_type (standard/weighted/difference)
  * capacity_breakeven_monthly_{aum}.csv  -- gated book monthly series (std & weighted)
  * capacity_breakeven_gate_diag.csv      -- per-month gate pass rates (AUM-independent)
  * capacity_breakeven_net_cumret_{aum}.png  (unless --no-figures)

Predictions are READ from outputs/formalanalysis/experiment/...; outputs go under
outputs/eval_realignment/analysis/{model}/{weight_spec}/. Script 42 and
capacity_portfolio.py are reused read-only and not modified.
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

# One directory deeper than scripts/*.py -> repo root is three parents up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402

from src.analysis.eval_realignment.breakeven_capacity import (  # noqa: E402
    _returns_by_month,
    build_breakeven_book,
    compute_alpha_maps,
    gate_diagnostics,
    metrics_from_book,
    plot_breakeven_net_cumret,
)
from src.analysis.eval_realignment.capacity_portfolio import (  # noqa: E402
    build_capacity_book,
    compare_standard_vs_weighted,
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
logger = logging.getLogger("43_breakeven_capacity_portfolio")

METRICS_COLS = [
    "book", "row_type", "weight_label", "aum_label", "n_months",
    "gross_sr_annual", "net_sr_annual", "gross_sr_monthly", "net_sr_monthly",
    "gross_mean_annual", "net_mean_annual",
    "gross_ce_annual_g1", "gross_ce_annual_g5", "gross_ce_annual_g10",
    "net_ce_annual_g1", "net_ce_annual_g5", "net_ce_annual_g10",
    "turnover_mean", "tc_mean_monthly",
    "net_sr_diff", "net_sr_diff_pval", "net_sr_diff_reject", "net_mean_diff_nw_t",
    "net_ce_diff_annual_g1", "net_ce_diff_annual_g5", "net_ce_diff_annual_g10",
]

MONTHLY_COLS = [
    "yyyymm", "row_type", "ret_gross", "ret_net",
    "transaction_cost", "turnover", "n_long", "n_short",
]

DIAG_COLS = [
    "row_type", "yyyymm", "n_names", "n_pass", "pass_frac",
    "gross_pass_frac", "median_abs_alpha", "median_half_spread",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Breakeven-gated capacity portfolio (parameter-free no-trade gate)"
    )
    add_experiment_filters(parser)
    parser.add_argument(
        "--aum", nargs="*", default=None,
        help="AUM scenarios in $M, 'proportional_tc', or 'all' (default).",
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    _, experiment_dir, analysis_dir = eval_realignment_output_dirs(config)
    specs = load_filtered_specs(experiment_dir, args, logger)

    logger.info("Loading processed panel")
    panel = load_processed_panel()
    aum_scenarios = parse_aum_millions(args.aum, config)
    tc_context = prepare_transaction_cost_context(panel, config)

    for spec in specs:
        logger.info("=== %s ===", spec["spec_label"])
        out_dir = eval_realignment_spec_dir(analysis_dir, spec)
        preds_std, preds_wt = load_predictions(spec)
        weight_label = formal_weight_label(spec, config)

        # Expensive inputs computed once per prediction set (gate is AUM-independent).
        books = {}
        diag_frames = []
        for row_type, preds in [("standard", preds_std), ("weighted", preds_wt)]:
            targets = build_capacity_book(preds, panel, config, spec)
            if not targets[1]:
                logger.warning(
                    "No capacity book for %s (%s); skipping spec",
                    spec["spec_label"], row_type,
                )
                books = {}
                break
            alpha_maps = compute_alpha_maps(preds, panel, config, spec)
            ret_by_month = _returns_by_month(panel, config, set(targets[1]))
            gated = build_breakeven_book(
                preds, panel, config, spec, tc_context,
                targets=targets, alpha_maps=alpha_maps, ret_by_month=ret_by_month,
            )
            books[row_type] = {"full_rebalance": targets, "breakeven_gate": gated}
            diag = gate_diagnostics(targets[1], alpha_maps, tc_context)
            diag.insert(0, "row_type", row_type)
            diag_frames.append(diag)
        if not books:
            continue

        diag_df = pd.concat(diag_frames, ignore_index=True).reindex(columns=DIAG_COLS)
        diag_df.to_csv(out_dir / "capacity_breakeven_gate_diag.csv", index=False)

        for aum in aum_scenarios:
            label = aum_label(aum)
            metric_rows = []
            monthly_frames = []
            cumret_series = {}
            for book in ("full_rebalance", "breakeven_gate"):
                gross_s, pos_s = books["standard"][book]
                gross_w, pos_w = books["weighted"][book]
                row_s, ser_s = metrics_from_book(gross_s, pos_s, tc_context, aum)
                row_w, ser_w = metrics_from_book(gross_w, pos_w, tc_context, aum)
                if row_s is None or row_w is None:
                    logger.warning(
                        "No %s book for %s @ %s", book, spec["spec_label"], label
                    )
                    continue
                cmp = compare_standard_vs_weighted(ser_s, ser_w, row_s, row_w, config)
                base = {"weight_label": weight_label, "aum_label": label, "book": book}
                metric_rows.append({"row_type": "standard", **base, **row_s})
                metric_rows.append({"row_type": "weighted", **base, **row_w})
                metric_rows.append({"row_type": "difference", **base, **cmp})

                short = "full" if book == "full_rebalance" else "gated"
                cumret_series[f"standard {short}"] = ser_s
                cumret_series[f"weighted {short}"] = ser_w
                if book == "breakeven_gate":
                    for row_type, ser in [("standard", ser_s), ("weighted", ser_w)]:
                        frame = ser.reset_index()
                        frame.insert(1, "row_type", row_type)
                        monthly_frames.append(frame)

            if not metric_rows:
                continue
            metrics_df = pd.DataFrame(metric_rows).reindex(columns=METRICS_COLS)
            metrics_df.to_csv(
                out_dir / f"capacity_breakeven_metrics_{label}.csv", index=False
            )
            if monthly_frames:
                monthly_df = pd.concat(monthly_frames, ignore_index=True).reindex(
                    columns=MONTHLY_COLS
                )
                monthly_df.to_csv(
                    out_dir / f"capacity_breakeven_monthly_{label}.csv", index=False
                )
            if not args.no_figures and cumret_series:
                plot_breakeven_net_cumret(
                    cumret_series,
                    out_dir / f"capacity_breakeven_net_cumret_{label}.png",
                    spec["model"], label,
                )

    logger.info("Done")


if __name__ == "__main__":
    main()
