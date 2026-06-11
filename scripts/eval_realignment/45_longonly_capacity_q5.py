"""
45 - Capacity-Weighted Long-Only Q5 Book (note section 4.3)
===========================================================

The most defensible implementable object: a long-only, capacity-weighted top-quintile
book drawn from the liquid universe (top-60% by dollar volume), dropping the illiquid
short leg entirely. Per month: Q5 = predictions at or above the 80th percentile within
the screen; weights are the spec's own w_tilde re-trued monthly (fully invested,
sum theta = 1). The cost-aware B column is the parameter-free cost-scaled membership
HYSTERESIS: a held name that left Q5 is kept while r_hat >= b_t - (half_spread_own +
median half_spread of the month's Q5), since selling it to buy the marginal entrant
only pays when the prediction gap covers both tolls. Entries are ungated; screen or
universe exit forces the sale. As spreads -> 0 the hysteresis book equals the plain
book. (The script-43 breakeven gate deliberately does not transfer: quantile-book
turnover is high-alpha boundary churn, which an |alpha| >= cost gate would not damp.)

Cells: 1A = standard x plain, 1B = standard x hysteresis,
       2A = weighted x plain, 2B = weighted x hysteresis.

For each spec it writes, under outputs/eval_realignment/analysis/{model}/{weight_spec}/:
  * longonly_q5_metrics_{aum}.csv      -- book (plain/hysteresis) x row_type
  * longonly_q5_monthly_{aum}.csv      -- stacked monthly series, all four cells
  * longonly_two_by_two_{aum}.csv      -- old-format metric,value rows (incl. alphas)
  * longonly_two_by_two_timeseries_{aum}.xlsx  -- one sheet per cell
  * longonly_hysteresis_diag.csv       -- per-month hysteresis diagnostics (AUM-free)
  * longonly_net_cumret_{aum}.png      -- unless --no-figures
and per model: outputs/eval_realignment/tables/{model}/longonly_two_by_two_tables.xlsx
(Panel C = hysteresis diagnostics).

Predictions are READ from outputs/formalanalysis/experiment/...; scripts 42-44 and
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
    metrics_from_book,
)
from src.analysis.eval_realignment.capacity_portfolio import (  # noqa: E402
    compare_standard_vs_weighted,
)
from src.analysis.eval_realignment.longonly_capacity import (  # noqa: E402
    build_longonly_book,
    compute_longonly_inputs,
    plot_longonly_net_cumret,
    summarize_hysteresis_diag,
)
from src.analysis.eval_realignment.script_utils import (  # noqa: E402
    add_experiment_filters,
    eval_realignment_output_dirs,
    eval_realignment_spec_dir,
    load_filtered_specs,
)
from src.analysis.eval_realignment.two_by_two_tables import (  # noqa: E402
    build_table_values,
    format_two_by_two_rows,
    set_two_by_two_widths,
    write_timeseries_workbook,
    write_two_by_two_block,
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
logger = logging.getLogger("45_longonly_capacity_q5")

BOOKS = ("plain", "hysteresis")
CELL_OF = {
    ("standard", "plain"): "1A",
    ("standard", "hysteresis"): "1B",
    ("weighted", "plain"): "2A",
    ("weighted", "hysteresis"): "2B",
}

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
    "yyyymm", "book", "row_type", "ret_gross", "ret_net",
    "transaction_cost", "turnover", "n_long", "n_short",
]

DIAG_COLS = [
    "row_type", "yyyymm", "n_members", "n_q5", "n_buffer_held",
    "n_forced_exits", "entry_threshold", "buffer_width_median",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capacity-weighted long-only Q5 book (top-60% DolVol screen)"
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
    base_dir, experiment_dir, analysis_dir = eval_realignment_output_dirs(config)
    specs = load_filtered_specs(experiment_dir, args, logger)

    logger.info("Loading processed panel")
    panel = load_processed_panel()
    aum_scenarios = parse_aum_millions(args.aum, config)
    tc_context = prepare_transaction_cost_context(panel, config)

    # blocks[model][weight_spec] = list of (title, subtitle, table, panel_c_rows)
    blocks: dict[str, dict[str, list]] = {}

    for spec in specs:
        logger.info("=== %s ===", spec["spec_label"])
        out_dir = eval_realignment_spec_dir(analysis_dir, spec)
        preds_std, preds_wt = load_predictions(spec)
        weight_label = formal_weight_label(spec, config)

        # Books are AUM-independent; build the four cells once per spec.
        books: dict[str, dict[str, tuple]] = {}
        diag_frames = []
        for row_type, preds in [("standard", preds_std), ("weighted", preds_wt)]:
            inputs = compute_longonly_inputs(preds, panel, config, spec)
            if not inputs:
                logger.warning("No liquid-universe inputs for %s (%s); skipping spec",
                               spec["spec_label"], row_type)
                books = {}
                break
            plain = build_longonly_book(
                preds, panel, config, spec, tc_context,
                hysteresis=False, inputs=inputs,
            )
            diag_rows: list[dict] = []
            hyst = build_longonly_book(
                preds, panel, config, spec, tc_context,
                hysteresis=True, inputs=inputs, diag_out=diag_rows,
            )
            if plain[0].empty or hyst[0].empty:
                logger.warning("Empty long-only book for %s (%s); skipping spec",
                               spec["spec_label"], row_type)
                books = {}
                break
            books[row_type] = {"plain": plain, "hysteresis": hyst}
            diag = pd.DataFrame(diag_rows)
            diag.insert(0, "row_type", row_type)
            diag_frames.append(diag)
        if not books:
            continue

        diag_df = pd.concat(diag_frames, ignore_index=True).reindex(columns=DIAG_COLS)
        diag_df.to_csv(out_dir / "longonly_hysteresis_diag.csv", index=False)
        panel_c_rows = []
        for row_type in ("standard", "weighted"):
            s = summarize_hysteresis_diag(diag_df[diag_df["row_type"] == row_type])
            panel_c_rows.extend([
                (f"Mean members ({row_type})", s.get("mean_members", float("nan"))),
                (f"Mean buffer holds / month ({row_type})", s.get("mean_buffer_held", float("nan"))),
                (f"Mean forced exits / month ({row_type})", s.get("mean_forced_exits", float("nan"))),
            ])
        panel_c_rows.append((
            "Median buffer width (return units)",
            summarize_hysteresis_diag(
                diag_df[diag_df["row_type"] == "standard"]
            ).get("median_buffer_width", float("nan")),
        ))

        for aum in aum_scenarios:
            label = aum_label(aum)
            metric_rows = []
            monthly_frames = []
            series_by_cell: dict[str, pd.DataFrame] = {}
            cumret_series: dict[str, pd.DataFrame] = {}
            ok = True
            for book in BOOKS:
                row_s, ser_s = metrics_from_book(*books["standard"][book], tc_context, aum)
                row_w, ser_w = metrics_from_book(*books["weighted"][book], tc_context, aum)
                if row_s is None or row_w is None:
                    logger.warning("No %s book for %s @ %s", book, spec["spec_label"], label)
                    ok = False
                    break
                cmp = compare_standard_vs_weighted(ser_s, ser_w, row_s, row_w, config)
                base = {"weight_label": weight_label, "aum_label": label, "book": book}
                metric_rows.append({"row_type": "standard", **base, **row_s})
                metric_rows.append({"row_type": "weighted", **base, **row_w})
                metric_rows.append({"row_type": "difference", **base, **cmp})
                for row_type, ser in [("standard", ser_s), ("weighted", ser_w)]:
                    series_by_cell[CELL_OF[(row_type, book)]] = ser
                    cumret_series[f"{row_type} {book}"] = ser
                    frame = ser.reset_index()
                    frame.insert(1, "book", book)
                    frame.insert(2, "row_type", row_type)
                    monthly_frames.append(frame)
            if not ok or not metric_rows:
                continue

            pd.DataFrame(metric_rows).reindex(columns=METRICS_COLS).to_csv(
                out_dir / f"longonly_q5_metrics_{label}.csv", index=False
            )
            pd.concat(monthly_frames, ignore_index=True).reindex(
                columns=MONTHLY_COLS
            ).to_csv(out_dir / f"longonly_q5_monthly_{label}.csv", index=False)

            # Old-format deliverables on the four aligned cells.
            common = sorted(
                set.intersection(*(set(s.index) for s in series_by_cell.values()))
            )
            if common:
                cells = {c: s.loc[common] for c, s in series_by_cell.items()}
                pd.DataFrame(format_two_by_two_rows(cells, config)).to_csv(
                    out_dir / f"longonly_two_by_two_{label}.csv", index=False
                )
                write_timeseries_workbook(
                    cells, out_dir / f"longonly_two_by_two_timeseries_{label}.xlsx"
                )
                table = build_table_values(cells, config)
                title = (
                    f"Long-only Q5 2x2 — {spec['model']} / {spec['weight_spec']} ({label})"
                )
                subtitle = (
                    f"weights: {weight_label}; top-60% DolVol screen; "
                    f"A = plain Q5, B = membership hysteresis"
                )
                blocks.setdefault(spec["model"], {}).setdefault(
                    spec["weight_spec"], []
                ).append((title, subtitle, table, panel_c_rows))

            if not args.no_figures:
                plot_longonly_net_cumret(
                    cumret_series,
                    out_dir / f"longonly_net_cumret_{label}.png",
                    spec["model"], label,
                )
            logger.info("Wrote longonly outputs @ %s", label)

    from openpyxl import Workbook

    for model, by_spec in blocks.items():
        tables_dir = base_dir / "tables" / model
        tables_dir.mkdir(parents=True, exist_ok=True)
        wb = Workbook()
        wb.remove(wb.active)
        for weight_spec, spec_blocks in by_spec.items():
            ws = wb.create_sheet(title=weight_spec[:31])
            set_two_by_two_widths(ws)
            row = 1
            for title, subtitle, table, pc_rows in spec_blocks:
                row = write_two_by_two_block(
                    ws, table, {}, row, title, subtitle,
                    column_headers=("Plain Q5", "Membership hysteresis"),
                    panel_c_title="Panel C: Hysteresis Diagnostics",
                    panel_c_rows=pc_rows,
                )
        out_path = tables_dir / "longonly_two_by_two_tables.xlsx"
        wb.save(out_path)
        logger.info("Saved %s", out_path)

    logger.info("Done")


if __name__ == "__main__":
    main()
