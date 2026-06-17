"""
44 - Capacity 2x2 Tables (old-format deliverables)
==================================================

Presentation layer for the capacity track, mirroring the formalanalysis portfolio
deliverables (21e + 22) with the 2x3 cell grid reduced to the 2x2:

    1A = standard x full rebalance     1B = standard x breakeven gate
    2A = weighted x full rebalance     2B = weighted x breakeven gate

Old metric names are kept verbatim; "Net portfolio effect annualized" now means the
breakeven-gate (cost-device) effect SR(1B) - SR(1A). The tc-target (C) column is out
of scope.

INPUTS (per spec dir, written by scripts 42 and 43 — both must have been run):
  capacity_portfolio_monthly_{aum}.csv, capacity_breakeven_monthly_{aum}.csv,
  capacity_breakeven_gate_diag.csv

OUTPUTS:
  per spec dir: two_by_two_{aum}.csv (long metric/value format incl. effect
    decomposition, Ledoit-Wolf p-values, factor alphas) and
    two_by_two_timeseries_{aum}.xlsx (one sheet per cell);
  per model: outputs/eval_realignment/tables/{model}/capacity_two_by_two_tables.xlsx
    (one sheet per weight spec, AUM blocks with Panels A/B/C).

No panel is loaded and no book is rebuilt; missing inputs are skipped with a warning.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# One directory deeper than scripts/*.py -> repo root is three parents up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402

from src.analysis.eval_realignment.script_utils import (  # noqa: E402
    add_experiment_filters,
    eval_realignment_output_dirs,
    eval_realignment_spec_dir,
    load_filtered_specs,
)
from src.analysis.eval_realignment.two_by_two_tables import (  # noqa: E402
    build_table_values,
    format_two_by_two_rows,
    load_cell_series,
    set_two_by_two_widths,
    summarize_gate_diag,
    write_timeseries_workbook,
    write_two_by_two_block,
)
from src.analysis.formal.common import aum_label, formal_weight_label  # noqa: E402
from src.analysis.formal.script_utils import parse_aum_millions  # noqa: E402
from src.config import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("44_capacity_two_by_two_tables")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Old-format 2x2 tables for the capacity track (reads 42/43 outputs)"
    )
    add_experiment_filters(parser)
    parser.add_argument(
        "--aum", nargs="*", default=None,
        help="AUM scenarios in $M, 'proportional_tc', or 'all' (default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    base_dir, experiment_dir, analysis_dir = eval_realignment_output_dirs(config)
    specs = load_filtered_specs(experiment_dir, args, logger)
    aum_scenarios = parse_aum_millions(args.aum, config)

    # blocks[model][weight_spec] = list of (title, subtitle, table, gate_diag)
    blocks: dict[str, dict[str, list]] = {}

    for spec in specs:
        logger.info("=== %s ===", spec["spec_label"])
        spec_dir = eval_realignment_spec_dir(analysis_dir, spec)
        weight_label = formal_weight_label(spec, config)

        diag_path = spec_dir / "capacity_breakeven_gate_diag.csv"
        if not diag_path.exists():
            logger.warning("Missing %s; run script 43 first — skipping spec", diag_path)
            continue
        gate_diag = summarize_gate_diag(pd.read_csv(diag_path))

        for aum in aum_scenarios:
            label = aum_label(aum)
            cells = load_cell_series(spec_dir, label)
            if cells is None:
                logger.warning(
                    "Inputs incomplete for %s @ %s (need 42 and 43 outputs); skipping",
                    spec["spec_label"], label,
                )
                continue

            rows = format_two_by_two_rows(cells, config)
            pd.DataFrame(rows).to_csv(spec_dir / f"two_by_two_{label}.csv", index=False)
            write_timeseries_workbook(
                cells, spec_dir / f"two_by_two_timeseries_{label}.xlsx"
            )

            table = build_table_values(cells, config)
            title = f"Capacity 2x2 — {spec['model']} / {spec['weight_spec']} ({label})"
            subtitle = f"weights: {weight_label}; A = full rebalance, B = breakeven gate"
            blocks.setdefault(spec["model"], {}).setdefault(
                spec["weight_spec"], []
            ).append((title, subtitle, table, gate_diag))
            logger.info("Wrote two_by_two_%s.csv / .xlsx", label)

    # Per-model formatted workbook: one sheet per weight spec, AUM blocks stacked.
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
            for title, subtitle, table, gate_diag in spec_blocks:
                row = write_two_by_two_block(
                    ws, table, gate_diag, row, title, subtitle,
                    star_effects=True, include_panel_c=False,
                )
        out_path = tables_dir / "capacity_two_by_two_tables.xlsx"
        wb.save(out_path)
        logger.info("Saved %s", out_path)

    logger.info("Done")


if __name__ == "__main__":
    main()
