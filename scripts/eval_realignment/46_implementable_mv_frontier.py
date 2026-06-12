"""
46 - Implementable Mean-Variance Frontier (note section 4.5; JKMP 2024, Static tier)
====================================================================================

Feeds the existing one-month predictions into the Static-ML allocator of Jensen,
Kelly, Malamud, and Pedersen (2024): per month,

    theta_t = (gamma*Sigma_t + Lambda)^{-1} (mu_hat_t + Lambda*theta_drift),

with Sigma from a JKMP-style characteristic-factor model by default (intercept +
8 broad feature themes; factor-return EWMA with 6m/18m variance/correlation
half-lives; per-name idio EWMA; Woodbury solves) on the within-month top-60%
DolVol universe — the same screen as script 45 — or, with --cov lw, Ledoit-Wolf
shrinkage on the liquid top-1000. The diagonal quadratic (Kyle-lambda) cost
Lambda_ii = 2*lam*sigma_i*AUM/ADV_i is built from the same Frazzini inputs as the
realized accounting. The 2x2:

    columns: A = cost-blind Markowitz (Lambda=0, memoryless) vs
             B = cost-aware Static-ML (Lambda on)
    rows:    standard vs weighted predictions through the same allocator.

Headline: net certainty-equivalent CE(gamma) with each book built AND evaluated at
its own gamma over the grid {10, 100, 500} (raw mu_hat through unconstrained MV
levers extremely at low gamma; the book scales as 1/gamma). Every book is reported
in two VIEWS: raw (the literal optimizer output, leverage as a diagnostic) and
unit_gross (rescaled to Sum|theta|=1 before evaluation - the 42/43/45-comparable
scale; used for the old-format two_by_two at gamma=10). Realized net returns always
use the true half-spread + sqrt-impact accounting. PropTC is not run (no impact =>
Lambda=0 => B == A). The full cross-section remains covered by script 41; all four
cells share the same universe, Sigma, and cost inputs each month, so the within-46
comparisons are clean.

For each spec it writes, under outputs/eval_realignment/analysis/{model}/{weight_spec}/:
  * implementable_mv_metrics_{aum}.csv     -- book x row_type x gamma (+ differences)
  * implementable_mv_monthly_{aum}.csv     -- stacked monthly series incl. leverage
  * mv_two_by_two_{aum}.csv + mv_two_by_two_timeseries_{aum}.xlsx  -- old format @ gamma=10
  * implementable_mv_frontier_{aum}.png    -- net mean vs net SD, points per gamma
  * implementable_mv_diag.csv              -- per-month universe sizes
and per model: outputs/eval_realignment/tables/{model}/implementable_mv_tables.xlsx.
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

from src.analysis.eval_realignment.capacity_portfolio import (  # noqa: E402
    compare_standard_vs_weighted,
)
from src.analysis.eval_realignment.implementable_mv import (  # noqa: E402
    TOP_N,
    mv_metrics,
    mv_net_returns,
    plot_implementable_frontier,
    rescale_positions_unit_gross,
    run_mv_books,
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
    is_proportional_tc_scenario,
    load_predictions,
)
from src.config import load_config  # noqa: E402
from src.data.loader import load_processed_panel  # noqa: E402
from src.portfolio.construction import prepare_transaction_cost_context  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("46_implementable_mv_frontier")

DEFAULT_GAMMAS = [10.0, 100.0, 500.0]
DEFAULT_AUMS_MILLIONS = [100, 500, 1000]
BASE_GAMMA = 10.0  # JKMP base case; used for the old-format two_by_two deliverable

METRICS_COLS = [
    "book", "row_type", "gamma", "view", "cov_model",
    "weight_label", "aum_label", "n_months",
    "net_ce_own_gamma", "gross_ce_own_gamma",
    "net_ce_annual_g1", "net_ce_annual_g5", "net_ce_annual_g10",
    "gross_ce_annual_g1", "gross_ce_annual_g5", "gross_ce_annual_g10",
    "gross_sr_annual", "net_sr_annual", "gross_mean_annual", "net_mean_annual",
    "net_sd_annual", "turnover_mean", "tc_mean_monthly", "leverage_mean",
    "net_sr_diff", "net_sr_diff_pval", "net_sr_diff_reject", "net_mean_diff_nw_t",
    "net_ce_diff_annual_g1", "net_ce_diff_annual_g5", "net_ce_diff_annual_g10",
]

MONTHLY_COLS = [
    "yyyymm", "book", "row_type", "gamma", "view", "ret_gross", "ret_net",
    "transaction_cost", "turnover", "n_long", "n_short", "leverage",
]

CELL_OF = {
    ("standard", "markowitz"): "1A",
    ("standard", "static_ml"): "1B",
    ("weighted", "markowitz"): "2A",
    ("weighted", "static_ml"): "2B",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Implementable MV frontier (JKMP Static tier) on the liquid universe"
    )
    add_experiment_filters(parser)
    parser.add_argument(
        "--aum", nargs="*", type=float, default=DEFAULT_AUMS_MILLIONS,
        help="AUM scenarios in $M (impact-bearing only; PropTC degenerates).",
    )
    parser.add_argument(
        "--gamma", nargs="*", type=float, default=DEFAULT_GAMMAS,
        help="Risk-aversion grid; each book is built and evaluated at its own gamma.",
    )
    parser.add_argument(
        "--cov", choices=["factor", "lw"], default="factor",
        help="Risk model: JKMP-style characteristic-factor (default) or Ledoit-Wolf.",
    )
    parser.add_argument(
        "--screen-pct", type=float, default=0.60,
        help="Factor mode: keep the within-month top fraction by DolVol (45's screen).",
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N,
        help="LW mode only: largest-N names by DolVol with sufficient history.",
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    base_dir, experiment_dir, analysis_dir = eval_realignment_output_dirs(config)
    specs = load_filtered_specs(experiment_dir, args, logger)

    aums = []
    for value in args.aum:
        if is_proportional_tc_scenario(value):
            logger.warning("PropTC skipped: Lambda=0 makes Static-ML == Markowitz")
            continue
        aums.append(float(value) * 1_000_000)
    gammas = sorted(set(float(g) for g in args.gamma))

    logger.info("Loading processed panel")
    panel = load_processed_panel()
    tc_context = prepare_transaction_cost_context(panel, config)

    blocks: dict[str, dict[str, list]] = {}

    for spec in specs:
        logger.info("=== %s ===", spec["spec_label"])
        out_dir = eval_realignment_spec_dir(analysis_dir, spec)
        preds_std, preds_wt = load_predictions(spec)
        weight_label = formal_weight_label(spec, config)

        books, diag_df = run_mv_books(
            {"standard": preds_std, "weighted": preds_wt},
            panel, config, tc_context, gammas, aums,
            top_n=args.top_n, cov_model=args.cov, screen_pct=args.screen_pct,
        )
        if diag_df.empty:
            logger.warning("No MV books for %s; skipping spec", spec["spec_label"])
            continue
        sfx = "" if args.cov == "factor" else f"_{args.cov}"
        diag_df.to_csv(out_dir / f"implementable_mv_diag{sfx}.csv", index=False)

        for aum in aums:
            label = aum_label(aum)
            metric_rows: list[dict] = []
            monthly_frames: list[pd.DataFrame] = []
            base_cells: dict[str, pd.DataFrame] = {}
            for book in ("markowitz", "static_ml"):
                for gamma in gammas:
                    for view in ("raw", "unit_gross"):
                        rows_by_type: dict[str, tuple] = {}
                        for row_type in ("standard", "weighted"):
                            aum_key = None if book == "markowitz" else aum
                            gross_df, positions = books[
                                (row_type, book, gamma, aum_key)
                            ]
                            if view == "unit_gross":
                                gross_df, positions = rescale_positions_unit_gross(
                                    gross_df, positions
                                )
                            net = mv_net_returns(gross_df, positions, tc_context, aum)
                            row, series = mv_metrics(net, gamma)
                            rows_by_type[row_type] = (row, series)
                            base = {
                                "book": book, "row_type": row_type, "view": view,
                                "cov_model": args.cov,
                                "weight_label": weight_label, "aum_label": label,
                            }
                            metric_rows.append({**base, **row})
                            frame = series.reset_index()
                            frame.insert(1, "book", book)
                            frame.insert(2, "row_type", row_type)
                            frame.insert(3, "gamma", gamma)
                            frame.insert(4, "view", view)
                            monthly_frames.append(frame)
                            if gamma == BASE_GAMMA and view == "unit_gross":
                                base_cells[CELL_OF[(row_type, book)]] = series
                        cmp = compare_standard_vs_weighted(
                            rows_by_type["standard"][1], rows_by_type["weighted"][1],
                            rows_by_type["standard"][0], rows_by_type["weighted"][0],
                            config,
                        )
                        metric_rows.append({
                            "book": book, "row_type": "difference", "gamma": gamma,
                            "view": view, "cov_model": args.cov,
                            "weight_label": weight_label, "aum_label": label, **cmp,
                        })

            pd.DataFrame(metric_rows).reindex(columns=METRICS_COLS).to_csv(
                out_dir / f"implementable_mv_metrics{sfx}_{label}.csv", index=False
            )
            pd.concat(monthly_frames, ignore_index=True).reindex(
                columns=MONTHLY_COLS
            ).to_csv(out_dir / f"implementable_mv_monthly{sfx}_{label}.csv", index=False)

            # Old-format deliverables at the JKMP base case gamma.
            if len(base_cells) == 4:
                common = sorted(
                    set.intersection(*(set(s.index) for s in base_cells.values()))
                )
                if common:
                    cells = {
                        c: s.loc[common][
                            ["ret_gross", "ret_net", "transaction_cost",
                             "turnover", "n_long", "n_short"]
                        ]
                        for c, s in base_cells.items()
                    }
                    pd.DataFrame(format_two_by_two_rows(cells, config)).to_csv(
                        out_dir / f"mv_two_by_two{sfx}_{label}.csv", index=False
                    )
                    write_timeseries_workbook(
                        cells, out_dir / f"mv_two_by_two_timeseries{sfx}_{label}.xlsx"
                    )
                    table = build_table_values(cells, config)
                    raw_lev = {
                        r["book"]: r["leverage_mean"]
                        for r in metric_rows
                        if r.get("view") == "raw" and r.get("gamma") == BASE_GAMMA
                        and r.get("row_type") == "weighted"
                    }
                    panel_c_rows = [
                        ("Raw-view mean leverage (wt Markowitz)",
                         raw_lev.get("markowitz", float("nan"))),
                        ("Raw-view mean leverage (wt Static-ML)",
                         raw_lev.get("static_ml", float("nan"))),
                        ("Mean universe size", float(diag_df["n_priced"].mean())),
                        ("gamma (base case)", BASE_GAMMA),
                    ]
                    title = (
                        f"Implementable MV 2x2 — {spec['model']} / "
                        f"{spec['weight_spec']} ({label}, γ={BASE_GAMMA:g}, "
                        f"unit-gross view)"
                    )
                    universe_desc = (
                        f"top-{args.screen_pct:.0%} DolVol universe"
                        if args.cov == "factor"
                        else f"top-{args.top_n} DolVol universe"
                    )
                    subtitle = (
                        f"weights: {weight_label}; {universe_desc}; "
                        f"cov: {args.cov}; A = Markowitz (Λ=0), B = Static-ML"
                    )
                    blocks.setdefault(spec["model"], {}).setdefault(
                        spec["weight_spec"], []
                    ).append((title, subtitle, table, panel_c_rows))

            if not args.no_figures:
                plot_implementable_frontier(
                    [r for r in metric_rows
                     if r["row_type"] != "difference" and r.get("view") == "raw"],
                    out_dir / f"implementable_mv_frontier{sfx}_{label}.png",
                    spec["model"], label,
                )
            logger.info("Wrote implementable_mv outputs @ %s", label)

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
                    column_headers=("Markowitz", "Static-ML"),
                    panel_c_title="Panel C: Optimizer Diagnostics",
                    panel_c_rows=pc_rows,
                )
        out_path = tables_dir / f"implementable_mv_tables{'' if args.cov == 'factor' else '_' + args.cov}.xlsx"
        wb.save(out_path)
        logger.info("Saved %s", out_path)

    logger.info("Done")


if __name__ == "__main__":
    main()
