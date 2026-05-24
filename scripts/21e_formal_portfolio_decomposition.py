"""
21e - Formal Portfolio Decomposition
====================================

Generate the formal Section 9 portfolio tables:
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/two_by_three_{aum}.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/two_by_three_timeseries_{aum}.xlsx
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/prediction_quantile_timeseries_{aum}.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/prediction_quantile_timeseries_{aum}.xlsx
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/within_quintile_portfolio.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/within_quintile_portfolio_tc_sort_{aum}.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/within_quintile_portfolio_scaled_aum_{aum}.csv

Only the within-liquidity-quintile/scissors outputs use liquidity breakpoints
and are written under:
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/liquidity_breakpoints/{mode}/
The 2x3 prediction-quantile decomposition stays in ``{portfolio_run}`` because
it does not use liquidity breakpoints.
"""

from __future__ import annotations

import argparse
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

from src.analysis.formal.common import (  # noqa: E402
    aum_label,
    is_proportional_tc_scenario,
    load_predictions,
    scenario_sizing_aum,
)
from src.analysis.formal.portfolio_tables import (  # noqa: E402
    compute_quintile_aum_scaling,
    compute_quintile_sr_scissors_tables,
    compute_two_by_three_decomposition,
    format_prediction_quantile_timeseries_rows,
    format_within_quintile_from_scissors,
    format_two_by_three_decomposition_rows,
    format_two_by_three_timeseries_rows,
    plot_sr_scissors_comparison,
    plot_sr_scissors_table,
)
from src.analysis.formal.script_utils import (  # noqa: E402
    add_experiment_filters,
    add_liquidity_breakpoint_option,
    config_for_liquidity_breakpoint,
    formal_output_dirs,
    formal_spec_dir,
    liquidity_breakpoint_dir,
    load_filtered_specs,
    namespace_liquidity_breakpoints,
    parse_aum_millions,
    resolve_liquidity_breakpoints,
)
from src.config import load_config  # noqa: E402
from src.data.loader import load_processed_panel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("21e_formal_portfolio_decomposition")

INTERNAL_PORTFOLIO_MODE = "long_short"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate formal portfolio decomposition tables"
    )
    add_experiment_filters(parser)
    add_liquidity_breakpoint_option(parser)
    parser.add_argument(
        "--aum",
        type=str,
        action="append",
        default=None,
        help=(
            "Net-return TC scenario. Use an AUM in $M, or proportional_tc for "
            "spread/2 only. Repeat for multiple values. Default runs all "
            "transaction_costs.aum_scenarios."
        ),
    )
    parser.add_argument(
        "--primary-aum",
        type=int,
        default=500,
        help="Primary AUM in $M for within-quintile tables.",
    )
    parser.add_argument(
        "--skip-within-quintile",
        action="store_true",
        help="Skip within-quintile portfolio tables.",
    )
    parser.add_argument(
        "--skip-decomposition",
        action="store_true",
        help="Skip the 2x3 decomposition and only generate within-quintile outputs.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write tables only and skip PNG figures.",
    )
    parser.add_argument(
        "--skip-tc-aware-scissors",
        action="store_true",
        help=(
            "Skip extra within-quintile/scissors outputs for weighted "
            "predictions with TC-aware sorting."
        ),
    )
    parser.add_argument(
        "--scaled-quintile-aum",
        action="store_true",
        help=(
            "Also write within-quintile/scissors robustness outputs that scale "
            "each quintile's AUM by its average selected-leg stock count "
            "relative to Q1."
        ),
    )
    parser.add_argument(
        "--portfolio-weighting",
        choices=["equal", "value", "both"],
        default="equal",
        help=(
            "Selected-stock weights inside each long/short leg. "
            "Default equal preserves existing outputs; value uses liq_me_raw; "
            "both writes equal and value-weighted output folders."
        ),
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
    portfolio_weightings = _portfolio_weightings(args.portfolio_weighting)
    breakpoint_modes = resolve_liquidity_breakpoints(
        config,
        args.liquidity_breakpoints,
    )
    use_breakpoint_namespace = namespace_liquidity_breakpoints(
        args.liquidity_breakpoints,
    )

    for spec in specs:
        spec_label = spec["spec_label"]
        logger.info("=== %s ===", spec_label)
        preds_standard, preds_weighted = load_predictions(spec)
        tc_target_predictions = _load_tc_target_predictions(spec)

        for portfolio_weighting in portfolio_weightings:
            _run_spec_portfolio_weighting(
                args=args,
                spec=spec,
                analysis_dir=analysis_dir,
                preds_standard=preds_standard,
                preds_weighted=preds_weighted,
                tc_target_predictions=tc_target_predictions,
                panel=panel,
                config=config,
                aum_scenarios=aum_scenarios,
                primary_aum=primary_aum,
                portfolio_weighting=portfolio_weighting,
                breakpoint_modes=breakpoint_modes,
                use_breakpoint_namespace=use_breakpoint_namespace,
            )

    logger.info("Done")


def _portfolio_weightings(choice: str) -> list[str]:
    """Expand CLI portfolio-weighting choice into concrete run names."""
    if choice == "both":
        return ["equal", "value"]
    return [choice]


def _portfolio_run_name(portfolio_weighting: str) -> str:
    """Return the output folder for one portfolio run."""
    if portfolio_weighting == "equal":
        return "prediction_quantile"
    if portfolio_weighting == "value":
        return "prediction_quantile_value_weight"
    raise ValueError(f"Unknown portfolio weighting: {portfolio_weighting!r}")


def _run_spec_portfolio_weighting(
    *,
    args: argparse.Namespace,
    spec: dict,
    analysis_dir: Path,
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    tc_target_predictions: tuple[pd.DataFrame, pd.DataFrame] | None,
    panel: pd.DataFrame,
    config: dict,
    aum_scenarios: list[int | float | str],
    primary_aum: int,
    portfolio_weighting: str,
    breakpoint_modes: list[str],
    use_breakpoint_namespace: bool,
) -> None:
    """Write 21e outputs for one experiment spec and one portfolio weighting."""
    spec_label = spec["spec_label"]
    portfolio_run = _portfolio_run_name(portfolio_weighting)
    logger.info("--- portfolio weighting: %s -> %s ---", portfolio_weighting, portfolio_run)
    out_dir = formal_spec_dir(analysis_dir, spec) / portfolio_run
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_decomposition:
        if tc_target_predictions is None:
            logger.warning(
                "Skipping 2x3 for %s: missing tc_target predictions",
                spec_label,
            )
        else:
            preds_tc_target_standard, preds_tc_target_weighted = (
                tc_target_predictions
            )
            for aum in aum_scenarios:
                label = aum_label(aum)
                proportional_tc_only = is_proportional_tc_scenario(aum)
                sizing_aum = scenario_sizing_aum(aum)
                logger.info("2x3 decomposition at AUM=%s", label)
                result_2x3 = compute_two_by_three_decomposition(
                    preds_standard,
                    preds_weighted,
                    preds_tc_target_standard,
                    preds_tc_target_weighted,
                    panel,
                    aum=sizing_aum,
                    config=config,
                    portfolio_mode=INTERNAL_PORTFOLIO_MODE,
                    portfolio_weighting=portfolio_weighting,
                    portfolio_design=portfolio_run,
                    proportional_tc_only=proportional_tc_only,
                )
                rows_2x3 = format_two_by_three_decomposition_rows(result_2x3)
                pd.DataFrame(rows_2x3).to_csv(
                    out_dir / f"two_by_three_{label}.csv",
                    index=False,
                )
                timeseries_2x3 = format_two_by_three_timeseries_rows(
                    result_2x3,
                    aum=aum,
                    label=label,
                )
                _write_timeseries_workbook(
                    timeseries_2x3,
                    out_dir / f"two_by_three_timeseries_{label}.xlsx",
                )
                quantile_timeseries = format_prediction_quantile_timeseries_rows(
                    result_2x3,
                    aum=aum,
                    label=label,
                )
                quantile_timeseries.to_csv(
                    out_dir / f"prediction_quantile_timeseries_{label}.csv",
                    index=False,
                )
                _write_timeseries_workbook(
                    quantile_timeseries,
                    out_dir / f"prediction_quantile_timeseries_{label}.xlsx",
                )

    if not args.skip_within_quintile:
        for breakpoint_mode in breakpoint_modes:
            logger.info(
                "Within-quintile/scissors portfolio tables: breakpoints=%s",
                breakpoint_mode,
            )
            breakpoint_config = config_for_liquidity_breakpoint(
                config,
                breakpoint_mode,
            )
            within_out_dir = liquidity_breakpoint_dir(
                out_dir,
                breakpoint_mode,
                use_breakpoint_namespace,
            )
            _write_within_quintile_outputs(
                args=args,
                spec=spec,
                out_dir=within_out_dir,
                preds_standard=preds_standard,
                preds_weighted=preds_weighted,
                panel=panel,
                config=breakpoint_config,
                aum_scenarios=aum_scenarios,
                primary_aum=primary_aum,
                portfolio_weighting=portfolio_weighting,
            )


def _write_within_quintile_outputs(
    *,
    args: argparse.Namespace,
    spec: dict,
    out_dir: Path,
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    aum_scenarios: list[int | float | str],
    primary_aum: int,
    portfolio_weighting: str,
) -> None:
    """Write breakpoint-dependent within-liquidity-quintile portfolio outputs."""
    scissors_tables = compute_quintile_sr_scissors_tables(
        preds_standard,
        preds_weighted,
        panel,
        aum_scenarios=aum_scenarios,
        config=config,
        tc_sort_aum=(None if args.skip_tc_aware_scissors else primary_aum),
        portfolio_mode=INTERNAL_PORTFOLIO_MODE,
        portfolio_weighting=portfolio_weighting,
    )
    within_quintile = format_within_quintile_from_scissors(
        scissors_tables,
        primary_aum,
    )
    if len(within_quintile) > 0:
        within_quintile.to_csv(
            out_dir / "within_quintile_portfolio.csv",
            index=False,
        )
    scissors_tables["std"].to_csv(
        out_dir / "table3_sr_quintile_std.csv",
        index=False,
    )
    scissors_tables["weighted"].to_csv(
        out_dir / "table3_sr_quintile_weighted.csv",
        index=False,
    )
    tc_sort_table = scissors_tables.get("weighted_tc_sort")
    tc_sort_label = aum_label(primary_aum)
    if tc_sort_table is not None and not tc_sort_table.empty:
        tc_sort_table.to_csv(
            out_dir / f"table3_sr_quintile_weighted_tc_sort_{tc_sort_label}.csv",
            index=False,
        )
        within_quintile_tc_sort = format_within_quintile_from_scissors(
            {
                "std": scissors_tables["std"],
                "weighted": tc_sort_table,
            },
            primary_aum,
        )
        if len(within_quintile_tc_sort) > 0:
            within_quintile_tc_sort.to_csv(
                out_dir / f"within_quintile_portfolio_tc_sort_{tc_sort_label}.csv",
                index=False,
            )
    if not args.no_figures:
        plot_sr_scissors_table(
            scissors_tables["std"],
            out_dir / "figure_sr_scissors_std.png",
            spec["model"],
            "Standard Training",
            aum_scenarios,
        )
        plot_sr_scissors_table(
            scissors_tables["weighted"],
            out_dir / "figure_sr_scissors_weighted.png",
            spec["model"],
            "Weighted Training",
            aum_scenarios,
        )
        plot_sr_scissors_comparison(
            scissors_tables["std"],
            scissors_tables["weighted"],
            out_dir / "figure_sr_scissors_comparison.png",
            spec["model"],
            aum_scenarios,
        )
        if tc_sort_table is not None and not tc_sort_table.empty:
            plot_sr_scissors_table(
                tc_sort_table,
                out_dir / f"figure_sr_scissors_weighted_tc_sort_{tc_sort_label}.png",
                spec["model"],
                f"Weighted Training + TC-Aware Sort ({tc_sort_label})",
                aum_scenarios,
            )
            plot_sr_scissors_comparison(
                scissors_tables["std"],
                tc_sort_table,
                out_dir / f"figure_sr_scissors_comparison_tc_sort_{tc_sort_label}.png",
                spec["model"],
                aum_scenarios,
                right_title=f"Weighted + TC-Aware Sort ({tc_sort_label})",
            )

    if args.scaled_quintile_aum:
        logger.info("Within-quintile/scissors scaled-AUM robustness tables")
        scale_table = compute_quintile_aum_scaling(
            preds_standard,
            panel,
            base_aum=primary_aum,
            config=config,
            reference_quintile=1,
        )
        scale_table.to_csv(
            out_dir / f"quintile_aum_scaling_{tc_sort_label}.csv",
            index=False,
        )
        scale_map = dict(zip(scale_table["quintile"], scale_table["AUM_Scale"]))
        scaled_tables = compute_quintile_sr_scissors_tables(
            preds_standard,
            preds_weighted,
            panel,
            aum_scenarios=aum_scenarios,
            config=config,
            tc_sort_aum=(None if args.skip_tc_aware_scissors else primary_aum),
            quintile_aum_scale=scale_map,
            portfolio_mode=INTERNAL_PORTFOLIO_MODE,
            portfolio_weighting=portfolio_weighting,
        )
        scaled_tables["std"].to_csv(
            out_dir / f"table3_sr_quintile_std_scaled_aum_{tc_sort_label}.csv",
            index=False,
        )
        scaled_tables["weighted"].to_csv(
            out_dir / f"table3_sr_quintile_weighted_scaled_aum_{tc_sort_label}.csv",
            index=False,
        )
        within_scaled = format_within_quintile_from_scissors(
            scaled_tables,
            primary_aum,
        )
        if len(within_scaled) > 0:
            within_scaled.to_csv(
                out_dir / f"within_quintile_portfolio_scaled_aum_{tc_sort_label}.csv",
                index=False,
            )

        scaled_tc_sort_table = scaled_tables.get("weighted_tc_sort")
        if scaled_tc_sort_table is not None and not scaled_tc_sort_table.empty:
            scaled_tc_sort_table.to_csv(
                out_dir / (
                    "table3_sr_quintile_weighted_tc_sort_scaled_aum_"
                    f"{tc_sort_label}.csv"
                ),
                index=False,
            )
            within_scaled_tc_sort = format_within_quintile_from_scissors(
                {
                    "std": scaled_tables["std"],
                    "weighted": scaled_tc_sort_table,
                },
                primary_aum,
            )
            if len(within_scaled_tc_sort) > 0:
                within_scaled_tc_sort.to_csv(
                    out_dir / (
                        "within_quintile_portfolio_tc_sort_scaled_aum_"
                        f"{tc_sort_label}.csv"
                    ),
                    index=False,
                )

        if not args.no_figures:
            plot_sr_scissors_comparison(
                scaled_tables["std"],
                scaled_tables["weighted"],
                out_dir / f"figure_sr_scissors_comparison_scaled_aum_{tc_sort_label}.png",
                spec["model"],
                aum_scenarios,
                left_title=f"Standard Training (Scaled AUM base {tc_sort_label})",
                right_title=f"Weighted Training (Scaled AUM base {tc_sort_label})",
            )
            if scaled_tc_sort_table is not None and not scaled_tc_sort_table.empty:
                plot_sr_scissors_comparison(
                    scaled_tables["std"],
                    scaled_tc_sort_table,
                    out_dir / (
                        "figure_sr_scissors_comparison_tc_sort_scaled_aum_"
                        f"{tc_sort_label}.png"
                    ),
                    spec["model"],
                    aum_scenarios,
                    left_title=(
                        f"Standard Training (Scaled AUM base {tc_sort_label})"
                    ),
                    right_title=(
                        "Weighted + TC-Aware Sort "
                        f"(Scaled AUM base {tc_sort_label})"
                    ),
                )


def _load_tc_target_predictions(
    spec: dict,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load tc_target standard and weighted predictions for one formal spec."""
    model_dir = spec["std_dir"].parent
    tc_target_root = model_dir / "tc_target"
    std_path = tc_target_root / "standard" / "predictions.parquet"
    wt_path = tc_target_root / spec["weight_spec"] / "predictions.parquet"
    if not std_path.exists() or not wt_path.exists():
        return None
    return pd.read_parquet(std_path), pd.read_parquet(wt_path)


def _write_timeseries_workbook(timeseries: pd.DataFrame, path: Path) -> None:
    """Write monthly portfolio time-series output with one worksheet per cell."""
    if timeseries.empty or "cell" not in timeseries:
        return

    redundant_sheet_cols = [
        "cell",
        "training_model",
        "portfolio_sort",
        "aum",
        "aum_label",
        "portfolio_design",
        "portfolio_weighting",
        "gross_return_pct",
        "transaction_cost_pct",
        "net_return_pct",
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for cell in sorted(timeseries["cell"].dropna().unique()):
            cell_df = timeseries[timeseries["cell"] == cell].copy()
            if cell_df.empty:
                continue
            cell_df = cell_df.drop(
                columns=[c for c in redundant_sheet_cols if c in cell_df.columns]
            )
            cell_df.to_excel(writer, sheet_name=cell, index=False)


if __name__ == "__main__":
    main()
