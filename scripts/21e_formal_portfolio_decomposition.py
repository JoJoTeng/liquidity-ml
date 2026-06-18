"""
21e - Formal Portfolio Decomposition
====================================

Generate the formal Section 9 portfolio tables:
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/two_by_three_{aum}.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/two_by_three_timeseries_{aum}.xlsx
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/prediction_quantile_timeseries_{aum}.csv
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/prediction_quantile_timeseries_{aum}.xlsx

NYSE-only portfolio runs are analysis-only: they reuse existing model
predictions and filter the portfolio-construction panel to ``exchcd == 1``.
Those outputs are written under:
    outputs/formalanalysis/analysis/{model}/{weight_spec}/{portfolio_run}/stock_universe/nyse/
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
    compute_two_by_three_decomposition,
    format_prediction_quantile_timeseries_rows,
    format_two_by_three_decomposition_rows,
    format_two_by_three_timeseries_rows,
)
from src.analysis.formal.script_utils import (  # noqa: E402
    add_experiment_filters,
    add_stock_universe_option,
    formal_output_dirs,
    formal_spec_dir,
    load_filtered_specs,
    namespace_stock_universe,
    parse_aum_millions,
    resolve_stock_universes,
    stock_universe_dir,
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
    add_stock_universe_option(parser)
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
        "--portfolio-weighting",
        choices=["equal", "value", "dolvol", "both"],
        default=None,
        help=(
            "Selected-stock weights inside each long/short leg. Default (None) "
            "reads config.portfolio.weighting (dolvol on this branch); equal uses "
            "1/N, value uses liq_me_raw, dolvol uses liq_dvol_21d; both writes the "
            "equal and value folders."
        ),
    )
    parser.add_argument(
        "--liquidity-screen-pct",
        type=float,
        default=None,
        help=(
            "Within-month dollar-volume percentile cutoff applied before the "
            "quantile sort (e.g. 0.40 keeps the top 60%%). Default (None) reads "
            "config.portfolio.liquidity_screen_pct; pass 0 to disable the screen."
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
    portfolio_cfg = config["portfolio"]
    if args.portfolio_weighting is None:
        args.portfolio_weighting = portfolio_cfg.get("weighting", "equal")
    if args.liquidity_screen_pct is None:
        args.liquidity_screen_pct = float(
            portfolio_cfg.get("liquidity_screen_pct", 0.0) or 0.0
        )
    portfolio_weightings = _portfolio_weightings(args.portfolio_weighting)
    stock_universes = resolve_stock_universes(args.stock_universe)
    use_stock_namespace = namespace_stock_universe(args.stock_universe)

    for spec in specs:
        spec_label = spec["spec_label"]
        logger.info("=== %s ===", spec_label)
        preds_standard, preds_weighted = load_predictions(spec)
        tc_target_predictions = _load_tc_target_predictions(spec)

        for stock_universe in stock_universes:
            panel_universe = _filter_panel_for_stock_universe(panel, stock_universe)
            logger.info(
                "Stock universe %s: %d rows, %d permnos",
                stock_universe,
                len(panel_universe),
                panel_universe["permno"].nunique(),
            )

            for portfolio_weighting in portfolio_weightings:
                _run_spec_portfolio_weighting(
                    args=args,
                    spec=spec,
                    analysis_dir=analysis_dir,
                    preds_standard=preds_standard,
                    preds_weighted=preds_weighted,
                    tc_target_predictions=tc_target_predictions,
                    panel=panel_universe,
                    config=config,
                    aum_scenarios=aum_scenarios,
                    portfolio_weighting=portfolio_weighting,
                    stock_universe=stock_universe,
                    use_stock_namespace=use_stock_namespace,
                )

    logger.info("Done")


def _portfolio_weightings(choice: str) -> list[str]:
    """Expand CLI portfolio-weighting choice into concrete run names."""
    if choice == "both":
        return ["equal", "value"]
    return [choice]


def _portfolio_run_name(
    portfolio_weighting: str, liquidity_screen_pct: float | None = None
) -> str:
    """Return the output folder for one portfolio run (weighting + optional screen)."""
    base = {
        "equal": "prediction_quantile",
        "value": "prediction_quantile_value_weight",
        "dolvol": "prediction_quantile_dolvol_weight",
    }.get(portfolio_weighting)
    if base is None:
        raise ValueError(f"Unknown portfolio weighting: {portfolio_weighting!r}")
    if liquidity_screen_pct:  # > 0 -> screened (top-(1-pct)) universe
        keep_pct = round((1.0 - float(liquidity_screen_pct)) * 100)
        base = f"{base}_liq{keep_pct}"
    return base


def _filter_panel_for_stock_universe(
    panel: pd.DataFrame,
    stock_universe: str,
) -> pd.DataFrame:
    """Return the panel slice used for portfolio construction."""
    if stock_universe == "full_sample":
        return panel
    if stock_universe != "nyse":
        raise ValueError(f"Unknown stock universe: {stock_universe!r}")
    if "exchcd" not in panel.columns:
        raise ValueError(
            "Cannot run --stock-universe nyse because processed_panel.parquet "
            "does not contain exchcd. Re-run 00/01 with exchcd preserved."
        )
    out = panel[panel["exchcd"] == 1].copy()
    if out.empty:
        raise ValueError("NYSE stock universe is empty after filtering exchcd == 1.")
    return out


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
    portfolio_weighting: str,
    stock_universe: str,
    use_stock_namespace: bool,
) -> None:
    """Write 21e outputs for one experiment spec and one portfolio weighting."""
    spec_label = spec["spec_label"]
    portfolio_run = _portfolio_run_name(portfolio_weighting, args.liquidity_screen_pct)
    logger.info(
        "--- stock universe: %s; portfolio weighting: %s -> %s ---",
        stock_universe,
        portfolio_weighting,
        portfolio_run,
    )
    base_out_dir = formal_spec_dir(analysis_dir, spec) / portfolio_run
    out_dir = stock_universe_dir(
        base_out_dir,
        stock_universe,
        use_stock_namespace,
    )

    if tc_target_predictions is None:
        logger.warning(
            "Skipping 2x3 for %s: missing tc_target predictions",
            spec_label,
        )
        return

    preds_tc_target_standard, preds_tc_target_weighted = tc_target_predictions
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
            liquidity_screen_pct=(args.liquidity_screen_pct or None),
            liquidity_screen_col=config["portfolio"].get("dolvol_col", "liq_dvol_21d"),
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
