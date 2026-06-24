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
    compute_formal_utility_weights,
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
        "--universe",
        nargs="*",
        choices=["full", "nyse", "top40", "all"],
        default=None,
        help=(
            "Investable universe(s) for the long-short — parallel options: 'full' "
            "(all stocks, unscreened), 'nyse' (exchcd==1, unscreened), 'top40' "
            "(top 40%% by dollar volume = config.portfolio.liquidity_screen_pct), "
            "or 'all'. Default reads config.portfolio.universe.default."
        ),
    )
    parser.add_argument(
        "--portfolio-weighting",
        choices=["equal", "value", "dolvol", "signal", "both", "all"],
        default=None,
        help=(
            "Selected-leg weights. Default (None) reads config.portfolio.weighting "
            "(signal on this branch): signal=the spec's w-tilde, equal=1/N, "
            "value=liq_me_raw, dolvol=liq_dvol_21d. 'both'=equal+value; "
            "'all'=equal+value+signal."
        ),
    )
    parser.add_argument(
        "--leg-capital",
        choices=["gross", "full"],
        default="gross",
        help=(
            "'gross' (default): $A gross long-short ($A/2 per leg, comparable with "
            "the capacity book); 'full': legacy $2A ($A per leg)."
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
        args.portfolio_weighting = portfolio_cfg.get("weighting", "signal")
    portfolio_weightings = _portfolio_weightings(args.portfolio_weighting)
    universes = _resolve_universes(args.universe, config)
    signal_col = portfolio_cfg.get("signal_weight_col", "w_tilde")

    for spec in specs:
        logger.info("=== %s ===", spec["spec_label"])
        preds_standard, preds_weighted = load_predictions(spec)
        # Merge the spec's deployment weight (w-tilde) so 'signal' leg weighting
        # can read it as a panel column (construction stays spec-agnostic).
        panel_spec = panel.copy()
        panel_spec[signal_col] = compute_formal_utility_weights(panel, config, spec)

        for universe in universes:
            panel_u, screen_pct = _universe_panel(panel_spec, universe, config)
            logger.info(
                "Universe %s: %d rows, %d permnos (screen_pct=%s)",
                universe,
                len(panel_u),
                panel_u["permno"].nunique(),
                screen_pct,
            )
            for portfolio_weighting in portfolio_weightings:
                _run_spec_portfolio_weighting(
                    args=args,
                    spec=spec,
                    analysis_dir=analysis_dir,
                    preds_standard=preds_standard,
                    preds_weighted=preds_weighted,
                    panel=panel_u,
                    config=config,
                    aum_scenarios=aum_scenarios,
                    portfolio_weighting=portfolio_weighting,
                    universe=universe,
                    screen_pct=screen_pct,
                    signal_col=signal_col,
                )

    logger.info("Done")


def _portfolio_weightings(choice: str) -> list[str]:
    """Expand the CLI portfolio-weighting choice into concrete leg-weightings."""
    if choice == "both":
        return ["equal", "value"]
    if choice == "all":
        return ["equal", "value", "signal"]
    return [choice]


def _resolve_universes(arg: list[str] | None, config: dict) -> list[str]:
    """Universes to run; default (or 'all') is full+nyse+top40."""
    allu = ["full", "nyse", "top40"]
    if not arg:
        default = config["portfolio"].get("universe", {}).get("default", "top40")
        return allu if default == "all" else [default]
    if "all" in arg:
        return allu
    return [u for u in allu if u in arg]


def _universe_panel(
    panel: pd.DataFrame, universe: str, config: dict
) -> tuple[pd.DataFrame, float | None]:
    """Return (panel_slice, screen_pct) for one universe.

    'full' and 'nyse' are UNSCREENED (screen_pct=None); 'top40' is the full
    sample with the top-40%% dollar-volume screen
    (config.portfolio.liquidity_screen_pct, shared with script 45).
    """
    screen = float(config["portfolio"].get("liquidity_screen_pct", 0.0) or 0.0)
    if universe == "full":
        return panel, None
    if universe == "top40":
        return panel, (screen or None)
    if universe == "nyse":
        if "exchcd" not in panel.columns:
            raise ValueError(
                "nyse universe needs exchcd in processed_panel.parquet; re-run 00/01."
            )
        out = panel[panel["exchcd"] == 1].copy()
        if out.empty:
            raise ValueError("NYSE universe is empty after filtering exchcd == 1.")
        return out, None
    raise ValueError(f"Unknown universe: {universe!r}")


def _portfolio_run_name(portfolio_weighting: str) -> str:
    """Output folder for one leg-weighting (the universe is a subdirectory)."""
    base = {
        "equal": "prediction_quantile",
        "value": "prediction_quantile_value_weight",
        "dolvol": "prediction_quantile_dolvol_weight",
        "signal": "prediction_quantile_signal_weight",
    }.get(portfolio_weighting)
    if base is None:
        raise ValueError(f"Unknown portfolio weighting: {portfolio_weighting!r}")
    return base


def _run_spec_portfolio_weighting(
    *,
    args: argparse.Namespace,
    spec: dict,
    analysis_dir: Path,
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    aum_scenarios: list[int | float | str],
    portfolio_weighting: str,
    universe: str,
    screen_pct: float | None,
    signal_col: str,
) -> None:
    """Write 21e 2x2 outputs for one spec, universe, and leg-weighting."""
    portfolio_run = _portfolio_run_name(portfolio_weighting)
    logger.info(
        "--- universe: %s; weighting: %s -> %s ---",
        universe,
        portfolio_weighting,
        portfolio_run,
    )
    out_dir = (
        formal_spec_dir(analysis_dir, spec)
        / portfolio_run
        / "stock_universe"
        / universe
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    screen_col = config["portfolio"].get("dolvol_col", "liq_dvol_21d")
    value_col = config["portfolio"].get("value_weight_col", "liq_me_raw")

    for aum in aum_scenarios:
        label = aum_label(aum)
        proportional_tc_only = is_proportional_tc_scenario(aum)
        sizing_aum = scenario_sizing_aum(aum)
        logger.info("2x2 decomposition at AUM=%s", label)
        result = compute_two_by_three_decomposition(
            preds_standard,
            preds_weighted,
            panel=panel,
            aum=sizing_aum,
            config=config,
            portfolio_mode=INTERNAL_PORTFOLIO_MODE,
            portfolio_weighting=portfolio_weighting,
            portfolio_design=portfolio_run,
            value_weight_col=value_col,
            signal_weight_col=signal_col,
            liquidity_screen_pct=screen_pct,
            liquidity_screen_col=screen_col,
            leg_capital=args.leg_capital,
            proportional_tc_only=proportional_tc_only,
        )
        pd.DataFrame(format_two_by_three_decomposition_rows(result)).to_csv(
            out_dir / f"two_by_two_{label}.csv", index=False
        )
        _write_timeseries_workbook(
            format_two_by_three_timeseries_rows(result, aum=aum, label=label),
            out_dir / f"two_by_two_timeseries_{label}.xlsx",
        )
        quantile_timeseries = format_prediction_quantile_timeseries_rows(
            result, aum=aum, label=label
        )
        quantile_timeseries.to_csv(
            out_dir / f"prediction_quantile_timeseries_{label}.csv", index=False
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
