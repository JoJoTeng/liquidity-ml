"""Small utilities shared by formal-analysis CLI scripts."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

from src.analysis.formal.common import (
    discover_experiments,
    parse_tc_reporting_scenario,
)

MODEL_CHOICES = ["elastic_net", "xgboost", "neural_network"]
WEIGHT_CHOICES = ["dolvol", "softmax_rank", "tc", "tc_rank"]
PRIMARY_WEIGHT_SPECS = [
    "dolvol",
    "softmax_rank_lam2",
    "tc_500m",
    "tc_rank_lam3_500m",
]
LIQUIDITY_BREAKPOINT_CHOICES = ["nyse", "full_sample", "both"]
STOCK_UNIVERSE_CHOICES = ["full_sample", "nyse", "both"]


def add_experiment_filters(parser: argparse.ArgumentParser) -> None:
    """Add common model and weight-spec filters to a formal-analysis parser."""
    parser.add_argument(
        "--model",
        default=None,
        choices=MODEL_CHOICES,
        help="Analyze only one model family.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        choices=WEIGHT_CHOICES,
        help="Analyze only one formal weighting family.",
    )
    parser.add_argument(
        "--weight-spec",
        default=None,
        help=(
            "Analyze exact formal weight-spec folder(s), e.g. dolvol, "
            "softmax_rank_lam2, tc_500m, or tc_rank_lam3_500m. Use "
            "'primary' for the four main specs, 'all' for no spec filter, "
            "or a comma-separated list."
        ),
    )


def add_liquidity_breakpoint_option(parser: argparse.ArgumentParser) -> None:
    """Add the common liquidity-breakpoint selector to a formal-analysis parser."""
    parser.add_argument(
        "--liquidity-breakpoints",
        default="nyse",
        choices=LIQUIDITY_BREAKPOINT_CHOICES,
        help=(
            "Liquidity-quintile breakpoint universe for breakpoint-dependent "
            "outputs. 'both' writes separate nyse and full_sample folders."
        ),
    )


def add_stock_universe_option(parser: argparse.ArgumentParser) -> None:
    """Add the common stock-universe selector to a formal-analysis parser."""
    parser.add_argument(
        "--stock-universe",
        default="full_sample",
        choices=STOCK_UNIVERSE_CHOICES,
        help=(
            "Stock universe for portfolio construction or table export. "
            "'full_sample' preserves the legacy full-sample behavior, 'nyse' "
            "restricts to NYSE-listed stocks, and 'both' processes both "
            "universes with separate outputs."
        ),
    )


def formal_output_dirs(config: dict) -> tuple[Path, Path, Path]:
    """Return formal base, experiment, and analysis directories."""
    base_dir = Path(config["project"]["output_dir"]) / "formalanalysis"
    experiment_dir = base_dir / "experiment"
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, experiment_dir, analysis_dir


def formal_spec_dir(analysis_dir: Path, spec: dict) -> Path:
    """Return outputs/formalanalysis/analysis/{model}/{weight_spec}/."""
    out_dir = analysis_dir / spec["model"] / spec["weight_spec"]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_liquidity_breakpoints(config: dict, choice: str) -> list[str]:
    """Resolve a CLI breakpoint choice to concrete config values."""
    if choice == "both":
        return ["nyse", "full_sample"]
    if choice in {"nyse", "full_sample"}:
        return [choice]
    valid = ", ".join(LIQUIDITY_BREAKPOINT_CHOICES)
    raise ValueError(f"Unknown liquidity breakpoint choice {choice!r}: {valid}")


def config_for_liquidity_breakpoint(config: dict, breakpoint_mode: str) -> dict:
    """Return a config copy using one concrete liquidity-breakpoint mode."""
    if breakpoint_mode not in {"nyse", "full_sample"}:
        raise ValueError(
            "breakpoint_mode must be 'nyse' or 'full_sample', "
            f"got {breakpoint_mode!r}"
        )
    out = copy.deepcopy(config)
    out.setdefault("liquidity", {})["quintile_breakpoints"] = breakpoint_mode
    return out


def namespace_liquidity_breakpoints(choice: str) -> bool:
    """Return whether outputs should be written under breakpoint subfolders."""
    return True


def liquidity_breakpoint_dir(
    base_dir: Path,
    breakpoint_mode: str,
    use_namespace: bool,
) -> Path:
    """Return a breakpoint-specific output directory when requested."""
    out_dir = (
        base_dir / "liquidity_breakpoints" / breakpoint_mode
        if use_namespace
        else base_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_stock_universes(choice: str) -> list[str]:
    """Resolve a CLI stock-universe choice to concrete universe values."""
    if choice == "both":
        return ["full_sample", "nyse"]
    if choice in {"full_sample", "nyse"}:
        return [choice]
    valid = ", ".join(STOCK_UNIVERSE_CHOICES)
    raise ValueError(f"Unknown stock universe choice {choice!r}: {valid}")


def namespace_stock_universe(choice: str) -> bool:
    """Return whether 21e should write stock-universe subfolders."""
    return choice != "full_sample"


def stock_universe_dir(
    base_dir: Path,
    stock_universe: str,
    use_namespace: bool,
) -> Path:
    """Return the output directory for a stock-universe run."""
    if stock_universe not in {"full_sample", "nyse"}:
        raise ValueError(
            "stock_universe must be 'full_sample' or 'nyse', "
            f"got {stock_universe!r}"
        )
    out_dir = (
        base_dir / "stock_universe" / stock_universe
        if use_namespace
        else base_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def stock_universe_read_dir(base_dir: Path, stock_universe: str) -> Path:
    """Return the 21e output folder for a stock universe.

    Full-sample outputs historically lived directly in ``base_dir``. If a
    newer namespaced full-sample folder exists, prefer it; otherwise read the
    legacy path. NYSE-only outputs are always namespaced.
    """
    if stock_universe not in {"full_sample", "nyse"}:
        raise ValueError(
            "stock_universe must be 'full_sample' or 'nyse', "
            f"got {stock_universe!r}"
        )
    namespaced = base_dir / "stock_universe" / stock_universe
    if stock_universe == "full_sample":
        return namespaced if namespaced.exists() else base_dir
    return namespaced


def load_filtered_specs(
    experiment_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> list[dict]:
    """Discover completed formal experiments and apply CLI filters."""
    specs = discover_experiments(experiment_dir)
    if args.model:
        specs = [spec for spec in specs if spec["model"] == args.model]
    if args.weights:
        specs = [spec for spec in specs if spec["weight_family"] == args.weights]
    if getattr(args, "weight_spec", None):
        selected_specs = _resolve_weight_spec_filter(args.weight_spec)
        if selected_specs is not None:
            specs = [spec for spec in specs if spec["weight_spec"] in selected_specs]
    if not specs:
        raise SystemExit(f"No completed experiments to analyze in {experiment_dir}")
    logger.info("Found %d specifications to analyze", len(specs))
    return specs


def _resolve_weight_spec_filter(value: str) -> set[str] | None:
    """Resolve a weight-spec CLI value to exact spec folder names.

    ``None`` means no filtering. This keeps ``--weight-spec primary`` aligned
    with the table-export script while still supporting exact folder names.
    """
    token = value.strip()
    if not token:
        return None
    if token.lower() == "all":
        return None
    if token.lower() == "primary":
        return set(PRIMARY_WEIGHT_SPECS)
    return {part.strip() for part in token.split(",") if part.strip()}


def parse_aum_millions(values: list[str] | None, config: dict) -> list[int | float | str]:
    """Return realized-return TC scenarios, defaulting to config settings."""
    if not values or any(str(value).strip().lower() == "all" for value in values):
        return [
            parse_tc_reporting_scenario(value)
            for value in config["transaction_costs"]["aum_scenarios"]
        ]

    scenarios: list[int | float | str] = []
    for value in values:
        parsed = parse_tc_reporting_scenario(value)
        if isinstance(parsed, str):
            scenarios.append(parsed)
        else:
            scenarios.append(int(parsed) * 1_000_000)
    return scenarios
