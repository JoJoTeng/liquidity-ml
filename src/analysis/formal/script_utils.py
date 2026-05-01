"""Small utilities shared by formal-analysis CLI scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.analysis.formal.common import discover_experiments

MODEL_CHOICES = ["elastic_net", "xgboost", "neural_network"]
WEIGHT_CHOICES = ["dolvol", "softmax_rank", "tc"]


def add_experiment_filters(parser: argparse.ArgumentParser) -> None:
    """Add common model/weight filters to a formal-analysis parser."""
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
    if not specs:
        raise SystemExit(f"No completed experiments to analyze in {experiment_dir}")
    logger.info("Found %d specifications to analyze", len(specs))
    return specs


def parse_aum_millions(values: list[int] | None, config: dict) -> list[int]:
    """Return AUM values in dollars, defaulting to config scenarios."""
    if not values:
        return list(config["transaction_costs"]["aum_scenarios"])
    return [int(v) * 1_000_000 for v in values]
