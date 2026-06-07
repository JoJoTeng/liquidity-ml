"""Output-directory helpers for the eval_realignment evaluation track.

Predictions are READ from the existing formalanalysis experiment cache; outputs
are written to a separate ``outputs/eval_realignment/`` root so the realigned
evaluation never collides with the formalanalysis pipeline. The CLI/breakpoint
helpers are re-exported from ``src.analysis.formal.script_utils`` so the
eval_realignment scripts import everything from this one module.
"""

from __future__ import annotations

from pathlib import Path

# Re-export the reusable formal CLI/breakpoint helpers unchanged.
from src.analysis.formal.script_utils import (  # noqa: F401
    add_experiment_filters,
    add_liquidity_breakpoint_option,
    config_for_liquidity_breakpoint,
    liquidity_breakpoint_dir,
    load_filtered_specs,
    namespace_liquidity_breakpoints,
    resolve_liquidity_breakpoints,
)


def eval_realignment_output_dirs(config: dict) -> tuple[Path, Path, Path]:
    """Return (base, formalanalysis experiment, eval_realignment analysis) dirs.

    The experiment directory points at the existing formalanalysis prediction
    cache (read-only inputs). The analysis directory is the separate
    eval_realignment output root. This is the single, deliberate deviation from
    ``formal_output_dirs`` (which roots both at ``formalanalysis``).
    """
    out_root = Path(config["project"]["output_dir"])
    experiment_dir = out_root / "formalanalysis" / "experiment"
    base_dir = out_root / "eval_realignment"
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, experiment_dir, analysis_dir


def eval_realignment_spec_dir(analysis_dir: Path, spec: dict) -> Path:
    """Return outputs/eval_realignment/analysis/{model}/{weight_spec}/."""
    out_dir = analysis_dir / spec["model"] / spec["weight_spec"]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
