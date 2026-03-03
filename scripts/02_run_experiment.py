"""
02 — Run 2×2 Experiment
========================
Run the rolling-window 2×2 experiment for all ML models.

For each model (XGBoost, Random Forest, Neural Network):
  - Train standard and liquidity-weighted models over rolling windows
  - Build 4 portfolio configurations (2×2 design matrix)
  - Compute effect decomposition with Ledoit-Wolf bootstrap tests

Input:  data/signed_predictors_all_wide.csv  (from 00_fetch_data.py)
Output: outputs/experiment/{model_name}/     (predictions, returns, stats)

Usage:
    python scripts/02_run_experiment.py            # full run (2000–2024)
    python scripts/02_run_experiment.py --quick     # quick test (2015–2024)
    python scripts/02_run_experiment.py --model xgboost  # single model
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── Project imports ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.loader import load_panel
from src.evaluation.two_by_two import run_all_models, run_two_by_two, save_results

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("02_run_experiment")


# ── CLI ───────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 2×2 liquidity-weighted ML experiment.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: OOS from 2015 instead of 2000 (~2.5× faster).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["xgboost", "random_forest", "neural_network"],
        help="Run a single model instead of all three.",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    t0 = time.time()

    config = load_config()
    np.random.seed(config["project"]["seed"])

    logger.info("=" * 60)
    logger.info("Step 02: Run 2×2 Experiment")
    logger.info("=" * 60)

    # ── Quick mode: shorten OOS period ──
    if args.quick:
        config = copy.deepcopy(config)
        config["training"]["oos_start"] = 201501
        logger.info("QUICK MODE: OOS from 201501 (last ~10 years)")

    # ── Load data ──
    logger.info("Loading panel …")
    panel = load_panel(config)
    logger.info(
        "Panel: %d rows, %d columns, months %d–%d",
        len(panel),
        len(panel.columns),
        panel["yyyymm"].min(),
        panel["yyyymm"].max(),
    )

    # ── Run experiment ──
    if args.model:
        logger.info("Running single model: %s", args.model)
        results = run_two_by_two(args.model, panel=panel, config=config)
    else:
        logger.info("Running all models: %s", config["models"]["run_models"])
        results = run_all_models(panel=panel, config=config)

    # ── Save ──
    save_results(results)

    # ── Summary ──
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE  (%.1f min)", elapsed / 60)
    logger.info("=" * 60)

    if "summary" in results:
        logger.info("\n%s", results["summary"].to_string(index=False, float_format="%.4f"))

    logger.info("Next step: python scripts/03_analyze_results.py")


if __name__ == "__main__":
    main()
