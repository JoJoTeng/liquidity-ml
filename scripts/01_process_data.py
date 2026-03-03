"""
01 — Process Data
==================
Load the raw panel from 00_fetch_data.py, normalize features,
compute liquidity weights, and save the analysis-ready panel.

Input:  data/signed_predictors_all_wide.csv  (from 00_fetch_data.py)
Output: data/processed_panel.parquet

Usage:
    python scripts/01_process_data.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project imports ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_data_dir
from src.data.loader import (
    load_panel,
    get_feature_names,
    normalize_features,
    LIQUIDITY_COLS,
    NON_FEATURE_COLS,
)
from src.weighting import compute_all_weights, get_available_schemes

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("01_process_data")


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()
    config = load_config()
    data_dir = get_data_dir()
    np.random.seed(config["project"]["seed"])

    logger.info("=" * 60)
    logger.info("Step 01: Process Data")
    logger.info("=" * 60)

    # ── 1. Load panel ─────────────────────────────────────────
    logger.info("Loading panel...")
    panel = load_panel(config)
    logger.info(
        "Panel loaded: %d rows, %d cols, %d permnos, dates %d–%d",
        len(panel),
        len(panel.columns),
        panel["permno"].nunique(),
        panel["yyyymm"].min(),
        panel["yyyymm"].max(),
    )

    # ── 2. Get feature names ──────────────────────────────────
    features = get_feature_names(panel)
    logger.info("Selected features: %d", len(features))

    # ── 3. Feature missingness (before normalization) ─────────
    miss = panel[features].isna().mean()
    high_miss = miss[miss > 0.3].sort_values(ascending=False)
    if len(high_miss) > 0:
        logger.warning(
            "%d features with >30%% missing:\n%s",
            len(high_miss),
            high_miss.to_string(),
        )
    logger.info(
        "Feature missingness: mean=%.1f%%, median=%.1f%%, max=%.1f%%",
        miss.mean() * 100,
        miss.median() * 100,
        miss.max() * 100,
    )

    # ── 4. Drop rows with >50% features missing ──────────────
    frac_missing = panel[features].isna().mean(axis=1)
    sparse_mask = frac_missing > 0.5
    n_sparse = sparse_mask.sum()
    if n_sparse > 0:
        logger.info("Dropping %d rows with >50%% features missing", n_sparse)
        panel = panel[~sparse_mask].reset_index(drop=True)

    # ── 5. Normalize features ─────────────────────────────────
    logger.info("Normalizing %d features (cross-sectional rank-quantile)...", len(features))
    panel = normalize_features(panel, features)

    # Verify range
    feat_min = panel[features].min().min()
    feat_max = panel[features].max().max()
    logger.info(
        "Normalized feature range: [%.4f, %.4f] (expected [-0.5, 0.5])",
        feat_min,
        feat_max,
    )

    # ── 6. Fill remaining NaN in features with 0.0 (neutral) ─
    n_nan_before = panel[features].isna().sum().sum()
    if n_nan_before > 0:
        logger.info(
            "Filling %d remaining NaN values in features with 0.0 (neutral)",
            n_nan_before,
        )
        panel[features] = panel[features].fillna(0.0)

    # ── 7. Compute liquidity weights ────────────────────────
    logger.info("Computing liquidity weights (4 schemes)...")
    weights_df = compute_all_weights(panel)
    panel = pd.concat([panel, weights_df], axis=1)

    weight_cols = [c for c in panel.columns if c.startswith("weight_")]
    for wc in weight_cols:
        logger.info(
            "  %s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            wc,
            panel[wc].mean(),
            panel[wc].std(),
            panel[wc].min(),
            panel[wc].max(),
        )

    # ── 8. Save processed panel ───────────────────────────────
    out_path = data_dir / "processed_panel.parquet"
    panel.to_parquet(out_path, index=False, engine="pyarrow")
    file_size_mb = out_path.stat().st_size / 1e6
    logger.info("Saved: %s (%.1f MB)", out_path, file_size_mb)

    # ── 9. Summary ────────────────────────────────────────────
    liq_cols = [c for c in panel.columns if c.startswith("liq_")]
    elapsed = time.time() - t0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Processing complete in %.1f seconds", elapsed)
    logger.info("=" * 60)
    logger.info("  Rows:            %d", len(panel))
    logger.info("  Columns:         %d", len(panel.columns))
    logger.info("  Permnos:         %d", panel["permno"].nunique())
    logger.info("  Date range:      %d – %d", panel["yyyymm"].min(), panel["yyyymm"].max())
    logger.info("  Features:        %d (normalized to [-0.5, 0.5])", len(features))
    logger.info("  Liquidity cols:  %d (%s)", len(liq_cols), ", ".join(liq_cols))
    logger.info("  Weight schemes:  %d (%s)", len(weight_cols), ", ".join(weight_cols))
    logger.info("  Target NaN:      %d", panel["excess_ret"].isna().sum())
    logger.info("  Feature NaN:     %d", panel[features].isna().sum().sum())
    logger.info("")
    logger.info("Next step: python scripts/02_run_experiment.py")


if __name__ == "__main__":
    main()
