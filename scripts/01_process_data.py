"""
01 — Process Data
==================
Load the raw panel from 00_fetch_data.py, select features from SignalDoc.csv
(Clear Predictors, ~142 features), rank-transform to [0, 1],
compute liquidity weights, and save the analysis-ready panel.

Input:  data/signed_predictors_all_wide.csv  (from 00_fetch_data.py)
        data/SignalDoc.csv                    (CZ Signal Documentation)
Output: data/processed_panel.parquet
        config/feature_categories.json        (feature → category mapping)

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
from src.data.loader import load_panel, LIQUIDITY_COLS, NON_FEATURE_COLS
from src.analysis.motivation import (
    load_signaldoc,
    get_motivation_features,
    build_feature_categories,
    rank_transform_01,
)

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

    # Verify exchcd is present
    if "exchcd" not in panel.columns:
        logger.error(
            "exchcd not in panel! Re-run scripts/00_fetch_data.py "
            "and delete data/signed_predictors_all_wide.parquet"
        )
        sys.exit(1)

    # ── 2. Feature selection from SignalDoc.csv ───────────────
    logger.info("Loading SignalDoc.csv for feature selection...")
    signaldoc = load_signaldoc("data/SignalDoc.csv")

    logger.info("Building feature categories...")
    categories = build_feature_categories(signaldoc, "config/feature_categories.json")

    features = get_motivation_features(signaldoc, panel)
    logger.info("Selected features: %d (Clear Predictors from SignalDoc)", len(features))

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

    # ── 4. Save raw copies of robustness liquidity measures ────
    # These get rank-transformed below, but NYSE quintile assignment
    # in 05_step1_divergence.py needs raw values.
    for col in ["Illiquidity", "BidAskSpread"]:
        if col in panel.columns:
            panel[f"raw_{col}"] = panel[col].copy()
            logger.info("Saved raw copy: raw_%s", col)

    # ── 6. Rank-transform features to [0, 1] ─────────────────
    logger.info("Rank-transforming %d features to [0, 1]...", len(features))
    panel = rank_transform_01(panel, features)

    # Verify range
    feat_min = panel[features].min().min()
    feat_max = panel[features].max().max()
    logger.info(
        "Normalized feature range: [%.4f, %.4f] (expected [0.0, 1.0])",
        feat_min,
        feat_max,
    )

    # ── 6. Fill remaining NaN in features with 0.5 (neutral rank) ─
    n_nan_before = panel[features].isna().sum().sum()
    if n_nan_before > 0:
        logger.info(
            "Filling %d remaining NaN values in features with 0.5 (neutral rank)",
            n_nan_before,
        )
        panel[features] = panel[features].fillna(0.5)

    # ── 7. (Weights computed in 05_step1_divergence.py, not here) ──

    # ── 8. Save processed panel ───────────────────────────────
    out_path = data_dir / "processed_panel.parquet"
    panel.to_parquet(out_path, index=False, engine="pyarrow")
    file_size_mb = out_path.stat().st_size / 1e6
    logger.info("Saved: %s (%.1f MB)", out_path, file_size_mb)

    # ── 9. Save feature list for downstream scripts ──────────
    import json
    meta = {
        "features": features,
        "n_features": len(features),
        "normalization": "[0, 1] rank transform",
        "nan_fill": 0.5,
    }
    meta_path = data_dir / "feature_list.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved feature list: %s", meta_path)

    # ── 10. Summary ───────────────────────────────────────────
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
    logger.info("  Features:        %d (rank-transformed to [0, 1])", len(features))
    logger.info("  Liquidity cols:  %d (%s)", len(liq_cols), ", ".join(liq_cols))
    logger.info("  Target NaN:      %d", panel["excess_ret"].isna().sum())
    logger.info("  Feature NaN:     %d", panel[features].isna().sum().sum())
    logger.info("")
    logger.info("Next step: python scripts/05_step1_divergence.py")


if __name__ == "__main__":
    main()
