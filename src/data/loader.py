"""
Data Loader
============
Load the processed panel, define feature lists, and provide
cross-sectional normalization.

This module is the data backbone. All downstream modules depend on it:
- Weighting schemes: need raw liq_* columns
- Models: need normalized feature arrays
- Portfolio construction: need permno/yyyymm identifiers
- Evaluation framework: needs per-window normalization
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import load_config, get_data_dir

logger = logging.getLogger(__name__)

# ── Module-level constants (from config) ─────────────────────

_cfg = load_config()

SELECTED_FEATURES: list[str] = _cfg["data"]["selected_features"]
ILLIQUIDITY_FEATURES: list[str] = _cfg["data"]["illiquidity_features"]
TRADABLE_FEATURES: list[str] = _cfg["data"]["tradable_features"]
ILLIQUIDITY_FEATURES_EXT: list[str] = _cfg["data"]["illiquidity_features_extended"]
TRADABLE_FEATURES_EXT: list[str] = _cfg["data"]["tradable_features_extended"]
LIQUIDITY_COLS: list[str] = [f"liq_{m}" for m in _cfg["liquidity"]["measures"]]

_WEIGHT_SCHEMES = ["softmax_rank", "linear_dolvol", "bid_ask_spread", "tc_stock", "quintile_discrete"]

NON_FEATURE_COLS: set[str] = {
    "permno", "yyyymm", "ret", "excess_ret",
    "me_raw", "dvol_monthly", "dvol_6m", "dvol_21d", "lambda_tc", "liu_lm",
    "daily_sigma", "exchcd",
} | {f"liq_{m}" for m in _cfg["liquidity"]["measures"]} | {f"weight_{s}" for s in _WEIGHT_SCHEMES}


# ── Public API ───────────────────────────────────────────────


def load_panel(config: dict | None = None) -> pd.DataFrame:
    """Load the full panel with forward excess returns and liq_ columns.

    Steps:
        1. Read signed_predictors_all_wide.csv (parquet cache if available)
        2. Merge Fama-French RF, compute excess_ret = ret - RF
        3. Create liq_ prefixed copies of raw liquidity columns
        4. Forward-shift target (month t features → month t+1 return)
        5. Filter date range and drop NaN targets

    Returns
    -------
    pd.DataFrame with columns:
        permno, yyyymm          — identifiers
        ret                     — raw return (kept for portfolio analysis)
        excess_ret              — forward 1-month excess return (target)
        ~210 CZ predictor cols  — unnormalized features
        liq_dvol_21d, etc.      — raw liquidity copies for weighting (config-driven)
    """
    if config is None:
        config = load_config()
    data_dir = get_data_dir()

    # ── 1. Load main panel ──────────────────────────────────
    csv_path = data_dir / config["data"]["signed_predictors_csv"]
    parquet_path = csv_path.with_suffix(".parquet")

    if (
        parquet_path.exists()
        and parquet_path.stat().st_mtime > csv_path.stat().st_mtime
    ):
        logger.info("Loading from parquet cache: %s", parquet_path.name)
        panel = pd.read_parquet(parquet_path)
    else:
        logger.info("Loading CSV: %s (this may take a few minutes)...", csv_path.name)
        panel = pd.read_csv(
            csv_path,
            dtype={"permno": "int64", "yyyymm": "int64"},
            low_memory=False,
        )
        logger.info("Saving parquet cache: %s", parquet_path.name)
        panel.to_parquet(parquet_path, index=False, engine="pyarrow")

    logger.info("Raw panel: %d rows, %d columns", len(panel), len(panel.columns))

    # ── 2. Deduplicate ──────────────────────────────────────
    n_dup = panel.duplicated(subset=["permno", "yyyymm"]).sum()
    if n_dup > 0:
        logger.warning(
            "Found %d duplicate (permno, yyyymm) pairs; keeping first", n_dup
        )
        panel = panel.drop_duplicates(subset=["permno", "yyyymm"], keep="first")

    # ── 3. Merge Fama-French RF ─────────────────────────────
    ff_path = data_dir / config["data"]["ff_factors_csv"]
    ff = pd.read_csv(ff_path)
    ff = ff.rename(columns={"Date": "yyyymm"})
    ff["RF"] = ff["RF"] / 100  # percent → decimal (match ret format)
    ff = ff[["yyyymm", "RF"]]

    panel = panel.merge(ff, on="yyyymm", how="left")

    n_missing_rf = panel["RF"].isna().sum()
    if n_missing_rf > 0:
        logger.warning("%d rows have no matching RF value", n_missing_rf)

    # ── 4. Compute excess return ────────────────────────────
    ret_col = config["data"]["return_col"]
    panel["excess_ret"] = panel[ret_col] - panel["RF"]
    panel = panel.drop(columns=["RF"])

    # ── 5. Create liq_ prefixed copies (before normalization)
    liq_measures = config["liquidity"]["measures"]
    for col in liq_measures:
        if col in panel.columns:
            panel[f"liq_{col}"] = panel[col].copy()
        else:
            logger.warning("Liquidity measure '%s' not found in panel", col)

    # ── 6. Forward-shift target ─────────────────────────────
    #   Month t features predict month t+1 return
    panel = panel.sort_values(["permno", "yyyymm"])
    panel["excess_ret"] = panel.groupby("permno")["excess_ret"].shift(-1)
    panel["ret"] = panel.groupby("permno")["ret"].shift(-1)

    # ── 7. Filter date range ────────────────────────────────
    start = config["data"]["start_yyyymm"]
    end = config["data"]["end_yyyymm"]
    panel = panel[(panel["yyyymm"] >= start) & (panel["yyyymm"] <= end)]
    logger.info("Date filter [%d, %d]: %d rows", start, end, len(panel))

    # ── 8. Drop NaN targets ─────────────────────────────────
    n_before = len(panel)
    panel = panel.dropna(subset=["excess_ret", "ret"])
    logger.info("Dropped %d rows with NaN target", n_before - len(panel))

    # ── Summary ─────────────────────────────────────────────
    non_feature_cols = {"permno", "yyyymm", "ret", "excess_ret", "exchcd",
                        "me_raw", "dvol_monthly", "dvol_21d", "dvol_6m",
                        "lambda_tc", "liu_lm", "daily_sigma", "BidAskSpread"}
    liq_cols = [c for c in panel.columns if c.startswith("liq_")]
    non_feature_cols.update(liq_cols)
    features = [c for c in panel.columns if c not in non_feature_cols]
    logger.info(
        "Panel ready: %d rows, %d permnos, dates %d–%d, "
        "%d features, %d liq_ columns",
        len(panel),
        panel["permno"].nunique(),
        panel["yyyymm"].min(),
        panel["yyyymm"].max(),
        len(features),
        len(liq_cols),
    )

    panel = panel.reset_index(drop=True)
    return panel


def get_feature_names(panel: pd.DataFrame) -> list[str]:
    """Return the curated predictor feature names present in the panel.

    Uses the hand-picked ``selected_features`` list from config (75
    predictors with coverage from 1972 onward).  Only returns features
    that actually exist in ``panel.columns`` so callers never get
    KeyError on missing columns.
    """
    panel_cols = set(panel.columns)
    present = [f for f in SELECTED_FEATURES if f in panel_cols]
    missing = [f for f in SELECTED_FEATURES if f not in panel_cols]
    if missing:
        logger.warning("Selected features missing from panel: %s", missing)
    return present


def normalize_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Cross-sectional rank normalization to [0, 1].

    For each feature and each month (yyyymm cross-section):
        rank non-NaN values → percentile rank in [0, 1].

    Following GKX (2020). NaN values remain NaN (neutral fill = 0.5
    is done downstream, not here).

    This function is intentionally separate from load_panel so that the
    rolling-window framework can normalize each train/val/test split
    independently, preventing look-ahead bias.

    Parameters
    ----------
    df : DataFrame with ``yyyymm`` column and feature columns.
    features : Column names to normalize.

    Returns
    -------
    DataFrame with normalized feature columns; other columns unchanged.
    NaN values remain NaN.
    """
    out = df.copy()
    for col in features:
        out[col] = out.groupby("yyyymm")[col].rank(pct=True)
    return out
