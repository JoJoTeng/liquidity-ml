"""
Data Loader
============
Load the raw/processed panels, define feature lists, and provide
cross-sectional normalization.

This module is the data backbone. All downstream modules depend on it:
- Weighting schemes: need raw liq_* columns
- Models: need normalized feature arrays from processed_panel.parquet
- Portfolio construction: need permno/yyyymm identifiers
- Evaluation framework: can normalize raw inputs or consume pre-normalized panels
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, get_data_dir

logger = logging.getLogger(__name__)

# ── Module-level constants (from config) ─────────────────────

_cfg = load_config()

ILLIQUIDITY_FEATURES: list[str] = _cfg["data"]["illiquidity_features"]
TRADABLE_FEATURES: list[str] = _cfg["data"]["tradable_features"]
ILLIQUIDITY_FEATURES_EXT: list[str] = _cfg["data"]["illiquidity_features_extended"]
TRADABLE_FEATURES_EXT: list[str] = _cfg["data"]["tradable_features_extended"]
LIQUIDITY_COLS: list[str] = [f"liq_{m}" for m in _cfg["liquidity"]["measures"]]

CORE_NON_FEATURE_COLS: set[str] = {
    "permno", "yyyymm", "ret", "excess_ret",
    "me_raw", "dvol_21d", "excess_sigma_12m", "excess_sigma_12m_daily", "exchcd",
}

NON_FEATURE_COLS: set[str] = (
    CORE_NON_FEATURE_COLS
    | {f"liq_{m}" for m in _cfg["liquidity"]["measures"]}
)


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

    # ── 5. Trailing volatility for transaction-cost impact ─────
    # Computed before the target shift, so month t uses returns through t.
    panel = panel.sort_values(["permno", "yyyymm"])
    panel["excess_sigma_12m"] = panel.groupby("permno")["excess_ret"].transform(
        lambda x: x.rolling(12, min_periods=6).std()
    )
    # Square-root impact uses daily ADV. Convert the monthly rolling sigma to a
    # daily scale so the volatility and volume horizons are internally aligned.
    panel["excess_sigma_12m_daily"] = panel["excess_sigma_12m"] / np.sqrt(21.0)

    # ── 6. Create liq_ prefixed copies (before normalization)
    liq_measures = config["liquidity"]["measures"]
    for col in liq_measures:
        if col in panel.columns:
            panel[f"liq_{col}"] = panel[col].copy()
        else:
            logger.warning("Liquidity measure '%s' not found in panel", col)

    # ── 7. Forward-shift target ─────────────────────────────
    #   Month t features predict month t+1 return
    panel["excess_ret"] = panel.groupby("permno")["excess_ret"].shift(-1)
    panel["ret"] = panel.groupby("permno")["ret"].shift(-1)

    # ── 8. Filter date range ────────────────────────────────
    start = config["data"]["start_yyyymm"]
    end = config["data"]["end_yyyymm"]
    panel = panel[(panel["yyyymm"] >= start) & (panel["yyyymm"] <= end)]
    logger.info("Date filter [%d, %d]: %d rows", start, end, len(panel))

    # ── 9. Drop NaN targets ─────────────────────────────────
    n_before = len(panel)
    panel = panel.dropna(subset=["excess_ret", "ret"])
    logger.info("Dropped %d rows with NaN target", n_before - len(panel))

    # ── Summary ─────────────────────────────────────────────
    liq_cols = [c for c in panel.columns if c.startswith("liq_")]
    features = [c for c in panel.columns if c not in NON_FEATURE_COLS]
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


def load_processed_panel(data_dir=None) -> pd.DataFrame:
    """Load the analysis-ready panel produced by ``scripts/01_process_data.py``.

    This panel keeps the selected model features rank-normalized to [0, 1]
    while preserving raw liquidity copies such as ``liq_dvol_21d`` for weights,
    quintiles, and transaction-cost calculations.
    """
    data_dir = get_data_dir() if data_dir is None else Path(data_dir)
    panel_path = data_dir / "processed_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} not found. Run scripts/01_process_data.py first."
        )
    logger.info("Loading processed panel: %s", panel_path)
    panel = pd.read_parquet(panel_path)
    logger.info("Processed panel: %d rows, %d columns", len(panel), len(panel.columns))
    return panel


def rank_to_unit_interval(x: pd.Series) -> pd.Series:
    """Map non-missing cross-sectional ranks to [0, 1].

    Average ranks are used for ties. Missing values remain missing so callers
    can decide whether to apply the neutral fill value of 0.5.
    """
    ranks = x.rank(method="average")
    n = int(ranks.notna().sum())
    if n == 0:
        return ranks
    if n == 1:
        return ranks.where(ranks.isna(), 0.5)
    return (ranks - 1.0) / (n - 1.0)


def normalize_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Cross-sectional quantile-rank normalization to [0, 1].

    For each feature and each month (yyyymm cross-section):
        average-rank non-NaN values, then map ranks to [0, 1] via
        (rank - 1) / (n_nonmissing - 1).

    Following GKX (2020). NaN values remain NaN (neutral fill = 0.5
    is done downstream, not here).

    This function is intentionally separate from ``load_panel``. It is used by
    ``01_process_data.py`` to create the processed panel, and can also be used
    directly by analyses that start from raw features.

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
        out[col] = out.groupby("yyyymm")[col].transform(rank_to_unit_interval)
    return out
