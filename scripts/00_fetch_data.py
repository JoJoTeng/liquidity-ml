#!/usr/bin/env python3
"""
Step 0: Fetch Data from WRDS + Merge with Chen & Zimmermann
=============================================================
Input file (place in data/temp/ before running):
    data/temp/signed_predictors_dl_wide.zip   — CZ predictors

Output:
    data/signed_predictors_all_wide.csv

What this script computes from CRSP monthly:
    ret             — delisting-adjusted return (Shumway 1997)
    STreversal      — signed predictor: -1 × ret
    Price           — signed predictor: -1 × log(|prc|)
    Size            — signed predictor: -1 × log(me)
    me_raw          — market equity (|prc| × shrout)

What this script computes from CRSP daily:
    dvol_21d        — 21-day trailing avg dollar volume (Eq. 14)

What comes from CZ predictors (already in the zip):
    BidAskSpread    — bid-ask spread (used for transaction-cost weighting)
    DolVol          — dollar volume signed predictor
    other signed firm characteristics

Usage:
    conda activate liquidml
    python scripts/00_fetch_data.py
"""

import os
import sys
import zipfile
import logging

import numpy as np
import pandas as pd
import wrds

# ── Config ──
WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "tengjo")
DVOL_21D_WINDOW = 21  # 21 trading days for daily dollar volume avg

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(DATA_DIR, "signed_predictors_all_wide.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _compute_daily_dollar_volume(crspd: pd.DataFrame) -> pd.DataFrame:
    """Compute daily CRSP dollar volume at the permno-month level.

    Parameters
    ----------
    crspd : Daily CRSP with columns: permno, date, prc, vol

    Returns
    -------
    DataFrame with columns: permno, yyyymm, dvol_21d
        One row per permno-month (end-of-month values).
    """
    df = crspd.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["prc"] = df["prc"].abs()
    df["vol"] = pd.to_numeric(
        df["vol"], errors="coerce"
    )  # DSF vol is already in shares
    df["yyyymm"] = df["date"].dt.year * 100 + df["date"].dt.month
    df = df.sort_values(["permno", "date"])

    # ── DollarVol 21-day trailing avg (Eq. 14) ───────────
    df["dvol_daily"] = df["prc"] * df["vol"]
    df["dvol_21d_rolling"] = df.groupby("permno")["dvol_daily"].transform(
        lambda x: x.rolling(DVOL_21D_WINDOW, min_periods=5).mean()
    )
    # Take last trading day of each month
    dvol_21d = (
        df.groupby(["permno", "yyyymm"])["dvol_21d_rolling"]
        .last()
        .rename("dvol_21d")
        .reset_index()
    )
    logger.info("  dvol_21d computed: %d permno-months", len(dvol_21d))
    return dvol_21d


def main():
    logger.info("=" * 60)
    logger.info("Step 0: Fetch CRSP + Merge CZ Predictors")
    logger.info("=" * 60)

    # ══════════════════════════════════════════════════════
    # 1. Connect to WRDS & Download CRSP Monthly
    # ══════════════════════════════════════════════════════
    logger.info("Connecting to WRDS (username: %s)...", WRDS_USERNAME)
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    logger.info("Downloading CRSP monthly stock file...")
    query = """
        SELECT a.permno, a.date, a.ret, a.shrout, a.prc,
               b.exchcd,
               c.dlstcd, c.dlret
        FROM crsp.msf AS a
        LEFT JOIN crsp.msenames AS b
            ON a.permno = b.permno
            AND b.namedt <= a.date
            AND a.date <= b.nameendt
        LEFT JOIN crsp.msedelist AS c
            ON a.permno = c.permno
            AND DATE_TRUNC('month', a.date) = DATE_TRUNC('month', c.dlstdt)
    """
    crspm = db.raw_sql(query)

    # ══════════════════════════════════════════════════════
    # 1b. Download CRSP Daily (for dvol_21d)
    # ══════════════════════════════════════════════════════
    logger.info("Downloading CRSP daily stock file (this is large, please wait)...")
    daily_query = """
        SELECT permno, date, ABS(prc) AS prc, vol
        FROM crsp.dsf
        WHERE prc IS NOT NULL
    """
    crspd = db.raw_sql(daily_query)
    db.close()
    logger.info("WRDS connection closed.")
    logger.info(
        "CRSP raw: %d rows, %d unique permnos", len(crspm), crspm["permno"].nunique()
    )

    # ══════════════════════════════════════════════════════
    # 2. Incorporate Delisting Returns
    # ══════════════════════════════════════════════════════
    logger.info("Adjusting delisting returns...")

    # Fill NaN in dlstcd/exchcd to avoid "boolean value of NA is ambiguous"
    dlstcd = crspm["dlstcd"].fillna(0)
    exchcd = crspm["exchcd"].fillna(0)
    dlret_missing = crspm["dlret"].isna()

    # Performance-related delisting codes: 500, 520–584
    perf_delist = (dlstcd == 500) | ((dlstcd >= 520) & (dlstcd <= 584))

    # NYSE/AMEX with missing dlret: assume -35%
    nyse_amex = exchcd.isin([1, 2])
    crspm.loc[dlret_missing & perf_delist & nyse_amex, "dlret"] = -0.35

    # NASDAQ with missing dlret: assume -55%
    nasdaq = exchcd == 3
    crspm.loc[dlret_missing & perf_delist & nasdaq, "dlret"] = -0.55

    # Floor at -100%, fill remaining NaN with 0
    crspm["dlret"] = crspm["dlret"].clip(lower=-1).fillna(0)

    # Compound return
    crspm["ret"] = (1 + crspm["ret"]) * (1 + crspm["dlret"]) - 1
    crspm["ret"] = np.where(
        crspm["ret"].isna() & (crspm["dlret"] != 0), crspm["dlret"], crspm["ret"]
    )

    # ══════════════════════════════════════════════════════
    # 3. Format CRSP Monthly Signals
    # ══════════════════════════════════════════════════════
    logger.info("Formatting CRSP monthly signals...")

    crspm["date"] = pd.to_datetime(crspm["date"])
    crspm["prc"] = crspm["prc"].abs()
    crspm["me"] = crspm["prc"] * crspm["shrout"]
    crspm["yyyymm"] = crspm["date"].dt.year * 100 + crspm["date"].dt.month

    # Summary
    logger.info("CRSP Summary:")
    logger.info("  Unique permnos: %d", crspm["permno"].nunique())
    logger.info("  Date range: %d – %d", crspm["yyyymm"].min(), crspm["yyyymm"].max())

    # ══════════════════════════════════════════════════════
    # 3b. Compute Daily-Based Dollar Volume
    # ══════════════════════════════════════════════════════
    logger.info("Computing daily-based dollar volume...")
    daily_liq = _compute_daily_dollar_volume(crspd)
    logger.info(
        "Daily dollar volume: %d rows, columns: %s",
        len(daily_liq),
        list(daily_liq.columns),
    )

    # Merge into monthly panel
    crspm = crspm.merge(daily_liq, on=["permno", "yyyymm"], how="left")
    logger.info(
        "After daily merge: dvol_21d %.1f%% non-null",
        crspm["dvol_21d"].notna().mean() * 100,
    )

    # Free daily data
    del crspd

    # ══════════════════════════════════════════════════════
    # 4. Filter Exchanges & Create CRSP Signals
    # ══════════════════════════════════════════════════════
    logger.info("Creating CRSP signals...")

    filtered_crspm = crspm[crspm["exchcd"].isin([1, 2, 3, -1, -2])]

    crspmsignal = filtered_crspm[["permno", "yyyymm", "ret", "exchcd"]].copy()

    # Signed predictors (Chen & Zimmermann convention)
    crspmsignal["STreversal"] = -1 * filtered_crspm["ret"].fillna(0)

    prc_adj = filtered_crspm["prc"].replace(0, np.nan)
    crspmsignal["Price"] = -1 * np.log(prc_adj)

    me_adj = filtered_crspm["me"].replace(0, np.nan)
    crspmsignal["Size"] = -1 * np.log(me_adj)

    # Raw liquidity columns (kept un-normalized for weighting & TC model)
    crspmsignal["me_raw"] = filtered_crspm["me"].values
    crspmsignal["dvol_21d"] = filtered_crspm["dvol_21d"].values

    logger.info("CRSP signals: %d rows", len(crspmsignal))

    # ══════════════════════════════════════════════════════
    # 5. Load Chen & Zimmermann Predictors
    # ══════════════════════════════════════════════════════
    logger.info("Loading Chen & Zimmermann predictors...")

    zip_path = os.path.join(TEMP_DIR, "signed_predictors_dl_wide.zip")
    if not os.path.exists(zip_path):
        logger.error(
            "signed_predictors_dl_wide.zip not found. Place it in: %s",
            TEMP_DIR,
        )
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(TEMP_DIR)

    wide_dl_raw = pd.read_csv(os.path.join(TEMP_DIR, "signed_predictors_dl_wide.csv"))
    os.remove(os.path.join(TEMP_DIR, "signed_predictors_dl_wide.csv"))

    logger.info(
        "CZ predictors: %d rows, %d columns", len(wide_dl_raw), len(wide_dl_raw.columns)
    )

    # ══════════════════════════════════════════════════════
    # 6. Merge & Export
    # ══════════════════════════════════════════════════════
    logger.info("Merging CZ predictors with CRSP signals...")

    signalwide = pd.merge(wide_dl_raw, crspmsignal, on=["permno", "yyyymm"], how="left")
    signalwide.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / 1e6

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info("=" * 60)
    logger.info("Output: %s (%.1f MB)", OUTPUT_CSV, size_mb)
    logger.info("Rows: %d | Columns: %d", len(signalwide), len(signalwide.columns))
    logger.info(
        "Date range: %d – %d", signalwide["yyyymm"].min(), signalwide["yyyymm"].max()
    )
    logger.info("Unique permnos: %d", signalwide["permno"].nunique())

    logger.info("\nKey column availability:")
    key_cols = [
        ("ret", "Delisting-adjusted return"),
        ("STreversal", "Short-term reversal signed predictor"),
        ("Price", "Price signed predictor"),
        ("Size", "Size signed predictor"),
        ("me_raw", "Market equity"),
        ("dvol_21d", "21-day trailing avg dollar volume (daily)"),
        ("BidAskSpread", "Bid-ask spread (from CZ)"),
        ("DolVol", "Dollar volume (from CZ)"),
        ("Mom12m", "12-month momentum (from CZ)"),
        ("BMdec", "Book-to-market (from CZ)"),
        ("GP", "Gross profitability (from CZ)"),
    ]
    for col, desc in key_cols:
        if col in signalwide.columns:
            pct = signalwide[col].notna().mean() * 100
            logger.info("  ✓ %-18s  %5.1f%% non-null   (%s)", col, pct, desc)
        else:
            logger.warning("  ✗ %-18s  MISSING            (%s)", col, desc)

    logger.info("\nRaw liquidity summary:")
    for col in [
        "me_raw",
        "dvol_21d",
    ]:
        s = signalwide[col].dropna()
        logger.info("  %s: mean=%.2e  median=%.2e", col, s.mean(), s.median())

    logger.info("\nNext step: python scripts/01_process_data.py")


if __name__ == "__main__":
    main()
