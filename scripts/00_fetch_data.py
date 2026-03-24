#!/usr/bin/env python3
"""
Step 0: Fetch Data from WRDS + Merge with Chen & Zimmermann
=============================================================
Input files (place in data/temp/ and data/ before running):
    data/temp/signed_predictors_dl_wide.zip   — CZ predictors
    data/FFResearch_Data_Factors.csv           — Fama-French factors

Output:
    data/signed_predictors_all_wide.csv

What this script computes from CRSP monthly:
    ret             — delisting-adjusted return (Shumway 1997)
    STreversal      — signed predictor: -1 × ret
    Price           — signed predictor: -1 × log(|prc|)
    Size            — signed predictor: -1 × log(me)
    me_raw          — market equity (|prc| × shrout)
    dvol_monthly    — monthly dollar volume (|prc| × vol_in_shares)
    dvol_6m         — 6-month rolling average of dvol_monthly
    lambda_tc       — price impact: 0.2 × 21 / dvol_6m

What this script computes from CRSP daily:
    dvol_21d        — 21-day trailing avg dollar volume (Eq. 14)
    liu_lm          — Liu (2006) composite illiquidity (LM_12)

What comes from CZ predictors (already in the zip):
    BidAskSpread    — bid-ask spread (used for Scheme C weighting)
    DolVol          — dollar volume signed predictor
    ~60+ other firm characteristics

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
TRADING_DAYS_PER_MONTH = 21
DVOL_ROLLING_WINDOW = 6
DVOL_21D_WINDOW = 21  # 21 trading days for daily dollar volume avg
LIU_LM_MONTHS = 12  # Liu (2006) lookback in months

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


def _compute_daily_liquidity_measures(crspd: pd.DataFrame) -> pd.DataFrame:
    """Compute dvol_21d and Liu (2006) LM_12 from daily CRSP data.

    Parameters
    ----------
    crspd : Daily CRSP with columns: permno, date, prc, vol, ret, shrout

    Returns
    -------
    DataFrame with columns: permno, yyyymm, dvol_21d, liu_lm
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

    # ── Daily return volatility (for Frazzini TC model) ──
    df["daily_sigma_rolling"] = df.groupby("permno")["ret"].transform(
        lambda x: x.rolling(DVOL_21D_WINDOW, min_periods=5).std()
    )
    daily_sigma = (
        df.groupby(["permno", "yyyymm"])["daily_sigma_rolling"]
        .last()
        .rename("daily_sigma")
        .reset_index()
    )
    logger.info("  daily_sigma computed: %d permno-months", len(daily_sigma))

    # ── Liu (2006) LM_12 composite illiquidity (daily 252-trading-day window) ──
    # LM_12 = [ZERO_252 + (1 / TURN_252) / DEF] * (252 / NoTD_252)
    # ZERO_252: # of zero-volume days in last 252 trading-day observations
    # TURN_252: sum of daily turnover = Σ(vol_d / shrout_d) over the window
    # NoTD_252: # valid trading-day observations in the window

    TRADING_DAYS_PER_MONTH = 21
    LIU_LM_MONTHS = 12
    WINDOW = TRADING_DAYS_PER_MONTH * LIU_LM_MONTHS  # 252
    MIN_MONTHS = 6
    MIN_OBS = TRADING_DAYS_PER_MONTH * MIN_MONTHS  # 126
    EPS = 1e-12

    # Assumes df is DAILY and has: permno, date, vol, shrout
    df = df.sort_values(["permno", "date"]).copy()

    # Match your original behavior: treat missing vol as zero-volume
    df["vol_filled"] = df["vol"].fillna(0)

    # If shrout is only populated periodically, forward-fill within permno
    df["shrout_ffill"] = df.groupby("permno")["shrout"].ffill()

    # Valid day = we have a positive shrout to compute turnover
    df["valid_day"] = df["shrout_ffill"].notna() & (df["shrout_ffill"] > 0)

    # Zero-volume indicator (restricted to valid days)
    df["is_zero_vol"] = df["valid_day"] & df["vol_filled"].eq(0)

    # Daily turnover = vol / shrout (NaN on invalid days)
    df["turnover_d"] = np.where(
        df["valid_day"],
        df["vol_filled"] / df["shrout_ffill"],
        np.nan,
    )

    g = df.groupby("permno", sort=False)

    # Rolling window stats (252 trading-day observations)
    df["n_days_252"] = (
        g["turnover_d"]
        .rolling(WINDOW, min_periods=MIN_OBS)
        .count()
        .reset_index(level=0, drop=True)
    )
    df["n_zero_252"] = (
        g["is_zero_vol"]
        .rolling(WINDOW, min_periods=MIN_OBS)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["turnover_252"] = (
        g["turnover_d"]
        .rolling(WINDOW, min_periods=MIN_OBS)
        .sum()
        .reset_index(level=0, drop=True)
    ).clip(lower=EPS)

    inv_turnover = 1.0 / df["turnover_252"]

    # Deflator: guarantee (inv_turnover / deflator) < 1 for all finite inv_turnover
    finite_inv = inv_turnover[np.isfinite(inv_turnover)]
    deflator = float(finite_inv.max() * 1.1) if len(finite_inv) else np.nan
    logger.info("  Liu deflator (max-based): %.3e", deflator)

    # Fractional tie-breaker (extra safety cap so it never reaches/exceeds 1)
    frac = (inv_turnover / deflator).clip(upper=1 - 1e-12)

    # LM12 at DAILY frequency
    df["liu_lm_daily"] = (df["n_zero_252"] + frac) * (WINDOW / df["n_days_252"])

    # Convert to permno-month: take the last trading day in each month
    if "yyyymm" not in df.columns:
        df["yyyymm"] = df["date"].dt.year * 100 + df["date"].dt.month

    liu = (
        df.dropna(subset=["liu_lm_daily"])
        .sort_values(["permno", "date"])
        .groupby(["permno", "yyyymm"], as_index=False)["liu_lm_daily"]
        .last()
        .rename(columns={"liu_lm_daily": "liu_lm"})
    )

    logger.info("  liu_lm computed: %d permno-months", len(liu))

    # ── Merge and return ──────────────────────────────────
    result = dvol_21d.merge(liu, on=["permno", "yyyymm"], how="outer")
    result = result.merge(daily_sigma, on=["permno", "yyyymm"], how="outer")
    return result


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
               a.vol,
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
    # 1b. Download CRSP Daily (for dvol_21d and Liu LM)
    # ══════════════════════════════════════════════════════
    logger.info("Downloading CRSP daily stock file (this is large, please wait)...")
    daily_query = """
        SELECT permno, date, ABS(prc) AS prc, vol, ret, shrout
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
    # 3. Format & Compute Liquidity Measures
    # ══════════════════════════════════════════════════════
    logger.info("Computing liquidity measures...")

    crspm["date"] = pd.to_datetime(crspm["date"])
    crspm["prc"] = crspm["prc"].abs()
    crspm["me"] = crspm["prc"] * crspm["shrout"]
    crspm["yyyymm"] = crspm["date"].dt.year * 100 + crspm["date"].dt.month

    # Volume: CRSP MSF vol is in hundreds of shares → multiply by 100
    crspm["vol"] = pd.to_numeric(crspm["vol"], errors="coerce")
    crspm["vol"] = crspm["vol"] * 100

    # Monthly dollar volume
    crspm["dvol_monthly"] = crspm["prc"] * crspm["vol"]

    # 6-month rolling average
    crspm = crspm.sort_values(["permno", "date"])
    crspm["dvol_6m"] = crspm.groupby("permno")["dvol_monthly"].transform(
        lambda x: x.rolling(window=DVOL_ROLLING_WINDOW, min_periods=1).mean()
    )

    # Price impact lambda: lambda = 0.2 × 21 / dvol_6m
    crspm["lambda_tc"] = (0.2 * TRADING_DAYS_PER_MONTH) / crspm["dvol_6m"]
    crspm.loc[crspm["dvol_6m"].isna() | (crspm["dvol_6m"] <= 0), "lambda_tc"] = np.nan

    # Summary
    logger.info("CRSP Summary:")
    logger.info("  Unique permnos: %d", crspm["permno"].nunique())
    logger.info("  Date range: %d – %d", crspm["yyyymm"].min(), crspm["yyyymm"].max())

    # ══════════════════════════════════════════════════════
    # 3b. Compute Daily-Based Liquidity Measures
    # ══════════════════════════════════════════════════════
    logger.info("Computing daily-based liquidity measures...")
    daily_liq = _compute_daily_liquidity_measures(crspd)
    logger.info(
        "Daily liquidity measures: %d rows, columns: %s",
        len(daily_liq),
        list(daily_liq.columns),
    )

    # Merge into monthly panel
    crspm = crspm.merge(daily_liq, on=["permno", "yyyymm"], how="left")
    logger.info(
        "After daily merge: dvol_21d %.1f%%, liu_lm %.1f%%, daily_sigma %.1f%% non-null",
        crspm["dvol_21d"].notna().mean() * 100,
        crspm["liu_lm"].notna().mean() * 100,
        crspm["daily_sigma"].notna().mean() * 100,
    )

    # Free daily data
    del crspd

    # ══════════════════════════════════════════════════════
    # 4. Filter Exchanges & Create CRSP Signals
    # ══════════════════════════════════════════════════════
    logger.info("Creating CRSP signals...")

    filtered_crspm = crspm[crspm["exchcd"].isin([1, 2, 3, -1, -2])]

    crspmsignal = filtered_crspm[["permno", "yyyymm", "ret", "lambda_tc", "exchcd"]].copy()

    # Signed predictors (Chen & Zimmermann convention)
    crspmsignal["STreversal"] = -1 * filtered_crspm["ret"].fillna(0)

    prc_adj = filtered_crspm["prc"].replace(0, np.nan)
    crspmsignal["Price"] = -1 * np.log(prc_adj)

    me_adj = filtered_crspm["me"].replace(0, np.nan)
    crspmsignal["Size"] = -1 * np.log(me_adj)

    # Raw liquidity columns (kept un-normalized for weighting & TC model)
    crspmsignal["me_raw"] = filtered_crspm["me"].values
    crspmsignal["dvol_monthly"] = filtered_crspm["dvol_monthly"].values
    crspmsignal["dvol_6m"] = filtered_crspm["dvol_6m"].values
    crspmsignal["dvol_21d"] = filtered_crspm["dvol_21d"].values
    crspmsignal["liu_lm"] = filtered_crspm["liu_lm"].values
    crspmsignal["daily_sigma"] = filtered_crspm["daily_sigma"].values

    logger.info("CRSP signals: %d rows", len(crspmsignal))

    # ══════════════════════════════════════════════════════
    # 5. Load Chen & Zimmermann Predictors
    # ══════════════════════════════════════════════════════
    logger.info("Loading Chen & Zimmermann predictors...")

    zip_path = os.path.join(TEMP_DIR, "signed_predictors_dl_wide.zip")
    if not os.path.exists(zip_path):
        for alt in [
            os.path.join(DATA_DIR, "signed_predictors_dl_wide.zip"),
            os.path.expanduser(
                "~/Desktop/PhD-Y3/Liquidity/Data/char/temp/signed_predictors_dl_wide.zip"
            ),
        ]:
            if os.path.exists(alt):
                zip_path = alt
                break
        else:
            logger.error(
                "signed_predictors_dl_wide.zip not found!\n" "Place it in: %s", TEMP_DIR
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
        ("lambda_tc", "Price impact lambda"),
        ("me_raw", "Market equity"),
        ("dvol_monthly", "Monthly dollar volume"),
        ("dvol_6m", "6-month avg monthly dollar volume"),
        ("dvol_21d", "21-day trailing avg dollar volume (daily)"),
        ("liu_lm", "Liu (2006) composite illiquidity LM_12"),
        ("daily_sigma", "21-day rolling daily return volatility"),
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

    logger.info("\nLiquidity summary:")
    for col in [
        "dvol_monthly",
        "dvol_6m",
        "dvol_21d",
        "lambda_tc",
        "liu_lm",
        "daily_sigma",
    ]:
        s = signalwide[col].dropna()
        logger.info("  %s: mean=%.2e  median=%.2e", col, s.mean(), s.median())

    logger.info("\nNext step: python scripts/01_process_data.py")


if __name__ == "__main__":
    main()
