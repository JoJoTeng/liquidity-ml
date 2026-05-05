"""
Audit zero / NaN / negative values in the panel's liquidity columns.

For every ``liq_*`` column in ``data/processed_panel.parquet``, reports
how many stock-months have NaN, exactly-zero, negative, or positive
values. Cross-references zero-liquidity rows against return data to
distinguish "delisted / no-trade" from "valid but tiny" stocks.

Run:
    python scripts/audit_liquidity_zeros.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_processed_panel  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def audit_column(panel: pd.DataFrame, col: str) -> dict:
    """Return one row of audit stats for a single liquidity column."""
    s = panel[col]
    n = len(s)
    n_nan = s.isna().sum()
    n_zero = (s == 0).sum()
    n_neg = (s < 0).sum()
    n_pos = (s > 0).sum()
    return {
        "column": col,
        "total": n,
        "nan": n_nan,
        "nan_pct": 100 * n_nan / n,
        "zero": n_zero,
        "zero_pct": 100 * n_zero / n,
        "negative": n_neg,
        "positive": n_pos,
        "min_positive": s[s > 0].min() if n_pos > 0 else np.nan,
        "median_positive": s[s > 0].median() if n_pos > 0 else np.nan,
        "max": s.max() if n_pos > 0 else np.nan,
    }


def cross_check_zero_returns(panel: pd.DataFrame, col: str) -> None:
    """For zero-liquidity rows, what does the return column look like?"""
    if (panel[col] == 0).sum() == 0:
        logger.info("%s: no exact zeros; nothing to cross-check", col)
        return

    zeros = panel[panel[col] == 0]
    n = len(zeros)
    ret_nan = zeros["ret"].isna().sum()
    ret_zero = (zeros["ret"] == 0).sum()
    ret_other = n - ret_nan - ret_zero

    logger.info(
        "%s: %d rows with %s == 0",
        col, n, col,
    )
    logger.info("  ret NaN     : %d (%.1f%%)", ret_nan, 100 * ret_nan / n)
    logger.info("  ret == 0    : %d (%.1f%%)", ret_zero, 100 * ret_zero / n)
    logger.info("  ret non-zero: %d (%.1f%%)", ret_other, 100 * ret_other / n)


def main() -> None:
    panel = load_processed_panel()
    logger.info(
        "Panel: %d rows, dates %d–%d",
        len(panel), panel["yyyymm"].min(), panel["yyyymm"].max(),
    )

    liq_cols = sorted(c for c in panel.columns if c.startswith("liq_"))
    if not liq_cols:
        logger.error("No liq_* columns in the panel — is this the right file?")
        return
    logger.info("Auditing %d liq_* columns: %s", len(liq_cols), liq_cols)

    rows = [audit_column(panel, c) for c in liq_cols]
    summary = pd.DataFrame(rows)

    pd.set_option("display.float_format", lambda x: f"{x:,.4g}")
    logger.info("\n=== Summary ===\n%s", summary.to_string(index=False))

    logger.info("\n=== Zero-liquidity vs return cross-check ===")
    for col in liq_cols:
        cross_check_zero_returns(panel, col)

    out_path = Path("outputs") / "audit_liquidity_zeros.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info("Saved summary to %s", out_path)


if __name__ == "__main__":
    main()
