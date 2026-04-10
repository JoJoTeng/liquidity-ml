"""
Regime-Conditional Distributional Analysis (Daniele feedback)
==============================================================
Produces density_comparison and weight_distribution plots for:
  - NBER recession vs. expansion
  - VIX above vs. below median
  - NFCI above vs. below zero

Usage:
  python scripts/11_regime_analysis.py --download-regime-data
  python scripts/11_regime_analysis.py
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config, get_data_dir, get_output_dir
from src.analysis.motivation import (
    compute_implementability_weights, rank_transform_01,
    plot_density_comparison, plot_weight_distribution,
    get_motivation_features, load_signaldoc, DENSITY_PLOT_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
REGIME_PATH = PROJECT_ROOT / "data" / "regime_indicators.csv"


def download_regime_data(output_path=REGIME_PATH):
    try:
        from pandas_datareader import data as pdr
        logger.info("Downloading FRED data...")
        vix = pdr.DataReader("VIXCLS", "fred", "1990-01-01").resample("ME").last().reset_index()
        vix.columns = ["date", "vix"]; vix["yyyymm"] = vix["date"].dt.year * 100 + vix["date"].dt.month
        nfci = pdr.DataReader("NFCI", "fred", "1971-01-01").resample("ME").last().reset_index()
        nfci.columns = ["date", "nfci"]; nfci["yyyymm"] = nfci["date"].dt.year * 100 + nfci["date"].dt.month
        rec = pdr.DataReader("USREC", "fred", "1960-01-01").resample("ME").last().reset_index()
        rec.columns = ["date", "recession"]; rec["yyyymm"] = rec["date"].dt.year * 100 + rec["date"].dt.month
        regime = vix[["yyyymm","vix"]].merge(nfci[["yyyymm","nfci"]], on="yyyymm", how="outer"
            ).merge(rec[["yyyymm","recession"]], on="yyyymm", how="outer").sort_values("yyyymm")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        regime.to_csv(output_path, index=False)
        logger.info("Saved %d rows to %s", len(regime), output_path)
        return regime
    except ImportError:
        logger.error("pip install pandas-datareader  OR manually create data/regime_indicators.csv")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Regime-Conditional Analysis")
    parser.add_argument("--download-regime-data", action="store_true")
    parser.add_argument("--liquidity", type=str, default="dvol", choices=["dvol","mcap"])
    args = parser.parse_args()

    LIQ = {"dvol": {"col": "liq_dvol_21d", "asc": True}, "mcap": {"col": "liq_me_raw", "asc": True}}
    liq = LIQ[args.liquidity]

    if args.download_regime_data:
        download_regime_data(); return

    if not REGIME_PATH.exists():
        logger.error("Run with --download-regime-data first."); sys.exit(1)

    regime = pd.read_csv(REGIME_PATH)
    config = load_config(); data_dir = get_data_dir()
    output_base = Path(get_output_dir()) / "motivation" / "step1_regime"
    panel = pd.read_parquet(data_dir / "processed_panel.parquet")
    panel = panel.merge(regime, on="yyyymm", how="left")

    signaldoc = load_signaldoc()
    all_features = get_motivation_features(signaldoc, panel)
    density_features = [f for f in DENSITY_PLOT_FEATURES if f in panel.columns]

    # Rank-transform on full panel (not per-regime) so ranks reflect
    # the cross-sectional distribution at each month.
    panel = rank_transform_01(panel, all_features)
    panel["w_tilde"] = compute_implementability_weights(panel, liq["col"])

    monthly_vix = panel[["yyyymm", "vix"]].drop_duplicates("yyyymm").set_index("yyyymm")["vix"]
    vix_median = monthly_vix.median()
    logger.info("VIX median: %.2f (over %d months)", vix_median, monthly_vix.notna().sum())

    regimes = {
        "recession": {"Recession": panel["recession"] == 1, "Expansion": panel["recession"] == 0},
        "vix": {"High VIX": panel["vix"] >= vix_median, "Low VIX": panel["vix"] < vix_median},
        "nfci": {"Tight (NFCI gt 0)": panel["nfci"] > 0, "Loose (NFCI le 0)": panel["nfci"] <= 0},
    }

    for rname, states in regimes.items():
        rdir = output_base / rname; rdir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 60); logger.info("Regime: %s", rname)
        for slabel, mask in states.items():
            sub = panel[mask].copy()
            nm = sub["yyyymm"].nunique()
            if nm < 12: logger.warning("  %s: %d months — skip", slabel, nm); continue
            logger.info("  %s: %d months, %d rows", slabel, nm, len(sub))
            # Re-normalize weights within regime subset so mean(w_tilde) = 1
            sub["w_tilde"] = compute_implementability_weights(sub, liq["col"])
            fn = slabel.lower().replace(" ","_").replace("(","").replace(")","").replace("<=","le").replace(">","gt")
            plot_density_comparison(sub, density_features, "w_tilde", rdir / f"density_{fn}.png",
                                   vw_col="liq_me_raw", title_suffix=f" — {slabel}")
            plot_weight_distribution(sub, "w_tilde", rdir / f"weights_{fn}.png",
                                    vw_col="liq_me_raw", title_suffix=f" — {slabel}")

    logger.info("Regime analysis complete: %s", output_base)

if __name__ == "__main__":
    main()
