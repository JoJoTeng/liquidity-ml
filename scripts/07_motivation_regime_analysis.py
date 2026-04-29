"""
Regime-Conditional Distributional Analysis
==========================================
Produces density_comparison and weight_distribution plots for:
  - NBER recession vs. expansion
  - VIX above vs. below median
  - NFCI above vs. below zero

This extension reuses Step 1 distribution plots. It does not train or load
model predictions; the --model argument only namespaces outputs so the regime
artifacts align with the model-specific motivation pipeline.

Usage:
  python scripts/07_motivation_regime_analysis.py --download-regime-data
  python scripts/07_motivation_regime_analysis.py
  python scripts/07_motivation_regime_analysis.py --model elastic_net
"""
from __future__ import annotations
import argparse, logging, os, sys, tempfile
import json
from pathlib import Path
import pandas as pd

# Keep matplotlib/fontconfig quiet on systems where the default user cache
# directories are not writable (common in sandboxed runs and cluster jobs).
_runtime_cache = Path(tempfile.gettempdir()) / "liquidity_ml_cache"
_runtime_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_runtime_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_runtime_cache / "xdg"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config, get_data_dir, get_output_dir
from src.data.loader import load_processed_panel
from src.models import get_all_model_names
from src.analysis.motivation import (
    ensure_motivation_weight_column,
    get_motivation_liquidity_choices,
    get_motivation_liquidity_config,
    get_motivation_liquidity_key,
    plot_density_comparison, plot_weight_distribution,
    DENSITY_PLOT_FEATURES,
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
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=get_all_model_names(),
        help="Model namespace for outputs; regime plots themselves are model-independent.",
    )
    parser.add_argument(
        "--liquidity",
        type=str,
        default="dvol",
        choices=get_motivation_liquidity_choices(),
    )
    parser.add_argument(
        "--aum",
        type=float,
        default=500.0,
        help="AUM in $M for --liquidity tc (default: 500)",
    )
    args = parser.parse_args()

    liq = get_motivation_liquidity_config(args.liquidity)
    liquidity_key = get_motivation_liquidity_key(args.liquidity, args.aum)

    if args.download_regime_data:
        download_regime_data(); return

    if not REGIME_PATH.exists():
        logger.error("Run with --download-regime-data first."); sys.exit(1)

    regime = pd.read_csv(REGIME_PATH)
    config = load_config(); data_dir = get_data_dir()
    output_base = (
        Path(get_output_dir()) / "motivation" / "step1_regime"
        / args.model / liquidity_key
    )
    panel = load_processed_panel(data_dir)
    panel = panel.merge(regime, on="yyyymm", how="left")
    logger.info("Model namespace: %s", args.model)
    logger.info("Liquidity: %s (%s)", args.liquidity, liq["label"])
    logger.info("Output: %s", output_base)
    if args.liquidity == "tc":
        logger.info("TC AUM: $%.0fM", args.aum)

    feature_list_path = data_dir / "feature_list.json"
    if not feature_list_path.exists():
        logger.error("feature_list.json not found! Run scripts/01_process_data.py first.")
        sys.exit(1)
    with open(feature_list_path) as f:
        feature_meta = json.load(f)
    all_features = feature_meta["features"] if isinstance(feature_meta, dict) else feature_meta
    missing_features = [f for f in all_features if f not in panel.columns]
    if missing_features:
        logger.warning("Features in feature_list.json but not in panel: %s", missing_features)
        all_features = [f for f in all_features if f in panel.columns]
    density_features = [f for f in DENSITY_PLOT_FEATURES if f in panel.columns]

    # processed_panel.parquet already has selected model features normalized.
    # Raw liq_* columns remain unchanged for weights, TC, and market-cap overlays.
    selected_weight_col = ensure_motivation_weight_column(
        panel, args.liquidity, config=config, aum_millions=args.aum
    )
    panel["w_tilde"] = panel[selected_weight_col]

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

        # Save regime summary: which months belong to which state
        regime_months = []
        for slabel, mask in states.items():
            sub = panel[mask].copy()
            nm = sub["yyyymm"].nunique()
            months = sorted(sub["yyyymm"].unique())
            for m in months:
                regime_months.append({"yyyymm": m, "state": slabel, "n_stocks": int((panel["yyyymm"] == m).sum())})
            if nm < 12: logger.warning("  %s: %d months — skip", slabel, nm); continue
            logger.info("  %s: %d months, %d rows", slabel, nm, len(sub))
            fn = slabel.lower().replace(" ","_").replace("(","").replace(")","").replace("<=","le").replace(">","gt")
            plot_density_comparison(sub, density_features, "w_tilde", rdir / f"density_{fn}.png",
                                   vw_col="liq_me_raw", title_suffix=f" — {slabel}")
            plot_weight_distribution(sub, "w_tilde", rdir / f"weights_{fn}.png",
                                    vw_col="liq_me_raw", title_suffix=f" — {slabel}")

        pd.DataFrame(regime_months).to_csv(rdir / "regime_months.csv", index=False)
        logger.info("  Saved regime_months.csv (%d rows)", len(regime_months))

    meta = {
        "model_namespace": args.model,
        "liquidity": args.liquidity,
        "liquidity_key": liquidity_key,
        "selected_weight_col": selected_weight_col,
        "aum_millions": args.aum if args.liquidity == "tc" else None,
        "feature_count": len(all_features),
        "density_features": density_features,
        "regime_path": str(REGIME_PATH),
        "vix_median": float(vix_median) if pd.notna(vix_median) else None,
        "note": "Regime plots are model-independent; model only namespaces outputs.",
    }
    with open(output_base / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Regime analysis complete: %s", output_base)

if __name__ == "__main__":
    main()
