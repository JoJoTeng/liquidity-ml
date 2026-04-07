"""
Step 3e: Quintile-Specific XGBoost Models (Section 5.2e)
=========================================================
Trains 5 separate XGBoost models (one per liquidity quintile).
Compares within-quintile R² vs. the pooled model.

Uses SAME hyperparameters as baseline (no retuning) per document spec.
Quintile assignment based on last month of training window.

Outputs (to outputs/motivation_raw/step3_quintile/{liquidity}/):
  - predictions_q{1-5}.parquet
  - r2_comparison.csv  (Output 3.10)
  - r2_comparison.png
  - meta.json

Usage:
  python scripts/10_quintile_specific_models.py
  python scripts/10_quintile_specific_models.py --quintile 5
  python scripts/10_quintile_specific_models.py --retune  # robustness: retune within quintile
  python scripts/10_quintile_specific_models.py --recompute
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, get_data_dir, get_output_dir
from src.data.loader import load_panel
from src.analysis.motivation import (
    assign_nyse_quintiles, rolling_xgboost_predict_quintile,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def pooled_r2_zero(preds):
    ss_res = (preds["y_true"] - preds["y_pred"]).pow(2).sum()
    ss_tot = preds["y_true"].pow(2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def main():
    parser = argparse.ArgumentParser(description="Step 3e: Quintile-Specific Models")
    parser.add_argument("--liquidity", type=str, default="dvol", choices=["dvol", "mcap"])
    parser.add_argument("--quintile", type=int, default=None, choices=[1,2,3,4,5])
    parser.add_argument("--retune", action="store_true", help="Retune within quintile (robustness)")
    parser.add_argument("--recompute", action="store_true", help="Skip training, recompute from saved")
    args = parser.parse_args()

    LIQ = {"dvol": {"col": "liq_dvol_21d", "asc": True}, "mcap": {"col": "liq_me_raw", "asc": True}}
    liq = LIQ[args.liquidity]
    config = load_config()
    data_dir = get_data_dir()
    output_dir = Path(get_output_dir()) / "motivation_raw" / "step3_quintile" / args.liquidity
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir = Path(get_output_dir()) / "motivation" / "step3" / args.liquidity

    # Load data
    panel_path = data_dir / "processed_panel.parquet"
    if not panel_path.exists():
        logger.error("processed_panel.parquet not found."); sys.exit(1)
    panel = pd.read_parquet(panel_path)
    logger.info("Panel: %d rows", len(panel))

    # Features — must match pooled model
    feat_path = data_dir / "feature_list.json"
    if not feat_path.exists():
        logger.error("feature_list.json not found at %s. Run 07_step3_ml_diagnostics.py first.", feat_path)
        sys.exit(1)
    with open(feat_path) as f:
        feat_meta = json.load(f)
    features = feat_meta["features"] if isinstance(feat_meta, dict) else feat_meta
    logger.info("Features: %d", len(features))

    # Quintiles
    panel["liq_quintile"] = assign_nyse_quintiles(panel, liq["col"], ascending=liq["asc"])

    # Fixed params from baseline (default: use baseline config, no retuning)
    fixed_params = None if args.retune else config["models"]["xgboost"]

    quintiles = [args.quintile] if args.quintile else [1,2,3,4,5]

    if not args.recompute:
        for q in quintiles:
            logger.info("=" * 60)
            logger.info("Training Q%d-specific XGBoost...", q)
            preds = rolling_xgboost_predict_quintile(
                panel, features, quintile=q, quintile_col="liq_quintile",
                config=config, fixed_params=fixed_params,
            )
            if len(preds) > 0:
                preds.to_parquet(output_dir / f"predictions_q{q}.parquet", index=False)
                logger.info("Q%d: %d predictions saved", q, len(preds))

    # Comparison
    logger.info("=" * 60)
    logger.info("Computing R² comparison...")

    # Pooled R² per quintile
    pooled_r2 = {}
    pooled_pred_path = pooled_dir / "predictions.parquet"
    if pooled_pred_path.exists():
        pp = pd.read_parquet(pooled_pred_path)
        qmap = panel[["permno", "yyyymm", "liq_quintile"]].drop_duplicates()
        pp = pp.merge(qmap, on=["permno", "yyyymm"], how="left")
        for q in range(1, 6):
            qp = pp[pp["liq_quintile"] == q]
            if len(qp) > 0: pooled_r2[q] = pooled_r2_zero(qp)

    rows = []
    for q in range(1, 6):
        path = output_dir / f"predictions_q{q}.parquet"
        r2_own = np.nan
        n = 0
        if path.exists():
            pq = pd.read_parquet(path)
            r2_own = pooled_r2_zero(pq)
            n = len(pq)
        r2_pooled = pooled_r2.get(q, np.nan)
        delta = (r2_own - r2_pooled) * 100 if not (np.isnan(r2_own) or np.isnan(r2_pooled)) else np.nan
        rows.append({"quintile": f"Q{q}", "r2_pooled_pct": r2_pooled*100 if not np.isnan(r2_pooled) else np.nan,
                      "r2_own_pct": r2_own*100, "delta_pp": delta, "n_predictions": n})
        logger.info("Q%d: pooled=%.3f%%, own=%.3f%%, Δ=%.3f pp", q,
                     r2_pooled*100 if not np.isnan(r2_pooled) else 0, r2_own*100, delta if not np.isnan(delta) else 0)

    comp = pd.DataFrame(rows)
    comp.to_csv(output_dir / "r2_comparison.csv", index=False)

    # Bar chart
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    valid = comp.dropna(subset=["r2_own_pct"])
    if len(valid) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(valid)); w = 0.35
        if valid["r2_pooled_pct"].notna().any():
            ax.bar(x - w/2, valid["r2_pooled_pct"], w, label="Pooled model", color="steelblue", alpha=0.85)
            ax.bar(x + w/2, valid["r2_own_pct"], w, label="Quintile-specific", color="darkorange", alpha=0.85)
        else:
            ax.bar(x, valid["r2_own_pct"], w, label="Quintile-specific", color="darkorange")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(valid["quintile"].tolist())
        ax.set_xlabel("Liquidity Quintile"); ax.set_ylabel("OOS R² (%)")
        ax.set_title("Pooled vs. Quintile-Specific XGBoost: Within-Quintile R²")
        ax.legend(); plt.tight_layout()
        fig.savefig(output_dir / "r2_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    with open(output_dir / "meta.json", "w") as f:
        json.dump({"quintiles_run": quintiles, "n_features": len(features), "fixed_params": fixed_params is not None,
                    "comparison": comp.to_dict(orient="records")}, f, indent=2, default=str)
    logger.info("Step 3e complete. Outputs: %s", output_dir)

if __name__ == "__main__":
    main()
