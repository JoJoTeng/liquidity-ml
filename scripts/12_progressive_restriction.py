"""
Step 3d: Progressive Universe Restriction (Section 5.2d)
=========================================================
Trains 4 additional XGBoost models with restricted training universes.
Uses SAME hyperparameters as baseline (no retuning).
Test set includes ALL stocks. Primary metric: R²(Q4-Q5).

Models: MQ2+ (drop Q1), MQ3+ (drop Q1-Q2), MQ4+ (drop Q1-Q3), MQ5 (Q5 only)

Outputs (to outputs/motivation/step3_restriction/{liquidity}/):
  - predictions_MQ{2,3,4,5}.parquet
  - restriction_curve.png       (Output 3.7)
  - restriction_comparison.csv  (Output 3.8)
  - restriction_by_quintile.csv (Output 3.9)
  - meta.json

Usage:
  python scripts/12_progressive_restriction.py
  python scripts/12_progressive_restriction.py --min-quintile 2
  python scripts/12_progressive_restriction.py --recompute
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, get_data_dir, get_output_dir
from src.analysis.motivation import (
    assign_nyse_quintiles, rolling_xgboost_predict_restricted,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def pooled_r2_zero(preds):
    ss_res = (preds["y_true"] - preds["y_pred"]).pow(2).sum()
    ss_tot = preds["y_true"].pow(2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def r2_for_quintiles(preds, panel, q_list):
    """R² for stocks in specified quintiles."""
    qmap = panel[["permno", "yyyymm", "liq_quintile"]].drop_duplicates()
    merged = preds.merge(qmap, on=["permno", "yyyymm"], how="left")
    sub = merged[merged["liq_quintile"].isin(q_list)].dropna(subset=["y_true", "y_pred"])
    return pooled_r2_zero(sub) if len(sub) > 0 else np.nan


def main():
    parser = argparse.ArgumentParser(description="Step 3d: Progressive Universe Restriction")
    parser.add_argument("--liquidity", type=str, default="dvol", choices=["dvol", "mcap"])
    parser.add_argument("--min-quintile", type=int, default=None, choices=[2,3,4,5],
                        help="Run single restriction level (for parallel runs)")
    parser.add_argument("--retune", action="store_true", help="Retune within restricted universe (robustness)")
    parser.add_argument("--use-baseline-params", action="store_true",
                        help="Use baseline's per-window tuned params (from tuned_params.csv)")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    LIQ = {"dvol": {"col": "liq_dvol_21d", "asc": True}, "mcap": {"col": "liq_me_raw", "asc": True}}
    liq = LIQ[args.liquidity]
    config = load_config()
    data_dir = get_data_dir()

    # Mode subdirectory: baseline / retune / default
    if args.use_baseline_params:
        mode = "baseline"
    elif args.retune:
        mode = "retune"
    else:
        mode = "default"

    output_dir = Path(get_output_dir()) / "motivation" / "step3_restriction_rerank" / args.liquidity / mode
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

    # Param strategy: --use-baseline-params > --retune > fixed config defaults
    baseline_tuned = None
    fixed_params = None
    if args.use_baseline_params:
        tp_path = pooled_dir / "tuned_params.csv"
        if not tp_path.exists():
            logger.error("tuned_params.csv not found at %s. Re-run 07_step3_ml_diagnostics.py first.", tp_path)
            sys.exit(1)
        baseline_tuned = pd.read_csv(tp_path, index_col="yyyymm")
        logger.info("Loaded %d baseline tuned param windows from %s", len(baseline_tuned), tp_path)
    elif not args.retune:
        fixed_params = config["models"]["xgboost"]
        logger.info("Using fixed config defaults (no retuning)")
    else:
        logger.info("Retuning within each restricted universe")

    levels = [args.min_quintile] if args.min_quintile else [2, 3, 4, 5]

    # Train restricted models
    if not args.recompute:
        for mq in levels:
            label = f"MQ{mq}+"
            pred_path = output_dir / f"predictions_{label}.parquet"
            logger.info("=" * 60)
            logger.info("Training %s (quintile >= %d)...", label, mq)
            preds = rolling_xgboost_predict_restricted(
                panel, features, min_quintile=mq, quintile_col="liq_quintile",
                config=config, fixed_params=fixed_params,
                baseline_tuned_params=baseline_tuned,
            )
            if len(preds) > 0:
                preds.to_parquet(pred_path, index=False)
                logger.info("%s: %d predictions saved", label, len(preds))

    # Comparison
    logger.info("=" * 60)
    logger.info("Computing restriction curve...")

    # Load baseline (Mall) predictions
    models = {"Mall": pooled_dir / "predictions.parquet"}
    for mq in [2, 3, 4, 5]:
        models[f"MQ{mq}+"] = output_dir / f"predictions_MQ{mq}+.parquet"

    # Compute avg N_train/month for each restriction level
    q_counts = panel.groupby(["yyyymm", "liq_quintile"]).size().unstack(fill_value=0)
    avg_n_train = {}
    avg_n_train["Mall"] = panel.groupby("yyyymm").size().mean()
    for mq in [2, 3, 4, 5]:
        cols = [c for c in q_counts.columns if c >= mq]
        avg_n_train[f"MQ{mq}+"] = q_counts[cols].sum(axis=1).mean()

    # Output 3.8: R² by training universe evaluated on Q4-Q5
    rows_38 = []
    for model_name, pred_path in models.items():
        if not pred_path.exists():
            rows_38.append({"model": model_name, "r2_q45_pct": np.nan, "r2_q4_pct": np.nan,
                            "r2_q5_pct": np.nan, "r2_full_pct": np.nan,
                            "N_train/month": avg_n_train.get(model_name, np.nan)})
            continue
        preds = pd.read_parquet(pred_path)
        r2_q45 = r2_for_quintiles(preds, panel, [4, 5])
        r2_q4 = r2_for_quintiles(preds, panel, [4])
        r2_q5 = r2_for_quintiles(preds, panel, [5])
        r2_full = pooled_r2_zero(preds)
        rows_38.append({"model": model_name, "r2_q45_pct": r2_q45*100, "r2_q4_pct": r2_q4*100,
                         "r2_q5_pct": r2_q5*100, "r2_full_pct": r2_full*100,
                         "N_train/month": avg_n_train.get(model_name, np.nan)})
        logger.info("%s: R²(Q4-Q5)=%.3f%%, N_train/month≈%.0f", model_name,
                     r2_q45*100 if not np.isnan(r2_q45) else 0, avg_n_train.get(model_name, 0))

    comp38 = pd.DataFrame(rows_38)
    comp38.to_csv(output_dir / "restriction_comparison.csv", index=False)

    # Output 3.9: R² by quintile × training universe (Table 7 format)
    rows_39 = []
    for model_name, pred_path in models.items():
        if not pred_path.exists(): continue
        preds = pd.read_parquet(pred_path)
        for q in range(1, 6):
            r2_q = r2_for_quintiles(preds, panel, [q])
            rows_39.append({"model": model_name, "quintile": f"Q{q}", "r2_pct": r2_q * 100})
        r2_full = pooled_r2_zero(preds)
        rows_39.append({"model": model_name, "quintile": "Full sample", "r2_pct": r2_full * 100})
        r2_q45 = r2_for_quintiles(preds, panel, [4, 5])
        rows_39.append({"model": model_name, "quintile": "Q4-Q5 combined", "r2_pct": r2_q45 * 100})

    comp39 = pd.DataFrame(rows_39)

    # Pivot to Table 7 format: rows=quintile, columns=model
    quintile_order = ["Q1", "Q2", "Q3", "Q4", "Q5", "Full sample", "Q4-Q5 combined"]
    quintile_labels = {"Q1": "Q1 (Illiquid)", "Q5": "Q5 (Liquid)"}
    pivot = comp39.pivot(index="quintile", columns="model", values="r2_pct")
    model_order = [m for m in ["Mall", "MQ2+", "MQ3+", "MQ4+", "MQ5+"] if m in pivot.columns]
    pivot = pivot[model_order]
    pivot = pivot.reindex([q for q in quintile_order if q in pivot.index])
    pivot.index = [quintile_labels.get(q, q) for q in pivot.index]
    pivot = pivot.round(3)
    pivot.to_csv(output_dir / "restriction_by_quintile.csv")
    if len(comp39) > 0:
        pivot = comp39.pivot(index="quintile", columns="model", values="r2_pct")
        logger.info("R² by quintile × model:\n%s", pivot.to_string())

    # Output 3.7: Restriction curve
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from src.analysis.motivation import _set_academic_style
    _set_academic_style()
    valid = comp38.dropna(subset=["r2_q45_pct"])
    if len(valid) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        x_labels = valid["model"].tolist()
        x = np.arange(len(x_labels))
        ax.plot(x, valid["r2_q45_pct"], "o-", color="steelblue", linewidth=2, markersize=8)
        for xi, yi, lab in zip(x, valid["r2_q45_pct"], x_labels):
            ax.annotate(f"{yi:.3f}%", (xi, yi), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="-")
        ax.set_xticks(x); ax.set_xticklabels(x_labels)
        ax.set_xlabel("Training Universe")
        ax.set_ylabel(r"OOS $R^2$ on Q4--Q5 (\%)")
        ax.set_title("Restriction Curve: Liquid-Stock $R^2$ as Training Universe Narrows")
        plt.tight_layout()
        fig.savefig(output_dir / "restriction_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        logger.info("Saved restriction_curve.png")

    with open(output_dir / "meta.json", "w") as f:
        param_source = "baseline_tuned" if baseline_tuned is not None else ("fixed_config" if fixed_params is not None else "retune_within_restriction")
        json.dump({"levels_run": levels, "n_features": len(features),
                    "param_source": param_source}, f, indent=2, default=str)
    logger.info("Step 3d complete. Outputs: %s", output_dir)

if __name__ == "__main__":
    main()
