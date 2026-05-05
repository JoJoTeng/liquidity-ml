"""
Step 3d: Progressive Universe Restriction (Section 5.2d)
=========================================================
Trains four additional models with restricted training universes.
Uses the Step 3 baseline tuned-parameter path by default. Use --retune for
the robustness version that retunes within each restricted universe.
Test set includes ALL stocks. Primary metric: R²(Q4-Q5).

Models: MQ2+ (drop Q1), MQ3+ (drop Q1-Q2), MQ4+ (drop Q1-Q3), MQ5 (Q5 only)

Outputs (to outputs/motivation/step3_restriction/{model}/{liquidity_key}/{normalization}/{mode}/):
  - predictions_MQ{2,3,4,5}.parquet
  - restriction_curve.png       (Output 3.7)
  - restriction_comparison.csv  (Output 3.8)
  - restriction_by_quintile.csv (Output 3.9)
  - meta.json

Usage:
  python scripts/05_motivation_step3d_progressive_restriction.py
  python scripts/05_motivation_step3d_progressive_restriction.py --model elastic_net
  python scripts/05_motivation_step3d_progressive_restriction.py --normalization rerank
  python scripts/05_motivation_step3d_progressive_restriction.py --min-quintile 2
  python scripts/05_motivation_step3d_progressive_restriction.py --use-cache
"""
from __future__ import annotations
import argparse, ast, json, logging, os, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

_runtime_cache = Path(tempfile.gettempdir()) / "liquidity_ml_cache"
_runtime_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_runtime_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_runtime_cache / "xdg"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, get_data_dir, get_output_dir
from src.models import get_all_model_names
from src.analysis.motivation import (
    assign_nyse_quintiles,
    ensure_motivation_weight_column,
    get_motivation_liquidity_choices,
    get_motivation_liquidity_config,
    get_motivation_liquidity_key,
    rolling_model_predict_restricted,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _coerce_param_value(value):
    """Restore list/dict/bool values that may have been serialized to CSV."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text in {"", "nan", "NaN", "None"}:
        return np.nan
    if text[0] in "[{(" or text in {"True", "False"}:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value
    return value


def load_tuned_params(path: Path) -> pd.DataFrame:
    params = pd.read_csv(path, index_col="yyyymm")
    for col in params.columns:
        params[col] = params[col].map(_coerce_param_value)
    return params


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
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=get_all_model_names(),
        help="Model family to use, matching Step 3 baseline artifacts.",
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
    parser.add_argument(
        "--normalization",
        type=str,
        default="global",
        choices=["rerank", "global"],
        help=(
            "global = keep processed full-cross-section ranks; "
            "rerank = re-rank features after restriction."
        ),
    )
    parser.add_argument("--min-quintile", type=int, default=None, choices=[2,3,4,5],
                        help="Run single restriction level (for parallel runs)")
    parser.add_argument("--retune", action="store_true", help="Retune within restricted universe (robustness)")
    parser.add_argument(
        "--use-baseline-params",
        action="store_true",
        help="Use baseline's per-window tuned params (default; kept for explicit run scripts)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip training and recompute tables/figures from saved predictions.",
    )
    args = parser.parse_args()

    liq = get_motivation_liquidity_config(args.liquidity)
    config = load_config()
    data_dir = get_data_dir()
    liquidity_key = get_motivation_liquidity_key(args.liquidity, args.aum)

    if args.retune and args.use_baseline_params:
        logger.error("--retune and --use-baseline-params are mutually exclusive")
        sys.exit(1)

    # Mode subdirectory: baseline / retune
    mode = "retune" if args.retune else "baseline"

    base_output = Path(get_output_dir())
    output_dir = (
        base_output / "motivation" / "step3_restriction"
        / args.model / liquidity_key / args.normalization / mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir = base_output / "motivation" / "step3" / args.model / liquidity_key

    # Load data
    panel_path = data_dir / "processed_panel.parquet"
    if not panel_path.exists():
        logger.error("processed_panel.parquet not found. Run scripts/01_process_data.py first.")
        sys.exit(1)
    panel = pd.read_parquet(panel_path)
    logger.info("Panel: %d rows", len(panel))
    if args.liquidity == "tc":
        logger.info("TC AUM: $%.0fM", args.aum)

    # Features — must match pooled model
    feat_path = data_dir / "feature_list.json"
    if not feat_path.exists():
        logger.error("feature_list.json not found at %s. Run scripts/01_process_data.py first.", feat_path)
        sys.exit(1)
    with open(feat_path) as f:
        feat_meta = json.load(f)
    features = feat_meta["features"] if isinstance(feat_meta, dict) else feat_meta
    logger.info("Features: %d", len(features))

    # Quintiles
    ensure_motivation_weight_column(
        panel, args.liquidity, config=config, aum_millions=args.aum
    )
    panel["liq_quintile"] = assign_nyse_quintiles(
        panel, liq["quintile_col"], ascending=liq["ascending"]
    )

    # Param strategy: baseline tuned params by default; --retune is robustness.
    baseline_tuned = None
    if not args.retune:
        tp_path = pooled_dir / "tuned_params.csv"
        if not tp_path.exists():
            logger.error("tuned_params.csv not found at %s. Run scripts/04_motivation_step3_ml_diagnostics.py first.", tp_path)
            sys.exit(1)
        baseline_tuned = load_tuned_params(tp_path)
        logger.info("Loaded %d baseline tuned param windows from %s", len(baseline_tuned), tp_path)
    else:
        logger.info("Retuning %s within each restricted universe", args.model)

    levels = [args.min_quintile] if args.min_quintile else [2, 3, 4, 5]
    rerank_after_filter = args.normalization == "rerank"

    # Train restricted models
    if not args.use_cache:
        for mq in levels:
            label = f"MQ{mq}+"
            pred_path = output_dir / f"predictions_{label}.parquet"
            logger.info("=" * 60)
            logger.info(
                "Training %s %s (quintile >= %d, normalization=%s)...",
                args.model, label, mq, args.normalization,
            )
            preds = rolling_model_predict_restricted(
                panel, features, min_quintile=mq, model_name=args.model,
                quintile_col="liq_quintile",
                config=config,
                baseline_tuned_params=baseline_tuned,
                rerank_after_filter=rerank_after_filter,
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
    q45_restriction_rows = []
    for model_name, pred_path in models.items():
        if not pred_path.exists():
            q45_restriction_rows.append({
                "model": model_name,
                "r2_q45_pct": np.nan,
                "r2_q4_pct": np.nan,
                "r2_q5_pct": np.nan,
                "r2_full_pct": np.nan,
                "N_train/month": avg_n_train.get(model_name, np.nan),
            })
            continue
        preds = pd.read_parquet(pred_path)
        r2_q45 = r2_for_quintiles(preds, panel, [4, 5])
        r2_q4 = r2_for_quintiles(preds, panel, [4])
        r2_q5 = r2_for_quintiles(preds, panel, [5])
        r2_full = pooled_r2_zero(preds)
        q45_restriction_rows.append({
            "model": model_name,
            "r2_q45_pct": r2_q45 * 100,
            "r2_q4_pct": r2_q4 * 100,
            "r2_q5_pct": r2_q5 * 100,
            "r2_full_pct": r2_full * 100,
            "N_train/month": avg_n_train.get(model_name, np.nan),
        })
        logger.info("%s: R²(Q4-Q5)=%.3f%%, N_train/month≈%.0f", model_name,
                     r2_q45*100 if not np.isnan(r2_q45) else 0, avg_n_train.get(model_name, 0))

    q45_restriction_comparison = pd.DataFrame(q45_restriction_rows)
    q45_restriction_comparison.to_csv(
        output_dir / "restriction_comparison.csv", index=False
    )

    # Output 3.9: R² by individual quintile × training universe.
    rows_39 = []
    for model_name, pred_path in models.items():
        if not pred_path.exists(): continue
        preds = pd.read_parquet(pred_path)
        for q in range(1, 6):
            r2_q = r2_for_quintiles(preds, panel, [q])
            rows_39.append({"model": model_name, "quintile": f"Q{q}", "r2_pct": r2_q * 100})

    comp39 = pd.DataFrame(rows_39)

    # Pivot to table format: rows=individual quintile, columns=model.
    if comp39.empty:
        logger.warning("No prediction files available for restriction_by_quintile.csv")
    else:
        quintile_order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        quintile_labels = {"Q1": "Q1 (Illiquid)", "Q5": "Q5 (Liquid)"}
        pivot = comp39.pivot(index="quintile", columns="model", values="r2_pct")
        model_order = [m for m in ["Mall", "MQ2+", "MQ3+", "MQ4+", "MQ5+"] if m in pivot.columns]
        pivot = pivot[model_order]
        pivot = pivot.reindex([q for q in quintile_order if q in pivot.index])
        pivot.index = [quintile_labels.get(q, q) for q in pivot.index]
        pivot = pivot.round(3)
        pivot.to_csv(output_dir / "restriction_by_quintile.csv")
        logger.info("R² by quintile × model:\n%s", pivot.to_string())

    # Output 3.7: Restriction curve
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from src.analysis.motivation import _set_academic_style
    _set_academic_style()
    valid = q45_restriction_comparison.dropna(subset=["r2_q45_pct"])
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
        param_source = "baseline_tuned" if baseline_tuned is not None else "retune_within_restriction"
        json.dump({
            "model": args.model,
            "liquidity": args.liquidity,
            "liquidity_key": liquidity_key,
            "normalization": args.normalization,
            "rerank_after_filter": rerank_after_filter,
            "levels_run": levels,
            "n_features": len(features),
            "param_source": param_source,
            "pooled_baseline_dir": str(pooled_dir),
        }, f, indent=2, default=str)
    logger.info("Step 3d complete. Outputs: %s", output_dir)

if __name__ == "__main__":
    main()
