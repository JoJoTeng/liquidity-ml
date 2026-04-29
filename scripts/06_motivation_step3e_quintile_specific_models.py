"""
Step 3e: Quintile-Specific Models (Section 5.2e)
================================================
Trains five separate models (one per liquidity quintile).
Compares within-quintile R² vs. the pooled model.

Uses the Step 3 baseline tuned-parameter path by default. Use --retune for
the robustness version that retunes within each quintile.
Quintile assignment based on last month of training window.

Outputs (to outputs/motivation/step3_quintile/{model}/{liquidity_key}/{normalization}/{mode}/):
  - predictions_q{1-5}.parquet
  - r2_comparison.csv  (Output 3.10)
  - r2_comparison.png
  - meta.json

Usage:
  python scripts/06_motivation_step3e_quintile_specific_models.py
  python scripts/06_motivation_step3e_quintile_specific_models.py --model elastic_net
  python scripts/06_motivation_step3e_quintile_specific_models.py --normalization rerank
  python scripts/06_motivation_step3e_quintile_specific_models.py --quintile 5
  python scripts/06_motivation_step3e_quintile_specific_models.py --retune  # robustness: retune within quintile
  python scripts/06_motivation_step3e_quintile_specific_models.py --use-cache
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
    rolling_model_predict_quintile,
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


def main():
    parser = argparse.ArgumentParser(description="Step 3e: Quintile-Specific Models")
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
            "rerank = re-rank features after quintile filtering."
        ),
    )
    parser.add_argument("--quintile", type=int, default=None, choices=[1,2,3,4,5])
    parser.add_argument("--retune", action="store_true", help="Retune within quintile (robustness)")
    parser.add_argument("--use-baseline-params", action="store_true",
                        help="Use baseline's per-window tuned params (default).")
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
        base_output / "motivation" / "step3_quintile"
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
        logger.info("Retuning %s within each quintile", args.model)

    quintiles = [args.quintile] if args.quintile else [1,2,3,4,5]
    rerank_after_filter = args.normalization == "rerank"

    if not args.use_cache:
        for q in quintiles:
            logger.info("=" * 60)
            logger.info(
                "Training %s Q%d-specific model (normalization=%s)...",
                args.model, q, args.normalization,
            )
            preds = rolling_model_predict_quintile(
                panel, features, quintile=q, model_name=args.model,
                quintile_col="liq_quintile", config=config,
                baseline_tuned_params=baseline_tuned,
                rerank_after_filter=rerank_after_filter,
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

    # Avg N_train/month per quintile
    avg_n_train = panel.groupby(["yyyymm", "liq_quintile"]).size().groupby("liq_quintile").mean()

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
        n_train = int(avg_n_train.get(q, 0))
        rows.append({"quintile": f"Q{q}", "r2_pooled_pct": r2_pooled*100 if not np.isnan(r2_pooled) else np.nan,
                      "r2_own_pct": r2_own*100, "delta_pp": delta, "N_train/month": n_train})
        logger.info("Q%d: pooled=%.3f%%, own=%.3f%%, Δ=%.3f pp, N_train≈%d", q,
                     r2_pooled*100 if not np.isnan(r2_pooled) else 0, r2_own*100,
                     delta if not np.isnan(delta) else 0, n_train)

    comp = pd.DataFrame(rows)
    comp.to_csv(output_dir / "r2_comparison.csv", index=False)

    # Bar chart
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from src.analysis.motivation import _set_academic_style
    _set_academic_style()
    valid = comp.dropna(subset=["r2_own_pct"])
    if len(valid) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(valid)); w = 0.35
        if valid["r2_pooled_pct"].notna().any():
            ax.bar(x - w/2, valid["r2_pooled_pct"], w, label="Pooled model", color="steelblue", alpha=0.85)
            ax.bar(x + w/2, valid["r2_own_pct"], w, label="Quintile-specific", color="darkorange", alpha=0.85)
        else:
            ax.bar(x, valid["r2_own_pct"], w, label="Quintile-specific", color="darkorange")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(valid["quintile"].tolist())
        ax.set_xlabel("Liquidity Quintile")
        ax.set_ylabel(r"OOS $R^2$ (\%)")
        ax.set_title(rf"Pooled vs.\ Quintile-Specific {args.model}: Within-Quintile $R^2$")
        ax.legend(fontsize=10); plt.tight_layout()
        fig.savefig(output_dir / "r2_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    with open(output_dir / "meta.json", "w") as f:
        param_source = "baseline_tuned" if baseline_tuned is not None else "retune_within_quintile"
        json.dump({
            "model": args.model,
            "liquidity": args.liquidity,
            "liquidity_key": liquidity_key,
            "normalization": args.normalization,
            "mode": mode,
            "quintiles_run": quintiles,
            "n_features": len(features),
            "param_source": param_source,
            "pooled_baseline_dir": str(pooled_dir),
            "comparison": comp.to_dict(orient="records"),
        }, f, indent=2, default=str)
    logger.info("Step 3e complete. Outputs: %s", output_dir)

if __name__ == "__main__":
    main()
