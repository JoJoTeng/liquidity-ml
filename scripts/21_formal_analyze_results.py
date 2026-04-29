"""
03 - Analyze Formal Experiment Results (LiquidityML v3)
========================================================
Reads predictions from 20_formal_run_experiment.py and produces all tables
and figures specified in Sections 8-9 of the v3 paper.

Outputs (all in outputs/formalanalysis/analysis/):
  Tables:
    prediction_1_r2_by_quintile_{spec}.csv
    prediction_1_utility_weighted_r2_{spec}.csv
    prediction_2_importance_shift_{spec}.csv
    prediction_4_se_differential_{spec}.csv
    table_11_within_quintile_{spec}.csv
    table_12_decomposition_{spec}_{aum}.csv
    hypothesis_tests.json

  Figures:
    squared_error_differential_{spec}.png   - Prediction 4
    importance_shift_{spec}.png             - Prediction 2

Usage:
    python scripts/21_formal_analyze_results.py
    python scripts/21_formal_analyze_results.py --model xgboost
    python scripts/21_formal_analyze_results.py --weights dolvol
    python scripts/21_formal_analyze_results.py --extra-benchmarks
    python scripts/21_formal_analyze_results.py --no-figures
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

# Keep matplotlib/fontconfig quiet on systems where the default user cache
# directories are not writable (common in sandboxed runs and cluster jobs).
_runtime_cache = Path(tempfile.gettempdir()) / "liquidity_ml_cache"
_runtime_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_runtime_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_runtime_cache / "xdg"))

# Suppress upstream library deprecation noise. These warnings come from
# pandas_datareader (date_parser) and pandas itself (groupby.apply future
# behaviour) — neither is actionable from our code.
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*date_parser.*"
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning,
    message=".*DataFrameGroupBy.apply operated on the grouping columns.*"
)

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.loader import load_processed_panel
from src.analysis.motivation import compute_quintile_oos_r2, compute_utility_weighted_r2
from src.evaluation.statistics import (
    sharpe_ratio,
    bootstrap_sharpe_test,
    factor_alpha,
    load_ff_factors,
    oos_r_squared,
    compute_effect_decomposition,
)
from src.portfolio.construction import (
    build_portfolio_timeseries,
    compute_net_returns,
)
from src.weighting.schemes import compute_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("03_analysis")


# -- CLI ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze formal experiment results")
    p.add_argument("--model", default=None,
                    choices=["elastic_net", "xgboost", "neural_network"])
    p.add_argument("--weights", default=None,
                    choices=["dolvol", "softmax_rank", "tc"])
    p.add_argument(
        "--extra-benchmarks",
        action="store_true",
        help=(
            "Also report cross-sectional-mean and historical-mean R² columns. "
            "Default formal Prediction 1 table reports zero-benchmark R² only."
        ),
    )
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


# -- Discovery & Loading -----------------------------------------------

def discover_experiments(base_dir: Path) -> list[dict]:
    """Discover all completed experiment specifications."""
    specs = []
    if not base_dir.exists():
        return specs
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        std_dir = model_dir / "standard"
        if not (std_dir / "predictions.parquet").exists():
            continue
        for wt_dir in sorted(model_dir.iterdir()):
            if wt_dir.name == "standard" or not wt_dir.is_dir():
                continue
            if not (wt_dir / "predictions.parquet").exists():
                continue
            dirname = wt_dir.name
            if dirname == "dolvol":
                weight_family = "dolvol"
                aum_label = None
                softmax_lambda = None
            elif dirname == "softmax_rank":
                # Backward-compatible default folder: lambda comes from config
                # at training time, historically lambda=2.
                weight_family = "softmax_rank"
                aum_label = None
                softmax_lambda = None
            elif dirname.startswith("softmax_rank_lam"):
                weight_family = "softmax_rank"
                aum_label = None
                token = dirname.removeprefix("softmax_rank_lam")
                try:
                    softmax_lambda = float(
                        token.replace("m", "-").replace("p", ".")
                    )
                except ValueError:
                    softmax_lambda = None
            elif dirname.startswith("tc_"):
                weight_family = "tc"
                aum_label = dirname
                softmax_lambda = None
            else:
                continue
            specs.append({
                "model": model_name,
                "weight_family": weight_family,
                "aum_label": aum_label,
                "softmax_lambda": softmax_lambda,
                "std_dir": std_dir,
                "wt_dir": wt_dir,
                "spec_label": f"{model_name}_{dirname}",
            })
    return specs


def load_predictions(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds_std = pd.read_parquet(spec["std_dir"] / "predictions.parquet")
    preds_wt = pd.read_parquet(spec["wt_dir"] / "predictions.parquet")
    return preds_std, preds_wt


def load_importance(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    imp_std = pd.read_csv(spec["std_dir"] / "importance_shap.csv")
    imp_wt = pd.read_csv(spec["wt_dir"] / "importance_shap.csv")
    return imp_std, imp_wt


def assign_liquidity_quintiles(panel: pd.DataFrame, config: dict) -> pd.Series:
    """NYSE-breakpoint liquidity quintiles per month (Section 2.3).

    Uses only NYSE stocks (exchcd=1) to compute 20/40/60/80 percentile
    breakpoints, then assigns all stocks (NYSE + NASDAQ + AMEX) to
    quintiles based on those breakpoints. Q1 = most illiquid, Q5 = most
    liquid. Delegates to src.analysis.motivation.assign_nyse_quintiles
    to match the motivation pipeline exactly.
    """
    from src.analysis.motivation import assign_nyse_quintiles

    liq_col = f"liq_{config['liquidity']['primary']}"
    if "exchcd" not in panel.columns:
        logger.warning(
            "exchcd column missing — falling back to full-sample quintiles. "
            "Re-run scripts/00_fetch_data.py to get NYSE breakpoints."
        )
        quintiles = pd.Series(np.nan, index=panel.index, name="liq_quintile")
        for yyyymm, group in panel.groupby("yyyymm"):
            liq = group[liq_col]
            breaks = liq.quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
            breaks[0] = -np.inf
            breaks[-1] = np.inf
            q = pd.cut(liq, bins=breaks, labels=[1, 2, 3, 4, 5], include_lowest=True)
            quintiles.loc[group.index] = q.astype(float)
        return quintiles

    return assign_nyse_quintiles(panel, sort_col=liq_col, ascending=True)


def parse_tc_aum(aum_label: str | None) -> float | None:
    """Parse formal TC folder labels such as ``tc_500m`` into dollars."""
    if aum_label is None:
        return None
    token = aum_label.removeprefix("tc_")
    if not token.endswith("m"):
        raise ValueError(f"Could not parse TC AUM label: {aum_label!r}")
    return float(token[:-1]) * 1_000_000


def formal_weight_label(spec: dict, config: dict) -> str:
    """Human-readable label for the utility weights in this formal spec."""
    family = spec["weight_family"]
    if family == "dolvol":
        return "dollar-volume"
    if family == "softmax_rank":
        lam = spec["softmax_lambda"]
        if lam is None:
            lam = config.get("weighting", {}).get("softmax_rank_lambda", 2.0)
        return f"softmax-rank(lambda={float(lam):g})"
    if family == "tc":
        return f"TC ({spec['aum_label']})"
    return family


def compute_formal_utility_weights(
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
) -> pd.Series:
    """Recreate the mean-one utility weights used by the weighted experiment."""
    cfg = copy.deepcopy(config)
    family = spec["weight_family"]
    aum = None

    if family == "softmax_rank" and spec["softmax_lambda"] is not None:
        cfg.setdefault("weighting", {})["softmax_rank_lambda"] = spec["softmax_lambda"]
    elif family == "tc":
        aum = parse_tc_aum(spec["aum_label"])

    return compute_weights(panel, scheme=family, config=cfg, aum=aum)


def predictions_for_utility_r2(
    predictions: pd.DataFrame,
    panel_sub: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Convert formal prediction files to the y_true/y_pred utility schema."""
    if "prediction" not in predictions.columns:
        raise ValueError("formal predictions must contain a 'prediction' column")

    return (
        predictions[["permno", "yyyymm", "prediction"]]
        .rename(columns={"prediction": "y_pred"})
        .merge(
            panel_sub[["permno", "yyyymm", target]],
            on=["permno", "yyyymm"],
            how="inner",
        )
        .rename(columns={target: "y_true"})
    )


# ------------------------------------------------------------------
# Prediction 1: Improved liquid-stock R^2
# ------------------------------------------------------------------

def compute_prediction_1(
    preds_std,
    preds_wt,
    panel,
    config,
    spec,
    include_extra_benchmarks: bool = False,
):
    target = config["data"]["target_col"]
    liq_col = f"liq_{config['liquidity']['primary']}"

    panel_sub = panel[["permno", "yyyymm", target, liq_col]].copy()
    panel_sub["liq_quintile"] = assign_liquidity_quintiles(panel, config)

    # R^2 by quintile for both models. This is an unweighted evaluation for
    # both M_std and M_w; training weights affect the fitted M_w predictions,
    # but not the loss used in this prediction table.
    hist_window = None
    if include_extra_benchmarks:
        hist_window = (
            config["training"]["train_window"]
            + config["training"]["validation_window"]
        )
    r2_std = compute_quintile_oos_r2(
        preds_std,
        panel_sub,
        "liq_quintile",
        return_col=target,
        hist_window=hist_window,
    )
    r2_wt = compute_quintile_oos_r2(
        preds_wt,
        panel_sub,
        "liq_quintile",
        return_col=target,
        hist_window=hist_window,
    )
    r2_table = r2_std.merge(r2_wt, on="quintile", suffixes=("_std", "_wt"))
    for bench in ["zero", "cs", "hist"]:
        std_col = f"pooled_r2_{bench}_std"
        wt_col = f"pooled_r2_{bench}_wt"
        if std_col not in r2_table or wt_col not in r2_table:
            continue
        r2_table[f"r2_{bench}_std_pct"] = r2_table[std_col] * 100
        r2_table[f"r2_{bench}_wt_pct"] = r2_table[wt_col] * 100
        r2_table[f"delta_{bench}_pct"] = (r2_table[wt_col] - r2_table[std_col]) * 100

    full_row = r2_table[r2_table["quintile"] == "Full"].iloc[0]
    r2_std_f = full_row["r2_zero_std_pct"] / 100
    r2_wt_f = full_row["r2_zero_wt_pct"] / 100

    # Backward-compatible names: the old table reported zero-benchmark R^2.
    r2_table["r2_std_pct"] = r2_table["r2_zero_std_pct"]
    r2_table["r2_wt_pct"] = r2_table["r2_zero_wt_pct"]
    r2_table["delta_pct"] = r2_table["delta_zero_pct"]
    r2_table["n_obs"] = r2_table["n_obs_std"]
    keep_cols = [
        "quintile",
        "r2_std_pct", "r2_wt_pct", "delta_pct",
        "avg_n_month_std", "avg_n_month_wt", "n_obs",
    ]
    if include_extra_benchmarks:
        keep_cols[4:4] = [
            "r2_zero_std_pct", "r2_zero_wt_pct", "delta_zero_pct",
            "r2_cs_std_pct", "r2_cs_wt_pct", "delta_cs_pct",
            "r2_hist_std_pct", "r2_hist_wt_pct", "delta_hist_pct",
        ]
    r2_table = r2_table[[c for c in keep_cols if c in r2_table.columns]]

    # Utility-weighted R^2 for both prediction sets. These weights match the
    # formal experiment spec (dolvol, softmax-rank lambda, or TC AUM), while
    # the benchmark remains zero.
    panel_utility = panel_sub.copy()
    panel_utility["w_tilde"] = compute_formal_utility_weights(panel, config, spec)
    std_eval = predictions_for_utility_r2(preds_std, panel_sub, target)
    wt_eval = predictions_for_utility_r2(preds_wt, panel_sub, target)
    r2_uw_std = compute_utility_weighted_r2(std_eval, panel_utility)["r2_weighted_zero"]
    r2_uw_wt = compute_utility_weighted_r2(wt_eval, panel_utility)["r2_weighted_zero"]
    utility_label = formal_weight_label(spec, config)

    utility_r2 = pd.DataFrame([
        {
            "metric": "Unweighted evaluation",
            "r2_std_pct": r2_std_f * 100,
            "r2_wt_pct": r2_wt_f * 100,
        },
        {
            "metric": f"Utility-weighted evaluation ({utility_label})",
            "r2_std_pct": r2_uw_std * 100,
            "r2_wt_pct": r2_uw_wt * 100,
        },
        {
            "metric": f"Gap (unweighted - {utility_label})",
            "r2_std_pct": (r2_std_f - r2_uw_std) * 100,
            "r2_wt_pct": (r2_wt_f - r2_uw_wt) * 100,
        },
    ])

    return {"r2_by_quintile": r2_table, "utility_weighted_r2": utility_r2}


# ------------------------------------------------------------------
# Prediction 2: Feature importance reallocation
# ------------------------------------------------------------------

def compute_prediction_2(
    imp_std,
    imp_wt,
    interaction_csv_path=None,
    quintile_csv_path=None,
):
    """Prediction 2: Feature importance reallocation under weighted training.

    Components (v3 document, Section 8):
      1. Per-feature SHAP shift table with paired-window t-stats
      2. Regression of Delta_I_j on gamma_bar_j from Step 2 (if available)
      3. Grouped importance-share analysis for Q1-only / Q5-only / both

    Parameters
    ----------
    imp_std, imp_wt : per-window SHAP DataFrames (one row per window).
    interaction_csv_path : Path to Step 2 interaction_regression.csv
        (columns: feature, beta_bar, beta_t, gamma_bar, gamma_t).
    quintile_csv_path : Path to Step 2 quintile_fm_coefficients.csv
        with per-feature Q1/Q5 coefficients and t-statistics.
    """
    common_months = set(imp_std["yyyymm"]) & set(imp_wt["yyyymm"])
    imp_std = imp_std[imp_std["yyyymm"].isin(common_months)].sort_values("yyyymm")
    imp_wt = imp_wt[imp_wt["yyyymm"].isin(common_months)].sort_values("yyyymm")
    features = [c for c in imp_std.columns if c != "yyyymm"]

    # -- 1. Per-feature SHAP shift table --
    rows = []
    for feat in features:
        sv = imp_std[feat].values
        wv = imp_wt[feat].values
        valid = ~(np.isnan(sv) | np.isnan(wv))
        if valid.sum() < 5:
            continue
        mean_std = np.nanmean(sv[valid])
        mean_wt = np.nanmean(wv[valid])
        delta = mean_wt - mean_std
        diffs = wv[valid] - sv[valid]
        se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
        t_stat = np.mean(diffs) / se if se > 0 else 0.0
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=len(diffs) - 1))
        rows.append({
            "feature": feat, "mean_shap_std": mean_std, "mean_shap_wt": mean_wt,
            "delta": delta, "t_stat": t_stat, "p_value": p_val, "n_windows": int(valid.sum()),
        })
    if not rows:
        logger.warning(
            "  SHAP CSVs have no feature columns — skipping Prediction 2. "
            "(Likely NN due to earlier SHAP bug; re-run NN jobs with the fix "
            "to populate importance_shap.csv.)"
        )
        return {
            "importance_shift": pd.DataFrame(
                columns=["feature", "mean_shap_std", "mean_shap_wt",
                         "delta", "t_stat", "p_value", "n_windows"]
            ),
        }

    shift_df = pd.DataFrame(rows).sort_values("delta", ascending=False)

    result = {"importance_shift": shift_df}

    # -- 2. Delta_I_j regression on gamma_bar_j --
    if interaction_csv_path is not None and Path(interaction_csv_path).exists():
        try:
            gamma_df = pd.read_csv(interaction_csv_path)
            merged = shift_df.merge(
                gamma_df[["feature", "gamma_bar"]],
                on="feature", how="inner",
            )
            if len(merged) >= 3:
                x = merged["gamma_bar"].values
                y = merged["delta"].values
                valid_xy = ~(np.isnan(x) | np.isnan(y))
                if valid_xy.sum() >= 3:
                    slope, intercept, r_val, p_val, se = stats.linregress(
                        x[valid_xy], y[valid_xy],
                    )
                    result["delta_vs_gamma_regression"] = {
                        "n_features": int(valid_xy.sum()),
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "r_value": float(r_val),
                        "r_squared": float(r_val ** 2),
                        "p_value": float(p_val),
                        "std_err": float(se),
                    }
                    result["delta_vs_gamma_data"] = merged[
                        ["feature", "gamma_bar", "delta", "t_stat"]
                    ]
        except Exception as e:
            logger.warning("  Delta vs gamma regression failed: %s", e)

    # -- 3. Grouped importance shares (Q1-only / Q5-only / both) --
    if quintile_csv_path is not None and Path(quintile_csv_path).exists():
        try:
            q_df = pd.read_csv(quintile_csv_path)
            # Parse "<beta> (<t>)" format from columns Q1, Q5
            def _parse_t(cell):
                if not isinstance(cell, str):
                    return np.nan
                try:
                    return float(cell.split("(")[1].rstrip(")"))
                except (IndexError, ValueError):
                    return np.nan

            q1_col = next((c for c in q_df.columns if "Q1" in c), None)
            q5_col = next((c for c in q_df.columns if "Q5" in c and "Q5-Q1" not in c), None)
            if q1_col and q5_col:
                q_df["t_Q1"] = q_df[q1_col].map(_parse_t)
                q_df["t_Q5"] = q_df[q5_col].map(_parse_t)
                feat_col = next((c for c in q_df.columns if c.lower() == "feature"), q_df.columns[0])
                q_df = q_df.rename(columns={feat_col: "feature"})

                sig_q1 = q_df["t_Q1"].abs() > 2
                sig_q5 = q_df["t_Q5"].abs() > 2
                q_df["group"] = np.select(
                    [sig_q1 & ~sig_q5, ~sig_q1 & sig_q5, sig_q1 & sig_q5],
                    ["Q1_only", "Q5_only", "both"],
                    default="neither",
                )

                shift_with_group = shift_df.merge(
                    q_df[["feature", "group"]], on="feature", how="inner",
                )
                total_std = shift_with_group["mean_shap_std"].sum()
                total_wt = shift_with_group["mean_shap_wt"].sum()

                group_rows = []
                for grp in ["Q1_only", "Q5_only", "both", "neither"]:
                    sub = shift_with_group[shift_with_group["group"] == grp]
                    if len(sub) == 0:
                        continue
                    share_std = sub["mean_shap_std"].sum() / total_std if total_std > 0 else np.nan
                    share_wt = sub["mean_shap_wt"].sum() / total_wt if total_wt > 0 else np.nan
                    group_rows.append({
                        "group": grp,
                        "n_features": len(sub),
                        "share_std_pct": share_std * 100,
                        "share_wt_pct": share_wt * 100,
                        "delta_share_pct": (share_wt - share_std) * 100,
                    })
                result["group_shares"] = pd.DataFrame(group_rows)
        except Exception as e:
            logger.warning("  Group shares analysis failed: %s", e)

    return result


# ------------------------------------------------------------------
# Prediction 3: Weighted model dominates the restriction curve
# ------------------------------------------------------------------

def compute_prediction_3(preds_wt, panel, config, restriction_csv_path):
    """Place the weighted model's liquid-stock R^2 against the Step 3d curve.

    Parameters
    ----------
    preds_wt : Weighted-model OOS predictions (DataFrame with permno, yyyymm, prediction).
    panel : Full panel (for liquidity quintile assignment + target).
    config : Full config dict.
    restriction_csv_path : Path to restriction_comparison.csv from Step 3d.

    Returns
    -------
    dict with:
      - restriction_df : augmented table with an extra row for the weighted model
      - weighted_r2_q45 : float, the weighted model's R^2 on Q4-Q5 stocks
      - dominates : bool, True if weighted R^2 exceeds the best point on curve
    """
    target = config["data"]["target_col"]
    panel_sub = panel[["permno", "yyyymm", target]].copy()
    panel_sub["liq_quintile"] = assign_liquidity_quintiles(panel, config)

    merged = preds_wt.merge(panel_sub, on=["permno", "yyyymm"], how="inner")
    q45 = merged[merged["liq_quintile"] >= 4]

    if len(q45) == 0:
        return {
            "restriction_df": None,
            "weighted_r2_q45": np.nan,
            "dominates": False,
        }

    # R^2 under zero benchmark (Eq. 8 in v3 document)
    ss_pred = ((q45[target] - q45["prediction"]) ** 2).sum()
    ss_total = (q45[target] ** 2).sum()
    r2_q45 = (1.0 - ss_pred / ss_total) * 100 if ss_total > 0 else np.nan

    if not restriction_csv_path.exists():
        logger.warning(
            "Restriction curve CSV not found at %s — skipping Prediction 3 overlay",
            restriction_csv_path,
        )
        return {
            "restriction_df": None,
            "weighted_r2_q45": r2_q45,
            "dominates": False,
        }

    curve = pd.read_csv(restriction_csv_path)
    best_on_curve = curve["r2_q45_pct"].max()
    dominates = r2_q45 > best_on_curve

    # Append weighted model row
    out_df = curve.copy()
    new_row = {col: np.nan for col in out_df.columns}
    new_row["model"] = "M_w (weighted)"
    new_row["r2_q45_pct"] = r2_q45
    out_df = pd.concat([out_df, pd.DataFrame([new_row])], ignore_index=True)

    return {
        "restriction_df": out_df,
        "weighted_r2_q45": r2_q45,
        "best_on_curve": best_on_curve,
        "dominates": dominates,
    }


def plot_restriction_curve_with_weighted(result, out_path, model, weight_family):
    """Bar chart of the restriction curve with the weighted model highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = result["restriction_df"]
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue"] * (len(df) - 1) + ["coral"]
    ax.bar(df["model"], df["r2_q45_pct"], color=colors)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylabel("OOS $R^2$ on Q4-Q5 (%)")
    ax.set_title(
        f"Prediction 3: Weighted model vs restriction curve ({model} / {weight_family})"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", out_path)


# ------------------------------------------------------------------
# Prediction 4: Cumulative squared error differential
# ------------------------------------------------------------------

def compute_prediction_4(preds_std, preds_wt, panel, config):
    target = config["data"]["target_col"]
    panel_sub = panel[["permno", "yyyymm", target]].copy()
    panel_sub["liq_quintile"] = assign_liquidity_quintiles(panel, config)

    std_m = preds_std.merge(panel_sub, on=["permno", "yyyymm"], how="inner")
    wt_m = preds_wt.merge(panel_sub, on=["permno", "yyyymm"], how="inner")

    # Q4-Q5 only
    std_liq = std_m[std_m["liq_quintile"] >= 4]
    wt_liq = wt_m[wt_m["liq_quintile"] >= 4]

    std_se = std_liq.groupby("yyyymm").apply(
        lambda g: ((g[target] - g["prediction"]) ** 2).mean()
    ).rename("se_std")
    wt_se = wt_liq.groupby("yyyymm").apply(
        lambda g: ((g[target] - g["prediction"]) ** 2).mean()
    ).rename("se_wt")

    diff = pd.DataFrame({"se_std": std_se, "se_wt": wt_se}).dropna()
    diff["se_diff"] = diff["se_std"] - diff["se_wt"]
    diff["cumulative_diff"] = diff["se_diff"].cumsum()
    return diff.reset_index()


# ------------------------------------------------------------------
# Table 11: Within-quintile portfolio performance
# ------------------------------------------------------------------

def compute_table_11(preds_std, preds_wt, panel, aum, config):
    panel_work = panel.copy()
    panel_work["liq_quintile"] = assign_liquidity_quintiles(panel_work, config)

    rows = []
    for q in [1, 2, 3, 4, 5]:
        panel_q = panel_work[panel_work["liq_quintile"] == q].copy()
        if len(panel_q) < 100:
            continue
        for label, preds in [("std", preds_std), ("wt", preds_wt)]:
            # Pass DataFrame directly — build_portfolio_timeseries merges
            # on (permno, yyyymm) so row alignment is guaranteed.
            ret_df, pos_hist = build_portfolio_timeseries(
                panel_q, preds, tc_penalised=False, config=config,
            )
            if len(ret_df) < 12:
                continue

            gross_sr = sharpe_ratio(ret_df["ret_long_short"].dropna())
            net_df = compute_net_returns(ret_df, pos_hist, panel, aum=aum, config=config)
            net_sr = sharpe_ratio(net_df["ret_long_short_net"].dropna())

            rows.append({
                "quintile": f"Q{q}", "model_type": label,
                "gross_sr": gross_sr, "net_sr": net_sr,
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="quintile", columns="model_type", values=["gross_sr", "net_sr"])
    pivot.columns = [f"{m}_{t}" for m, t in pivot.columns]
    return pivot.reset_index()


# ------------------------------------------------------------------
# Table 12: 2x2 Decomposition
# ------------------------------------------------------------------

def compute_table_12(preds_std, preds_wt, panel, aum, config):
    """Build the 2x2 decomposition at a given AUM level.

    Returns a dict with:
      - decomposition : output of compute_effect_decomposition (sharpe ratios,
        effects, LW tests, factor alphas per cell)
      - training_share : training_effect / total_effect * 100
      - cells : {cell: DataFrame with gross + net returns per month}
      - turnover : drift-adjusted conventional turnover per cell, computed as
        0.5 * sum(|target_w - drifted_w|) from compute_net_returns
    """
    logger.info("  Building 2x2 portfolios at AUM=$%.0fM...", aum / 1e6)

    cells = {}
    for cell_name, preds, tc_pen in [
        ("1A", preds_std, False), ("1B", preds_std, True),
        ("2A", preds_wt, False), ("2B", preds_wt, True),
    ]:
        # Pass the DataFrame directly — build_portfolio_timeseries merges
        # on (permno, yyyymm) so row alignment is guaranteed.
        ret_df, pos_hist = build_portfolio_timeseries(
            panel, preds, tc_penalised=tc_pen,
            aum=aum if tc_pen else None, config=config,
        )
        net_df = compute_net_returns(ret_df, pos_hist, panel, aum=aum, config=config)
        cells[cell_name] = net_df

    # Align to common months
    common = set(cells["1A"]["yyyymm"])
    for k in ["1B", "2A", "2B"]:
        common &= set(cells[k]["yyyymm"])
    common = sorted(common)

    aligned = {}
    for k, df in cells.items():
        aligned[k] = df[df["yyyymm"].isin(common)].sort_values("yyyymm")

    return_summary = {}
    for cell_name, df in aligned.items():
        return_summary[cell_name] = {
            "gross_return_monthly": df["ret_long_short"].mean(),
            "gross_return_annual": df["ret_long_short"].mean() * 12,
            "net_return_monthly": df["ret_long_short_net"].mean(),
            "net_return_annual": df["ret_long_short_net"].mean() * 12,
            "gross_sr_monthly": sharpe_ratio(
                df["ret_long_short"].values,
                annualize=False,
            ),
            "net_sr_monthly": sharpe_ratio(
                df["ret_long_short_net"].values,
                annualize=False,
            ),
            "gross_sr_annual": sharpe_ratio(
                df["ret_long_short"].values,
                annualize=True,
            ),
            "net_sr_annual": sharpe_ratio(
                df["ret_long_short_net"].values,
                annualize=True,
            ),
            "tc_mean_monthly": df["transaction_cost"].mean(),
            "tc_median_monthly": df["transaction_cost"].median(),
        }

    # Net decomposition (primary — Section 9.5)
    decomp_net = compute_effect_decomposition(
        returns_1a=aligned["1A"]["ret_long_short_net"].values,
        returns_1b=aligned["1B"]["ret_long_short_net"].values,
        returns_2a=aligned["2A"]["ret_long_short_net"].values,
        returns_2b=aligned["2B"]["ret_long_short_net"].values,
        yyyymm=aligned["1A"]["yyyymm"].values,
        config=config,
    )

    # Gross decomposition (secondary — Section 9.4 gross/net pair)
    decomp_gross = compute_effect_decomposition(
        returns_1a=aligned["1A"]["ret_long_short"].values,
        returns_1b=aligned["1B"]["ret_long_short"].values,
        returns_2a=aligned["2A"]["ret_long_short"].values,
        returns_2b=aligned["2B"]["ret_long_short"].values,
        yyyymm=aligned["1A"]["yyyymm"].values,
        config=config,
    )

    total = decomp_net["total_effect"]
    training_share = (decomp_net["training_effect"] / total * 100) if abs(total) > 1e-10 else np.nan

    # Turnover from the same drift-adjusted trades used by the TC engine.
    # Exclude the first month because it is initial portfolio formation rather
    # than a rebalance.
    turnover = {}
    raw_trade_sum = {}
    for cell_name, df in aligned.items():
        turnover_values = df["turnover"].iloc[1:] if "turnover" in df else pd.Series(dtype=float)
        raw_trade_values = df["raw_trade_sum"].iloc[1:] if "raw_trade_sum" in df else pd.Series(dtype=float)
        turnover[cell_name] = turnover_values.mean() if len(turnover_values) else np.nan
        raw_trade_sum[cell_name] = raw_trade_values.mean() if len(raw_trade_values) else np.nan

    return {
        "decomposition": decomp_net,
        "decomposition_gross": decomp_gross,
        "training_share": training_share,
        "cells": aligned,
        "return_summary": return_summary,
        "turnover": turnover,
        "raw_trade_sum": raw_trade_sum,
    }


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_squared_error_differential(diff_df, out_path, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(diff_df)), diff_df["cumulative_diff"], color="steelblue", lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylabel("Cumulative SE(M_std) - SE(M_w)")
    ax.set_title(f"Prediction 4: Squared Error Differential ({model}, Q4-Q5)")
    n = len(diff_df)
    tick_every = max(n // 10, 1)
    ax.set_xticks(range(0, n, tick_every))
    ax.set_xticklabels(diff_df["yyyymm"].astype(str).iloc[::tick_every], rotation=45, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", out_path)


def plot_importance_shift(shift_df, out_path, model, top_n=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = shift_df.head(top_n).copy()
    df = df.reindex(df["delta"].abs().sort_values(ascending=True).index)
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.25)))
    colors = ["steelblue" if d > 0 else "coral" for d in df["delta"]]
    ax.barh(df["feature"], df["delta"], color=colors)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Delta SHAP (weighted - standard)")
    ax.set_title(f"Prediction 2: Importance Reallocation ({model})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", out_path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config()

    base_dir = Path(config["project"]["output_dir"]) / "formalanalysis"
    experiment_dir = base_dir / "experiment"
    tables_dir = base_dir / "analysis" / "tables"
    figures_dir = base_dir / "analysis" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    specs = discover_experiments(experiment_dir)
    if not specs:
        logger.error("No completed experiments in %s", experiment_dir)
        sys.exit(1)

    if args.model:
        specs = [s for s in specs if s["model"] == args.model]
    if args.weights:
        specs = [s for s in specs if s["weight_family"] == args.weights]

    logger.info("Found %d specifications to analyze", len(specs))
    logger.info("Loading panel...")
    panel = load_processed_panel()

    aum_primary = 500_000_000
    aum_scenarios = config["transaction_costs"]["aum_scenarios"]

    all_results = {}

    for spec in specs:
        sl = spec["spec_label"]
        logger.info("=== %s ===", sl)
        preds_std, preds_wt = load_predictions(spec)
        logger.info("  Predictions: std=%d, wt=%d", len(preds_std), len(preds_wt))

        # Prediction 1
        logger.info("  Prediction 1...")
        p1 = compute_prediction_1(
            preds_std,
            preds_wt,
            panel,
            config,
            spec,
            include_extra_benchmarks=args.extra_benchmarks,
        )
        p1["r2_by_quintile"].to_csv(tables_dir / f"prediction_1_r2_{sl}.csv", index=False)
        p1["utility_weighted_r2"].to_csv(tables_dir / f"prediction_1_utility_r2_{sl}.csv", index=False)

        # Prediction 2 (SHAP shift + delta/gamma regression + group shares)
        logger.info("  Prediction 2...")
        try:
            imp_std, imp_wt = load_importance(spec)
            step2_dir = (
                Path(config["project"]["output_dir"])
                / "motivation" / "step2" / "dvol"
            )
            interaction_csv = step2_dir / "interaction_regression_full.csv"
            quintile_csv = step2_dir / "quintile_fm_coefficients_full.csv"

            p2 = compute_prediction_2(
                imp_std, imp_wt,
                interaction_csv_path=interaction_csv,
                quintile_csv_path=quintile_csv,
            )
            p2["importance_shift"].to_csv(tables_dir / f"prediction_2_{sl}.csv", index=False)

            if "delta_vs_gamma_regression" in p2:
                reg = p2["delta_vs_gamma_regression"]
                logger.info(
                    "  P2 delta vs gamma: slope=%.4f, R^2=%.3f, p=%.4f (n=%d)",
                    reg["slope"], reg["r_squared"], reg["p_value"], reg["n_features"],
                )
                with open(tables_dir / f"prediction_2_regression_{sl}.json", "w") as f:
                    json.dump(reg, f, indent=2)
                if "delta_vs_gamma_data" in p2:
                    p2["delta_vs_gamma_data"].to_csv(
                        tables_dir / f"prediction_2_delta_gamma_{sl}.csv", index=False,
                    )

            if "group_shares" in p2:
                p2["group_shares"].to_csv(
                    tables_dir / f"prediction_2_group_shares_{sl}.csv", index=False,
                )
                logger.info("  P2 group shares:\n%s", p2["group_shares"].to_string(index=False))

            if not args.no_figures:
                plot_importance_shift(p2["importance_shift"], figures_dir / f"importance_shift_{sl}.png", spec["model"])
        except FileNotFoundError as e:
            logger.warning("  Prediction 2 partial: %s", e)

        # Prediction 3 (weighted model vs restriction curve from Step 3d)
        logger.info("  Prediction 3...")
        restriction_csv = (
            Path(config["project"]["output_dir"])
            / "motivation" / "step3_restriction" / spec["model"] / "dvol" / "global" / "baseline"
            / "restriction_comparison.csv"
        )
        p3 = compute_prediction_3(preds_wt, panel, config, restriction_csv)
        if p3["restriction_df"] is not None:
            p3["restriction_df"].to_csv(
                tables_dir / f"prediction_3_{sl}.csv", index=False
            )
            logger.info(
                "  Prediction 3: weighted R^2(Q4-Q5)=%.3f%%, best on curve=%.3f%%, dominates=%s",
                p3["weighted_r2_q45"], p3.get("best_on_curve", np.nan), p3["dominates"],
            )
            if not args.no_figures:
                plot_restriction_curve_with_weighted(
                    p3, figures_dir / f"restriction_curve_{sl}.png",
                    spec["model"], spec["weight_family"],
                )

        # Prediction 4
        logger.info("  Prediction 4...")
        p4 = compute_prediction_4(preds_std, preds_wt, panel, config)
        p4.to_csv(tables_dir / f"prediction_4_{sl}.csv", index=False)
        if not args.no_figures:
            plot_squared_error_differential(p4, figures_dir / f"se_diff_{sl}.png", spec["model"])

        # Table 12 per AUM (Section 9.5 — gross + net + factor alphas)
        for aum in aum_scenarios:
            al = f"{aum // 1_000_000}M" if aum < 1_000_000_000 else f"{aum // 1_000_000_000}B"
            logger.info("  Table 12 (AUM=$%s)...", al)
            t12 = compute_table_12(preds_std, preds_wt, panel, aum=aum, config=config)
            d_net = t12["decomposition"]
            d_gross = t12["decomposition_gross"]

            rows = []
            # Panel A: returns, transaction costs, and Sharpe ratios per cell
            ret_summary = t12["return_summary"]
            for cell in ["1A", "1B", "2A", "2B"]:
                rows.append({
                    "metric": f"Gross return monthly ({cell})",
                    "value": ret_summary[cell]["gross_return_monthly"],
                })
                rows.append({
                    "metric": f"Gross return annualized ({cell})",
                    "value": ret_summary[cell]["gross_return_annual"],
                })
                rows.append({
                    "metric": f"Net return monthly ({cell})",
                    "value": ret_summary[cell]["net_return_monthly"],
                })
                rows.append({
                    "metric": f"Net return annualized ({cell})",
                    "value": ret_summary[cell]["net_return_annual"],
                })
                rows.append({
                    "metric": f"TC mean monthly ({cell})",
                    "value": ret_summary[cell]["tc_mean_monthly"],
                })
                rows.append({
                    "metric": f"TC median monthly ({cell})",
                    "value": ret_summary[cell]["tc_median_monthly"],
                })
                rows.append({
                    "metric": f"SR_gross_monthly({cell})",
                    "value": ret_summary[cell]["gross_sr_monthly"],
                })
                rows.append({
                    "metric": f"SR_net_monthly({cell})",
                    "value": ret_summary[cell]["net_sr_monthly"],
                })
                rows.append({
                    "metric": f"SR_gross_annualized({cell})",
                    "value": ret_summary[cell]["gross_sr_annual"],
                })
                rows.append({
                    "metric": f"SR_net_annualized({cell})",
                    "value": ret_summary[cell]["net_sr_annual"],
                })
                # Backward-compatible aliases used by earlier table exports.
                rows.append({
                    "metric": f"SR_gross({cell})",
                    "value": ret_summary[cell]["gross_sr_annual"],
                })
                rows.append({
                    "metric": f"SR_net({cell})",
                    "value": ret_summary[cell]["net_sr_annual"],
                })

            # Panel B: Net decomposition (primary — Section 9.5)
            rows += [
                {"metric": "Net training effect", "value": d_net["training_effect"]},
                {"metric": "Net portfolio effect", "value": d_net["portfolio_effect"]},
                {"metric": "Net total effect", "value": d_net["total_effect"]},
                {"metric": "Net interaction", "value": d_net["interaction"]},
                {"metric": "Training share (%)", "value": t12["training_share"]},
                {"metric": "LW p-val (training, net)", "value": d_net["lw_training"].get("p_value", np.nan)},
                {"metric": "LW p-val (total, net)", "value": d_net["lw_total"].get("p_value", np.nan)},
                {"metric": "LW p-val (H1, net)", "value": d_net["lw_h3"].get("p_value", np.nan)},
            ]

            # Panel C: Gross decomposition (secondary)
            rows += [
                {"metric": "Gross training effect", "value": d_gross["training_effect"]},
                {"metric": "Gross portfolio effect", "value": d_gross["portfolio_effect"]},
                {"metric": "Gross total effect", "value": d_gross["total_effect"]},
                {"metric": "LW p-val (training, gross)", "value": d_gross["lw_training"].get("p_value", np.nan)},
                {"metric": "LW p-val (total, gross)", "value": d_gross["lw_total"].get("p_value", np.nan)},
                {"metric": "LW p-val (H1, gross)", "value": d_gross["lw_h3"].get("p_value", np.nan)},
            ]

            # Panel D: Factor alphas (Section 9.4 — Eq. 26, FF5+Mom primary)
            # Uses NET returns as the regression target.
            alphas = d_net.get("factor_alphas", {})
            for model_name in ["capm", "ff3", "ff5", "ff5_mom"]:
                if model_name not in alphas:
                    continue
                for cell in ["1A", "1B", "2A", "2B"]:
                    a = alphas[model_name].get(cell, {})
                    rows.append({
                        "metric": f"alpha_{model_name}({cell})_annual",
                        "value": a.get("alpha_annual", np.nan),
                    })
                    rows.append({
                        "metric": f"alpha_{model_name}({cell})_tstat",
                        "value": a.get("alpha_tstat", np.nan),
                    })
                    rows.append({
                        "metric": f"alpha_{model_name}({cell})_pvalue",
                        "value": a.get("alpha_pvalue", np.nan),
                    })

            # Panel E: drift-adjusted turnover.
            for cell in ["1A", "1B", "2A", "2B"]:
                rows.append({
                    "metric": f"Turnover ({cell})",
                    "value": t12["turnover"].get(cell, np.nan),
                })
                rows.append({
                    "metric": f"Raw trade sum ({cell})",
                    "value": t12["raw_trade_sum"].get(cell, np.nan),
                })

            pd.DataFrame(rows).to_csv(tables_dir / f"table_12_{sl}_{al}.csv", index=False)
            if aum == aum_primary:
                all_results[sl] = t12

        # Table 11
        logger.info("  Table 11...")
        t11 = compute_table_11(preds_std, preds_wt, panel, aum=aum_primary, config=config)
        if len(t11) > 0:
            t11.to_csv(tables_dir / f"table_11_{sl}.csv", index=False)

    # Consolidated hypothesis tests
    logger.info("=== Hypothesis tests ===")
    hyp = {}
    for sl, t12 in all_results.items():
        d = t12["decomposition"]
        hyp[sl] = {
            "H1_training_share_pct": t12["training_share"],
            "H1_lw_pvalue": d["lw_h3"].get("p_value"),
            "H3_total_effect": d["total_effect"],
            "H3_lw_pvalue": d["lw_total"].get("p_value"),
            "sharpe_ratios": d["sharpe_ratios"],
            "training_effect": d["training_effect"],
            "portfolio_effect": d["portfolio_effect"],
            "interaction": d["interaction"],
        }

    with open(tables_dir / "hypothesis_tests.json", "w") as f:
        json.dump(hyp, f, indent=2, default=str)

    logger.info("=== Done. Outputs in %s ===", base_dir / "analysis")


if __name__ == "__main__":
    main()
