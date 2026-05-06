"""Feature-importance reallocation analysis for formal experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.statistics import newey_west_tstat

logger = logging.getLogger(__name__)


def analyze_importance_reallocation(
    importance_standard: pd.DataFrame,
    importance_weighted: pd.DataFrame,
    interaction_csv_path: Path | None = None,
    quintile_raw_csv_path: Path | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Measure raw and share-based feature-importance reallocation."""
    common_months = set(importance_standard["yyyymm"]) & set(
        importance_weighted["yyyymm"]
    )
    importance_standard = importance_standard[
        importance_standard["yyyymm"].isin(common_months)
    ].sort_values("yyyymm")
    importance_weighted = importance_weighted[
        importance_weighted["yyyymm"].isin(common_months)
    ].sort_values("yyyymm")
    features = [c for c in importance_standard.columns if c != "yyyymm"]

    rows = []
    for feature in features:
        std_values = importance_standard[feature].values
        wt_values = importance_weighted[feature].values
        valid = ~(np.isnan(std_values) | np.isnan(wt_values))
        if valid.sum() < 5:
            continue

        mean_std = np.nanmean(std_values[valid])
        mean_wt = np.nanmean(wt_values[valid])
        diffs = wt_values[valid] - std_values[valid]
        nw = newey_west_tstat(diffs, lags=6)
        delta = nw["mean"]

        rows.append(
            {
                "feature": feature,
                "mean_importance_std": mean_std,
                "mean_importance_wt": mean_wt,
                "delta": delta,
                "delta_t": nw["t_stat"],
                "p_value": nw["p_value"],
                "n_windows": int(valid.sum()),
            }
        )

    if not rows:
        logger.warning(
            "Importance CSVs have no feature columns; skipping importance reallocation."
        )
        return {
            "importance_shift": pd.DataFrame(
                columns=[
                    "feature",
                    "mean_importance_std",
                    "mean_importance_wt",
                    "delta",
                    "delta_t",
                    "p_value",
                    "n_windows",
                ]
            )
        }

    shift_df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    result: dict[str, pd.DataFrame | dict] = {"importance_shift": shift_df}

    if interaction_csv_path is not None and Path(interaction_csv_path).exists():
        _add_delta_gamma_regression(result, Path(interaction_csv_path))

    if quintile_raw_csv_path is not None and Path(quintile_raw_csv_path).exists():
        _add_grouped_importance_shares(result, Path(quintile_raw_csv_path))

    result["importance_shift"] = result["importance_shift"].sort_values(
        "delta",
        ascending=False,
    )
    ordered_cols = [
        "feature",
        "group",
        "gamma_bar",
        "gamma_t",
        "mean_importance_std",
        "mean_importance_wt",
        "delta",
        "delta_t",
        "p_value",
        "n_windows",
        "share_std_pct",
        "share_wt_pct",
        "delta_share_pct",
    ]
    existing_ordered = [
        col for col in ordered_cols if col in result["importance_shift"].columns
    ]
    remaining_cols = [
        col for col in result["importance_shift"].columns if col not in existing_ordered
    ]
    result["importance_shift"] = result["importance_shift"][
        existing_ordered + remaining_cols
    ]

    return result


def _add_delta_gamma_regression(
    result: dict[str, pd.DataFrame | dict],
    interaction_csv_path: Path,
) -> None:
    """Regress feature-importance changes on Step 2 liquid-minus-illiquid slopes."""
    try:
        gamma_df = pd.read_csv(interaction_csv_path)
        gamma_cols = [
            col for col in ["feature", "gamma_bar", "gamma_t"] if col in gamma_df
        ]
        merged = result["importance_shift"].merge(
            gamma_df[gamma_cols],
            on="feature",
            how="left",
        )
        result["importance_shift"] = merged
        regression_df = merged.dropna(subset=["gamma_bar", "delta"])
        if len(regression_df) < 3:
            return

        x = regression_df["gamma_bar"].values
        y = regression_df["delta"].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 3:
            return

        slope, intercept, r_value, p_value, se = stats.linregress(
            x[valid],
            y[valid],
        )
        result["delta_vs_gamma_regression"] = {
            "n_features": int(valid.sum()),
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "std_err": float(se),
        }
    except Exception as exc:
        logger.warning("Delta vs gamma regression failed: %s", exc)


def _add_grouped_importance_shares(
    result: dict[str, pd.DataFrame | dict],
    quintile_raw_csv_path: Path,
) -> None:
    """Group importance shares by Q1-only, Q5-only, both, and neither signals."""
    try:
        q_df = pd.read_csv(quintile_raw_csv_path)
        required_cols = {"feature", "t_Q1", "t_Q5"}
        missing = required_cols - set(q_df.columns)
        if missing:
            logger.warning(
                "Grouped importance-share analysis skipped; %s missing from %s",
                sorted(missing),
                quintile_raw_csv_path,
            )
            return

        q_df["t_Q1"] = pd.to_numeric(q_df["t_Q1"], errors="coerce")
        q_df["t_Q5"] = pd.to_numeric(q_df["t_Q5"], errors="coerce")

        sig_q1 = q_df["t_Q1"].abs() > 2
        sig_q5 = q_df["t_Q5"].abs() > 2
        q_df["group"] = np.select(
            [sig_q1 & ~sig_q5, ~sig_q1 & sig_q5, sig_q1 & sig_q5],
            ["Q1_only", "Q5_only", "both"],
            default="neither",
        )

        shift_with_group = result["importance_shift"].merge(
            q_df[["feature", "group"]],
            on="feature",
            how="inner",
        )
        total_std = shift_with_group["mean_importance_std"].sum()
        total_wt = shift_with_group["mean_importance_wt"].sum()
        if total_std > 0:
            shift_with_group["share_std_pct"] = (
                shift_with_group["mean_importance_std"] / total_std * 100
            )
        else:
            shift_with_group["share_std_pct"] = np.nan
        if total_wt > 0:
            shift_with_group["share_wt_pct"] = (
                shift_with_group["mean_importance_wt"] / total_wt * 100
            )
        else:
            shift_with_group["share_wt_pct"] = np.nan
        shift_with_group["delta_share_pct"] = (
            shift_with_group["share_wt_pct"] - shift_with_group["share_std_pct"]
        )

        group_rows = []
        for group in ["Q1_only", "Q5_only", "both", "neither"]:
            sub = shift_with_group[shift_with_group["group"] == group]
            if len(sub) == 0:
                continue
            share_std = sub["share_std_pct"].sum()
            share_wt = sub["share_wt_pct"].sum()
            group_rows.append(
                {
                    "group": group,
                    "n_features": len(sub),
                    "share_std_pct": share_std,
                    "share_wt_pct": share_wt,
                    "delta_share_pct": share_wt - share_std,
                }
            )

        result["importance_shift"] = shift_with_group
        result["group_shares"] = pd.DataFrame(group_rows)
    except Exception as exc:
        logger.warning("Grouped importance-share analysis failed: %s", exc)


def plot_importance_reallocation(
    shift_df: pd.DataFrame,
    out_path: Path,
    model: str,
    importance_source: str = "native",
    top_n: int | None = 30,
) -> None:
    """Plot the largest feature-importance share shifts when available."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from src.analysis.motivation import _set_academic_style

    _set_academic_style()
    metric_col = "delta_share_pct" if "delta_share_pct" in shift_df else "delta"
    df = shift_df.reindex(
        shift_df[metric_col].abs().sort_values(ascending=False).index
    )
    if top_n is not None:
        df = df.head(top_n)
    df = df.copy()
    df = df.reindex(df[metric_col].abs().sort_values(ascending=True).index)

    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.22)))
    group_palette = {
        "Q1_only": "#D55E00",
        "Q5_only": "#0072B2",
        "both": "#009E73",
        "neither": "#999999",
    }
    if "group" in df.columns:
        colors = df["group"].map(group_palette).fillna("#999999")
    else:
        colors = ["steelblue" if delta > 0 else "coral" for delta in df[metric_col]]
    ax.barh(df["feature"], df[metric_col], color=colors)
    ax.axvline(0, color="black", lw=0.5)
    source_label = "SHAP" if importance_source == "shap" else "native importance"
    if metric_col == "delta_share_pct":
        ax.set_xlabel(
            rf"$\Delta$ {source_label} share (weighted $-$ standard, pp)"
        )
        ax.set_title(f"Importance-share reallocation ({model}, {source_label})")
    else:
        ax.set_xlabel(rf"$\Delta$ {source_label} (weighted $-$ standard)")
        ax.set_title(f"Importance reallocation ({model}, {source_label})")
    if "group" in df.columns:
        present_groups = [g for g in group_palette if g in set(df["group"])]
        handles = [
            Patch(facecolor=group_palette[group], label=group.replace("_", " "))
            for group in present_groups
        ]
        ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
