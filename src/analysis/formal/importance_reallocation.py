"""Feature-importance reallocation analysis for formal experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def analyze_importance_reallocation(
    importance_standard: pd.DataFrame,
    importance_weighted: pd.DataFrame,
    interaction_csv_path: Path | None = None,
    quintile_raw_csv_path: Path | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Measure how weighted training reallocates feature importance."""
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
        delta = mean_wt - mean_std
        diffs = wt_values[valid] - std_values[valid]
        se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
        t_stat = np.mean(diffs) / se if se > 0 else 0.0
        p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=len(diffs) - 1))

        rows.append(
            {
                "feature": feature,
                "mean_shap_std": mean_std,
                "mean_shap_wt": mean_wt,
                "delta": delta,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_windows": int(valid.sum()),
            }
        )

    if not rows:
        logger.warning(
            "SHAP CSVs have no feature columns; skipping importance reallocation."
        )
        return {
            "importance_shift": pd.DataFrame(
                columns=[
                    "feature",
                    "mean_shap_std",
                    "mean_shap_wt",
                    "delta",
                    "t_stat",
                    "p_value",
                    "n_windows",
                ]
            )
        }

    shift_df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    result: dict[str, pd.DataFrame | dict] = {"importance_shift": shift_df}

    if interaction_csv_path is not None and Path(interaction_csv_path).exists():
        _add_delta_gamma_regression(result, shift_df, Path(interaction_csv_path))

    if quintile_raw_csv_path is not None and Path(quintile_raw_csv_path).exists():
        _add_grouped_importance_shares(result, shift_df, Path(quintile_raw_csv_path))

    return result


def _add_delta_gamma_regression(
    result: dict[str, pd.DataFrame | dict],
    shift_df: pd.DataFrame,
    interaction_csv_path: Path,
) -> None:
    """Regress feature-importance changes on Step 2 liquid-minus-illiquid slopes."""
    try:
        gamma_df = pd.read_csv(interaction_csv_path)
        merged = shift_df.merge(
            gamma_df[["feature", "gamma_bar"]],
            on="feature",
            how="inner",
        )
        if len(merged) < 3:
            return

        x = merged["gamma_bar"].values
        y = merged["delta"].values
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
        result["delta_vs_gamma_data"] = merged[
            ["feature", "gamma_bar", "delta", "t_stat"]
        ]
    except Exception as exc:
        logger.warning("Delta vs gamma regression failed: %s", exc)


def _add_grouped_importance_shares(
    result: dict[str, pd.DataFrame | dict],
    shift_df: pd.DataFrame,
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

        shift_with_group = shift_df.merge(
            q_df[["feature", "group"]],
            on="feature",
            how="inner",
        )
        total_std = shift_with_group["mean_shap_std"].sum()
        total_wt = shift_with_group["mean_shap_wt"].sum()

        group_rows = []
        for group in ["Q1_only", "Q5_only", "both", "neither"]:
            sub = shift_with_group[shift_with_group["group"] == group]
            if len(sub) == 0:
                continue
            share_std = sub["mean_shap_std"].sum() / total_std if total_std > 0 else np.nan
            share_wt = sub["mean_shap_wt"].sum() / total_wt if total_wt > 0 else np.nan
            group_rows.append(
                {
                    "group": group,
                    "n_features": len(sub),
                    "share_std_pct": share_std * 100,
                    "share_wt_pct": share_wt * 100,
                    "delta_share_pct": (share_wt - share_std) * 100,
                }
            )

        result["group_shares"] = pd.DataFrame(group_rows)
    except Exception as exc:
        logger.warning("Grouped importance-share analysis failed: %s", exc)


def plot_importance_reallocation(
    shift_df: pd.DataFrame,
    out_path: Path,
    model: str,
    top_n: int = 30,
) -> None:
    """Plot the largest feature-importance shifts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = shift_df.head(top_n).copy()
    df = df.reindex(df["delta"].abs().sort_values(ascending=True).index)

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.25)))
    colors = ["steelblue" if delta > 0 else "coral" for delta in df["delta"]]
    ax.barh(df["feature"], df["delta"], color=colors)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Delta SHAP (weighted - standard)")
    ax.set_title(f"Importance reallocation ({model})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)
