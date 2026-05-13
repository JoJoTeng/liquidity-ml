"""
Feature Importance & SHAP Analysis
====================================
Analysis module for Hypothesis 2: does liquidity-weighted training
shift feature importance toward tradable factors?

Provides:
  1. SHAP computation dispatcher (model-agnostic wrapper)
  2. Loading SHAP/importance data from experiment output
  3. Aggregation across rolling windows
  4. H2 hypothesis testing (paired t-tests, ratio tests)
  5. Comparison visualization (standard vs weighted)
  6. Per-model and cross-model analysis

Usage (after running scripts/20_formal_run_experiment.py):
    from src.analysis.feature_importance import (
        load_importance, test_h2_group_shift, cross_model_h2_summary,
    )

    data = load_importance("xgboost", weight_spec="dolvol", importance_type="shap")
    h2 = test_h2_group_shift(data["standard"], data["weighted"])
    summary = cross_model_h2_summary()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_output_dir
from src.data.loader import (
    ILLIQUIDITY_FEATURES,
    ILLIQUIDITY_FEATURES_EXT,
    TRADABLE_FEATURES,
    TRADABLE_FEATURES_EXT,
)
from src.evaluation.statistics import paired_ttest

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. SHAP Computation (dispatcher / fallback)
# ══════════════════════════════════════════════════════════════


def compute_shap_values(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    X_background: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Compute SHAP values for any model type.

    Dispatches to ``model.get_shap_values()`` if available,
    otherwise uses a fallback based on *model_name*.

    Parameters
    ----------
    model : Fitted model instance (BaseReturnPredictor subclass).
    model_name : ``"elastic_net"``, ``"xgboost"``, or ``"neural_network"``.
    X_test : (N, P) observations to explain.
    X_background : (M, P) background data (required for NN).
    feature_names : Feature names for columns.
    config : SHAP config dict.

    Returns
    -------
    pd.DataFrame of shape (N, P) with SHAP values.
    """
    if hasattr(model, "get_shap_values"):
        return model.get_shap_values(
            X_test, X_background, feature_names, config
        )

    # Fallback dispatcher
    import shap

    cfg = config or {}
    cols = feature_names or [f"f{i}" for i in range(X_test.shape[1])]

    if model_name == "xgboost":
        inner = getattr(model, "model", model)
        explainer = shap.TreeExplainer(inner)
        sv = explainer.shap_values(X_test)
    elif model_name == "neural_network":
        inner = getattr(model, "model", model)
        n_bg = cfg.get("background_samples", 100)
        if X_background is None:
            raise ValueError("X_background required for neural_network")
        if len(X_background) > n_bg:
            idx = np.random.RandomState(42).choice(
                len(X_background), n_bg, replace=False
            )
            X_bg = X_background[idx]
        else:
            X_bg = X_background
        explainer = shap.DeepExplainer(inner, X_bg)
        sv = explainer.shap_values(X_test)
        if isinstance(sv, list):
            sv = sv[0]
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return pd.DataFrame(sv, columns=cols)


def compute_mean_abs_shap(shap_df: pd.DataFrame) -> pd.Series:
    """Aggregate raw SHAP values to mean |SHAP| per feature.

    Parameters
    ----------
    shap_df : (N, P) raw SHAP values from one window.

    Returns
    -------
    pd.Series of length P, indexed by feature name.
    """
    return shap_df.abs().mean(axis=0)


# ══════════════════════════════════════════════════════════════
# 2. Loading from Disk
# ══════════════════════════════════════════════════════════════


def load_importance(
    model_name: str,
    weight_spec: str = "dolvol",
    importance_type: str = "shap",
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load per-window importance data from experiment output.

    Parameters
    ----------
    model_name : ``"elastic_net"``, ``"xgboost"``, or ``"neural_network"``.
    weight_spec : Weighted formal-output folder, e.g. ``"dolvol"``,
        ``"softmax_rank_lam2"``, ``"tc_500m"``, or ``"tc_rank_lam3_500m"``.
    importance_type : ``"shap"`` or ``"native"``/``"gain"``.
    output_dir : Override experiment root. Defaults to
        ``outputs/formalanalysis/experiment``.

    Returns
    -------
    dict with keys ``"standard"`` and ``"weighted"``, each a DataFrame
    with rows = months (yyyymm index), columns = features.
    """
    if output_dir is None:
        base = get_output_dir() / "formalanalysis" / "experiment" / model_name
    else:
        base = Path(output_dir) / model_name

    if importance_type == "shap":
        filename = "importance_shap.csv"
    elif importance_type in {"native", "gain"}:
        filename = "importance_native.csv"
    else:
        raise ValueError("importance_type must be 'shap', 'native', or 'gain'")

    std_file = base / "standard" / filename
    wt_file = base / weight_spec / filename

    std_df = _read_importance_file(std_file)
    wt_df = _read_importance_file(wt_file)

    return {"standard": std_df, "weighted": wt_df}


def _read_importance_file(path: Path) -> pd.DataFrame:
    """Read current formal importance CSVs and index by yyyymm when present."""
    df = pd.read_csv(path)
    if "yyyymm" in df.columns:
        df = df.set_index("yyyymm")
    return df


# ══════════════════════════════════════════════════════════════
# 3. Aggregation
# ══════════════════════════════════════════════════════════════


def aggregate_importance(
    importance_df: pd.DataFrame,
    method: str = "mean",
) -> pd.Series:
    """Aggregate per-window importance across all windows.

    Parameters
    ----------
    importance_df : (n_windows, n_features) importance values.
    method : ``"mean"`` or ``"median"``.

    Returns
    -------
    pd.Series of aggregated importance per feature, sorted descending.
    """
    if method == "mean":
        result = importance_df.mean(axis=0)
    elif method == "median":
        result = importance_df.median(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}")
    return result.sort_values(ascending=False)


def compute_group_importance(
    importance_df: pd.DataFrame,
    group: str = "tradable",
    extended: bool = False,
) -> pd.Series:
    """Sum importance within a feature group per window.

    Parameters
    ----------
    importance_df : (n_windows, n_features).
    group : ``"tradable"`` or ``"illiquidity"``.
    extended : Use the empirical Step 2 groups for importance reallocation:
        ``"illiquidity"`` means Q1-only predictors, and ``"tradable"``
        means Q5-only plus both-Q1-and-Q5 predictors.

    Returns
    -------
    pd.Series of length n_windows — total group importance per window.
    """
    if group == "tradable":
        features = TRADABLE_FEATURES_EXT if extended else TRADABLE_FEATURES
    elif group == "illiquidity":
        features = ILLIQUIDITY_FEATURES_EXT if extended else ILLIQUIDITY_FEATURES
    else:
        raise ValueError(f"Unknown group: {group!r}")

    present = [f for f in features if f in importance_df.columns]
    if not present:
        logger.warning("No features from group %r found in data", group)
        return pd.Series(0.0, index=importance_df.index, dtype=float)

    return importance_df[present].sum(axis=1)


def compute_importance_ratio(
    importance_df: pd.DataFrame,
    extended: bool = False,
) -> pd.Series:
    """Compute tradable / (tradable + illiquidity) ratio per window.

    Higher ratio = model relies more on tradable features.

    Parameters
    ----------
    importance_df : (n_windows, n_features).
    extended : Use extended feature lists.

    Returns
    -------
    pd.Series of length n_windows with ratio values in [0, 1].
    """
    tradable_sum = compute_group_importance(importance_df, "tradable", extended)
    illiq_sum = compute_group_importance(importance_df, "illiquidity", extended)
    total = tradable_sum + illiq_sum
    return (tradable_sum / total).fillna(0.5)


# ══════════════════════════════════════════════════════════════
# 4. H2 Hypothesis Testing
# ══════════════════════════════════════════════════════════════


def test_h2_group_shift(
    importance_std: pd.DataFrame,
    importance_wt: pd.DataFrame,
    extended: bool = False,
) -> dict[str, Any]:
    """Test H2: weighted training increases tradable importance share.

    Computes tradable / (tradable + illiquidity) ratio for each window,
    then tests whether weighted > standard using a paired t-test.

    Parameters
    ----------
    importance_std : (n_windows, n_features) for standard training.
    importance_wt : (n_windows, n_features) for weighted training.
    extended : Use extended feature groups.

    Returns
    -------
    dict with:
        ratio_std_mean, ratio_wt_mean : average ratios
        ratio_diff : mean difference (wt - std)
        ratio_test : output of paired_ttest (t_stat, p_value, etc.)
        tradable_test : paired t-test on total tradable importance
        illiquidity_test : paired t-test on total illiquidity importance
    """
    ratio_std = compute_importance_ratio(importance_std, extended)
    ratio_wt = compute_importance_ratio(importance_wt, extended)

    ratio_test = paired_ttest(ratio_wt, ratio_std, alternative="greater")

    # Also test individual group sums
    trad_std = compute_group_importance(importance_std, "tradable", extended)
    trad_wt = compute_group_importance(importance_wt, "tradable", extended)
    tradable_test = paired_ttest(trad_wt, trad_std, alternative="greater")

    illiq_std = compute_group_importance(importance_std, "illiquidity", extended)
    illiq_wt = compute_group_importance(importance_wt, "illiquidity", extended)
    illiq_test = paired_ttest(illiq_std, illiq_wt, alternative="greater")

    return {
        "ratio_std_mean": float(ratio_std.mean()),
        "ratio_wt_mean": float(ratio_wt.mean()),
        "ratio_diff": float((ratio_wt - ratio_std).mean()),
        "ratio_test": ratio_test,
        "tradable_test": tradable_test,
        "illiquidity_test": illiq_test,
        "n_windows": len(ratio_std),
    }


def test_h2_per_feature(
    importance_std: pd.DataFrame,
    importance_wt: pd.DataFrame,
) -> pd.DataFrame:
    """Per-feature paired t-test: weighted vs standard importance.

    Parameters
    ----------
    importance_std, importance_wt : (n_windows, n_features).

    Returns
    -------
    DataFrame with one row per feature and columns:
        feature, mean_std, mean_wt, mean_diff, t_stat, p_value,
        significant, category
    """
    rows = []
    for feat in importance_std.columns:
        if feat not in importance_wt.columns:
            continue
        test = paired_ttest(
            importance_wt[feat], importance_std[feat], alternative="two-sided"
        )
        rows.append({
            "feature": feat,
            "mean_std": float(importance_std[feat].mean()),
            "mean_wt": float(importance_wt[feat].mean()),
            "mean_diff": test["mean_diff"],
            "t_stat": test["t_stat"],
            "p_value": test["p_value"],
            "significant": test["p_value"] < 0.05,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["category"] = result["feature"].apply(_categorize_feature)
    return result.sort_values("p_value").reset_index(drop=True)


def _categorize_feature(feature: str) -> str:
    """Categorize a feature using empirical Step 2 groups."""
    if feature in set(TRADABLE_FEATURES_EXT):
        return "tradable"
    if feature in set(ILLIQUIDITY_FEATURES_EXT):
        return "illiquidity"
    return "other"


# ══════════════════════════════════════════════════════════════
# 5. Cross-Model Analysis
# ══════════════════════════════════════════════════════════════


def cross_model_h2_summary(
    model_names: list[str] | None = None,
    weight_spec: str = "dolvol",
    importance_type: str = "shap",
    output_dir: str | Path | None = None,
    extended: bool = False,
) -> pd.DataFrame:
    """H2 test results across all models.

    Parameters
    ----------
    model_names : Models to include. Default: all active models.
    weight_spec : Weighted formal-output folder to compare against standard.
    importance_type : ``"shap"`` or ``"gain"``.
    output_dir : Override output directory.
    extended : Use extended feature groups.

    Returns
    -------
    DataFrame with one row per model: ratio_std, ratio_wt,
    ratio_diff, t_stat, p_value, n_windows.
    """
    if model_names is None:
        model_names = ["elastic_net", "xgboost", "neural_network"]

    rows = []
    for name in model_names:
        try:
            data = load_importance(
                name,
                weight_spec=weight_spec,
                importance_type=importance_type,
                output_dir=output_dir,
            )
            h2 = test_h2_group_shift(
                data["standard"], data["weighted"], extended=extended
            )
            rows.append({
                "model": name,
                "ratio_std": h2["ratio_std_mean"],
                "ratio_wt": h2["ratio_wt_mean"],
                "ratio_diff": h2["ratio_diff"],
                "t_stat": h2["ratio_test"]["t_stat"],
                "p_value": h2["ratio_test"]["p_value"],
                "n_windows": h2["n_windows"],
            })
        except FileNotFoundError:
            logger.warning("No %s data for model %s", importance_type, name)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 6. Visualization
# ══════════════════════════════════════════════════════════════


def plot_importance_comparison(
    importance_std: pd.DataFrame,
    importance_wt: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance: Standard vs Weighted Training",
    save_path: str | Path | None = None,
) -> Any:
    """Side-by-side horizontal bar chart of mean importance, top N features.

    Parameters
    ----------
    importance_std, importance_wt : (n_windows, n_features).
    top_n : Number of top features to show.
    title : Plot title.
    save_path : If provided, save figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    mean_std = aggregate_importance(importance_std)
    mean_wt = aggregate_importance(importance_wt)

    # Union of top features from both
    top_std = set(mean_std.head(top_n).index)
    top_wt = set(mean_wt.head(top_n).index)
    top_features = sorted(
        top_std | top_wt, key=lambda f: -mean_std.get(f, 0)
    )[:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(top_features))
    width = 0.35

    vals_std = [mean_std.get(f, 0) for f in top_features]
    vals_wt = [mean_wt.get(f, 0) for f in top_features]

    colors_std = [
        _category_color(_categorize_feature(f), alpha=0.6)
        for f in top_features
    ]
    colors_wt = [
        _category_color(_categorize_feature(f), alpha=0.9)
        for f in top_features
    ]

    ax.barh(x + width, vals_std, width, label="Standard", color=colors_std)
    ax.barh(x, vals_wt, width, label="Weighted", color=colors_wt)

    ax.set_yticks(x + width / 2)
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_importance_over_time(
    importance_std: pd.DataFrame,
    importance_wt: pd.DataFrame,
    group: str = "tradable",
    extended: bool = False,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """Time series of group importance (tradable or illiquidity).

    Parameters
    ----------
    importance_std, importance_wt : (n_windows, n_features).
    group : ``"tradable"`` or ``"illiquidity"``.
    extended : Use extended feature groups.
    title : Plot title.
    save_path : If provided, save figure.

    Returns
    -------
    matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    group_std = compute_group_importance(importance_std, group, extended)
    group_wt = compute_group_importance(importance_wt, group, extended)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(group_std.index, group_std.values, label="Standard", alpha=0.8)
    ax.plot(group_wt.index, group_wt.values, label="Weighted", alpha=0.8)
    ax.set_xlabel("Test Month (yyyymm)")
    ax.set_ylabel(f"Total {group.title()} Importance")
    ax.set_title(title or f"{group.title()} Feature Importance Over Time")
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ratio_over_time(
    importance_std: pd.DataFrame,
    importance_wt: pd.DataFrame,
    extended: bool = False,
    title: str = "Tradable / (Tradable + Illiquidity) Ratio Over Time",
    save_path: str | Path | None = None,
) -> Any:
    """Time series of the tradable importance ratio.

    Parameters
    ----------
    importance_std, importance_wt : (n_windows, n_features).
    extended : Use extended feature groups.
    title : Plot title.
    save_path : If provided, save figure.

    Returns
    -------
    matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    ratio_std = compute_importance_ratio(importance_std, extended)
    ratio_wt = compute_importance_ratio(importance_wt, extended)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ratio_std.index, ratio_std.values, label="Standard", alpha=0.8)
    ax.plot(ratio_wt.index, ratio_wt.values, label="Weighted", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Neutral")
    ax.set_xlabel("Test Month (yyyymm)")
    ax.set_ylabel("Tradable Ratio")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _category_color(category: str, alpha: float = 1.0) -> tuple:
    """Return RGBA color for a feature category."""
    colors = {
        "tradable": (0.2, 0.6, 0.2, alpha),
        "illiquidity": (0.8, 0.2, 0.2, alpha),
        "other": (0.5, 0.5, 0.5, alpha),
    }
    return colors.get(category, colors["other"])
