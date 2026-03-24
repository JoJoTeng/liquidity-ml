"""
Distributional Divergence Analysis (Step 1)
=============================================
Establishes that P_train(x) != P_deploy(x) across a broad range of
stock characteristics, not just mechanical liquidity dimensions.

Produces Outputs 1.1--1.5 from the motivation document:
  1.1  Ranked bar chart of marginal divergences
  1.2  Divergence summary by economic category
  1.3  Fama-MacBeth weight regression (log w~ on characteristics)
  1.4  Density comparison plots (equal-wt vs volume-wt)
  1.5  Weight distribution histogram
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.config import load_config
from src.data.loader import NON_FEATURE_COLS
from src.evaluation.statistics import newey_west_tstat
from src.weighting import compute_weights

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

# 15 focal characteristics from Table 1 of the motivation document.
# Keys = CZ Acronym (or CRSP-derived name), values = (category, rationale).
FOCAL_CHARACTERISTICS = {
    "STreversal": ("Momentum", "Microstructure / limits to arbitrage"),
    "Mom12m": ("Momentum", "Core anomaly"),
    "BM": ("Value", "Core anomaly"),
    "EP": ("Value", "Fundamentals-based predictor"),
    "GP": ("Profitability", "Quality factor"),
    "AssetGrowth": ("Investment", "Investment anomaly"),
    "RoE": ("Profitability", "Fundamentals-based"),
    "Accruals": ("Quality", "Earnings quality / mispricing"),
    "IdioVol3F": ("Risk", "Illiquidity-correlated risk"),
    "Beta": ("Risk", "Systematic risk"),
    "Illiquidity": ("Liquidity", "Amihud illiquidity"),
    "zerotrade12M": ("Liquidity", "Zero-trading days"),
    "Size": ("Other", "Log market capitalization"),
    "ChNAnalyst": ("Other", "Analyst coverage"),
    "BidAskSpread": ("Liquidity", "Transaction cost proxy"),
}

# 6 focal characteristics for density plots (Output 1.4)
DENSITY_PLOT_FEATURES = [
    "Illiquidity",
    "IdioVol3F",
    "Mom12m",
    "BM",
    "ChNAnalyst",
    "STreversal",
]

# Mapping from CZ fine categories to 8 broad categories (for Table 2)
# Mapping from CZ Cat.Economic (fine) to 8 broad categories for Table 2.
# Follows the document's 8-category grouping (Section 3.3, Table 2).
CZ_TO_BROAD = {
    # Momentum
    "momentum": "Momentum",
    "short-term reversal": "Momentum",
    "long term reversal": "Momentum",
    "lead lag": "Momentum",
    # Value
    "valuation": "Value",
    # Profitability
    "profitability": "Profitability",
    "profitability alt": "Profitability",
    "earnings growth": "Profitability",
    "earnings forecast": "Profitability",
    "earnings event": "Profitability",
    "sales growth": "Profitability",
    # Investment
    "investment": "Investment",
    "investment alt": "Investment",
    "investment growth": "Investment",
    "external financing": "Investment",
    "asset composition": "Investment",
    # Liquidity
    "liquidity": "Liquidity",
    "volume": "Liquidity",
    "informed trading": "Liquidity",
    "turnover": "Liquidity",
    # Risk
    "risk": "Risk",
    "volatility": "Risk",
    "market risk": "Risk",
    "default risk": "Risk",
    "optionrisk": "Risk",
    "cash flow risk": "Risk",
    # Quality
    "accruals": "Quality",
    "composite accounting": "Quality",
    "leverage": "Quality",
    "R&D": "Quality",
    "payout indicator": "Quality",
    # Other — everything not listed above
    # (other, size, short sale constraints, ownership, recommendation, info proxy)
}


# ═════════════════════════════════════════════════════════════
# Data Preparation
# ═════════════════════════════════════════════════════════════


def load_signaldoc(path: str | Path = "data/SignalDoc.csv") -> pd.DataFrame:
    """Load the CZ Signal Documentation file."""
    doc = pd.read_csv(path, encoding="latin-1")
    logger.info("SignalDoc: %d rows, columns: %s", len(doc), doc.columns.tolist())
    return doc


def get_motivation_features(
    signaldoc: pd.DataFrame,
    panel: pd.DataFrame,
    exclude_binary_threshold: int = 5,
) -> list[str]:
    """Select Clear + Likely predictors, excluding binary and circular features.

    Parameters
    ----------
    signaldoc : SignalDoc.csv DataFrame with columns Acronym, Cat.Predictor, etc.
    panel : The full data panel (to check which columns exist and detect binary).
    exclude_binary_threshold : Features with <= this many unique values
        in a typical month are excluded as binary/discrete.

    Returns
    -------
    list[str] : Sorted list of usable feature names.
    """
    # Filter to Clear Predictors only
    # SignalDoc column "Predictability in OP": 1_clear (165), 2_likely (47)
    mask = (signaldoc["Cat.Signal"] == "Predictor") & (
        signaldoc["Predictability in OP"] == "1_clear"
    )
    candidates = set(signaldoc.loc[mask, "Acronym"].tolist())
    logger.info("SignalDoc: %d Clear Predictor signals", len(candidates))

    # Exclude discrete/binary features (Cat.Form == "discrete")
    # They produce degenerate rank distributions and distort regressions.
    discrete = set(
        signaldoc.loc[signaldoc["Cat.Form"] == "discrete", "Acronym"].tolist()
    )
    n_removed = len(candidates & discrete)
    candidates -= discrete
    if n_removed:
        logger.info("Excluded %d discrete features", n_removed)

    # Add CRSP-derived features that aren't in CZ file but are valid predictors
    crsp_features = {"STreversal", "Size", "Price"}
    candidates |= crsp_features

    # Always include focal characteristics (some are "Likely" or "discrete"
    # but required for Table 1 analyses)
    focal = set(FOCAL_CHARACTERISTICS.keys())
    focal_added = focal - candidates
    if focal_added:
        logger.info("Adding %d focal characteristics: %s", len(focal_added), sorted(focal_added))
    candidates |= focal

    # Remove features not in the panel
    panel_cols = set(panel.columns)
    available = candidates & panel_cols
    missing = candidates - panel_cols
    if missing:
        logger.warning("Features in SignalDoc but not in panel: %s", sorted(missing))

    # Remove NON_FEATURE_COLS that might have slipped in
    available -= NON_FEATURE_COLS

    # Discrete features excluded via Cat.Form above. Focal chars added back.

    result = sorted(available)
    logger.info("Final motivation feature set: %d features", len(result))
    return result


def load_feature_categories(
    path: str | Path = "config/feature_categories.json",
) -> dict:
    """Load feature → category mapping from JSON."""
    with open(path) as f:
        return json.load(f)


def build_feature_categories(
    signaldoc: pd.DataFrame,
    output_path: str | Path = "config/feature_categories.json",
) -> dict:
    """Build and save feature category mapping from SignalDoc.csv.

    Returns dict with keys: 'fine', 'broad', 'cz_to_broad'.
    """
    # Fine-grained: CZ Acronym → Cat.Economic
    fine = dict(zip(signaldoc["Acronym"], signaldoc["Cat.Economic"]))

    # Add CRSP-derived features (not in SignalDoc)
    fine["STreversal"] = "short-term reversal"
    fine["Size"] = "size"
    fine["Price"] = "liquidity"

    # Broad: map fine categories to 8 broad categories
    broad = {}
    for acronym, cat in fine.items():
        broad[acronym] = CZ_TO_BROAD.get(cat, "Other")

    result = {
        "fine": fine,
        "broad": broad,
        "cz_to_broad": CZ_TO_BROAD,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved feature categories to %s", output_path)

    return result


def rank_transform_01(
    panel: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Cross-sectional percentile rank within each month → [0, 1].

    Following Gu et al. (2020). NaN values remain NaN.
    """
    out = panel.copy()
    for col in feature_cols:
        out[col] = out.groupby("yyyymm")[col].rank(pct=True)
    return out


def compute_implementability_weights(
    panel: pd.DataFrame,
    liq_col: str = "liq_dvol_21d",
) -> pd.Series:
    """Compute w̃_it = liquidity_it / mean(liquidity_t) per month.

    Parameters
    ----------
    liq_col : Raw liquidity column to use. Default is 'liq_dvol_21d'
        (21-day trailing avg dollar volume). Alternatives:
        'liq_me_raw' (market cap), 'raw_Illiquidity' (Amihud),
        'raw_BidAskSpread' (bid-ask spread).
        For Amihud/spread, higher = less liquid, so invert before normalizing.
    """
    invert = liq_col in ("raw_Illiquidity", "raw_BidAskSpread")

    results = []
    for _, group in panel.groupby("yyyymm"):
        vals = group[liq_col].copy()
        median_val = vals.median()
        if pd.isna(median_val):
            median_val = 1.0
        vals = vals.fillna(median_val).clip(lower=1e-8)

        if invert:
            vals = 1.0 / vals  # higher illiquidity → lower weight

        mean_w = vals.mean()
        if mean_w == 0 or pd.isna(mean_w):
            results.append(pd.Series(1.0, index=group.index))
        else:
            results.append(vals / mean_w)

    return pd.concat(results)


def assign_nyse_quintiles(
    panel: pd.DataFrame,
    sort_col: str,
    exchcd_col: str = "exchcd",
    ascending: bool = True,
) -> pd.Series:
    """Assign quintiles using NYSE breakpoints.

    Parameters
    ----------
    sort_col : Column to sort on (e.g., 'liq_dvol_21d', 'Illiquidity').
    ascending : If True, higher values = more liquid (dvol, me_raw).
        If False, higher values = more illiquid (Amihud, spread).
        Q1 = most illiquid, Q5 = most liquid regardless of direction.
    """

    def _per_month(group):
        vals = group[sort_col]
        nyse = group[group[exchcd_col] == 1][sort_col].dropna()

        if len(nyse) < 20:
            return pd.Series(np.nan, index=group.index, dtype="float64")

        breakpoints = nyse.quantile([0.2, 0.4, 0.6, 0.8]).values
        raw_quintile = np.digitize(vals.values, breakpoints) + 1  # 1-5
        raw_quintile = np.where(vals.isna(), np.nan, raw_quintile.astype(float))

        if not ascending:
            # Reverse: high value = illiquid = Q1
            raw_quintile = np.where(
                np.isnan(raw_quintile), np.nan, 6 - raw_quintile
            )

        return pd.Series(raw_quintile, index=group.index, dtype="float64")

    return panel.groupby("yyyymm", group_keys=False).apply(_per_month)


# ═════════════════════════════════════════════════════════════
# Output 1.1 + 1.2: Marginal Divergence
# ═════════════════════════════════════════════════════════════


def compute_marginal_divergence(
    panel: pd.DataFrame,
    feature_cols: list[str],
    w_col: str = "w_tilde",
) -> pd.DataFrame:
    """Compute monthly divergence d_jt = x̄_deploy - x̄_train for each feature.

    Returns DataFrame with rows=yyyymm, columns=features, values=d_jt.
    """
    months = sorted(panel["yyyymm"].unique())
    results = []

    for m in months:
        mdf = panel[panel["yyyymm"] == m]
        w = mdf[w_col].values
        row = {"yyyymm": m}

        for col in feature_cols:
            vals = mdf[col].values
            valid = ~np.isnan(vals) & ~np.isnan(w)
            if valid.sum() < 50:
                row[col] = np.nan
                continue

            v = vals[valid]
            wv = w[valid]
            x_bar_train = np.mean(v)
            x_bar_deploy = np.average(v, weights=wv)
            row[col] = x_bar_deploy - x_bar_train

        results.append(row)

    return pd.DataFrame(results).set_index("yyyymm")


def compute_divergence_stats(divergence_df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-series mean d̄_j and Newey-West t-statistics per feature.

    Returns DataFrame with columns:
        feature, d_bar, std_err, t_stat, p_value, n_obs, abs_d_bar
    """
    results = []
    for col in divergence_df.columns:
        series = divergence_df[col].dropna()
        if len(series) < 12:
            continue
        nw = newey_west_tstat(series.values, lags=6)
        results.append(
            {
                "feature": col,
                "d_bar": nw["mean"],
                "std_err": nw["std_err"],
                "t_stat": nw["t_stat"],
                "p_value": nw["p_value"],
                "n_obs": nw["n_obs"],
                "abs_d_bar": abs(nw["mean"]),
            }
        )

    df = pd.DataFrame(results).sort_values("abs_d_bar", ascending=False)
    return df.reset_index(drop=True)


def summarize_divergence_by_category(
    stats_df: pd.DataFrame,
    broad_categories: dict[str, str],
) -> pd.DataFrame:
    """Aggregate divergence stats by broad economic category.

    Parameters
    ----------
    stats_df : Output of compute_divergence_stats().
    broad_categories : dict mapping feature name → broad category.
    """
    df = stats_df.copy()
    df["category"] = df["feature"].map(broad_categories).fillna("Other")
    df["significant"] = df["t_stat"].abs() > 2.0

    summary = (
        df.groupby("category")
        .agg(
            avg_abs_d_bar=("abs_d_bar", "mean"),
            n_significant=("significant", "sum"),
            n_features=("feature", "count"),
        )
        .sort_values("avg_abs_d_bar", ascending=False)
        .reset_index()
    )

    # Clean up types and formatting
    summary["n_significant"] = summary["n_significant"].astype(int)
    summary["avg_abs_d_bar"] = summary["avg_abs_d_bar"].round(4)

    # Rename columns to match Table 2 template
    summary = summary.rename(columns={
        "category": "Category",
        "avg_abs_d_bar": "Avg. |d_bar|",
        "n_significant": "# Significant (|t| > 2)",
        "n_features": "# Characteristics",
    })

    return summary


# ═════════════════════════════════════════════════════════════
# Output 1.3: Fama-MacBeth Weight Regression
# ═════════════════════════════════════════════════════════════


def fama_macbeth_weight_regression(
    panel: pd.DataFrame,
    feature_cols: list[str],
    w_col: str = "w_tilde",
    min_stocks: int = 200,
    min_feature_coverage: float = 0.5,
) -> dict:
    """Monthly OLS of log(w̃) on all rank-transformed characteristics.

    Fama-MacBeth averaging of coefficients with Newey-West t-stats.

    Returns
    -------
    dict with keys:
        r2_mean : float — average R² across months
        r2_series : list[float] — monthly R² values
        coef_stats : pd.DataFrame — FM coefficients ranked by |δ̄_j|
        n_months : int
    """
    months = sorted(panel["yyyymm"].unique())
    coefs_list = []
    r2_list = []

    for m in months:
        mdf = panel[panel["yyyymm"] == m]

        # Dependent variable: log(w̃), floor to avoid log(0)
        w = mdf[w_col].values
        valid_w = ~np.isnan(w) & (w > 0)
        if valid_w.sum() < min_stocks:
            continue

        y = np.log(np.clip(w[valid_w], 1e-8, None))

        # Features: drop columns with poor coverage this month
        X_df = mdf.loc[valid_w, feature_cols].copy()
        good_cols = X_df.columns[X_df.notna().mean() > min_feature_coverage]
        good_cols = good_cols.tolist()
        if len(good_cols) < 10:
            continue

        X_df = X_df[good_cols].fillna(0.0)
        X = np.column_stack([np.ones(len(X_df)), X_df.values])

        # OLS via numpy
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # R-squared
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        r2_list.append({"yyyymm": m, "R2": r2, "N_stocks": int(valid_w.sum()),
                        "N_features": len(good_cols)})

        # Store coefficients (skip intercept at index 0)
        coef_row = {"yyyymm": m}
        for i, col in enumerate(good_cols):
            coef_row[col] = beta[i + 1]
        coefs_list.append(coef_row)

    if not coefs_list:
        logger.warning("Fama-MacBeth: no valid months")
        return {
            "r2_mean": np.nan,
            "r2_median": np.nan,
            "r2_df": pd.DataFrame(columns=["yyyymm", "R2", "N_stocks", "N_features"]),
            "coef_stats": pd.DataFrame(),
            "n_months": 0,
        }

    coefs_df = pd.DataFrame(coefs_list).set_index("yyyymm")
    r2_df = pd.DataFrame(r2_list)
    r2_vals = r2_df["R2"].values

    # FM averaging with NW t-stats per feature
    coef_results = []
    for col in feature_cols:
        if col not in coefs_df.columns:
            continue
        series = coefs_df[col].dropna()
        if len(series) < 12:
            continue
        nw = newey_west_tstat(series.values, lags=6)
        coef_results.append(
            {
                "feature": col,
                "delta_bar": nw["mean"],
                "std_err": nw["std_err"],
                "t_stat": nw["t_stat"],
                "p_value": nw["p_value"],
                "n_months": nw["n_obs"],
                "abs_delta_bar": abs(nw["mean"]),
            }
        )

    coef_stats = (
        pd.DataFrame(coef_results)
        .sort_values("abs_delta_bar", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "r2_mean": float(np.mean(r2_vals)),
        "r2_median": float(np.median(r2_vals)),
        "r2_df": r2_df,
        "coef_stats": coef_stats,
        "n_months": len(r2_df),
    }


# ═════════════════════════════════════════════════════════════
# Plotting Functions
# ═════════════════════════════════════════════════════════════


def plot_divergence_bar_chart(
    stats_df: pd.DataFrame,
    broad_categories: dict[str, str],
    output_path: str | Path,
    top_n: int | None = None,
) -> None:
    """Horizontal bar chart of d̄_j sorted by |d̄_j|, color-coded by category.

    Significance marking:
      - |t| > 2: full category color + dark edge → visually prominent
      - |t| ≤ 2: light gray fill, no edge → clearly "faded out"
      - Y-axis labels: bold for significant, normal for not significant

    Parameters
    ----------
    stats_df : Output of compute_divergence_stats().
    broad_categories : dict mapping feature name → broad category.
    top_n : If set, show only top N features. None = show all.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df = stats_df.copy()
    df["category"] = df["feature"].map(broad_categories).fillna("Other")
    df["significant"] = df["t_stat"].abs() > 2
    df = df.sort_values("abs_d_bar", ascending=True)  # ascending for horizontal bars

    if top_n is not None:
        df = df.tail(top_n)

    # Category colors
    categories = sorted(df["category"].unique())
    cmap = plt.cm.get_cmap("tab10", len(categories))
    cat_colors = {cat: cmap(i) for i, cat in enumerate(categories)}

    # Bar colors: category color if significant, light gray if not
    NONSIG_COLOR = "#D3D3D3"  # light gray
    bar_colors = [
        cat_colors[row["category"]] if row["significant"] else NONSIG_COLOR
        for _, row in df.iterrows()
    ]
    edge_colors = [
        "#333333" if row["significant"] else "none"
        for _, row in df.iterrows()
    ]
    edge_widths = [
        0.5 if row["significant"] else 0
        for _, row in df.iterrows()
    ]

    n_bars = len(df)
    fig_height = max(8, n_bars * 0.22)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bars = ax.barh(
        range(n_bars),
        df["d_bar"].values,
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=edge_widths,
    )

    # Y-axis labels: bold for significant
    ax.set_yticks(range(n_bars))
    labels = ax.set_yticklabels(df["feature"].values, fontsize=7)
    for label, sig in zip(labels, df["significant"].values):
        if sig:
            label.set_fontweight("bold")
        else:
            label.set_fontweight("normal")
            label.set_color("#999999")

    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Mean divergence d̄ (deploy − train)")

    n_sig = df["significant"].sum()
    n_total = len(df)
    ax.set_title(
        "Marginal Divergence: Equal-Weighted vs Volume-Weighted Means\n"
        f"(colored = significant |t| > 2: {n_sig}/{n_total}; "
        "gray = not significant)"
    )

    # Legend: category colors + gray for not significant
    handles = [Patch(facecolor=cat_colors[c], edgecolor="#333333",
                     linewidth=0.5, label=c) for c in categories]
    handles.append(Patch(facecolor=NONSIG_COLOR, edgecolor="none",
                         label="Not significant"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved divergence bar chart to %s", output_path)


def plot_density_comparison(
    panel: pd.DataFrame,
    focal_features: list[str],
    w_col: str,
    output_path: str | Path,
) -> None:
    """2×3 panel of KDE plots: equal-weighted vs volume-weighted densities."""
    import matplotlib.pyplot as plt

    n_features = len(focal_features)
    nrows = (n_features + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    x_grid = np.linspace(0, 1, 200)

    for i, feat in enumerate(focal_features):
        ax = axes[i]
        valid = panel[feat].notna() & panel[w_col].notna()
        vals = panel.loc[valid, feat].values
        w = panel.loc[valid, w_col].values

        if len(vals) < 100:
            ax.set_title(f"{feat} (insufficient data)")
            continue

        # Equal-weighted KDE
        kde_ew = gaussian_kde(vals)
        # Volume-weighted KDE
        kde_vw = gaussian_kde(vals, weights=w)

        ax.plot(x_grid, kde_ew(x_grid), "-", linewidth=1.5, label="Training (equal-wt)")
        ax.plot(
            x_grid,
            kde_vw(x_grid),
            "--",
            linewidth=1.5,
            label="Deployment (vol-wt)",
        )
        ax.set_title(feat)
        ax.set_xlabel("Characteristic rank [0, 1]")
        ax.set_ylabel("Density")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Training vs Deployment Distributions (6 Focal Characteristics)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved density comparison to %s", output_path)


def plot_weight_distribution(
    panel: pd.DataFrame,
    w_col: str,
    output_path: str | Path,
) -> None:
    """Histogram of log₁₀(w̃) with percentile annotations."""
    import matplotlib.pyplot as plt

    w = panel[w_col].dropna()
    w = w[w > 0]

    log_w = np.log10(w.values)

    # Percentiles of raw w̃ (not log)
    pcts = [5, 25, 50, 75, 95]
    pct_vals = np.percentile(w.values, pcts)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(log_w, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="none")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="log₁₀(1) = 0 (average stock)")

    # Percentile annotation
    pct_text = "Percentiles of w̃:\n" + "\n".join(
        f"  {p}th: {v:.3f}" for p, v in zip(pcts, pct_vals)
    )
    ax.text(
        0.98,
        0.95,
        pct_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel("log₁₀(w̃)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Implementability Weights (w̃ = dvol / mean(dvol))")
    ax.legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved weight distribution to %s", output_path)
    logger.info(
        "Weight percentiles: %s",
        {p: f"{v:.4f}" for p, v in zip(pcts, pct_vals)},
    )


# ═════════════════════════════════════════════════════════════
# Step 2: Heterogeneous Predictability
# ═════════════════════════════════════════════════════════════


def quintile_fama_macbeth(
    panel: pd.DataFrame,
    focal_features: list[str],
    quintile_col: str = "liq_quintile",
    return_col: str = "excess_ret",
) -> dict:
    """Quintile-specific Fama-MacBeth regressions (Eq. 5-6 in document).

    For each quintile q=1..5 and each month t:
        r_i,t+1 = α_q,t + x'_it β_q,t + ε   for i ∈ Q_q,t

    Returns
    -------
    dict with keys:
        coef_table : pd.DataFrame — rows=focal chars, cols include
            β̄_Q1..β̄_Q5, t_Q1..t_Q5, se_Q1..se_Q5, β̄_Q5-Q1, t_Q5-Q1
        monthly_coefs : dict[int, pd.DataFrame] — per-quintile monthly coefficients
    """
    quintiles = sorted(panel[quintile_col].dropna().unique())
    quintiles = [int(q) for q in quintiles]
    months = sorted(panel["yyyymm"].unique())

    # Collect monthly coefficients per quintile
    monthly_coefs = {q: [] for q in quintiles}

    for m in months:
        mdf = panel[panel["yyyymm"] == m]

        for q in quintiles:
            qdf = mdf[mdf[quintile_col] == q]
            y = qdf[return_col].values
            X_df = qdf[focal_features].copy()

            valid = ~np.isnan(y)
            for col in focal_features:
                valid &= X_df[col].notna()

            if valid.sum() < 30:
                continue

            y_v = y[valid]
            X_v = X_df.loc[valid].fillna(0.5).values
            X_v = np.column_stack([np.ones(len(X_v)), X_v])

            try:
                beta, _, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)
            except np.linalg.LinAlgError:
                continue

            row = {"yyyymm": m}
            for i, col in enumerate(focal_features):
                row[col] = beta[i + 1]
            monthly_coefs[q].append(row)

    # FM averaging with NW t-stats per quintile per feature
    results = []
    for feat in focal_features:
        row = {"feature": feat}

        q_means = {}
        q_ses = {}
        for q in quintiles:
            df_q = pd.DataFrame(monthly_coefs[q])
            if feat not in df_q.columns or len(df_q) < 12:
                row[f"beta_Q{q}"] = np.nan
                row[f"t_Q{q}"] = np.nan
                row[f"se_Q{q}"] = np.nan
                continue

            series = df_q[feat].dropna()
            if len(series) < 12:
                row[f"beta_Q{q}"] = np.nan
                row[f"t_Q{q}"] = np.nan
                row[f"se_Q{q}"] = np.nan
                continue

            nw = newey_west_tstat(series.values, lags=6)
            row[f"beta_Q{q}"] = nw["mean"]
            row[f"t_Q{q}"] = nw["t_stat"]
            row[f"se_Q{q}"] = nw["std_err"]
            q_means[q] = nw["mean"]
            q_ses[q] = nw["std_err"]

        # Q5 - Q1 difference
        if 5 in q_means and 1 in q_means:
            # Compute NW t-stat on the Q5-Q1 difference series
            df_q5 = pd.DataFrame(monthly_coefs[5]).set_index("yyyymm")
            df_q1 = pd.DataFrame(monthly_coefs[1]).set_index("yyyymm")
            if feat in df_q5.columns and feat in df_q1.columns:
                common = df_q5.index.intersection(df_q1.index)
                diff = df_q5.loc[common, feat] - df_q1.loc[common, feat]
                diff = diff.dropna()
                if len(diff) >= 12:
                    nw_diff = newey_west_tstat(diff.values, lags=6)
                    row["beta_Q5-Q1"] = nw_diff["mean"]
                    row["t_Q5-Q1"] = nw_diff["t_stat"]
                else:
                    row["beta_Q5-Q1"] = np.nan
                    row["t_Q5-Q1"] = np.nan
            else:
                row["beta_Q5-Q1"] = np.nan
                row["t_Q5-Q1"] = np.nan
        else:
            row["beta_Q5-Q1"] = np.nan
            row["t_Q5-Q1"] = np.nan

        results.append(row)

    coef_table = pd.DataFrame(results)
    return {
        "coef_table": coef_table,
        "monthly_coefs": monthly_coefs,
    }


def format_quintile_table(
    coef_table: pd.DataFrame,
    beta_fmt: str = ".4f",
    t_fmt: str = ".2f",
) -> pd.DataFrame:
    """Format quintile FM coefficients into academic presentation style.

    Produces a table matching the document's Output 2.1 template:
        Feature | Q1 (illiq) | Q2 | Q3 | Q4 | Q5 (liq) | Q5-Q1

    Each cell shows β̄ with t-statistic in parentheses, e.g. "0.0313 (10.91)".
    """
    rows = []
    for _, r in coef_table.iterrows():
        row = {"Feature": r["feature"]}
        for q in [1, 2, 3, 4, 5]:
            b = r.get(f"beta_Q{q}", np.nan)
            t = r.get(f"t_Q{q}", np.nan)
            if pd.notna(b) and pd.notna(t):
                row[f"Q{q}"] = f"{b:{beta_fmt}} ({t:{t_fmt}})"
            else:
                row[f"Q{q}"] = ""
        # Q5-Q1
        b = r.get("beta_Q5-Q1", np.nan)
        t = r.get("t_Q5-Q1", np.nan)
        if pd.notna(b) and pd.notna(t):
            row["Q5-Q1"] = f"{b:{beta_fmt}} ({t:{t_fmt}})"
        else:
            row["Q5-Q1"] = ""
        rows.append(row)

    formatted = pd.DataFrame(rows)
    formatted = formatted.rename(columns={
        "Q1": "Q1 (illiq)",
        "Q5": "Q5 (liq)",
    })
    return formatted


def interaction_fama_macbeth(
    panel: pd.DataFrame,
    focal_features: list[str],
    liq_col: str = "liq_rank",
    return_col: str = "excess_ret",
    use_dummy: bool = False,
) -> dict:
    """Interaction Fama-MacBeth regression (Eq. 7 in document).

    Each month t, full sample:
        r_i,t+1 = α_t + x'_it β_t + (x_it · L_it)' γ_t + ε_it

    Parameters
    ----------
    liq_col : Liquidity rank column (continuous [0,1] or dummy).
    use_dummy : If True, L_it = 1 if above median, 0 otherwise.

    Returns
    -------
    dict with keys:
        coef_table : pd.DataFrame — rows=focal chars, cols: beta_bar, beta_t,
            gamma_bar, gamma_t
        f_test_pvalue : float — joint F-test p-value for H0: all γ = 0
        n_months : int
    """
    months = sorted(panel["yyyymm"].unique())
    beta_list = []
    gamma_list = []
    n_features = len(focal_features)

    for m in months:
        mdf = panel[panel["yyyymm"] == m].copy()

        y = mdf[return_col].values
        L = mdf[liq_col].values

        if use_dummy:
            L = (L > np.nanmedian(L)).astype(float)

        # Build X = [1, x_1..x_15, x_1*L..x_15*L]
        X_main = mdf[focal_features].values
        X_interact = X_main * L[:, np.newaxis]

        # Valid mask
        valid = ~np.isnan(y) & ~np.isnan(L)
        for j in range(n_features):
            valid &= ~np.isnan(X_main[:, j])

        if valid.sum() < 100:
            continue

        y_v = y[valid]
        X_m = X_main[valid]
        X_i = X_interact[valid]
        X_full = np.column_stack([np.ones(valid.sum()), X_m, X_i])

        try:
            beta_all, _, _, _ = np.linalg.lstsq(X_full, y_v, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Extract: beta_all[0]=intercept, [1..15]=main, [16..30]=interaction
        beta_row = {"yyyymm": m}
        gamma_row = {"yyyymm": m}
        for i, feat in enumerate(focal_features):
            beta_row[feat] = beta_all[1 + i]
            gamma_row[feat] = beta_all[1 + n_features + i]
        beta_list.append(beta_row)
        gamma_list.append(gamma_row)

    if not beta_list:
        logger.warning("Interaction FM: no valid months")
        return {
            "coef_table": pd.DataFrame(),
            "f_test_pvalue": np.nan,
            "n_months": 0,
        }

    beta_df = pd.DataFrame(beta_list).set_index("yyyymm")
    gamma_df = pd.DataFrame(gamma_list).set_index("yyyymm")

    # FM averaging with NW t-stats
    results = []
    for feat in focal_features:
        row = {"feature": feat}

        if feat in beta_df.columns:
            b_series = beta_df[feat].dropna()
            if len(b_series) >= 12:
                nw_b = newey_west_tstat(b_series.values, lags=6)
                row["beta_bar"] = nw_b["mean"]
                row["beta_t"] = nw_b["t_stat"]
            else:
                row["beta_bar"] = np.nan
                row["beta_t"] = np.nan
        else:
            row["beta_bar"] = np.nan
            row["beta_t"] = np.nan

        if feat in gamma_df.columns:
            g_series = gamma_df[feat].dropna()
            if len(g_series) >= 12:
                nw_g = newey_west_tstat(g_series.values, lags=6)
                row["gamma_bar"] = nw_g["mean"]
                row["gamma_t"] = nw_g["t_stat"]
            else:
                row["gamma_bar"] = np.nan
                row["gamma_t"] = np.nan
        else:
            row["gamma_bar"] = np.nan
            row["gamma_t"] = np.nan

        results.append(row)

    coef_table = pd.DataFrame(results)

    # Joint F-test: H0: all γ = 0
    # Use time-series F-statistic on the monthly γ estimates
    gamma_mat = gamma_df[focal_features].dropna()
    if len(gamma_mat) >= 12:
        gamma_means = gamma_mat.mean().values
        T = len(gamma_mat)
        k = len(focal_features)
        # Covariance of mean estimates (simple, not NW — for F-test)
        cov_mat = gamma_mat.cov().values / T
        try:
            cov_inv = np.linalg.inv(cov_mat)
            f_stat = float(gamma_means @ cov_inv @ gamma_means / k)
            from scipy.stats import f as f_dist
            f_pvalue = float(1 - f_dist.cdf(f_stat, k, T - k))
        except np.linalg.LinAlgError:
            f_stat = np.nan
            f_pvalue = np.nan
    else:
        f_stat = np.nan
        f_pvalue = np.nan

    return {
        "coef_table": coef_table,
        "f_test_stat": f_stat,
        "f_test_pvalue": f_pvalue,
        "f_test_df": (k, T - k),
        "n_months": len(beta_df),
    }


def plot_quintile_coefficients(
    quintile_results: dict,
    focal_features: list[str],
    output_path: str | Path,
) -> None:
    """3×5 panel of coefficient plots: β̄_j,q by quintile with 95% CI.

    Each subplot shows one focal characteristic.
    x-axis = quintile (1-5), y-axis = β̄_j,q.
    Shaded 95% CI bands. Flat = homogeneous, sloping = heterogeneous.
    """
    import matplotlib.pyplot as plt

    ct = quintile_results["coef_table"]
    n = len(focal_features)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows))
    axes = axes.flatten()
    quintiles = [1, 2, 3, 4, 5]

    for i, feat in enumerate(focal_features):
        ax = axes[i]
        row = ct[ct["feature"] == feat]
        if row.empty:
            ax.set_title(feat)
            continue
        row = row.iloc[0]

        betas = [row.get(f"beta_Q{q}", np.nan) for q in quintiles]
        ses = [row.get(f"se_Q{q}", np.nan) for q in quintiles]
        ci_lo = [b - 1.96 * s for b, s in zip(betas, ses)]
        ci_hi = [b + 1.96 * s for b, s in zip(betas, ses)]

        ax.plot(quintiles, betas, "o-", color="steelblue", linewidth=1.5, markersize=5)
        ax.fill_between(quintiles, ci_lo, ci_hi, alpha=0.2, color="steelblue")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xticks(quintiles)
        ax.set_xticklabels(["Q1\n(illiq)", "Q2", "Q3", "Q4", "Q5\n(liq)"], fontsize=7)
        ax.set_title(feat, fontsize=9, fontweight="bold")

        # Annotate Q5-Q1
        diff = row.get("beta_Q5-Q1", np.nan)
        t_diff = row.get("t_Q5-Q1", np.nan)
        if not np.isnan(diff):
            sig_marker = "*" if abs(t_diff) > 2 else ""
            ax.text(
                0.95, 0.05,
                f"Q5-Q1={diff:.4f}{sig_marker}\n(t={t_diff:.2f})",
                transform=ax.transAxes, fontsize=6,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7),
            )

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Fama-MacBeth Coefficients by Liquidity Quintile\n"
        "(95% CI bands; Q1 = illiquid, Q5 = liquid)",
        fontsize=12,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved quintile coefficient plots to %s", output_path)


def plot_divergence_vs_heterogeneity(
    divergence_stats: pd.DataFrame,
    interaction_results: dict,
    focal_features: list[str],
    output_path: str | Path,
) -> float:
    """Scatter: |d̄_j| (Step 1) vs |γ̄_j| (Step 2) for focal characteristics.

    Returns Spearman rank correlation.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    int_table = interaction_results["coef_table"]

    points = []
    for feat in focal_features:
        div_row = divergence_stats[divergence_stats["feature"] == feat]
        int_row = int_table[int_table["feature"] == feat]
        if div_row.empty or int_row.empty:
            continue
        d_bar = div_row.iloc[0]["abs_d_bar"]
        gamma = abs(int_row.iloc[0]["gamma_bar"])
        points.append({"feature": feat, "abs_d_bar": d_bar, "abs_gamma": gamma})

    if not points:
        logger.warning("No matching features for divergence vs heterogeneity scatter")
        return np.nan

    df = pd.DataFrame(points)
    rho, p_val = spearmanr(df["abs_d_bar"], df["abs_gamma"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["abs_d_bar"], df["abs_gamma"], s=60, color="steelblue", zorder=3)

    # Label each point
    for _, row in df.iterrows():
        ax.annotate(
            row["feature"],
            (row["abs_d_bar"], row["abs_gamma"]),
            fontsize=8,
            textcoords="offset points",
            xytext=(5, 5),
        )

    ax.set_xlabel("|d̄_j| (distributional divergence, Step 1)", fontsize=11)
    ax.set_ylabel("|γ̄_j| (predictability heterogeneity, Step 2)", fontsize=11)
    ax.set_title(
        "Distributional Divergence vs Predictability Heterogeneity\n"
        f"(Spearman ρ = {rho:.3f}; N = {len(df)} focal characteristics)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved divergence vs heterogeneity scatter to %s (ρ=%.3f)", output_path, rho)
    return rho
