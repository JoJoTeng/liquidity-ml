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
    "AnnouncementReturn": ("Other", "Earnings announcement return"),
    "BidAskSpread": ("Liquidity", "Transaction cost proxy"),
}

# 6 focal characteristics for density plots (Output 1.4)
DENSITY_PLOT_FEATURES = [
    "Illiquidity",
    "IdioVol3F",
    "Mom12m",
    "BM",
    "AnnouncementReturn",
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

        X_df = X_df[good_cols].fillna(0.5)  # neutral rank for sparse remaining NaN
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


def _set_academic_style():
    """Set matplotlib rcParams for clean academic figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _clean_axes(ax):
    """Remove top and right spines for clean academic style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    _set_academic_style()
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
    ax.set_xlabel(r"Mean divergence $\bar{d}$ (deploy $-$ train)")

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


def plot_divergence_by_category(
    cat_summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Horizontal bar chart of average |d̄| aggregated by economic category.

    This is the main-paper version of Output 1.1. Each bar represents one
    economic category (8 bars total), showing the average absolute divergence
    across all characteristics in that category. Annotation shows the fraction
    of significant characteristics.

    For the full per-characteristic version, see plot_divergence_bar_chart()
    (intended for the appendix).

    Parameters
    ----------
    cat_summary : Output of summarize_divergence_by_category().
        Expected columns: Category, Avg. |d_bar|, # Significant (|t| > 2),
        # Characteristics
    """
    import matplotlib.pyplot as plt
    _set_academic_style()

    df = cat_summary.copy()
    df = df.sort_values("Avg. |d_bar|", ascending=True)  # ascending for horizontal bars

    n_bars = len(df)
    categories = df["Category"].values
    avg_d = df["Avg. |d_bar|"].values

    fig, ax = plt.subplots(figsize=(8, 8))

    # Gradient color: darker = larger divergence
    norm_vals = avg_d / avg_d.max()
    colors = [plt.cm.Blues(0.35 + 0.55 * v) for v in norm_vals]

    ax.barh(
        range(n_bars), avg_d,
        color=colors, edgecolor="white", linewidth=0.3,
    )

    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel(r"Average $|\bar{d}|$ across characteristics in category", fontsize=11)

    n_sig = df["# Significant (|t| > 2)"].values
    n_total = df["# Characteristics"].values
    total_sig = int(n_sig.sum())
    total_feat = int(n_total.sum())
    ax.set_title(
        "Distributional Divergence by Economic Category\n"
        f"({total_sig}/{total_feat} characteristics significant at $|t| > 2$)",
        fontsize=13,
    )


    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved category divergence bar chart to %s", output_path)


def plot_density_comparison(
    panel: pd.DataFrame,
    focal_features: list[str],
    w_col: str,
    output_path: str | Path,
    vw_col: str | None = None,
    title_suffix: str = "",
) -> None:
    """Two-panel density comparison: training (flat line) vs deployment (KDE).

    Panel A (left):  Characteristics related to size/liquidity
                     (Illiquidity, BM, STreversal)
    Panel B (right): Characteristics less obviously related to size
                     (IdioVol3F, Mom12m, AnnouncementReturn)

    The training distribution is drawn as a flat line at y=1.0 because
    rank-transforming to [0,1] produces Uniform(0,1) by construction.
    Using KDE for the training line creates boundary artifacts (drops
    near 0 and 1) that are misleading.

    Parameters
    ----------
    vw_col : Optional column for value-weight (market cap) KDE overlay.
    title_suffix : Optional suffix for the figure title (e.g., " — Recession").
    """
    import matplotlib.pyplot as plt
    _set_academic_style()

    # Define the two panels
    panel_a_label = "Panel A: Liquidity-related characteristics"
    panel_b_label = "Panel B: Non-liquidity characteristics"

    # Map features to panels — order matters for subplot placement
    panel_a_features = ["Illiquidity", "BM", "STreversal"]
    panel_b_features = ["IdioVol3F", "Mom12m", "AnnouncementReturn"]

    # Fall back to whatever is available
    panel_a = [f for f in panel_a_features if f in focal_features]
    panel_b = [f for f in panel_b_features if f in focal_features]

    # If the exact features aren't available, fill from remaining
    used = set(panel_a + panel_b)
    remaining = [f for f in focal_features if f not in used]
    while len(panel_a) < 3 and remaining:
        panel_a.append(remaining.pop(0))
    while len(panel_b) < 3 and remaining:
        panel_b.append(remaining.pop(0))

    nrows = max(len(panel_a), len(panel_b))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    x_grid = np.linspace(0, 1, 300)

    def _plot_one(ax, feat, panel_data, w_col_name, vw_col_name=None):
        valid = panel_data[feat].notna() & panel_data[w_col_name].notna()
        vals = panel_data.loc[valid, feat].values
        w = panel_data.loc[valid, w_col_name].values

        if len(vals) < 100:
            ax.set_title(f"{feat} (insufficient data)")
            return

        # Training distribution: flat line at y=1.0
        ax.axhline(1.0, color="steelblue", linewidth=2.0, label="Training (equal-wt)")

        # Deployment distribution: volume-weighted KDE
        kde_vw = gaussian_kde(vals, weights=w)
        deploy_y = kde_vw(x_grid)
        ax.plot(x_grid, deploy_y, "--", color="darkorange", linewidth=2.0,
                label="Deployment (vol-wt)")

        # Shade the gap between training and deployment
        ax.fill_between(x_grid, 1.0, deploy_y, alpha=0.15, color="darkorange")

        # Value-weighted density (market cap) — optional third line
        if vw_col_name is not None and vw_col_name in panel_data.columns:
            valid_vw = valid & panel_data[vw_col_name].notna()
            vals_vw = panel_data.loc[valid_vw, feat].values
            w_vw_raw = panel_data.loc[valid_vw, vw_col_name].values
            if len(vals_vw) >= 100 and w_vw_raw.sum() > 0:
                w_vw = w_vw_raw / w_vw_raw.mean()
                kde_mcap = gaussian_kde(vals_vw, weights=w_vw)
                mcap_y = kde_mcap(x_grid)
                ax.plot(x_grid, mcap_y, ":", color="seagreen", linewidth=2.0,
                        label="Value-weighted")

        ax.set_title(feat, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Characteristic rank [0, 1]")
        ax.set_ylabel("Density")

    # Plot Panel A (left column)
    for i, feat in enumerate(panel_a):
        _plot_one(axes[i, 0], feat, panel, w_col, vw_col_name=vw_col)

    # Plot Panel B (right column)
    for i, feat in enumerate(panel_b):
        _plot_one(axes[i, 1], feat, panel, w_col, vw_col_name=vw_col)

    # Hide unused subplots
    for i in range(len(panel_a), nrows):
        axes[i, 0].set_visible(False)
    for i in range(len(panel_b), nrows):
        axes[i, 1].set_visible(False)

    # Panel labels
    axes[0, 0].annotate(
        panel_a_label, xy=(0.5, 1.15), xycoords="axes fraction",
        ha="center", fontsize=11, fontstyle="italic",
    )
    axes[0, 1].annotate(
        panel_b_label, xy=(0.5, 1.15), xycoords="axes fraction",
        ha="center", fontsize=11, fontstyle="italic",
    )

    # Shared legend from first subplot
    axes[0, 0].legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"Training vs Deployment Distributions{title_suffix}",
        fontsize=14, y=1.02,
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
    vw_col: str | None = None,
    title_suffix: str = "",
) -> None:
    """Histogram of log₁₀(w̃) with value-weight comparison.

    Shows dollar-volume weights as the main histogram, with an optional
    overlay of value-weights (market cap) to address the "why not just
    value-weight?" objection.

    Parameters
    ----------
    w_col : Dollar-volume implementability weight column.
    title_suffix : Optional suffix for the plot title.
    vw_col : Optional value-weight (market cap) column for comparison.
        If provided, its normalized weights are overlaid as a second density.
    """
    import matplotlib.pyplot as plt
    _set_academic_style()

    w = panel[w_col].dropna()
    w = w[w > 0]
    log_w = np.log10(w.values)

    # Percentiles of raw w̃ (not log)
    pcts = [5, 25, 50, 75, 95]
    pct_vals = np.percentile(w.values, pcts)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Dollar-volume weights
    ax.hist(log_w, bins=80, density=True, alpha=0.6, color="steelblue",
            edgecolor="none", label="Dollar volume weights")

    # Value-weight comparison (if provided)
    if vw_col is not None and vw_col in panel.columns:
        # Compute normalized VW weights: me / mean(me) per month
        vw_raw = panel.groupby("yyyymm")[vw_col].transform(
            lambda x: x / x.mean()
        )
        vw_valid = vw_raw.dropna()
        vw_valid = vw_valid[vw_valid > 0]
        log_vw = np.log10(vw_valid.values)

        ax.hist(log_vw, bins=80, density=True, color="green", alpha=0.2,
                edgecolor="green", linewidth=1.5, histtype="stepfilled",
                label="Value weights (market cap)")

        # VW percentiles
        vw_pcts = np.percentile(vw_valid.values, pcts)
        vw_median_log = np.log10(vw_pcts[2])  # 50th percentile

        ax.axvline(vw_median_log, color="green", linestyle=":", linewidth=1.5,
                   label=f"VW median = {vw_pcts[2]:.3f}")

    # Equal-weight reference
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5,
               label=r"Equal-weight: $\log_{10}(1) = 0$")

    # DV median
    dv_median_log = np.log10(pct_vals[2])
    ax.axvline(dv_median_log, color="steelblue", linestyle=":", linewidth=1.5,
               label=f"DV median = {pct_vals[2]:.3f}")

    # Percentile annotation box
    pct_text = "Dollar vol. percentiles:\n" + "\n".join(
        f"  {p}th: {v:.3f}" for p, v in zip(pcts, pct_vals)
    )
    if vw_col is not None and vw_col in panel.columns:
        pct_text += "\n\nValue-wt percentiles:\n" + "\n".join(
            f"  {p}th: {v:.3f}" for p, v in zip(pcts, vw_pcts)
        )
    ax.text(
        0.98, 0.95, pct_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel(r"$\log_{10}(\tilde{w})$")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of Normalized Weights: Dollar Volume vs Value{title_suffix}")
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved weight distribution to %s", output_path)
    logger.info(
        "DV weight percentiles: %s",
        {p: f"{v:.4f}" for p, v in zip(pcts, pct_vals)},
    )
    if vw_col is not None and vw_col in panel.columns:
        logger.info(
            "VW weight percentiles: %s",
            {p: f"{v:.4f}" for p, v in zip(pcts, vw_pcts)},
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
    _set_academic_style()

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
    _set_academic_style()
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

    fig, ax = plt.subplots(figsize=(8, 8))
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

    ax.set_xlabel(r"$|\bar{d}_j|$ (distributional divergence)", fontsize=11)
    ax.set_ylabel(r"$|\bar{\gamma}_j|$ (predictability heterogeneity)", fontsize=11)
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


# ═════════════════════════════════════════════════════════════
# Step 3: Standard ML Is Affected
# ═════════════════════════════════════════════════════════════


def rolling_xgboost_predict(
    panel: pd.DataFrame,
    features: list[str],
    config: dict | None = None,
    return_col: str = "excess_ret",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train baseline XGBoost with rolling windows, collect OOS predictions.

    Protocol:
      - 120 months train + 12 months validation + 1 month test (rolling)
      - Train starts from 198901
      - Retune every 12 months using validation set MSE
      - Between retunes, refit with fixed best params on train set
      - Per-window rank normalization to [0,1] (no look-ahead)
      - NaN fill with 0.5 after ranking

    Returns
    -------
    predictions : DataFrame [permno, yyyymm, y_true, y_pred]
    importances : DataFrame rows=yyyymm (test month), cols=features (gain)
    """
    import itertools
    import time as _time
    from src.models import create_model

    if config is None:
        config = load_config()

    train_cfg = config["training"]
    train_window = train_cfg["train_window"]           # 120
    val_window = train_cfg["validation_window"]        # 12
    oos_start = train_cfg.get("oos_start", 200001)
    seed = config["project"]["seed"]

    all_months = sorted(panel["yyyymm"].unique())
    oos_months = [m for m in all_months if m >= oos_start]

    retune_freq = 12  # retune every year

    predictions_list = []
    importance_list = []
    tuned_params_list = []
    best_params = None
    months_since_retune = retune_freq  # force retune on first window
    t_start = _time.time()
    n_done = 0

    logger.info(
        "Rolling XGBoost: %d OOS months, retune every %d months, "
        "train %d + val %d months, grid size %d",
        len(oos_months), retune_freq, train_window, val_window,
        np.prod([len(v) for v in config["models"]["xgboost"].get("search_space", {}).values()]),
    )

    for i, test_month in enumerate(oos_months):
        test_idx = all_months.index(test_month)

        if test_idx < train_window + val_window:
            continue

        train_start = test_idx - train_window - val_window
        val_start = test_idx - val_window

        train_months_list = all_months[train_start:val_start]
        val_months_list = all_months[val_start:test_idx]

        train_df = panel[panel["yyyymm"].isin(train_months_list)].copy()
        val_df = panel[panel["yyyymm"].isin(val_months_list)].copy()
        test_df = panel[panel["yyyymm"] == test_month].copy()

        if len(train_df) < 100 or len(test_df) < 50:
            continue

        # Per-window rank normalization to [0,1] (no look-ahead)
        for col in features:
            train_df[col] = train_df.groupby("yyyymm")[col].rank(pct=True)
            val_df[col] = val_df.groupby("yyyymm")[col].rank(pct=True)
            test_df[col] = test_df.groupby("yyyymm")[col].rank(pct=True)

        # Fill NaN with 0.5 (neutral rank)
        train_df[features] = train_df[features].fillna(0.5)
        val_df[features] = val_df[features].fillna(0.5)
        test_df[features] = test_df[features].fillna(0.5)

        X_train = train_df[features].values
        y_train = train_df[return_col].values
        X_val = val_df[features].values
        y_val = val_df[return_col].values
        X_test = test_df[features].values
        y_test = test_df[return_col].values

        # Drop NaN targets
        valid_train = ~np.isnan(y_train)
        valid_val = ~np.isnan(y_val)
        valid_test = ~np.isnan(y_test)
        if valid_train.sum() < 100 or valid_test.sum() < 30:
            continue

        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_val, y_val = X_val[valid_val], y_val[valid_val]
        X_test, y_test = X_test[valid_test], y_test[valid_test]

        # Retune hyperparameters every 12 months using validation set
        if months_since_retune >= retune_freq:
            xgb_cfg = config["models"]["xgboost"]
            search_space = xgb_cfg.get("search_space", {})
            keys = list(search_space.keys())
            values = list(search_space.values())
            n_trials = xgb_cfg.get("n_random_search", 150)

            full_grid_size = 1
            for v in values:
                full_grid_size *= len(v)

            rng = np.random.RandomState(seed)
            if full_grid_size <= n_trials:
                param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
            else:
                param_grid = []
                for _ in range(n_trials):
                    combo = {k: v[rng.randint(len(v))] for k, v in zip(keys, values)}
                    param_grid.append(combo)

            best_mse = np.inf
            for params in param_grid:
                trial_cfg = {**xgb_cfg, **params}
                m = create_model("xgboost", config=trial_cfg, seed=seed)
                m.fit(X_train, y_train)
                preds = m.predict(X_val)
                mse = np.mean((y_val - preds) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_params = trial_cfg

            months_since_retune = 0
            logger.info(
                "Month %d: RETUNED XGBoost via validation set (best params: %s)",
                test_month,
                {k: best_params.get(k) for k in keys},
            )

        # Train XGBoost with best params on full train set
        model = create_model("xgboost", config=best_params, seed=seed)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Collect predictions
        test_valid = test_df[valid_test]
        pred_df = pd.DataFrame({
            "permno": test_valid["permno"].values,
            "yyyymm": test_valid["yyyymm"].values,
            "y_true": y_test,
            "y_pred": y_pred,
        })
        predictions_list.append(pred_df)

        # Feature importance (gain-based)
        imp = model.get_feature_importance(features)
        imp_row = {"yyyymm": test_month}
        imp_row.update(imp.to_dict())
        importance_list.append(imp_row)

        # Save tuned params for this window
        tuned_params_list.append({"yyyymm": test_month, **best_params})

        months_since_retune += 1
        n_done += 1

        # Progress with ETA
        elapsed = _time.time() - t_start
        avg_per_month = elapsed / n_done
        remaining = avg_per_month * (len(oos_months) - i - 1)
        remaining_min = remaining / 60

        if (i + 1) % 6 == 0 or i == 0 or months_since_retune == 1:
            logger.info(
                "Progress: %d/%d months (%.0f%%) | month %d%s | "
                "%.1f s/month | ETA: %.0f min",
                i + 1, len(oos_months), 100 * (i + 1) / len(oos_months),
                test_month,
                " [RETUNED]" if months_since_retune == 1 else "",
                avg_per_month, remaining_min,
            )

    if not predictions_list:
        logger.warning("No valid OOS months produced")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    predictions = pd.concat(predictions_list, ignore_index=True)
    importances = pd.DataFrame(importance_list).set_index("yyyymm")
    tuned_params = pd.DataFrame(tuned_params_list).set_index("yyyymm")

    logger.info(
        "Rolling XGBoost complete: %d OOS months, %d predictions",
        len(importances), len(predictions),
    )
    return predictions, importances, tuned_params


def _rolling_xgboost_core(
    panel: pd.DataFrame,
    features: list[str],
    config: dict,
    return_col: str,
    train_filter_fn,
    test_filter_fn,
    fixed_params: dict | None = None,
    baseline_tuned_params: pd.DataFrame | None = None,
    label: str = "XGB",
) -> pd.DataFrame:
    """Shared rolling-window XGBoost core for restricted/quintile models.

    Parameters
    ----------
    train_filter_fn : callable(train_df, val_df, all_months, train_months, val_months)
        Returns filtered (train_df, val_df).
    test_filter_fn : callable(test_df)
        Returns filtered test_df.
    fixed_params : If provided, use these same params every month (no retuning).
    baseline_tuned_params : DataFrame indexed by yyyymm with per-window params
        from the baseline model. If provided, uses baseline's tuned params for each
        test month. Takes precedence over fixed_params.
    """
    import itertools
    import time as _time
    from src.models import create_model

    train_cfg = config["training"]
    train_window = train_cfg["train_window"]
    val_window = train_cfg["validation_window"]
    oos_start = train_cfg.get("oos_start", 200001)
    seed = config["project"]["seed"]

    all_months = sorted(panel["yyyymm"].unique())
    oos_months = [m for m in all_months if m >= oos_start]

    retune_freq = 12
    predictions_list = []
    best_params = fixed_params
    months_since_retune = retune_freq
    t_start = _time.time()
    n_done = 0

    for i, test_month in enumerate(oos_months):
        test_idx = all_months.index(test_month)
        if test_idx < train_window + val_window:
            continue

        train_start = test_idx - train_window - val_window
        val_start = test_idx - val_window

        train_months_list = all_months[train_start:val_start]
        val_months_list = all_months[val_start:test_idx]

        train_df = panel[panel["yyyymm"].isin(train_months_list)].copy()
        val_df = panel[panel["yyyymm"].isin(val_months_list)].copy()
        test_df = panel[panel["yyyymm"] == test_month].copy()

        # Filter FIRST to restricted/quintile universe,
        # then re-rank WITHIN the filtered sample.
        # This avoids feature compression (e.g., Size compressed to [0.8, 1.0]
        # when only Q5 stocks remain after full-cross-section ranking).
        train_df, val_df = train_filter_fn(
            train_df, val_df, all_months, train_months_list, val_months_list
        )
        test_df = test_filter_fn(test_df)

        if len(train_df) < 100 or len(test_df) < 20:
            continue

        # Re-rank within filtered universe so features are uniform [0,1]
        for col in features:
            train_df[col] = train_df.groupby("yyyymm")[col].rank(pct=True)
            val_df[col] = val_df.groupby("yyyymm")[col].rank(pct=True)
            test_df[col] = test_df.groupby("yyyymm")[col].rank(pct=True)

        train_df[features] = train_df[features].fillna(0.5)
        val_df[features] = val_df[features].fillna(0.5)
        test_df[features] = test_df[features].fillna(0.5)

        X_train = train_df[features].values
        y_train = train_df[return_col].values
        X_val = val_df[features].values
        y_val = val_df[return_col].values
        X_test = test_df[features].values
        y_test = test_df[return_col].values

        valid_train = ~np.isnan(y_train)
        valid_val = ~np.isnan(y_val)
        valid_test = ~np.isnan(y_test)
        if valid_train.sum() < 100 or valid_test.sum() < 10:
            continue

        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_val, y_val = X_val[valid_val], y_val[valid_val]
        X_test, y_test = X_test[valid_test], y_test[valid_test]

        # Determine params: baseline_tuned > fixed > retune
        if baseline_tuned_params is not None and test_month in baseline_tuned_params.index:
            best_params = baseline_tuned_params.loc[test_month].to_dict()
        elif fixed_params is None and months_since_retune >= retune_freq:
            xgb_cfg = config["models"]["xgboost"]
            search_space = xgb_cfg.get("search_space", {})
            keys = list(search_space.keys())
            values = list(search_space.values())
            n_trials = xgb_cfg.get("n_random_search", 150)

            full_grid_size = 1
            for v in values:
                full_grid_size *= len(v)

            rng = np.random.RandomState(seed)
            if full_grid_size <= n_trials:
                param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
            else:
                param_grid = []
                for _ in range(n_trials):
                    combo = {k: v[rng.randint(len(v))] for k, v in zip(keys, values)}
                    param_grid.append(combo)

            best_mse = np.inf
            for params in param_grid:
                trial_cfg = {**xgb_cfg, **params}
                m = create_model("xgboost", config=trial_cfg, seed=seed)
                m.fit(X_train, y_train)
                preds = m.predict(X_val)
                mse = np.mean((y_val - preds) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_params = trial_cfg

            months_since_retune = 0
            logger.info(
                "%s month %d: RETUNED (best: %s)", label, test_month,
                {k: best_params.get(k) for k in keys},
            )
        elif fixed_params is not None and best_params is None:
            best_params = fixed_params

        model = create_model("xgboost", config=best_params, seed=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_valid = test_df[valid_test]
        pred_df = pd.DataFrame({
            "permno": test_valid["permno"].values,
            "yyyymm": test_valid["yyyymm"].values,
            "y_true": y_test,
            "y_pred": y_pred,
        })
        predictions_list.append(pred_df)

        months_since_retune += 1
        n_done += 1

        elapsed = _time.time() - t_start
        avg_per_month = elapsed / n_done
        remaining = avg_per_month * (len(oos_months) - i - 1)

        if (i + 1) % 12 == 0 or months_since_retune == 1:
            logger.info(
                "%s progress: %d/%d (%.0f%%) | month %d%s | ETA: %.0f min",
                label, i + 1, len(oos_months),
                100 * (i + 1) / len(oos_months), test_month,
                " [RETUNED]" if months_since_retune == 1 and fixed_params is None else "",
                remaining / 60,
            )

    if not predictions_list:
        logger.warning("%s: no valid OOS months produced", label)
        return pd.DataFrame()

    predictions = pd.concat(predictions_list, ignore_index=True)
    logger.info(
        "%s complete: %d predictions over %d months",
        label, len(predictions), predictions["yyyymm"].nunique(),
    )
    return predictions


def rolling_xgboost_predict_restricted(
    panel: pd.DataFrame,
    features: list[str],
    min_quintile: int,
    quintile_col: str = "liq_quintile",
    config: dict | None = None,
    return_col: str = "excess_ret",
    fixed_params: dict | None = None,
    baseline_tuned_params: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Train XGBoost on a restricted universe (drop illiquid quintiles).

    Per Section 5.2(d): filters train/val to quintile >= min_quintile.
    Test set includes ALL stocks (no filter). Uses fixed baseline
    hyperparameters to isolate training composition effect.

    Quintile assignment uses the last month of the training window.
    """
    if config is None:
        config = load_config()

    def train_filter(train_df, val_df, all_months, train_months, val_months):
        last_train_month = train_months[-1]
        last_month_q = train_df.loc[
            train_df["yyyymm"] == last_train_month,
            ["permno", quintile_col]
        ].drop_duplicates()
        valid_permnos = set(
            last_month_q.loc[last_month_q[quintile_col] >= min_quintile, "permno"]
        )
        return (
            train_df[train_df["permno"].isin(valid_permnos)].copy(),
            val_df[val_df["permno"].isin(valid_permnos)].copy(),
        )

    def test_filter(test_df):
        return test_df  # no filter on test set

    return _rolling_xgboost_core(
        panel, features, config, return_col,
        train_filter_fn=train_filter,
        test_filter_fn=test_filter,
        fixed_params=fixed_params,
        baseline_tuned_params=baseline_tuned_params,
        label=f"XGB_MQ{min_quintile}+",
    )


def rolling_xgboost_predict_quintile(
    panel: pd.DataFrame,
    features: list[str],
    quintile: int,
    quintile_col: str = "liq_quintile",
    config: dict | None = None,
    return_col: str = "excess_ret",
    fixed_params: dict | None = None,
    baseline_tuned_params: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Train XGBoost on a single liquidity quintile.

    Per Section 5.2(e): filters train/val/test to the specified quintile.
    Uses fixed baseline hyperparameters.

    Quintile assignment uses the last month of the training window.
    """
    if config is None:
        config = load_config()

    def train_filter(train_df, val_df, all_months, train_months, val_months):
        last_train_month = train_months[-1]
        last_month_q = train_df.loc[
            train_df["yyyymm"] == last_train_month,
            ["permno", quintile_col]
        ].drop_duplicates()
        valid_permnos = set(
            last_month_q.loc[last_month_q[quintile_col] == quintile, "permno"]
        )
        return (
            train_df[train_df["permno"].isin(valid_permnos)].copy(),
            val_df[val_df["permno"].isin(valid_permnos)].copy(),
        )

    def test_filter(test_df):
        return test_df[test_df[quintile_col] == quintile].copy()

    return _rolling_xgboost_core(
        panel, features, config, return_col,
        train_filter_fn=train_filter,
        test_filter_fn=test_filter,
        fixed_params=fixed_params,
        baseline_tuned_params=baseline_tuned_params,
        label=f"XGB_Q{quintile}",
    )


def rolling_elasticnet_predict(
    panel: pd.DataFrame,
    features: list[str],
    config: dict | None = None,
    return_col: str = "excess_ret",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train ElasticNetCV with rolling windows.

    Protocol:
      - 120 months train, 1 month test (no validation gap)
      - Per-window rank normalization to [0,1], fillna(0.5)
      - ElasticNetCV auto-tunes alpha + l1_ratio every month via TimeSeriesSplit CV
        (l1_ratio searched over [0.5, 0.7, 0.9, 0.95, 1.0])

    Returns
    -------
    predictions : DataFrame [permno, yyyymm, y_true, y_pred]
    importances : DataFrame rows=yyyymm (test month), cols=features (|coef|)
    """
    import time as _time
    from sklearn.linear_model import ElasticNetCV

    if config is None:
        from src.config import load_config
        config = load_config()

    train_cfg = config["training"]
    train_window = train_cfg["train_window"]           # 120
    oos_start = train_cfg.get("oos_start", 200001)
    seed = config["project"]["seed"]

    all_months = sorted(panel["yyyymm"].unique())
    oos_months = [m for m in all_months if m >= oos_start]

    logger.info(
        "Rolling ElasticNetCV: %d OOS months, auto-tune every month",
        len(oos_months),
    )

    predictions_list = []
    importance_list = []
    t_start = _time.time()

    for i, test_month in enumerate(oos_months):
        test_idx = all_months.index(test_month)

        if test_idx < train_window:
            continue

        train_start = test_idx - train_window
        train_months_list = all_months[train_start:test_idx]

        train_df = panel[panel["yyyymm"].isin(train_months_list)].copy()
        test_df = panel[panel["yyyymm"] == test_month].copy()

        if len(train_df) < 100 or len(test_df) < 50:
            continue

        # Per-window rank normalization to [0,1]
        for col in features:
            train_df[col] = train_df.groupby("yyyymm")[col].rank(pct=True)
            test_df[col] = test_df.groupby("yyyymm")[col].rank(pct=True)

        # Fill NaN with 0.5 (ElasticNet cannot handle NaN)
        train_df[features] = train_df[features].fillna(0.5)
        test_df[features] = test_df[features].fillna(0.5)

        X_train = train_df[features].values
        y_train = train_df[return_col].values
        X_test = test_df[features].values
        y_test = test_df[return_col].values

        # Drop NaN targets
        valid_train = ~np.isnan(y_train)
        valid_test = ~np.isnan(y_test)
        if valid_train.sum() < 100 or valid_test.sum() < 30:
            continue

        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test, y_test = X_test[valid_test], y_test[valid_test]

        # ElasticNetCV: auto-tunes alpha + l1_ratio via TimeSeriesSplit CV
        from sklearn.model_selection import TimeSeriesSplit
        model = ElasticNetCV(
            l1_ratio=[0.5, 0.7, 0.9, 0.95, 1.0],
            cv=TimeSeriesSplit(n_splits=5),
            max_iter=5000,
            random_state=seed,
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Store predictions
        test_valid = test_df.iloc[valid_test]
        pred_df = pd.DataFrame({
            "permno": test_valid["permno"].values,
            "yyyymm": test_valid["yyyymm"].values,
            "y_true": y_test,
            "y_pred": y_pred,
        })
        predictions_list.append(pred_df)

        # Feature importance = |coefficients|
        imp = pd.Series(np.abs(model.coef_), index=features)
        imp_row = {"yyyymm": test_month}
        imp_row.update(imp.to_dict())
        importance_list.append(imp_row)

        n_done = i + 1
        elapsed = _time.time() - t_start
        avg_per_month = elapsed / n_done
        remaining_min = avg_per_month * (len(oos_months) - n_done) / 60

        if (n_done) % 12 == 0 or i == 0:
            logger.info(
                "Progress: %d/%d months (%.0f%%) | month %d | "
                "%.1f s/month | ETA: %.0f min",
                n_done, len(oos_months), 100 * n_done / len(oos_months),
                test_month,
                avg_per_month, remaining_min,
            )

    if not predictions_list:
        logger.warning("No valid OOS months produced")
        return pd.DataFrame(), pd.DataFrame()

    predictions = pd.concat(predictions_list, ignore_index=True)
    importances = pd.DataFrame(importance_list).set_index("yyyymm")

    logger.info(
        "Rolling ElasticNetCV complete: %d OOS months, %d predictions",
        len(importances), len(predictions),
    )
    return predictions, importances


def expanding_xgboost_predict(
    panel: pd.DataFrame,
    features: list[str],
    config: dict | None = None,
    return_col: str = "excess_ret",
    train_start: int = 199001,
    oos_start: int = 200001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train baseline XGBoost with expanding windows (GKX-style).

    Protocol:
      - Expanding training: starts at train_start, grows by 12 months each year
      - No validation gap; retune via TimeSeriesSplit(5) CV on train each year
      - Test = 12 months (yearly), starting from oos_start
      - Per-window rank normalization to [0,1], fillna(0.5)

    Returns
    -------
    predictions : DataFrame [permno, yyyymm, y_true, y_pred]
    importances : DataFrame rows=yyyymm (test month), cols=features (gain)
    """
    import itertools
    import time as _time
    from src.models import create_model

    if config is None:
        config = load_config()

    seed = config["project"]["seed"]
    all_months = sorted(panel["yyyymm"].unique())

    # Build yearly test blocks: [200001..200012], [200101..200112], ...
    oos_months = [m for m in all_months if m >= oos_start]
    test_years = sorted(set(m // 100 for m in oos_months))

    predictions_list = []
    importance_list = []
    t_start = _time.time()

    logger.info(
        "Expanding XGBoost: %d test years (%d-%d), train starts %d, grid size %d",
        len(test_years), test_years[0], test_years[-1], train_start,
        np.prod([len(v) for v in config["models"]["xgboost"].get("search_space", {}).values()]),
    )

    for yi, test_year in enumerate(test_years):
        test_months_list = [m for m in all_months if m // 100 == test_year]
        if not test_months_list:
            continue

        # Expanding train: train_start up to (but not including) first test month
        first_test = test_months_list[0]
        train_months_list = [m for m in all_months if train_start <= m < first_test]

        if len(train_months_list) < 60:
            continue

        train_df = panel[panel["yyyymm"].isin(train_months_list)].copy()
        test_df = panel[panel["yyyymm"].isin(test_months_list)].copy()

        if len(train_df) < 100 or len(test_df) < 50:
            continue

        # Per-window rank normalization to [0,1]
        for col in features:
            train_df[col] = train_df.groupby("yyyymm")[col].rank(pct=True)
            test_df[col] = test_df.groupby("yyyymm")[col].rank(pct=True)

        train_df[features] = train_df[features].fillna(0.5)
        test_df[features] = test_df[features].fillna(0.5)

        X_train = train_df[features].values
        y_train = train_df[return_col].values
        X_test = test_df[features].values
        y_test = test_df[return_col].values

        valid_train = ~np.isnan(y_train)
        valid_test = ~np.isnan(y_test)
        if valid_train.sum() < 100 or valid_test.sum() < 30:
            continue

        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test, y_test = X_test[valid_test], y_test[valid_test]

        # Retune hyperparameters each year via TimeSeriesSplit CV
        from sklearn.model_selection import TimeSeriesSplit
        xgb_cfg = config["models"]["xgboost"]
        search_space = xgb_cfg.get("search_space", {})
        keys = list(search_space.keys())
        values = list(search_space.values())
        n_trials = xgb_cfg.get("n_random_search", 150)

        full_grid_size = 1
        for v in values:
            full_grid_size *= len(v)

        rng = np.random.RandomState(seed)
        if full_grid_size <= n_trials:
            param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
        else:
            param_grid = []
            for _ in range(n_trials):
                combo = {k: v[rng.randint(len(v))] for k, v in zip(keys, values)}
                param_grid.append(combo)

        tscv = TimeSeriesSplit(n_splits=5)
        best_mse = np.inf
        best_params = None
        for params in param_grid:
            trial_cfg = {**xgb_cfg, **params}
            cv_mses = []
            for tr_idx, va_idx in tscv.split(X_train):
                m = create_model("xgboost", config=trial_cfg, seed=seed)
                m.fit(X_train[tr_idx], y_train[tr_idx])
                preds = m.predict(X_train[va_idx])
                cv_mses.append(np.mean((y_train[va_idx] - preds) ** 2))
            avg_mse = np.mean(cv_mses)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = trial_cfg

        # Train on full expanding train set
        model = create_model("xgboost", config=best_params, seed=seed)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        test_valid = test_df[valid_test]
        pred_df = pd.DataFrame({
            "permno": test_valid["permno"].values,
            "yyyymm": test_valid["yyyymm"].values,
            "y_true": y_test,
            "y_pred": y_pred,
        })
        predictions_list.append(pred_df)

        # Feature importance (gain-based)
        imp = model.get_feature_importance(features)
        for tm in test_months_list:
            imp_row = {"yyyymm": tm}
            imp_row.update(imp.to_dict())
            importance_list.append(imp_row)

        elapsed = _time.time() - t_start
        avg_per_year = elapsed / (yi + 1)
        remaining_min = avg_per_year * (len(test_years) - yi - 1) / 60

        logger.info(
            "Year %d/%d | test %d | train %d-%d (%d months) | "
            "%.1f s/year | ETA: %.0f min",
            yi + 1, len(test_years), test_year,
            train_months_list[0], train_months_list[-1], len(train_months_list),
            avg_per_year, remaining_min,
        )

    if not predictions_list:
        logger.warning("No valid OOS years produced")
        return pd.DataFrame(), pd.DataFrame()

    predictions = pd.concat(predictions_list, ignore_index=True)
    importances = pd.DataFrame(importance_list).set_index("yyyymm")

    logger.info(
        "Expanding XGBoost complete: %d OOS months, %d predictions",
        len(importances), len(predictions),
    )
    return predictions, importances


def expanding_elasticnet_predict(
    panel: pd.DataFrame,
    features: list[str],
    config: dict | None = None,
    return_col: str = "excess_ret",
    train_start: int = 199001,
    oos_start: int = 200001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train ElasticNet with expanding windows (GKX-style).

    Protocol:
      - Expanding training: starts at train_start, grows by 12 months each year
      - No validation gap; retune via TimeSeriesSplit(5) CV on train each year
      - Test = 12 months (yearly), starting from oos_start
      - Per-window rank normalization to [0,1], fillna(0.5)
      - ElasticNetCV auto-tunes alpha + l1_ratio each year

    Returns
    -------
    predictions : DataFrame [permno, yyyymm, y_true, y_pred]
    importances : DataFrame rows=yyyymm (test month), cols=features (|coef|)
    """
    import time as _time
    from sklearn.linear_model import ElasticNetCV

    if config is None:
        from src.config import load_config
        config = load_config()

    seed = config["project"]["seed"]
    all_months = sorted(panel["yyyymm"].unique())

    oos_months = [m for m in all_months if m >= oos_start]
    test_years = sorted(set(m // 100 for m in oos_months))

    predictions_list = []
    importance_list = []
    t_start = _time.time()

    logger.info(
        "Expanding ElasticNetCV: %d test years (%d-%d), train starts %d",
        len(test_years), test_years[0], test_years[-1], train_start,
    )

    for yi, test_year in enumerate(test_years):
        test_months_list = [m for m in all_months if m // 100 == test_year]
        if not test_months_list:
            continue

        first_test = test_months_list[0]
        train_months_list = [m for m in all_months if train_start <= m < first_test]

        if len(train_months_list) < 60:
            continue

        train_df = panel[panel["yyyymm"].isin(train_months_list)].copy()
        test_df = panel[panel["yyyymm"].isin(test_months_list)].copy()

        if len(train_df) < 100 or len(test_df) < 50:
            continue

        # Per-window rank normalization to [0,1]
        for col in features:
            train_df[col] = train_df.groupby("yyyymm")[col].rank(pct=True)
            test_df[col] = test_df.groupby("yyyymm")[col].rank(pct=True)

        train_df[features] = train_df[features].fillna(0.5)
        test_df[features] = test_df[features].fillna(0.5)

        X_train = train_df[features].values
        y_train = train_df[return_col].values
        X_test = test_df[features].values
        y_test = test_df[return_col].values

        valid_train = ~np.isnan(y_train)
        valid_test = ~np.isnan(y_test)
        if valid_train.sum() < 100 or valid_test.sum() < 30:
            continue

        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test, y_test = X_test[valid_test], y_test[valid_test]

        # ElasticNetCV: auto-tunes alpha + l1_ratio via TimeSeriesSplit CV
        from sklearn.model_selection import TimeSeriesSplit
        model = ElasticNetCV(
            l1_ratio=[0.5, 0.7, 0.9, 0.95, 1.0],
            cv=TimeSeriesSplit(n_splits=5),
            max_iter=5000,
            random_state=seed,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        test_valid = test_df.iloc[valid_test]
        pred_df = pd.DataFrame({
            "permno": test_valid["permno"].values,
            "yyyymm": test_valid["yyyymm"].values,
            "y_true": y_test,
            "y_pred": y_pred,
        })
        predictions_list.append(pred_df)

        # Feature importance = |coefficients|
        imp = pd.Series(np.abs(model.coef_), index=features)
        for tm in test_months_list:
            imp_row = {"yyyymm": tm}
            imp_row.update(imp.to_dict())
            importance_list.append(imp_row)

        elapsed = _time.time() - t_start
        avg_per_year = elapsed / (yi + 1)
        remaining_min = avg_per_year * (len(test_years) - yi - 1) / 60

        logger.info(
            "Year %d/%d | test %d | train %d-%d (%d months) | "
            "%.1f s/year | ETA: %.0f min",
            yi + 1, len(test_years), test_year,
            train_months_list[0], train_months_list[-1], len(train_months_list),
            avg_per_year, remaining_min,
        )

    if not predictions_list:
        logger.warning("No valid OOS years produced")
        return pd.DataFrame(), pd.DataFrame()

    predictions = pd.concat(predictions_list, ignore_index=True)
    importances = pd.DataFrame(importance_list).set_index("yyyymm")

    logger.info(
        "Expanding ElasticNetCV complete: %d OOS months, %d predictions",
        len(importances), len(predictions),
    )
    return predictions, importances


def compute_topk_frequency(
    importances: pd.DataFrame,
    K: int = 10,
) -> pd.Series:
    """Fraction of rolling windows in which feature j is among the top K by gain.

    More robust than mean gain because it captures persistent capacity
    allocation rather than being diluted by noisy small splits.

    Parameters
    ----------
    importances : DataFrame with rows=yyyymm, cols=features, values=gain
    K : number of top features to count per window

    Returns
    -------
    Series indexed by feature name, values in [0, 1].
    """
    def _topk_row(row):
        return (row.rank(ascending=False) <= K).astype(float)

    topk_flags = importances.apply(_topk_row, axis=1)
    return topk_flags.mean()


def compute_illiquidity_relatedness_aligned(
    panel: pd.DataFrame,
    features: list[str],
    importances: pd.DataFrame,
    liq_col: str = "liq_dvol_21d",
    train_window: int = 120,
    val_window: int = 12,
) -> pd.Series:
    """Illiquidity-relatedness computed on the same transformed inputs the model saw.

    For each OOS month in importances.index, reconstructs the train window,
    applies the same rank normalization and fillna(0.5), then computes
    Spearman correlation between each transformed feature and liquidity.

    Returns Series: average relatedness per feature across windows.
    """
    from scipy.stats import spearmanr

    all_months = sorted(panel["yyyymm"].unique())
    oos_months = sorted(importances.index.unique())

    corr_list = []
    for test_month in oos_months:
        test_idx = all_months.index(test_month)
        if test_idx < train_window + val_window:
            continue

        train_start = test_idx - train_window - val_window
        val_start = test_idx - val_window
        train_months_list = all_months[train_start:val_start]

        train_df = panel[panel["yyyymm"].isin(train_months_list)].copy()

        # Apply same transformation as the model
        for col in features:
            train_df[col] = train_df.groupby("yyyymm")[col].rank(pct=True)
        train_df[features] = train_df[features].fillna(0.5)

        # Compute relatedness on transformed features
        liq_vals = train_df[liq_col].values
        valid_liq = ~np.isnan(liq_vals)

        row = {}
        for feat in features:
            vals = train_df[feat].values
            valid = valid_liq & ~np.isnan(vals)
            if valid.sum() < 50:
                row[feat] = np.nan
                continue
            rho, _ = spearmanr(vals[valid], liq_vals[valid])
            row[feat] = rho
        corr_list.append(row)

    corr_df = pd.DataFrame(corr_list)
    return corr_df.mean()


def compute_illiquidity_relatedness(
    panel: pd.DataFrame,
    features: list[str],
    liq_col: str = "liq_dvol_21d",
) -> pd.Series:
    """Monthly Spearman correlation of each feature with dollar volume.

    Returns Series: ρ̄_j per feature (averaged across months).
    A large negative ρ̄_j means the feature is associated with illiquid stocks.
    """
    from scipy.stats import spearmanr

    months = sorted(panel["yyyymm"].unique())
    corr_list = []

    for m in months:
        mdf = panel[panel["yyyymm"] == m]
        liq = mdf[liq_col].values
        valid_liq = ~np.isnan(liq)

        row = {}
        for feat in features:
            vals = mdf[feat].values
            valid = valid_liq & ~np.isnan(vals)
            if valid.sum() < 50:
                row[feat] = np.nan
                continue
            rho, _ = spearmanr(vals[valid], liq[valid])
            row[feat] = rho
        corr_list.append(row)

    corr_df = pd.DataFrame(corr_list)
    return corr_df.mean()


def compute_quintile_oos_r2(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    quintile_col: str = "liq_quintile",
) -> pd.DataFrame:
    """Pooled OOS R² per liquidity quintile (Eq. 8 in document).

    r̄_t = full-sample cross-sectional mean (not within-quintile).

    Returns DataFrame with columns:
        quintile, pooled_r2_cs, pooled_r2_zero, avg_monthly_r2, avg_n_month
    """
    pred = predictions.merge(
        panel[["permno", "yyyymm", quintile_col]],
        on=["permno", "yyyymm"],
        how="left",
    )

    # Benchmark: cross-sectional mean r̄_t (Eq. 8 in document, GKX 2020)
    # Computed over the FULL sample each month (not within quintile).
    # R² = 1 - Σ(r - r̂)² / Σ(r - r̄_t)²
    monthly_mean = pred.groupby("yyyymm")["y_true"].mean().rename("r_bar_t")
    pred = pred.merge(monthly_mean, on="yyyymm", how="left")

    results = []
    quintiles = sorted(pred[quintile_col].dropna().unique())

    for q in quintiles:
        qdf = pred[pred[quintile_col] == q].dropna(subset=["y_true", "y_pred"])

        ss_res = ((qdf["y_true"] - qdf["y_pred"]) ** 2).sum()

        # Cross-sectional mean benchmark
        ss_tot_cs = ((qdf["y_true"] - qdf["r_bar_t"]) ** 2).sum()
        r2_cs = 1 - ss_res / ss_tot_cs if ss_tot_cs > 0 else np.nan

        # Zero benchmark
        ss_tot_zero = (qdf["y_true"] ** 2).sum()
        r2_zero = 1 - ss_res / ss_tot_zero if ss_tot_zero > 0 else np.nan

        # Average monthly R²
        monthly_r2_cs = []
        monthly_r2_zero = []
        for _, mdf in qdf.groupby("yyyymm"):
            ss_r = ((mdf["y_true"] - mdf["y_pred"]) ** 2).sum()
            ss_t_cs = ((mdf["y_true"] - mdf["r_bar_t"]) ** 2).sum()
            ss_t_zero = (mdf["y_true"] ** 2).sum()
            if ss_t_cs > 0:
                monthly_r2_cs.append(1 - ss_r / ss_t_cs)
            if ss_t_zero > 0:
                monthly_r2_zero.append(1 - ss_r / ss_t_zero)
        avg_monthly_r2_cs = np.mean(monthly_r2_cs) if monthly_r2_cs else np.nan
        avg_monthly_r2_zero = np.mean(monthly_r2_zero) if monthly_r2_zero else np.nan

        avg_n = qdf.groupby("yyyymm").size().mean()

        results.append({
            "quintile": int(q),
            "pooled_r2_cs": r2_cs,
            "pooled_r2_zero": r2_zero,
            "avg_monthly_r2_cs": avg_monthly_r2_cs,
            "avg_monthly_r2_zero": avg_monthly_r2_zero,
            "avg_n_month": avg_n,
        })

    # Full sample
    ss_res_all = ((pred["y_true"] - pred["y_pred"]) ** 2).sum()
    ss_tot_cs_all = ((pred["y_true"] - pred["r_bar_t"]) ** 2).sum()
    ss_tot_zero_all = (pred["y_true"] ** 2).sum()

    results.append({
        "quintile": "Full",
        "pooled_r2_cs": 1 - ss_res_all / ss_tot_cs_all if ss_tot_cs_all > 0 else np.nan,
        "pooled_r2_zero": 1 - ss_res_all / ss_tot_zero_all if ss_tot_zero_all > 0 else np.nan,
        "avg_monthly_r2_cs": np.nan,
        "avg_monthly_r2_zero": np.nan,
        "avg_n_month": pred.groupby("yyyymm").size().mean(),
    })

    return pd.DataFrame(results)


def compute_monthly_quintile_r2(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    quintile_col: str = "liq_quintile",
) -> pd.DataFrame:
    """Monthly OOS R² per liquidity quintile + full sample.

    Returns DataFrame with columns:
        yyyymm, Q1_cs, Q2_cs, ..., Q5_cs, Full_cs,
                Q1_zero, Q2_zero, ..., Q5_zero, Full_zero
    """
    pred = predictions.merge(
        panel[["permno", "yyyymm", quintile_col]],
        on=["permno", "yyyymm"],
        how="left",
    )
    monthly_mean = pred.groupby("yyyymm")["y_true"].mean().rename("r_bar_t")
    pred = pred.merge(monthly_mean, on="yyyymm", how="left")

    quintiles = sorted(pred[quintile_col].dropna().unique())
    months = sorted(pred["yyyymm"].unique())

    rows = []
    for m in months:
        mdf = pred[pred["yyyymm"] == m].dropna(subset=["y_true", "y_pred"])
        row = {"yyyymm": m}

        for q in quintiles:
            qdf = mdf[mdf[quintile_col] == q]
            if len(qdf) < 10:
                row[f"Q{int(q)}_cs"] = np.nan
                row[f"Q{int(q)}_zero"] = np.nan
                continue
            ss_res = ((qdf["y_true"] - qdf["y_pred"]) ** 2).sum()
            ss_tot_cs = ((qdf["y_true"] - qdf["r_bar_t"]) ** 2).sum()
            ss_tot_zero = (qdf["y_true"] ** 2).sum()
            row[f"Q{int(q)}_cs"] = 1 - ss_res / ss_tot_cs if ss_tot_cs > 0 else np.nan
            row[f"Q{int(q)}_zero"] = 1 - ss_res / ss_tot_zero if ss_tot_zero > 0 else np.nan

        # Full sample
        ss_res_all = ((mdf["y_true"] - mdf["y_pred"]) ** 2).sum()
        ss_tot_cs_all = ((mdf["y_true"] - mdf["r_bar_t"]) ** 2).sum()
        ss_tot_zero_all = (mdf["y_true"] ** 2).sum()
        row["Full_cs"] = 1 - ss_res_all / ss_tot_cs_all if ss_tot_cs_all > 0 else np.nan
        row["Full_zero"] = 1 - ss_res_all / ss_tot_zero_all if ss_tot_zero_all > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compute_utility_weighted_r2(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    w_col: str = "w_tilde",
) -> dict:
    """Utility-weighted R² (Eq. 9 in document).

    R²_w = 1 − Σ w̃(r−r̂)² / Σ w̃(r−r̄_t)²
    Cross-sectional mean benchmark.
    """
    pred = predictions.merge(
        panel[["permno", "yyyymm", w_col]],
        on=["permno", "yyyymm"],
        how="left",
    )
    pred = pred.dropna(subset=["y_true", "y_pred", w_col])

    # Cross-sectional mean per month
    monthly_mean = pred.groupby("yyyymm")["y_true"].mean().rename("r_bar_t")
    pred = pred.merge(monthly_mean, on="yyyymm", how="left")

    w = pred[w_col].values
    r = pred["y_true"].values
    r_hat = pred["y_pred"].values
    r_bar = pred["r_bar_t"].values

    ss_res_w = np.sum(w * (r - r_hat) ** 2)
    ss_res = np.sum((r - r_hat) ** 2)

    # Cross-sectional mean benchmark
    ss_tot_w_cs = np.sum(w * (r - r_bar) ** 2)
    ss_tot_cs = np.sum((r - r_bar) ** 2)
    r2_w_cs = 1 - ss_res_w / ss_tot_w_cs if ss_tot_w_cs > 0 else np.nan
    r2_std_cs = 1 - ss_res / ss_tot_cs if ss_tot_cs > 0 else np.nan

    # Zero benchmark
    ss_tot_w_zero = np.sum(w * r ** 2)
    ss_tot_zero = np.sum(r ** 2)
    r2_w_zero = 1 - ss_res_w / ss_tot_w_zero if ss_tot_w_zero > 0 else np.nan
    r2_std_zero = 1 - ss_res / ss_tot_zero if ss_tot_zero > 0 else np.nan

    return {
        "r2_standard_cs": r2_std_cs,
        "r2_weighted_cs": r2_w_cs,
        "gap_cs": r2_std_cs - r2_w_cs,
        "r2_standard_zero": r2_std_zero,
        "r2_weighted_zero": r2_w_zero,
        "gap_zero": r2_std_zero - r2_w_zero,
    }


def compute_monthly_utility_weighted_r2(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    w_col: str = "w_tilde",
) -> pd.DataFrame:
    """Monthly utility-weighted R² time series.

    Returns DataFrame with columns:
        yyyymm, r2_std_cs, r2_wtd_cs, r2_std_zero, r2_wtd_zero
    """
    pred = predictions.merge(
        panel[["permno", "yyyymm", w_col]],
        on=["permno", "yyyymm"],
        how="left",
    )
    pred = pred.dropna(subset=["y_true", "y_pred", w_col])

    monthly_mean = pred.groupby("yyyymm")["y_true"].mean().rename("r_bar_t")
    pred = pred.merge(monthly_mean, on="yyyymm", how="left")

    rows = []
    for m, mdf in pred.groupby("yyyymm"):
        w = mdf[w_col].values
        r = mdf["y_true"].values
        r_hat = mdf["y_pred"].values
        r_bar = mdf["r_bar_t"].values

        ss_res = np.sum((r - r_hat) ** 2)
        ss_res_w = np.sum(w * (r - r_hat) ** 2)

        ss_tot_cs = np.sum((r - r_bar) ** 2)
        ss_tot_w_cs = np.sum(w * (r - r_bar) ** 2)
        ss_tot_zero = np.sum(r ** 2)
        ss_tot_w_zero = np.sum(w * r ** 2)

        rows.append({
            "yyyymm": m,
            "r2_std_cs": 1 - ss_res / ss_tot_cs if ss_tot_cs > 0 else np.nan,
            "r2_wtd_cs": 1 - ss_res_w / ss_tot_w_cs if ss_tot_w_cs > 0 else np.nan,
            "r2_std_zero": 1 - ss_res / ss_tot_zero if ss_tot_zero > 0 else np.nan,
            "r2_wtd_zero": 1 - ss_res_w / ss_tot_w_zero if ss_tot_w_zero > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def compute_univariate_liquid_r2(
    panel: pd.DataFrame,
    features: list[str],
    quintile_col: str = "liq_quintile",
    return_col: str = "excess_ret",
) -> pd.Series:
    """Per-feature univariate predictive R² among liquid stocks (Q4-Q5).

    Uses FM slope within Q4-Q5, then R² ≈ t²/(t²+T-1).
    """
    liquid = panel[panel[quintile_col].isin([4, 5])].copy()
    months = sorted(liquid["yyyymm"].unique())

    slopes = {feat: [] for feat in features}

    for m in months:
        mdf = liquid[liquid["yyyymm"] == m]
        y = mdf[return_col].values

        for feat in features:
            x = mdf[feat].values
            valid = ~np.isnan(y) & ~np.isnan(x)
            if valid.sum() < 30:
                continue
            xv, yv = x[valid], y[valid]
            with np.errstate(invalid="ignore"):
                xv_dm = xv - xv.mean()
                var_x = np.sum(xv_dm ** 2)
            if var_x < 1e-12 or np.isnan(var_x):
                continue
            beta = np.sum(xv_dm * (yv - yv.mean())) / var_x
            slopes[feat].append(beta)

    r2_dict = {}
    for feat in features:
        s = np.array(slopes[feat])
        s = s[~np.isnan(s)]
        if len(s) < 12:
            r2_dict[feat] = np.nan
            continue
        nw = newey_west_tstat(s, lags=6)
        t = nw["t_stat"]
        T = nw["n_obs"]
        r2_dict[feat] = t ** 2 / (t ** 2 + T - 1)

    return pd.Series(r2_dict)


def plot_importance_vs_illiquidity(
    avg_importance: pd.Series,
    illiq_relatedness: pd.Series,
    focal_features: list[str],
    output_path: str | Path,
) -> float:
    """Scatter: Ī_j (y) vs -ρ̄_j (x). 143 points, label 15 focal."""
    import matplotlib.pyplot as plt
    _set_academic_style()
    from scipy.stats import spearmanr

    common = avg_importance.index.intersection(illiq_relatedness.index)
    imp = avg_importance[common]
    rho_neg = -illiq_relatedness[common]

    valid = ~(imp.isna() | rho_neg.isna())
    imp = imp[valid]
    rho_neg = rho_neg[valid]

    spearman_rho, _ = spearmanr(imp.values, rho_neg.values)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(rho_neg.values, imp.values, s=20, alpha=0.5, color="steelblue")

    for feat in focal_features:
        if feat in imp.index:
            ax.annotate(
                feat,
                (rho_neg[feat], imp[feat]),
                fontsize=7, fontweight="bold", color="red",
                textcoords="offset points", xytext=(4, 4),
            )

    ax.set_xlabel(r"$-\bar{\rho}_j$ (illiquidity-relatedness; higher = more illiquid-stock related)")
    ax.set_ylabel("Ī_j (average XGBoost gain importance)")
    ax.set_title(
        "Feature Importance vs Illiquidity-Relatedness\n"
        f"(Spearman ρ = {spearman_rho:.3f}; N = {len(imp)} features)",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved importance vs illiquidity scatter (ρ=%.3f)", spearman_rho)
    return spearman_rho


def plot_importance_vs_liquid_r2(
    avg_importance: pd.Series,
    liquid_r2: pd.Series,
    focal_features: list[str],
    output_path: str | Path,
) -> None:
    """Scatter: Ī_j (y) vs R²_j(liquid) (x). Label 15 focal."""
    import matplotlib.pyplot as plt
    _set_academic_style()

    common = avg_importance.index.intersection(liquid_r2.index)
    imp = avg_importance[common]
    r2 = liquid_r2[common]

    valid = ~(imp.isna() | r2.isna())
    imp = imp[valid]
    r2 = r2[valid]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(r2.values, imp.values, s=20, alpha=0.5, color="steelblue")

    for feat in focal_features:
        if feat in imp.index:
            ax.annotate(
                feat,
                (r2[feat], imp[feat]),
                fontsize=7, fontweight="bold", color="red",
                textcoords="offset points", xytext=(4, 4),
            )

    ax.set_xlabel(r"$R^2_j$(liquid) — univariate predictive $R^2$ among Q4--Q5 stocks")
    ax.set_ylabel("Ī_j (average XGBoost gain importance)")
    ax.set_title(r"Feature Importance vs Liquid-Stock Predictive $R^2$")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved importance vs liquid R² scatter")


def plot_r2_by_quintile(
    quintile_r2: pd.DataFrame,
    output_path: str | Path,
    r2_col: str = "pooled_r2_cs",
) -> None:
    """Bar chart: OOS R² by liquidity quintile with full-sample reference."""
    import matplotlib.pyplot as plt
    _set_academic_style()

    benchmark_label = "CS-mean" if "cs" in r2_col else "Zero"
    q_data = quintile_r2[quintile_r2["quintile"] != "Full"]
    full_r2 = quintile_r2[quintile_r2["quintile"] == "Full"][r2_col].values[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    bars = ax.bar(
        [f"Q{int(q)}" for q in q_data["quintile"]],
        q_data[r2_col].values * 100,
        color="steelblue", edgecolor="black", linewidth=0.5,
    )
    ax.axhline(
        full_r2 * 100, color="red", linestyle="--", linewidth=1.5,
        label=f"Full sample R² = {full_r2*100:.2f}%",
    )

    ax.set_xlabel("Liquidity Quintile (Q1=illiquid, Q5=liquid)")
    ax.set_ylabel(r"Pooled OOS $R^2$ (%)")
    ax.set_title(rf"Out-of-Sample $R^2$ by Liquidity Quintile ({benchmark_label} benchmark)")
    ax.legend()

    for bar, val in zip(bars, q_data[r2_col].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val*100:.2f}%", ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved R² by quintile bar chart")
