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
from src.data.loader import NON_FEATURE_COLS, normalize_features
from src.evaluation.statistics import newey_west_tstat
from src.weighting import compute_weights

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

MOTIVATION_LIQUIDITY_OPTIONS = {
    "dvol": {
        "scheme": "raw_level",
        "source_col": "liq_dvol_21d",
        "weight_col": "w_tilde_dvol",
        "quintile_col": "liq_dvol_21d",
        "ascending": True,
        "label": "Dollar Volume (21-day)",
    },
    "mcap": {
        "scheme": "raw_level",
        "source_col": "liq_me_raw",
        "weight_col": "w_tilde_mcap",
        "quintile_col": "liq_me_raw",
        "ascending": True,
        "label": "Market Capitalization",
    },
    "tc": {
        "scheme": "tc",
        "source_col": None,
        "weight_col": "w_tilde_tc",
        "quintile_col": "w_tilde_tc",
        "ascending": True,
        "label": "Transaction-cost weight",
        "requires_aum": True,
    },
}


def get_motivation_liquidity_config(name: str) -> dict:
    """Return metadata for one active motivation liquidity option.

    Active choices are ``dvol``, ``mcap``, and computed ``tc``.
    """
    try:
        return MOTIVATION_LIQUIDITY_OPTIONS[name].copy()
    except KeyError as exc:
        choices = ", ".join(MOTIVATION_LIQUIDITY_OPTIONS)
        raise ValueError(
            f"Unknown motivation liquidity option {name!r}. Available: {choices}"
        ) from exc


def get_motivation_liquidity_choices() -> list[str]:
    """Return active motivation liquidity option names for CLI choices."""
    return list(MOTIVATION_LIQUIDITY_OPTIONS.keys())


def get_motivation_liquidity_key(name: str, aum_millions: float | None = None) -> str:
    """Return a stable output-folder key for the selected motivation proxy."""
    if name != "tc":
        return name
    aum_millions = 500.0 if aum_millions is None else float(aum_millions)
    if aum_millions.is_integer():
        aum_label = str(int(aum_millions))
    else:
        aum_label = ("%g" % aum_millions).replace(".", "p")
    return f"tc_{aum_label}m"


def ensure_motivation_weight_column(
    panel: pd.DataFrame,
    name: str,
    config: dict | None = None,
    aum_millions: float | None = None,
) -> str:
    """Ensure the requested motivation weight column exists and return its name.

    For ``dvol`` and ``tc``, this reuses the formal weighting implementation so
    the motivation pipeline and formal experiment share the same formulas and
    missing-value handling. For ``mcap``, this creates mean-one weights from the
    raw market-cap level because market cap is a motivation robustness proxy,
    not a formal training-weight family.
    """
    liq = get_motivation_liquidity_config(name)
    weight_col = liq["weight_col"]
    if weight_col in panel.columns:
        return weight_col
    config = config or load_config()

    if name == "dvol":
        panel[weight_col] = compute_weights(panel, scheme="dolvol", config=config)
        return weight_col

    if liq["scheme"] == "raw_level":
        panel[weight_col] = _compute_raw_level_mean_one_weights(
            panel, liq_col=liq["source_col"]
        )
        return weight_col

    if liq["scheme"] != "tc":
        raise ValueError(f"Unsupported motivation liquidity scheme: {liq['scheme']}")

    aum_millions = 500.0 if aum_millions is None else float(aum_millions)
    aum_dollars = aum_millions * 1_000_000

    tc_cfg = config["transaction_costs"]
    required = [
        f"liq_{tc_cfg['spread_col']}",
        f"liq_{tc_cfg['sigma_col']}",
        f"liq_{tc_cfg['adv_col']}",
    ]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise KeyError(
            "TC motivation weights require columns missing from panel: "
            + ", ".join(missing)
        )

    panel[weight_col] = compute_weights(
        panel, scheme="tc", config=config, aum=aum_dollars
    )
    return weight_col

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
    "Size": ("Liquidity", "Log market capitalization / capacity proxy"),
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
    "size": "Liquidity",
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
    # (other, short sale constraints, ownership, recommendation, info proxy)
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
    """Select the final motivation feature set.

    Start from SignalDoc Clear Predictors, exclude discrete predictors, add the
    CRSP-derived predictors and required focal characteristics, then keep only
    columns available in the panel. The >70% missingness filter is applied later
    in ``scripts/01_process_data.py``.

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


def _compute_raw_level_mean_one_weights(
    panel: pd.DataFrame,
    liq_col: str,
) -> pd.Series:
    """Compute raw-level mean-one weights within each month.

    This is currently used for the market-cap robustness proxy. Formal dvol
    and TC weights should use ``src.weighting.compute_weights`` instead.
    """
    results = []
    for _, group in panel.groupby("yyyymm"):
        vals = group[liq_col].copy()
        median_val = vals.median()
        if pd.isna(median_val):
            median_val = 1.0
        vals = vals.fillna(median_val).clip(lower=1e-8)

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
    sort_col : Column to sort on (e.g., 'liq_dvol_21d' or 'liq_me_raw').
    ascending : If True, higher values = more liquid (dvol, me_raw).
        If False, higher values = less implementable or more costly.
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

    kde_data = {}  # collect KDE curve data for CSV export

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

        # Collect KDE data for CSV
        feat_data = {"x": x_grid, "training_density": np.ones_like(x_grid),
                     "deployment_density_volvw": deploy_y}

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
                feat_data["deployment_density_mcap"] = mcap_y

        kde_data[feat] = feat_data

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

    # No in-figure suptitle: the paper caption names the figure
    # (title_suffix retained in the signature for backward compatibility).
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Export underlying KDE data to CSV
    csv_rows = []
    for feat, data in kde_data.items():
        for i, x_val in enumerate(data["x"]):
            row = {"feature": feat, "x": x_val,
                   "training_density": data["training_density"][i],
                   "deployment_density_volvw": data["deployment_density_volvw"][i]}
            if "deployment_density_mcap" in data:
                row["deployment_density_mcap"] = data["deployment_density_mcap"][i]
            csv_rows.append(row)
    if csv_rows:
        csv_path = Path(output_path).with_suffix(".csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        logger.info("Saved density data to %s", csv_path)

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
    title_suffix : Retained for backward compatibility; no in-figure title
        is drawn (the paper caption names the figure).
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
        vw_raw = panel.groupby("yyyymm")[vw_col].transform(lambda x: x / x.mean())
        vw_valid = vw_raw.dropna()
        vw_valid = vw_valid[vw_valid > 0]
        log_vw = np.log10(vw_valid.values)

        ax.hist(log_vw, bins=80, density=True, color="green", alpha=0.2,
                edgecolor="green", linewidth=1.5, histtype="stepfilled",
                label="Value weights (market cap)")

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
    # Percentile box in the top-right corner, above the low right tail
    # (the clipped x-axis leaves that corner empty).
    ax.text(
        0.98, 0.95, pct_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Clip the view to where the data lives: the near-zero tail (~0.2% of
    # mass) otherwise stretches the axis to log10(w) < -14 and leaves most
    # of the panel empty. The omitted tail is quantified in the caption.
    ax.set_xlim(-6, 3)
    ax.set_xlabel(r"$\log_{10}(\tilde{w})$")
    ax.set_ylabel("Density")
    # No in-figure title: the LaTeX caption names the figure (fig:weight_dist).
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Export underlying data to CSV
    csv_data = {"percentile": pcts,
                "dv_weight": list(pct_vals)}
    if vw_col is not None and vw_col in panel.columns:
        csv_data["vw_weight"] = list(vw_pcts)
    csv_path = Path(output_path).with_suffix(".csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    logger.info("Saved weight distribution data to %s", csv_path)

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

    Missing feature values are filled with the neutral rank 0.5, matching the
    ML pipeline convention. Rows are dropped only for missing returns.

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
    n_regressors = 1 + len(focal_features)
    min_obs = max(30, 2 * n_regressors)

    # Collect monthly coefficients per quintile
    monthly_coefs = {q: [] for q in quintiles}

    for m in months:
        mdf = panel[panel["yyyymm"] == m]

        for q in quintiles:
            qdf = mdf[mdf[quintile_col] == q]
            y = qdf[return_col].values
            X_df = qdf[focal_features].copy()

            valid = ~np.isnan(y)
            if valid.sum() < min_obs:
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


def _joint_gamma_f_test(
    gamma_df: pd.DataFrame,
    features: list[str],
    min_months: int = 12,
) -> tuple[float, float, tuple[int, int]]:
    """Hotelling-style joint test that the mean gamma vector equals zero.

    This implements the time-series test described as regressing the monthly
    gamma-vector estimates on a constant and testing that the constant vector
    is zero:

        T² = T * g_bar' S^{-1} g_bar
        F  = ((T - k) / (k * (T - 1))) * T²  ~  F(k, T-k)
    """
    gamma_mat = gamma_df[features].dropna()
    T = len(gamma_mat)
    k = len(features)
    df = (k, T - k)

    if T < min_months or T <= k:
        return np.nan, np.nan, df

    gamma_means = gamma_mat.mean().values
    cov_mat = gamma_mat.cov().values
    try:
        cov_inv = np.linalg.inv(cov_mat)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, df

    hotelling_t2 = float(T * gamma_means @ cov_inv @ gamma_means)
    f_stat = float((T - k) / (k * (T - 1)) * hotelling_t2)

    from scipy.stats import f as f_dist

    f_pvalue = float(1 - f_dist.cdf(f_stat, k, T - k))
    return f_stat, f_pvalue, df


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
    n_regressors = 1 + 2 * n_features
    min_obs = max(100, 2 * n_regressors)

    for m in months:
        mdf = panel[panel["yyyymm"] == m].copy()

        y = mdf[return_col].values
        L = mdf[liq_col].values
        valid = ~np.isnan(y) & ~np.isnan(L)

        if use_dummy:
            L = (L > np.nanmedian(L)).astype(float)

        # Build X = [1, x_1..x_p, x_1*L..x_p*L].
        # Missing features use neutral rank 0.5, as in the ML pipeline.
        X_main = mdf[focal_features].fillna(0.5).values
        X_interact = X_main * L[:, np.newaxis]

        if valid.sum() < min_obs:
            continue

        y_v = y[valid]
        X_m = X_main[valid]
        X_i = X_interact[valid]
        X_full = np.column_stack([np.ones(valid.sum()), X_m, X_i])

        try:
            beta_all, _, _, _ = np.linalg.lstsq(X_full, y_v, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Extract: beta_all[0]=intercept, then p main effects and p interactions.
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

    # Joint F-test: H0 that the time-series mean gamma vector is zero.
    f_stat, f_pvalue, f_df = _joint_gamma_f_test(gamma_df, focal_features)

    return {
        "coef_table": coef_table,
        "f_test_stat": f_stat,
        "f_test_pvalue": f_pvalue,
        "f_test_df": f_df,
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

    # Export underlying scatter data to CSV
    df["spearman_rho"] = rho
    df["spearman_pval"] = p_val
    csv_path = Path(output_path).with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved divergence vs heterogeneity data to %s", csv_path)

    logger.info("Saved divergence vs heterogeneity scatter to %s (ρ=%.3f)", output_path, rho)
    return rho


# ═════════════════════════════════════════════════════════════
# Step 3: Standard ML Is Affected
# ═════════════════════════════════════════════════════════════


def _renormalize_weights_by_month(
    weights: pd.Series,
    yyyymm: pd.Series,
) -> pd.Series:
    """Rescale ``weights`` so each ``yyyymm`` cross-section has mean 1.0.

    Used after restricted-universe filtering: the input weights were normalised
    to mean=1 on the full universe; after filtering, the surviving rows may no
    longer average to 1 within a month. Replace NaNs with 1.0 first, then
    divide by the per-month mean. Zero-mean months (very rare) fall back to 1.0
    to avoid division by zero.
    """
    w = weights.copy()
    w = w.fillna(1.0)
    monthly_mean = w.groupby(yyyymm.values).transform("mean")
    monthly_mean = monthly_mean.where(monthly_mean > 0, 1.0)
    return w / monthly_mean


def _rolling_model_filter_core(
    panel: pd.DataFrame,
    features: list[str],
    model_name: str,
    config: dict,
    return_col: str,
    train_filter_fn,
    test_filter_fn,
    baseline_tuned_params: pd.DataFrame | None = None,
    rerank_after_filter: bool = True,
    label: str | None = None,
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Shared rolling-window core for filtered-universe motivation models.

    Parameters
    ----------
    train_filter_fn : callable(train_df, val_df, all_months, train_months, val_months)
        Returns filtered (train_df, val_df).
    test_filter_fn : callable(test_df)
        Returns filtered test_df.
    baseline_tuned_params : DataFrame indexed by yyyymm with per-window params
        from the baseline model. If provided, uses baseline's tuned params for each
        test month. If omitted, the model retunes inside the filtered universe.
    rerank_after_filter
        If True, re-rank-normalize features after applying the restricted
        universe filter. If False, keep the already processed full-cross-section
        ranks from ``processed_panel.parquet``.
    weights : pd.Series | None
        Optional per-row sample weights aligned to ``panel.index``. When
        provided, they are propagated through the restricted universe filter
        and renormalised within each surviving month so that the per-cross-section
        mean is 1.0 (matching the formal weighted-training convention). The
        resulting array is passed to ``model.fit(sample_weight=...)``. When
        ``None`` (default), the model fits unweighted exactly as before.
    """
    import time as _time
    from src.models import create_model

    label = label or model_name
    train_cfg = config["training"]
    train_window = train_cfg["train_window"]
    val_window = train_cfg["validation_window"]
    retune_freq = train_cfg.get("retune_frequency", 12)
    oos_start = train_cfg.get("oos_start", 200001)
    oos_end = train_cfg.get("oos_end", 999999)
    seed = config["project"]["seed"]

    all_months = sorted(panel["yyyymm"].unique())
    oos_months = [m for m in all_months if oos_start <= m <= oos_end]

    predictions_list = []
    best_params = None
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

        # Filter first to the restricted/quintile universe. The caller chooses
        # whether to re-rank features inside that filtered universe or keep the
        # already processed full-cross-section ranks.
        train_df, val_df = train_filter_fn(
            train_df, val_df, all_months, train_months_list, val_months_list
        )
        test_df = test_filter_fn(test_df)

        if len(train_df) < 100 or len(test_df) < 20:
            continue

        if rerank_after_filter:
            train_df = normalize_features(train_df, features)
            val_df = normalize_features(val_df, features)
            test_df = normalize_features(test_df, features)

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
        if valid_train.sum() < 100 or valid_val.sum() < 10 or valid_test.sum() < 10:
            continue

        # Carry weights through the same filter + valid-mask pipeline. When
        # ``weights`` is None we leave w_train/w_val as None so the original
        # unweighted code path is exercised byte-for-byte.
        w_train_arr = None
        w_val_arr = None
        if weights is not None:
            w_train_series = weights.reindex(train_df.index)
            w_val_series = weights.reindex(val_df.index)
            # Renormalise within each surviving month so per-cross-section
            # mean is 1.0 in the filtered universe (matches the formal
            # weighted-training convention of normalising per yyyymm).
            w_train_series = _renormalize_weights_by_month(
                w_train_series, train_df["yyyymm"]
            )
            w_val_series = _renormalize_weights_by_month(
                w_val_series, val_df["yyyymm"]
            )
            w_train_arr = w_train_series.values

        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_val, y_val = X_val[valid_val], y_val[valid_val]
        X_test, y_test = X_test[valid_test], y_test[valid_test]
        if w_train_arr is not None:
            w_train_arr = w_train_arr[valid_train]
            w_val_arr = w_val_series.values[valid_val]

        # Determine params. Baseline tuned files only record retune months, so
        # carry the most recent baseline parameter row forward between tuning
        # dates. If no baseline params are supplied, retune inside the filtered
        # universe.
        if baseline_tuned_params is not None:
            available_months = baseline_tuned_params.index[
                baseline_tuned_params.index <= test_month
            ]
            if len(available_months) == 0:
                continue
            row = baseline_tuned_params.loc[available_months.max()]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            best_params = {}
            for k, v in row.to_dict().items():
                if k == "yyyymm" or v is None:
                    continue
                if isinstance(v, (float, np.floating)) and np.isnan(v):
                    continue
                best_params[k] = v
        elif months_since_retune >= retune_freq:
            tuning_model = create_model(model_name, seed=seed)
            best_params = tuning_model.tune_hyperparameters(
                X_train, y_train, X_val, y_val,
                sample_weight=w_train_arr,
                sample_weight_val=w_val_arr if w_train_arr is not None else None,
            )
            months_since_retune = 0
            logger.info(
                "%s month %d: RETUNED", label, test_month,
            )

        model = create_model(model_name, config=best_params, seed=seed)
        model.fit(X_train, y_train, sample_weight=w_train_arr)
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
                " [RETUNED]" if months_since_retune == 1 and baseline_tuned_params is None else "",
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


def rolling_model_predict_restricted(
    panel: pd.DataFrame,
    features: list[str],
    min_quintile: int,
    model_name: str = "xgboost",
    quintile_col: str = "liq_quintile",
    config: dict | None = None,
    return_col: str = "excess_ret",
    baseline_tuned_params: pd.DataFrame | None = None,
    rerank_after_filter: bool = True,
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Train a model on a restricted universe (drop illiquid quintiles).

    Per Section 5.2(d): filters train/val to quintile >= min_quintile.
    Test set includes ALL stocks (no filter). The caller decides whether to use
    baseline tuned params or retune inside the restricted universe.

    Quintile assignment uses the last month of the training window.

    When ``weights`` is provided, the same series is propagated through the
    restricted-universe filter and renormalised within each surviving month so
    the per-cross-section mean is 1.0 in the filtered universe. The resulting
    array is passed to ``model.fit(sample_weight=...)`` so the restricted
    model is fit with the requested implementability weighting (e.g.,
    softmax_rank or tc weights). When ``weights`` is None the function fits
    unweighted exactly as before.
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

    return _rolling_model_filter_core(
        panel, features, model_name, config, return_col,
        train_filter_fn=train_filter,
        test_filter_fn=test_filter,
        baseline_tuned_params=baseline_tuned_params,
        rerank_after_filter=rerank_after_filter,
        label=f"{model_name}_MQ{min_quintile}+",
        weights=weights,
    )


def rolling_model_predict_quintile(
    panel: pd.DataFrame,
    features: list[str],
    quintile: int,
    model_name: str = "xgboost",
    quintile_col: str = "liq_quintile",
    config: dict | None = None,
    return_col: str = "excess_ret",
    baseline_tuned_params: pd.DataFrame | None = None,
    rerank_after_filter: bool = True,
) -> pd.DataFrame:
    """Train a model on a single liquidity quintile.

    Per Section 5.2(e): filters train/val/test to the specified quintile.
    The caller decides whether to use baseline tuned params or retune inside
    the quintile.

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

    return _rolling_model_filter_core(
        panel, features, model_name, config, return_col,
        train_filter_fn=train_filter,
        test_filter_fn=test_filter,
        baseline_tuned_params=baseline_tuned_params,
        rerank_after_filter=rerank_after_filter,
        label=f"{model_name}_Q{quintile}",
    )


def compute_illiquidity_relatedness(
    panel: pd.DataFrame,
    features: list[str],
    liq_col: str = "liq_dvol_21d",
) -> pd.Series:
    """Monthly Spearman correlation with the selected liquidity sort variable.

    Returns Series: ρ̄_j per feature (averaged across months).
    Step 3 passes rank-normalized model features and raw liquidity levels. This
    is intentional: Spearman correlation re-ranks both variables within the
    valid monthly cross-section, so units do not affect the statistic.
    A large negative ρ̄_j means the feature is associated with less liquid or
    less implementable stocks when the liquidity variable is oriented so higher
    values are more liquid/implementable.
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


def attach_quintile_benchmarks(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    quintile_col: str = "liq_quintile",
    return_col: str = "excess_ret",
    hist_window: int | None = None,
    min_hist_periods: int = 12,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Attach quintiles, true returns, and OOS benchmarks to predictions.

    Accepts either formal prediction schema (``prediction``) or Step 3 schema
    (``y_pred``/``y_true``). The cross-sectional benchmark is the full predicted
    sample mean in each month, not the within-quintile mean.
    """
    pred = predictions.copy()
    if "y_pred" not in pred.columns:
        if "prediction" not in pred.columns:
            raise ValueError("predictions must contain either 'y_pred' or 'prediction'")
        pred = pred.rename(columns={"prediction": "y_pred"})

    merge_cols = ["permno", "yyyymm", quintile_col]
    if "y_true" not in pred.columns:
        merge_cols.append(return_col)
    if extra_cols:
        merge_cols.extend(c for c in extra_cols if c not in merge_cols)

    meta = panel[merge_cols].drop_duplicates(["permno", "yyyymm"])
    pred = pred.merge(meta, on=["permno", "yyyymm"], how="left")
    if "y_true" not in pred.columns:
        pred = pred.rename(columns={return_col: "y_true"})

    r_cs = pred.groupby("yyyymm")["y_true"].mean().rename("r_cs")
    pred = pred.merge(r_cs, on="yyyymm", how="left")
    pred["r_bar_t"] = pred["r_cs"]

    if hist_window is not None:
        effective_min_periods = min(min_hist_periods, hist_window)
        full_ret = panel[["permno", "yyyymm", return_col]].sort_values(
            ["permno", "yyyymm"]
        )
        full_ret["r_hist"] = (
            full_ret.groupby("permno")[return_col]
            .rolling(hist_window, min_periods=effective_min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
        full_ret["r_hist"] = full_ret.groupby("permno")["r_hist"].shift(1)
        pred = pred.merge(
            full_ret[["permno", "yyyymm", "r_hist"]],
            on=["permno", "yyyymm"],
            how="left",
        )

    return pred


def compute_quintile_oos_r2(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    quintile_col: str = "liq_quintile",
    return_col: str = "excess_ret",
    hist_window: int | None = None,
    min_hist_periods: int = 12,
    pooled_quintile_groups: dict[str, list[int | float]] | None = None,
) -> pd.DataFrame:
    """Pooled unweighted OOS R² per liquidity quintile.

    Computes zero-return and full-sample cross-sectional-mean benchmarks. If
    ``hist_window`` is provided, also computes stock-level rolling historical
    mean benchmark using information through t-1. The squared-error sums are
    unweighted; this is appropriate for comparing standard and weighted models
    under the same evaluation loss. Utility-weighted R² is computed separately.

    If ``pooled_quintile_groups`` is provided, each entry adds one extra pooled
    row. For example ``{"Q4-Q5": [4, 5]}`` evaluates liquid stocks jointly.

    Returns DataFrame with columns:
        quintile, pooled_r2_zero, pooled_r2_cs, optional pooled_r2_hist,
        average monthly versions, and avg_n_month.
    """
    pred = attach_quintile_benchmarks(
        predictions,
        panel,
        quintile_col=quintile_col,
        return_col=return_col,
        hist_window=hist_window,
        min_hist_periods=min_hist_periods,
    )

    def _compute_row(qdf: pd.DataFrame, label) -> dict:
        qdf = qdf.dropna(subset=["y_true", "y_pred"])

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
        monthly_r2_hist = []
        for _, mdf in qdf.groupby("yyyymm"):
            ss_r = ((mdf["y_true"] - mdf["y_pred"]) ** 2).sum()
            ss_t_cs = ((mdf["y_true"] - mdf["r_bar_t"]) ** 2).sum()
            ss_t_zero = (mdf["y_true"] ** 2).sum()
            if ss_t_cs > 0:
                monthly_r2_cs.append(1 - ss_r / ss_t_cs)
            if ss_t_zero > 0:
                monthly_r2_zero.append(1 - ss_r / ss_t_zero)
            if "r_hist" in mdf.columns:
                mdf_hist = mdf.dropna(subset=["r_hist"])
                ss_r_hist = ((mdf_hist["y_true"] - mdf_hist["y_pred"]) ** 2).sum()
                ss_t_hist = ((mdf_hist["y_true"] - mdf_hist["r_hist"]) ** 2).sum()
                if ss_t_hist > 0:
                    monthly_r2_hist.append(1 - ss_r_hist / ss_t_hist)
        avg_monthly_r2_cs = np.mean(monthly_r2_cs) if monthly_r2_cs else np.nan
        avg_monthly_r2_zero = np.mean(monthly_r2_zero) if monthly_r2_zero else np.nan
        avg_monthly_r2_hist = np.mean(monthly_r2_hist) if monthly_r2_hist else np.nan

        avg_n = qdf.groupby("yyyymm").size().mean()

        row = {
            "quintile": label,
            "pooled_r2_zero": r2_zero,
            "pooled_r2_cs": r2_cs,
            "avg_monthly_r2_zero": avg_monthly_r2_zero,
            "avg_monthly_r2_cs": avg_monthly_r2_cs,
            "avg_n_month": avg_n,
            "n_obs": len(qdf),
        }
        if "r_hist" in qdf.columns:
            qdf_hist = qdf.dropna(subset=["r_hist"])
            ss_res_hist = ((qdf_hist["y_true"] - qdf_hist["y_pred"]) ** 2).sum()
            ss_tot_hist = ((qdf_hist["y_true"] - qdf_hist["r_hist"]) ** 2).sum()
            row["pooled_r2_hist"] = (
                1 - ss_res_hist / ss_tot_hist if ss_tot_hist > 0 else np.nan
            )
            row["avg_monthly_r2_hist"] = avg_monthly_r2_hist
        return row

    results = []
    quintiles = sorted(pred[quintile_col].dropna().unique())

    for q in quintiles:
        qdf = pred[pred[quintile_col] == q]
        results.append(_compute_row(qdf, int(q)))

    for label, group_values in (pooled_quintile_groups or {}).items():
        qdf = pred[pred[quintile_col].isin(group_values)]
        results.append(_compute_row(qdf, label))

    # Full sample
    pred_valid = pred.dropna(subset=["y_true", "y_pred"])
    ss_res_all = ((pred_valid["y_true"] - pred_valid["y_pred"]) ** 2).sum()
    ss_tot_cs_all = ((pred_valid["y_true"] - pred_valid["r_bar_t"]) ** 2).sum()
    ss_tot_zero_all = (pred_valid["y_true"] ** 2).sum()

    full_row = {
        "quintile": "Full",
        "pooled_r2_zero": 1 - ss_res_all / ss_tot_zero_all if ss_tot_zero_all > 0 else np.nan,
        "pooled_r2_cs": 1 - ss_res_all / ss_tot_cs_all if ss_tot_cs_all > 0 else np.nan,
        "avg_monthly_r2_zero": np.nan,
        "avg_monthly_r2_cs": np.nan,
        "avg_n_month": pred_valid.groupby("yyyymm").size().mean(),
        "n_obs": len(pred_valid),
    }
    if "r_hist" in pred_valid.columns:
        pred_hist = pred_valid.dropna(subset=["r_hist"])
        ss_res_hist_all = ((pred_hist["y_true"] - pred_hist["y_pred"]) ** 2).sum()
        ss_tot_hist_all = ((pred_hist["y_true"] - pred_hist["r_hist"]) ** 2).sum()
        full_row["pooled_r2_hist"] = (
            1 - ss_res_hist_all / ss_tot_hist_all if ss_tot_hist_all > 0 else np.nan
        )
        full_row["avg_monthly_r2_hist"] = np.nan
    results.append(full_row)

    return pd.DataFrame(results)


def compute_utility_weighted_r2(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    w_col: str = "w_tilde",
) -> dict:
    """Zero-benchmark utility-weighted evaluation R² for one prediction set.

    This function does not compare standard-trained versus weighted-trained
    models. It takes one set of predictions and evaluates it twice: once with
    unweighted squared-error sums and once with utility/implementability weights.

    The benchmark is zero, matching the motivation document:

        R²_w = 1 − Σ w̃(r−r̂)² / Σ w̃r²
    """
    pred = predictions.merge(
        panel[["permno", "yyyymm", w_col]],
        on=["permno", "yyyymm"],
        how="left",
    )
    pred = pred.dropna(subset=["y_true", "y_pred", w_col])

    w = pred[w_col].values
    r = pred["y_true"].values
    r_hat = pred["y_pred"].values

    ss_res_w = np.sum(w * (r - r_hat) ** 2)
    ss_res = np.sum((r - r_hat) ** 2)

    ss_tot_w_zero = np.sum(w * r ** 2)
    ss_tot_zero = np.sum(r ** 2)
    r2_w_zero = 1 - ss_res_w / ss_tot_w_zero if ss_tot_w_zero > 0 else np.nan
    r2_std_zero = 1 - ss_res / ss_tot_zero if ss_tot_zero > 0 else np.nan

    return {
        "r2_standard_zero": r2_std_zero,
        "r2_weighted_zero": r2_w_zero,
        "gap_zero": r2_std_zero - r2_w_zero,
    }


def compute_univariate_liquid_r2(
    panel: pd.DataFrame,
    features: list[str],
    quintile_col: str = "liq_quintile",
    return_col: str = "excess_ret",
) -> pd.Series:
    """Per-feature univariate predictive strength among liquid stocks (Q4-Q5).

    The caller controls feature scaling. In Step 3, the supplied panel has the
    final feature set rank-normalized to [0, 1], matching the formal rolling
    model inputs.
    For each feature, estimate monthly univariate FM slopes within Q4-Q5, then
    summarize the Newey-West t-statistic as R² ≈ t²/(t²+T-1).
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
    importance_label: str = "average model importance",
) -> float:
    """Scatter: Ī_j (y) vs -ρ̄_j (x). Label focal characteristics."""
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
    ax.set_ylabel(f"Ī_j ({importance_label})")
    ax.set_title(
        "Feature Importance vs Illiquidity-Relatedness\n"
        f"(Spearman ρ = {spearman_rho:.3f}; N = {len(imp)} features)",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Export underlying scatter data to CSV
    scatter_df = pd.DataFrame({
        "feature": imp.index,
        "avg_importance": imp.values,
        "neg_illiq_relatedness": rho_neg.values,
    })
    scatter_df["is_focal"] = scatter_df["feature"].isin(focal_features)
    scatter_df["spearman_rho"] = spearman_rho
    csv_path = Path(output_path).with_suffix(".csv")
    scatter_df.to_csv(csv_path, index=False)
    logger.info("Saved importance vs illiquidity data to %s", csv_path)

    logger.info("Saved importance vs illiquidity scatter (ρ=%.3f)", spearman_rho)
    return spearman_rho


def plot_importance_vs_liquid_r2(
    avg_importance: pd.Series,
    liquid_r2: pd.Series,
    focal_features: list[str],
    output_path: str | Path,
    importance_label: str = "average model importance",
) -> None:
    """Scatter: Ī_j (y) vs R²_j(liquid) (x). Label focal characteristics."""
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

    from scipy.stats import spearmanr as _spearmanr
    spearman_rho, _ = _spearmanr(imp.values, r2.values)

    ax.set_xlabel(r"$R^2_j$(liquid) — univariate predictive $R^2$ among Q4--Q5 stocks")
    ax.set_ylabel(f"Ī_j ({importance_label})")
    ax.set_title(
        r"Feature Importance vs Liquid-Stock Predictive $R^2$"
        f"\n(Spearman ρ = {spearman_rho:.3f}; N = {len(imp)} features)"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Export underlying scatter data to CSV
    scatter_df = pd.DataFrame({
        "feature": imp.index,
        "avg_importance": imp.values,
        "liquid_r2": r2.values,
    })
    scatter_df["is_focal"] = scatter_df["feature"].isin(focal_features)
    scatter_df["spearman_rho"] = spearman_rho
    csv_path = Path(output_path).with_suffix(".csv")
    scatter_df.to_csv(csv_path, index=False)
    logger.info("Saved importance vs liquid R² data to %s", csv_path)

    logger.info("Saved importance vs liquid R² scatter")


def plot_r2_by_quintile(
    quintile_r2: pd.DataFrame,
    output_path: str | Path,
    r2_col: str = "pooled_r2_cs",
) -> None:
    """Bar chart: OOS R² by liquidity quintile with full-sample reference."""
    import matplotlib.pyplot as plt
    _set_academic_style()

    benchmark_label = "CS-mean" if "cs" in r2_col else ("Historical mean" if "hist" in r2_col else "Zero")
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
