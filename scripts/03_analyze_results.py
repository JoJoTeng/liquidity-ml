"""
03_analyze_results.py — Tables, Figures, and Hypothesis Tests (H1–H4)
=====================================================================
Final script in the build order. Loads experiment results saved by
02_run_experiment.py and generates:

  - Publication-ready tables (CSV + LaTeX)
  - Figures (PNG)
  - Consolidated hypothesis test summary (JSON)

Hypotheses (proposal Section 5):
  H1 (Training Dominance):   Training effect > Portfolio effect
  H2 (Feature Reallocation): SHAP shifts toward tradable features
  H3 (Sharpe Improvement):   SR(2B) − SR(1A) ≥ 0.20
  H4 (Capacity Improvement): Break-even AUM increases ≥ 25×

Usage:
  python scripts/03_analyze_results.py              # full analysis
  python scripts/03_analyze_results.py --no-figures  # tables only
  python scripts/03_analyze_results.py --model xgboost
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_output_dir
from src.evaluation.statistics import bootstrap_sharpe_test, sharpe_ratio
from src.analysis.feature_importance import (
    load_importance,
    cross_model_h2_summary,
    test_h2_group_shift,
    test_h2_per_feature,
    plot_importance_comparison,
    plot_importance_over_time,
    plot_ratio_over_time,
)

logger = logging.getLogger(__name__)


# ── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze experiment results: tables, figures, hypothesis tests."
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation (faster).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Analyze a single model only (e.g. xgboost).",
    )
    parser.add_argument(
        "--importance-type", type=str, default="shap",
        choices=["shap", "gain"],
        help="Feature importance type (default: shap, falls back to gain).",
    )
    return parser.parse_args()


# ── Loading ──────────────────────────────────────────────────────


def _load_model_results(
    model_name: str, experiment_dir: Path
) -> dict:
    """Load all saved outputs for a single model."""
    model_dir = experiment_dir / model_name

    # Effect decomposition
    with open(model_dir / "effect_decomposition.json") as f:
        decomp = json.load(f)

    # OOS R²
    with open(model_dir / "oos_r2.json") as f:
        oos_r2 = json.load(f)

    # Gross returns per cell
    gross = {}
    for cell in ["1A", "1B", "2A", "2B"]:
        gross[cell] = pd.read_csv(model_dir / f"gross_returns_{cell}.csv")

    # Net returns per cell
    net = {}
    for cell in ["1A", "1B", "2A", "2B"]:
        net[cell] = pd.read_csv(model_dir / f"net_returns_{cell}.csv")

    # Monthly OOS R²
    r2_monthly_path = model_dir / "oos_r2_monthly.csv"
    oos_r2_monthly = (
        pd.read_csv(r2_monthly_path) if r2_monthly_path.exists() else None
    )

    return {
        "model": model_name,
        "effect_decomposition": decomp,
        "oos_r2": oos_r2,
        "oos_r2_monthly": oos_r2_monthly,
        "gross_returns": gross,
        "net_returns": net,
    }


# ── Table Builders ───────────────────────────────────────────────


def build_main_results_table(
    all_results: dict[str, dict],
    summary_df: pd.DataFrame | None = None,
    net_tests: dict[str, dict] | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Build results table: SR per cell, effects, H1/H3 p-values (net primary, gross secondary)."""
    # Determine primary AUM label for net results
    primary_aum = _get_aum_labels(config)[1] if config else "500M"

    rows = []
    for model_name, res in all_results.items():
        decomp = res["effect_decomposition"]
        sharpes = decomp["sharpe_ratios"]

        training = decomp["training_effect"]
        portfolio = decomp["portfolio_effect"]
        total = decomp["total_effect"]
        interaction = decomp["interaction"]

        # Gross H1/H3
        training_share = training / total if total != 0 else np.nan
        h1_gross_pval = decomp["lw_h3"]["p_value"]
        h3_gross_pval = decomp["lw_total"]["p_value"]

        row = {
            "model": model_name,
            # Gross Sharpe ratios and effects
            "SR_1A": sharpes["1A"],
            "SR_1B": sharpes["1B"],
            "SR_2A": sharpes["2A"],
            "SR_2B": sharpes["2B"],
            "training_effect": training,
            "portfolio_effect": portfolio,
            "total_effect": total,
            "interaction": interaction,
            "training_share": training_share,
            "h1_gross_pval": h1_gross_pval,
            "h3_gross_total_effect": total,
            "h3_gross_pval": h3_gross_pval,
        }

        # Net primary H1/H3 at primary AUM
        if net_tests and model_name in net_tests:
            aum_tests = net_tests[model_name].get(primary_aum, {})
            if aum_tests:
                row["net_SR_1A"] = aum_tests["sharpe_ratios"]["1A"]
                row["net_SR_1B"] = aum_tests["sharpe_ratios"]["1B"]
                row["net_SR_2A"] = aum_tests["sharpe_ratios"]["2A"]
                row["net_SR_2B"] = aum_tests["sharpe_ratios"]["2B"]
                row["net_training_effect"] = aum_tests["training_effect"]
                row["net_portfolio_effect"] = aum_tests["portfolio_effect"]
                row["net_total_effect"] = aum_tests["total_effect"]
                row["net_interaction"] = aum_tests["interaction"]
                row["net_training_share"] = (
                    aum_tests["training_share"]
                    if aum_tests["training_share"] is not None else np.nan
                )
                row["h1_net_pval"] = aum_tests["lw_h3"]["p_value"]
                row["h3_net_total_effect"] = aum_tests["total_effect"]
                row["h3_net_pval"] = aum_tests["lw_total"]["p_value"]

        rows.append(row)

    # Ensemble row from summary.csv
    if summary_df is not None and len(summary_df) > 1:
        ens = {
            "model": "ensemble",
            "SR_1A": summary_df["SR_1A"].mean(),
            "SR_1B": summary_df["SR_1B"].mean(),
            "SR_2A": summary_df["SR_2A"].mean(),
            "SR_2B": summary_df["SR_2B"].mean(),
            "training_effect": summary_df["training_effect"].mean(),
            "portfolio_effect": summary_df["portfolio_effect"].mean(),
            "total_effect": summary_df["total_effect"].mean(),
            "interaction": summary_df["interaction"].mean(),
        }
        ens["training_share"] = (
            ens["training_effect"] / ens["total_effect"]
            if ens["total_effect"] != 0 else np.nan
        )
        ens["h1_gross_pval"] = np.nan
        ens["h3_gross_total_effect"] = ens["total_effect"]
        ens["h3_gross_pval"] = np.nan
        rows.append(ens)

    return pd.DataFrame(rows)


def _get_aum_labels(config: dict) -> list[str]:
    """Convert AUM integers to labels like '100M', '500M', '1B', '5B'."""
    labels = []
    for aum in config["transaction_costs"]["aum_scenarios"]:
        if aum < 1_000_000_000:
            labels.append(f"{aum // 1_000_000}M")
        else:
            labels.append(f"{aum // 1_000_000_000}B")
    return labels


def build_net_results_table(
    all_results: dict[str, dict],
    config: dict,
    net_tests: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Build net results table: net SR per AUM scenario + net effects + p-values."""
    aum_labels = _get_aum_labels(config)
    cells = ["1A", "1B", "2A", "2B"]
    rows = []

    for model_name, res in all_results.items():
        row: dict[str, object] = {"model": model_name}

        for aum_label in aum_labels:
            col = f"ret_ls_net_{aum_label}"
            for cell in cells:
                net_df = res["net_returns"][cell]
                if col in net_df.columns:
                    sr = sharpe_ratio(net_df[col].dropna())
                else:
                    sr = np.nan
                row[f"net_SR_{cell}_{aum_label}"] = sr

            # Net effect decomposition at this AUM
            sr_1a = row.get(f"net_SR_1A_{aum_label}", np.nan)
            sr_1b = row.get(f"net_SR_1B_{aum_label}", np.nan)
            sr_2a = row.get(f"net_SR_2A_{aum_label}", np.nan)
            sr_2b = row.get(f"net_SR_2B_{aum_label}", np.nan)
            net_training = sr_2a - sr_1a
            net_portfolio = sr_1b - sr_1a
            net_total = sr_2b - sr_1a
            net_interaction = net_total - net_training - net_portfolio
            row[f"net_training_{aum_label}"] = net_training
            row[f"net_portfolio_{aum_label}"] = net_portfolio
            row[f"net_total_{aum_label}"] = net_total
            row[f"net_interaction_{aum_label}"] = net_interaction
            row[f"net_training_share_{aum_label}"] = (
                net_training / net_total if net_total != 0 else np.nan
            )

            # P-values from bootstrap tests (if available)
            if net_tests and model_name in net_tests:
                aum_tests = net_tests[model_name].get(aum_label, {})
                row[f"h1_pval_{aum_label}"] = (
                    aum_tests["lw_h3"]["p_value"] if aum_tests else np.nan
                )
                row[f"h3_pval_{aum_label}"] = (
                    aum_tests["lw_total"]["p_value"] if aum_tests else np.nan
                )

        rows.append(row)

    return pd.DataFrame(rows)


def build_capacity_table(
    all_results: dict[str, dict],
    config: dict,
) -> pd.DataFrame:
    """Build H4 capacity table: break-even AUM per cell per model."""
    from scipy.interpolate import interp1d

    aum_scenarios = config["transaction_costs"]["aum_scenarios"]
    aum_labels = _get_aum_labels(config)
    log_aums = np.log10(aum_scenarios)
    cells = ["1A", "1B", "2A", "2B"]
    rows = []

    for model_name, res in all_results.items():
        cell_breakevens = {}

        for cell in cells:
            net_df = res["net_returns"][cell]
            srs = []
            for aum_label in aum_labels:
                col = f"ret_ls_net_{aum_label}"
                if col in net_df.columns:
                    srs.append(sharpe_ratio(net_df[col].dropna()))
                else:
                    srs.append(np.nan)

            srs = np.array(srs)
            row_data: dict[str, object] = {
                "model": model_name,
                "cell": cell,
            }
            for i, aum_label in enumerate(aum_labels):
                row_data[f"net_SR_{aum_label}"] = srs[i]

            # Interpolate break-even AUM (where net SR crosses 0)
            valid = ~np.isnan(srs)
            if valid.sum() >= 2:
                valid_srs = srs[valid]
                valid_log_aums = log_aums[valid]

                if np.all(valid_srs > 0):
                    breakeven = float("inf")  # SR positive at all AUM levels
                elif np.all(valid_srs <= 0):
                    breakeven = 0.0  # SR non-positive at all levels
                else:
                    try:
                        f_interp = interp1d(
                            valid_log_aums, valid_srs,
                            kind="linear", fill_value="extrapolate",
                        )
                        # Search for zero crossing
                        from scipy.optimize import brentq
                        # Find interval containing zero
                        found = False
                        for j in range(len(valid_srs) - 1):
                            if valid_srs[j] * valid_srs[j + 1] <= 0:
                                log_be = brentq(
                                    f_interp,
                                    valid_log_aums[j],
                                    valid_log_aums[j + 1],
                                )
                                breakeven = 10 ** log_be
                                found = True
                                break
                        if not found:
                            # Extrapolate
                            breakeven = float("inf") if valid_srs[-1] > 0 else 0.0
                    except Exception:
                        breakeven = np.nan
            else:
                breakeven = np.nan

            row_data["breakeven_aum"] = breakeven
            cell_breakevens[cell] = breakeven
            rows.append(row_data)

        # Compute H4 uplift: breakeven_2B / breakeven_1A
        be_1a = cell_breakevens.get("1A", np.nan)
        be_2b = cell_breakevens.get("2B", np.nan)
        if (
            be_1a is not None
            and be_2b is not None
            and be_1a > 0
            and np.isfinite(be_1a)
            and np.isfinite(be_2b)
        ):
            uplift = be_2b / be_1a
        else:
            uplift = np.nan

        # Tag uplift on the 2B row
        for r in rows:
            if r["model"] == model_name and r["cell"] == "2B":
                r["h4_uplift"] = uplift

    return pd.DataFrame(rows)


def build_factor_alpha_table(
    all_results: dict[str, dict],
) -> pd.DataFrame:
    """Build factor alpha table from effect_decomposition.json."""
    rows = []
    for model_name, res in all_results.items():
        alphas = res["effect_decomposition"].get("factor_alphas", {})
        for factor_model, cells in alphas.items():
            for cell, alpha_data in cells.items():
                rows.append({
                    "model": model_name,
                    "factor_model": factor_model,
                    "cell": cell,
                    "alpha_annual": alpha_data.get("alpha_annual"),
                    "alpha_tstat": alpha_data.get("alpha_tstat"),
                    "alpha_pval": alpha_data.get("alpha_pvalue"),
                })
    return pd.DataFrame(rows)


def build_oos_r2_table(
    all_results: dict[str, dict],
) -> pd.DataFrame:
    """Build OOS R² table."""
    rows = []
    for model_name, res in all_results.items():
        rows.append({
            "model": model_name,
            "R2_std": res["oos_r2"]["standard"],
            "R2_wt": res["oos_r2"]["weighted"],
        })
    return pd.DataFrame(rows)


# ── Net Return Statistical Tests ─────────────────────────────────


def compute_net_effect_tests(
    all_results: dict[str, dict],
    config: dict,
) -> dict[str, dict]:
    """Run Ledoit-Wolf bootstrap tests on net return series.

    For each model and AUM level, tests:
      - lw_portfolio: net SR(1B) vs net SR(1A)
      - lw_training:  net SR(2A) vs net SR(1A)
      - lw_total:     net SR(2B) vs net SR(1A)  (H3 net)
      - lw_h3:        net SR(2A) vs net SR(1B), one-sided  (H1 net)

    Returns
    -------
    dict : {model_name: {aum_label: {lw_portfolio, lw_training, lw_total, lw_h3}}}
    """
    aum_labels = _get_aum_labels(config)
    results: dict[str, dict] = {}

    for model_name, res in all_results.items():
        model_tests: dict[str, dict] = {}

        for aum_label in aum_labels:
            col = f"ret_ls_net_{aum_label}"

            # Load net return series for all 4 cells
            net_series = {}
            for cell in ["1A", "1B", "2A", "2B"]:
                net_df = res["net_returns"][cell]
                if col not in net_df.columns:
                    break
                net_series[cell] = net_df[["yyyymm", col]].dropna()
            else:
                # Align to common months
                common_months = sorted(
                    set(net_series["1A"]["yyyymm"])
                    & set(net_series["1B"]["yyyymm"])
                    & set(net_series["2A"]["yyyymm"])
                    & set(net_series["2B"]["yyyymm"])
                )
                if len(common_months) < 12:
                    logger.warning(
                        "%s / %s: only %d common months — skipping net tests.",
                        model_name, aum_label, len(common_months),
                    )
                    continue

                def _aligned(df: pd.DataFrame) -> np.ndarray:
                    return (
                        df[df["yyyymm"].isin(common_months)]
                        .sort_values("yyyymm")[col]
                        .values
                    )

                r_1a = _aligned(net_series["1A"])
                r_1b = _aligned(net_series["1B"])
                r_2a = _aligned(net_series["2A"])
                r_2b = _aligned(net_series["2B"])

                # Net Sharpe ratios
                net_sharpes = {
                    "1A": sharpe_ratio(r_1a),
                    "1B": sharpe_ratio(r_1b),
                    "2A": sharpe_ratio(r_2a),
                    "2B": sharpe_ratio(r_2b),
                }

                # Net effect decomposition
                net_portfolio_effect = net_sharpes["1B"] - net_sharpes["1A"]
                net_training_effect = net_sharpes["2A"] - net_sharpes["1A"]
                net_total_effect = net_sharpes["2B"] - net_sharpes["1A"]
                net_interaction = (
                    net_total_effect - net_portfolio_effect - net_training_effect
                )

                model_tests[aum_label] = {
                    "sharpe_ratios": net_sharpes,
                    "portfolio_effect": net_portfolio_effect,
                    "training_effect": net_training_effect,
                    "total_effect": net_total_effect,
                    "interaction": net_interaction,
                    "training_share": (
                        net_training_effect / net_total_effect
                        if net_total_effect != 0 else None
                    ),
                    "lw_portfolio": bootstrap_sharpe_test(
                        r_1b, r_1a, config=config
                    ),
                    "lw_training": bootstrap_sharpe_test(
                        r_2a, r_1a, config=config
                    ),
                    "lw_total": bootstrap_sharpe_test(
                        r_2b, r_1a, config=config
                    ),
                    "lw_h3": bootstrap_sharpe_test(
                        r_2a, r_1b, one_sided=True, config=config
                    ),
                }

                logger.info(
                    "  %s / %s: net SR(1A)=%.3f, SR(2B)=%.3f, "
                    "training=%.3f, portfolio=%.3f, total=%.3f, "
                    "H1 p=%.4f, H3 p=%.4f",
                    model_name, aum_label,
                    net_sharpes["1A"], net_sharpes["2B"],
                    net_training_effect, net_portfolio_effect, net_total_effect,
                    model_tests[aum_label]["lw_h3"]["p_value"],
                    model_tests[aum_label]["lw_total"]["p_value"],
                )

        results[model_name] = model_tests

    return results


# ── Hypothesis Test Summary ──────────────────────────────────────


def build_hypothesis_summary(
    all_results: dict[str, dict],
    h2_summary: pd.DataFrame | None,
    capacity_df: pd.DataFrame | None,
    net_tests: dict[str, dict] | None = None,
    config: dict | None = None,
) -> dict:
    """Consolidate H1–H4 results into a single JSON-serializable dict.

    Primary tests use net returns at the primary AUM ($500M).
    Gross results are kept as secondary under ``gross_results``.
    """
    primary_aum = _get_aum_labels(config)[1] if config else "500M"

    # H1: Training Dominance
    h1_per_model = {}
    for model_name, res in all_results.items():
        decomp = res["effect_decomposition"]
        gross_training = decomp["training_effect"]
        gross_portfolio = decomp["portfolio_effect"]
        gross_total = decomp["total_effect"]

        # Gross (secondary)
        gross_h1 = {
            "training_effect": gross_training,
            "portfolio_effect": gross_portfolio,
            "total_effect": gross_total,
            "training_share": gross_training / gross_total if gross_total != 0 else None,
            "p_value": decomp["lw_h3"]["p_value"],
        }

        # Primary: net at primary AUM
        primary_tests = (
            net_tests.get(model_name, {}).get(primary_aum)
            if net_tests else None
        )
        if primary_tests:
            h1_entry = {
                "training_effect": primary_tests["training_effect"],
                "portfolio_effect": primary_tests["portfolio_effect"],
                "total_effect": primary_tests["total_effect"],
                "training_share": primary_tests["training_share"],
                "p_value": primary_tests["lw_h3"]["p_value"],
                "primary_aum": primary_aum,
                "gross_results": gross_h1,
            }
        else:
            h1_entry = gross_h1
            h1_entry["primary_aum"] = "gross (net unavailable)"

        # Per-AUM net results
        if net_tests and model_name in net_tests:
            net_h1 = {}
            for aum_label, tests in net_tests[model_name].items():
                net_h1[aum_label] = {
                    "training_effect": tests["training_effect"],
                    "portfolio_effect": tests["portfolio_effect"],
                    "total_effect": tests["total_effect"],
                    "interaction": tests["interaction"],
                    "training_share": tests["training_share"],
                    "p_value": tests["lw_h3"]["p_value"],
                    "difference": tests["lw_h3"]["difference"],
                }
            h1_entry["net_results"] = net_h1
        h1_per_model[model_name] = h1_entry

    # H2: Feature Reallocation
    h2_per_model = {}
    if h2_summary is not None and len(h2_summary) > 0:
        for _, row in h2_summary.iterrows():
            h2_per_model[row["model"]] = {
                "ratio_std_mean": row.get("ratio_std"),
                "ratio_wt_mean": row.get("ratio_wt"),
                "ratio_diff": row.get("ratio_diff"),
                "t_stat": row.get("t_stat"),
                "p_value": row.get("p_value"),
            }

    # H3: Sharpe Improvement
    h3_per_model = {}
    for model_name, res in all_results.items():
        decomp = res["effect_decomposition"]
        gross_total = decomp["total_effect"]

        # Gross (secondary)
        gross_h3 = {
            "total_effect": gross_total,
            "p_value": decomp["lw_total"]["p_value"],
            "meets_target": gross_total >= 0.20,
        }

        # Primary: net at primary AUM
        primary_tests = (
            net_tests.get(model_name, {}).get(primary_aum)
            if net_tests else None
        )
        if primary_tests:
            net_total = primary_tests["total_effect"]
            h3_entry = {
                "total_effect": net_total,
                "p_value": primary_tests["lw_total"]["p_value"],
                "meets_target": net_total >= 0.20,
                "primary_aum": primary_aum,
                "gross_results": gross_h3,
            }
        else:
            h3_entry = gross_h3
            h3_entry["primary_aum"] = "gross (net unavailable)"

        # Per-AUM net results
        if net_tests and model_name in net_tests:
            net_h3 = {}
            for aum_label, tests in net_tests[model_name].items():
                net_h3[aum_label] = {
                    "sharpe_ratios": tests["sharpe_ratios"],
                    "total_effect": tests["total_effect"],
                    "meets_target": tests["total_effect"] >= 0.20,
                    "p_value": tests["lw_total"]["p_value"],
                    "difference": tests["lw_total"]["difference"],
                }
            h3_entry["net_results"] = net_h3
        h3_per_model[model_name] = h3_entry

    # H4: Capacity Improvement
    h4_per_model = {}
    if capacity_df is not None and len(capacity_df) > 0:
        for model_name in all_results:
            model_cap = capacity_df[capacity_df["model"] == model_name]
            be_1a_row = model_cap[model_cap["cell"] == "1A"]
            be_2b_row = model_cap[model_cap["cell"] == "2B"]

            be_1a = (
                float(be_1a_row["breakeven_aum"].iloc[0])
                if len(be_1a_row) > 0 else None
            )
            be_2b = (
                float(be_2b_row["breakeven_aum"].iloc[0])
                if len(be_2b_row) > 0 else None
            )
            uplift = (
                float(be_2b_row["h4_uplift"].iloc[0])
                if len(be_2b_row) > 0 and "h4_uplift" in be_2b_row.columns
                else None
            )

            h4_per_model[model_name] = {
                "breakeven_1A": be_1a,
                "breakeven_2B": be_2b,
                "uplift_ratio": uplift,
            }

    return {
        "H1_training_dominance": {
            "description": (
                "Training effect exceeds portfolio effect: "
                "[SR(2A)-SR(1A)] > [SR(1B)-SR(1A)] (net returns, primary AUM)"
            ),
            "target": "Training effect = 60-70% of total improvement",
            "per_model": h1_per_model,
        },
        "H2_feature_reallocation": {
            "description": (
                "Weighted training shifts feature importance "
                "from illiquidity to tradable features"
            ),
            "target": "Illiquidity total 43% -> 12%, Mom12m 10% -> 24%",
            "per_model": h2_per_model,
        },
        "H3_sharpe_improvement": {
            "description": "Net SR(2B) - Net SR(1A) >= 0.20 annualized (primary AUM)",
            "target": "Net total effect >= 0.20",
            "per_model": h3_per_model,
        },
        "H4_capacity_improvement": {
            "description": (
                "Break-even AUM of combined approach (2B) vs baseline (1A)"
            ),
            "target": "25x increase (~$200M to ~$5B)",
            "per_model": h4_per_model,
        },
    }


# ── Figures ──────────────────────────────────────────────────────


def _setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


CELL_LABELS = {
    "1A": "Std Train / Std Portfolio",
    "1B": "Std Train / Liq Portfolio",
    "2A": "Wt Train / Std Portfolio",
    "2B": "Wt Train / Liq Portfolio",
}
CELL_COLORS = {"1A": "#1f77b4", "1B": "#ff7f0e", "2A": "#2ca02c", "2B": "#d62728"}


def plot_cumulative_returns(
    gross: dict[str, pd.DataFrame],
    model_name: str,
    save_path: Path,
):
    """Plot cumulative gross returns for 4 cells."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots()

    for cell in ["1A", "1B", "2A", "2B"]:
        df = gross[cell].sort_values("yyyymm")
        cum = (1 + df["ret_long_short"]).cumprod()
        ax.plot(
            df["yyyymm"], cum,
            label=CELL_LABELS[cell], color=CELL_COLORS[cell],
        )

    ax.set_xlabel("Date (yyyymm)")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"Cumulative Gross Returns — {model_name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("  Saved %s", save_path.name)


def plot_cumulative_returns_net(
    net: dict[str, pd.DataFrame],
    model_name: str,
    primary_aum: str,
    save_path: Path,
):
    """Plot cumulative net returns (primary AUM) for 4 cells."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots()
    col = f"ret_ls_net_{primary_aum}"

    for cell in ["1A", "1B", "2A", "2B"]:
        df = net[cell].sort_values("yyyymm")
        if col not in df.columns:
            continue
        cum = (1 + df[col]).cumprod()
        ax.plot(
            df["yyyymm"], cum,
            label=CELL_LABELS[cell], color=CELL_COLORS[cell],
        )

    ax.set_xlabel("Date (yyyymm)")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"Cumulative Net Returns ({primary_aum} AUM) — {model_name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("  Saved %s", save_path.name)


def plot_effect_decomposition(
    main_df: pd.DataFrame,
    net: bool,
    save_path: Path,
    aum_label: str | None = None,
):
    """Grouped bar chart of training/portfolio/total effects per model."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots()

    models = main_df[main_df["model"] != "ensemble"]["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    if net and aum_label:
        train_col = f"net_training_{aum_label}"
        port_col = f"net_portfolio_{aum_label}"
        total_col = f"net_total_{aum_label}"
        title = f"Net Effect Decomposition ({aum_label} AUM)"
    else:
        train_col = "training_effect"
        port_col = "portfolio_effect"
        total_col = "total_effect"
        title = "Gross Effect Decomposition"

    model_rows = main_df[main_df["model"].isin(models)]
    ax.bar(x - width, model_rows[train_col], width, label="Training Effect", color="#2ca02c")
    ax.bar(x, model_rows[port_col], width, label="Portfolio Effect", color="#ff7f0e")
    ax.bar(x + width, model_rows[total_col], width, label="Total Effect", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Sharpe Ratio Difference")
    ax.set_title(title)
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("  Saved %s", save_path.name)


def plot_capacity_curve(
    capacity_df: pd.DataFrame,
    config: dict,
    save_path: Path,
):
    """Plot net SR vs AUM for each cell, subplot per model (H4)."""
    plt = _setup_matplotlib()

    aum_labels = _get_aum_labels(config)
    aum_scenarios = config["transaction_costs"]["aum_scenarios"]
    models = capacity_df["model"].unique()
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)

    for i, model_name in enumerate(models):
        ax = axes[0, i]
        model_data = capacity_df[capacity_df["model"] == model_name]

        for cell in ["1A", "2B"]:
            cell_row = model_data[model_data["cell"] == cell]
            if cell_row.empty:
                continue
            srs = [float(cell_row[f"net_SR_{lbl}"].iloc[0]) for lbl in aum_labels]
            ax.plot(
                aum_scenarios, srs,
                marker="o", label=CELL_LABELS[cell], color=CELL_COLORS[cell],
            )

            # Mark break-even
            be = float(cell_row["breakeven_aum"].iloc[0])
            if np.isfinite(be) and be > 0:
                ax.axvline(
                    x=be, color=CELL_COLORS[cell],
                    linestyle="--", alpha=0.5,
                )

        ax.set_xscale("log")
        ax.set_xlabel("AUM ($)")
        ax.set_ylabel("Net Sharpe Ratio")
        ax.set_title(model_name)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.legend(fontsize=8)

    fig.suptitle("Capacity Curve: Net SR vs AUM (H4)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("  Saved %s", save_path.name)


def plot_oos_r2_monthly(
    all_results: dict[str, dict],
    save_path: Path,
):
    """Plot monthly OOS R² time series for all models (std vs wt)."""
    plt = _setup_matplotlib()
    n_models = len(all_results)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 4 * n_models), squeeze=False)

    for idx, (model_name, res) in enumerate(all_results.items()):
        ax = axes[idx, 0]
        r2m = res.get("oos_r2_monthly")
        if r2m is None or r2m.empty:
            ax.set_title(f"OOS R² — {model_name} (no data)")
            continue

        r2m = r2m.sort_values("yyyymm")
        ax.plot(r2m["yyyymm"], r2m["R2_std"], label="Standard", alpha=0.7)
        ax.plot(r2m["yyyymm"], r2m["R2_wt"], label="Weighted", alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Date (yyyymm)")
        ax.set_ylabel("OOS R²")
        ax.set_title(f"Monthly OOS R² — {model_name}")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("  Saved %s", save_path.name)


# ── LaTeX helpers ────────────────────────────────────────────────


def _save_latex(df: pd.DataFrame, path: Path, float_format: str = "%.4f"):
    """Save DataFrame as LaTeX table."""
    latex = df.to_latex(index=False, float_format=float_format, escape=False)
    path.write_text(latex)


# ── Main ─────────────────────────────────────────────────────────


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()
    output_dir = get_output_dir()
    experiment_dir = output_dir / "experiment"

    if not experiment_dir.exists():
        logger.error("Experiment directory not found: %s", experiment_dir)
        logger.error("Run scripts/02_run_experiment.py first.")
        sys.exit(1)

    # Determine models to analyze
    if args.model:
        model_names = [args.model]
    else:
        model_names = config["models"]["run_models"]

    # Filter to models that actually have results
    model_names = [m for m in model_names if (experiment_dir / m).exists()]
    if not model_names:
        logger.error("No model results found in %s", experiment_dir)
        sys.exit(1)

    logger.info("Analyzing results for models: %s", model_names)

    # Create output directories
    analysis_dir = output_dir / "analysis"
    tables_dir = analysis_dir / "tables"
    figures_dir = analysis_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load results ──
    logger.info("Loading experiment results...")
    all_results: dict[str, dict] = {}
    for name in model_names:
        try:
            all_results[name] = _load_model_results(name, experiment_dir)
            logger.info("  Loaded %s", name)
        except Exception as e:
            logger.warning("  Failed to load %s: %s", name, e)

    if not all_results:
        logger.error("No results loaded successfully.")
        sys.exit(1)

    # Load summary.csv if exists (multi-model run)
    summary_path = experiment_dir / "summary.csv"
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else None

    # ── 2. Net return statistical tests (Ledoit-Wolf bootstrap) ──
    logger.info("Running Ledoit-Wolf bootstrap tests on net returns...")
    net_tests = compute_net_effect_tests(all_results, config)

    # ── 3. Main results table (gross + net primary) ──
    logger.info("Building main results table...")
    main_df = build_main_results_table(
        all_results, summary_df, net_tests=net_tests, config=config,
    )
    main_df.to_csv(tables_dir / "main_results.csv", index=False)
    _save_latex(main_df, tables_dir / "main_results.tex")
    logger.info("  Saved main_results.csv / .tex")

    # ── 4. Net results table ──
    logger.info("Building net results table...")
    net_df = build_net_results_table(all_results, config, net_tests=net_tests)
    net_df.to_csv(tables_dir / "net_results.csv", index=False)
    _save_latex(net_df, tables_dir / "net_results.tex")
    logger.info("  Saved net_results.csv / .tex")

    # ── 5. Capacity analysis (H4) ──
    logger.info("Computing capacity / break-even AUM (H4)...")
    capacity_df = build_capacity_table(all_results, config)
    capacity_df.to_csv(tables_dir / "capacity.csv", index=False)
    logger.info("  Saved capacity.csv")

    # Log H4 summary
    for model_name in all_results:
        model_cap = capacity_df[capacity_df["model"] == model_name]
        be_1a_row = model_cap[model_cap["cell"] == "1A"]
        be_2b_row = model_cap[model_cap["cell"] == "2B"]
        if not be_1a_row.empty and not be_2b_row.empty:
            be_1a = be_1a_row["breakeven_aum"].iloc[0]
            be_2b = be_2b_row["breakeven_aum"].iloc[0]
            uplift = be_2b_row.get("h4_uplift", pd.Series([np.nan])).iloc[0]
            logger.info(
                "  %s: break-even 1A=$%.0f, 2B=$%.0f, uplift=%.1fx",
                model_name, be_1a, be_2b,
                uplift if pd.notna(uplift) else 0,
            )

    # ── 6. Factor alpha table ──
    logger.info("Building factor alpha table...")
    alpha_df = build_factor_alpha_table(all_results)
    if len(alpha_df) > 0:
        alpha_df.to_csv(tables_dir / "factor_alphas.csv", index=False)
        _save_latex(alpha_df, tables_dir / "factor_alphas.tex")
        logger.info("  Saved factor_alphas.csv / .tex")
    else:
        logger.info("  No factor alpha data found (yyyymm may not have been provided).")

    # ── 7. OOS R² table ──
    logger.info("Building OOS R² table...")
    r2_df = build_oos_r2_table(all_results)
    r2_df.to_csv(tables_dir / "oos_r2.csv", index=False)
    logger.info("  Saved oos_r2.csv")

    # Monthly OOS R² CSVs
    for model_name, res in all_results.items():
        r2m = res.get("oos_r2_monthly")
        if r2m is not None and not r2m.empty:
            r2m.to_csv(tables_dir / f"oos_r2_monthly_{model_name}.csv", index=False)
            logger.info("  Saved oos_r2_monthly_%s.csv", model_name)

    # ── 8. H2 analysis (feature importance) ──
    logger.info("Running H2 analysis (feature importance)...")
    importance_type = args.importance_type
    h2_summary = None
    h2_per_feature_all = []

    try:
        h2_summary = cross_model_h2_summary(
            model_names=model_names,
            importance_type=importance_type,
            output_dir=experiment_dir,
        )
        if len(h2_summary) > 0:
            h2_summary.to_csv(tables_dir / "h2_summary.csv", index=False)
            logger.info("  Saved h2_summary.csv")
        else:
            logger.info("  No H2 summary data (importance files may be missing).")
    except Exception as e:
        logger.warning("  H2 summary failed: %s", e)
        if importance_type == "shap":
            logger.info("  Falling back to gain importance...")
            try:
                h2_summary = cross_model_h2_summary(
                    model_names=model_names,
                    importance_type="gain",
                    output_dir=experiment_dir,
                )
                if len(h2_summary) > 0:
                    h2_summary.to_csv(tables_dir / "h2_summary.csv", index=False)
                    logger.info("  Saved h2_summary.csv (gain fallback)")
                    importance_type = "gain"
            except Exception as e2:
                logger.warning("  Gain fallback also failed: %s", e2)

    # Per-feature H2 analysis
    for model_name in model_names:
        try:
            imp = load_importance(
                model_name, importance_type=importance_type,
                output_dir=experiment_dir,
            )
            per_feat = test_h2_per_feature(imp["standard"], imp["weighted"])
            per_feat["model"] = model_name
            h2_per_feature_all.append(per_feat)
        except Exception as e:
            logger.warning("  Per-feature H2 failed for %s: %s", model_name, e)

    if h2_per_feature_all:
        h2_per_feature_df = pd.concat(h2_per_feature_all, ignore_index=True)
        h2_per_feature_df.to_csv(tables_dir / "h2_per_feature.csv", index=False)
        logger.info("  Saved h2_per_feature.csv")

    # ── 9. Hypothesis test summary (H1–H4) ──
    logger.info("Building hypothesis test summary...")
    hyp_summary = build_hypothesis_summary(
        all_results, h2_summary, capacity_df,
        net_tests=net_tests, config=config,
    )
    with open(tables_dir / "hypothesis_tests.json", "w") as f:
        json.dump(hyp_summary, f, indent=2, default=str)
    logger.info("  Saved hypothesis_tests.json")

    # ── 10. Figures ──
    if not args.no_figures:
        logger.info("Generating figures...")

        # Primary AUM label for net return plots
        primary_aum = _get_aum_labels(config)[1]  # 500M (second scenario)

        for model_name, res in all_results.items():
            # Cumulative returns (gross)
            plot_cumulative_returns(
                res["gross_returns"], model_name,
                figures_dir / f"cumulative_returns_{model_name}.png",
            )

            # Cumulative returns (net)
            plot_cumulative_returns_net(
                res["net_returns"], model_name, primary_aum,
                figures_dir / f"cumulative_returns_net_{model_name}.png",
            )

            # Feature importance plots
            try:
                imp = load_importance(
                    model_name, importance_type=importance_type,
                    output_dir=experiment_dir,
                )
                plot_importance_comparison(
                    imp["standard"], imp["weighted"],
                    title=f"Feature Importance: Std vs Wt — {model_name}",
                    save_path=figures_dir / f"importance_comparison_{model_name}.png",
                )
                plot_ratio_over_time(
                    imp["standard"], imp["weighted"],
                    title=f"Tradable Ratio Over Time — {model_name}",
                    save_path=figures_dir / f"importance_ratio_{model_name}.png",
                )
                plot_importance_over_time(
                    imp["standard"], imp["weighted"],
                    group="tradable",
                    title=f"Tradable Group Importance — {model_name}",
                    save_path=figures_dir / f"importance_group_{model_name}.png",
                )
            except Exception as e:
                logger.warning("  Importance plots failed for %s: %s", model_name, e)

        # Effect decomposition (gross) — all models
        plot_effect_decomposition(
            main_df, net=False,
            save_path=figures_dir / "effect_decomposition.png",
        )

        # Effect decomposition (net) — all models
        plot_effect_decomposition(
            net_df, net=True, aum_label=primary_aum,
            save_path=figures_dir / "effect_decomposition_net.png",
        )

        # Capacity curve (H4)
        if len(capacity_df) > 0:
            plot_capacity_curve(
                capacity_df, config,
                save_path=figures_dir / "capacity_curve.png",
            )

        # Monthly OOS R² time series
        plot_oos_r2_monthly(
            all_results,
            save_path=figures_dir / "oos_r2_monthly.png",
        )
    else:
        logger.info("Skipping figure generation (--no-figures).")

    # ── 11. Console summary ──
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info("Tables saved to: %s", tables_dir)
    logger.info("Figures saved to: %s", figures_dir)

    # Print hypothesis verdicts
    logger.info("\n--- Hypothesis Summary ---")
    aum_labels = _get_aum_labels(config)
    primary_aum = aum_labels[1]  # 500M
    for model_name, res in all_results.items():
        decomp = res["effect_decomposition"]
        gross_training = decomp["training_effect"]
        gross_total = decomp["total_effect"]
        gross_share = gross_training / gross_total if gross_total != 0 else 0

        logger.info("\n%s:", model_name)

        # H1 — primary: net at primary AUM; secondary: gross
        if model_name in net_tests and primary_aum in net_tests[model_name]:
            nt = net_tests[model_name][primary_aum]
            net_share = (
                nt["training_share"] * 100
                if nt["training_share"] is not None else 0
            )
            logger.info(
                "  H1 (Training Dominance): share=%.1f%%, training=%.3f, "
                "portfolio=%.3f, total=%.3f, p=%.4f [net %s]",
                net_share,
                nt["training_effect"], nt["portfolio_effect"],
                nt["total_effect"], nt["lw_h3"]["p_value"],
                primary_aum,
            )
        logger.info(
            "    H1 (gross): share=%.1f%%, p=%.4f",
            gross_share * 100, decomp["lw_h3"]["p_value"],
        )
        # Other AUM levels
        if model_name in net_tests:
            for aum_label in aum_labels:
                if aum_label == primary_aum:
                    continue
                if aum_label in net_tests[model_name]:
                    nt = net_tests[model_name][aum_label]
                    net_share = (
                        nt["training_share"] * 100
                        if nt["training_share"] is not None else 0
                    )
                    logger.info(
                        "    H1 (net %s): share=%.1f%%, training=%.3f, "
                        "portfolio=%.3f, total=%.3f, p=%.4f",
                        aum_label, net_share,
                        nt["training_effect"], nt["portfolio_effect"],
                        nt["total_effect"], nt["lw_h3"]["p_value"],
                    )

        # H2
        if h2_summary is not None and len(h2_summary) > 0:
            h2_row = h2_summary[h2_summary["model"] == model_name]
            if len(h2_row) > 0:
                logger.info(
                    "  H2 (Feature Reallocation): ratio_diff=%.3f, p=%.4f",
                    h2_row["ratio_diff"].iloc[0], h2_row["p_value"].iloc[0],
                )

        # H3 — primary: net at primary AUM; secondary: gross
        if model_name in net_tests and primary_aum in net_tests[model_name]:
            nt = net_tests[model_name][primary_aum]
            logger.info(
                "  H3 (Sharpe Improvement): total_effect=%.3f (target: >=0.20), "
                "p=%.4f, SR(1A)=%.3f, SR(2B)=%.3f [net %s]",
                nt["total_effect"], nt["lw_total"]["p_value"],
                nt["sharpe_ratios"]["1A"], nt["sharpe_ratios"]["2B"],
                primary_aum,
            )
        logger.info(
            "    H3 (gross): total_effect=%.3f, p=%.4f",
            gross_total, decomp["lw_total"]["p_value"],
        )
        # Other AUM levels
        if model_name in net_tests:
            for aum_label in aum_labels:
                if aum_label == primary_aum:
                    continue
                if aum_label in net_tests[model_name]:
                    nt = net_tests[model_name][aum_label]
                    logger.info(
                        "    H3 (net %s): total_effect=%.3f, p=%.4f, "
                        "SR(1A)=%.3f, SR(2B)=%.3f",
                        aum_label, nt["total_effect"],
                        nt["lw_total"]["p_value"],
                        nt["sharpe_ratios"]["1A"], nt["sharpe_ratios"]["2B"],
                    )

        # H4
        model_cap = capacity_df[capacity_df["model"] == model_name]
        be_2b_row = model_cap[model_cap["cell"] == "2B"]
        if not be_2b_row.empty and "h4_uplift" in be_2b_row.columns:
            uplift = be_2b_row["h4_uplift"].iloc[0]
            if pd.notna(uplift):
                logger.info(
                    "  H4 (Capacity): uplift=%.1fx (target: 25x)",
                    uplift,
                )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
