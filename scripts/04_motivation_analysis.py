"""
Motivation Analysis — Section 2: The Implementability Imbalance Problem
========================================================================
Produces empirical evidence that standard ML training is misaligned with
the implementable investment universe.  Uses ElasticNet only (linear
benchmark) with standard (unweighted) training.

Outputs (saved to outputs/motivation/):
  - table1_distribution.csv       Distribution mismatch by quintile
  - table2_oos_r2_quintile.csv    OOS R² by liquidity quintile
  - table3_sr_quintile.csv        Gross vs net SR by quintile
  - predictions_elasticnet.parquet  Raw OOS predictions
  - figure_r2_by_quintile.png     Bar chart of R² by quintile
  - figure_sr_scissors.png        Gross vs net SR scissors chart

Usage:
  python scripts/04_motivation_analysis.py           # Full run (2000-2024)
  python scripts/04_motivation_analysis.py --quick   # Quick run (2015-2024)
  python scripts/04_motivation_analysis.py --recompute  # Recompute tables from saved predictions
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV

from src.config import load_config, get_output_dir
from src.data.loader import load_panel, get_feature_names, normalize_features

logger = logging.getLogger(__name__)


# ── ElasticNet wrapper ────────────────────────────────────────


class ElasticNetPredictor:
    """Minimal wrapper around sklearn ElasticNetCV for consistency."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model: ElasticNetCV | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            sample_weight=None, sample_weight_val=None):
        self.model = ElasticNetCV(
            l1_ratio=[0.5, 0.7, 0.9, 0.95, 1.0],
            cv=5,
            max_iter=5000,
            random_state=self.seed,
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)


# ── Window helpers (reused from two_by_two.py) ───────────────


def _get_oos_months(panel: pd.DataFrame, config: dict) -> list[int]:
    """Return sorted list of yyyymm test months in the OOS period."""
    train_cfg = config["training"]
    oos_start = train_cfg["oos_start"]
    oos_end = train_cfg["oos_end"]
    min_history = train_cfg["train_window"] + train_cfg["validation_window"]

    all_months = sorted(panel["yyyymm"].unique())
    oos_months = []
    for idx, m in enumerate(all_months):
        if m < oos_start or m > oos_end:
            continue
        if idx >= min_history:
            oos_months.append(m)

    logger.info("OOS months: %d (from %d to %d)",
                len(oos_months), oos_months[0] if oos_months else 0,
                oos_months[-1] if oos_months else 0)
    return oos_months


def _split_window(panel, test_yyyymm, all_months, config):
    """Split panel into train / val / test for a given test month."""
    train_cfg = config["training"]
    train_window = train_cfg["train_window"]
    val_window = train_cfg["validation_window"]

    months_before = [m for m in all_months if m < test_yyyymm]
    val_months = set(months_before[-val_window:])
    train_months = set(months_before[-(train_window + val_window):-val_window])

    train_df = panel[panel["yyyymm"].isin(train_months)].copy()
    val_df = panel[panel["yyyymm"].isin(val_months)].copy()
    test_df = panel[panel["yyyymm"] == test_yyyymm].copy()

    return train_df, val_df, test_df


def _prepare_xy(train_df, val_df, test_df, features, config):
    """Normalize features and extract arrays (standard training only)."""
    target_col = config["data"]["target_col"]

    # Normalize: train+val together, test independently
    train_val = pd.concat([train_df, val_df], ignore_index=False)
    train_val_norm = normalize_features(train_val, features)
    test_norm = normalize_features(test_df, features)

    train_norm = train_val_norm.loc[train_df.index]
    val_norm = train_val_norm.loc[val_df.index]

    # Fill NaN with 0.0 (neutral rank)
    for df in [train_norm, val_norm, test_norm]:
        df[features] = df[features].fillna(0.0)

    # Drop rows with NaN target
    train_norm = train_norm.dropna(subset=[target_col])
    val_norm = val_norm.dropna(subset=[target_col])
    test_norm = test_norm.dropna(subset=[target_col])

    return {
        "X_train": train_norm[features].values,
        "y_train": train_norm[target_col].values,
        "X_val": val_norm[features].values,
        "y_val": val_norm[target_col].values,
        "X_test": test_norm[features].values,
        "y_test": test_norm[target_col].values,
        "test_norm": test_norm,
    }


# ── Table 1: Distribution Mismatch ───────────────────────────


def compute_table1(panel: pd.DataFrame, config: dict, oos_months: list[int]) -> pd.DataFrame:
    """Compute implementable stock distribution by quintile at multiple AUM levels."""
    tc_cfg = config["transaction_costs"]
    spread_scale = tc_cfg["spread_scale"]
    lam = tc_cfg["lambda_market_impact"]
    aum_scenarios = {
        "100M": 100_000_000,
        "500M": 500_000_000,
        "1B": 1_000_000_000,
        "5B": 5_000_000_000,
    }

    # {aum_label: [monthly pct_impl Series, ...]}
    monthly_results = {label: [] for label in aum_scenarios}

    for month in oos_months:
        month_df = panel[panel["yyyymm"] == month].copy()

        # Skip stocks missing dvol_21d (can't assign quintile)
        month_df = month_df.dropna(subset=["liq_dvol_21d"])
        if len(month_df) < 50:
            continue

        n_stocks = len(month_df)
        n_portfolio = max(n_stocks // 5, 1)

        # Assign quintiles by liq_dvol_21d
        month_df["quintile"] = pd.qcut(
            month_df["liq_dvol_21d"], 5, labels=[1, 2, 3, 4, 5]
        ).astype(int)

        # Fill missing TC inputs with cross-sectional median
        spread = month_df["liq_BidAskSpread"].fillna(month_df["liq_BidAskSpread"].median())
        sigma = month_df["liq_daily_sigma"].fillna(month_df["liq_daily_sigma"].median())
        adv = month_df["liq_dvol_21d"]

        half_spread = spread * spread_scale / 2.0

        for label, aum in aum_scenarios.items():
            q_i = aum / n_portfolio
            market_impact = lam * sigma * np.sqrt(q_i / adv.clip(lower=1e3))
            tc = half_spread + market_impact

            month_df["implementable"] = (tc < 0.01).values

            impl_counts = month_df.groupby("quintile")["implementable"].sum()
            total_impl = impl_counts.sum()
            if total_impl == 0:
                continue

            monthly_results[label].append(impl_counts / total_impl * 100)

    # Average across months per AUM
    table1 = pd.DataFrame({
        "Quintile": [f"Q{q}" for q in range(1, 6)],
        "Pct_Training": [20.0] * 5,
    })
    for label in aum_scenarios:
        avg = pd.DataFrame(monthly_results[label]).mean()
        table1[f"Pct_Impl_{label}"] = avg.values

    table1.loc[0, "Quintile"] = "Q1 (Most Illiquid)"
    table1.loc[4, "Quintile"] = "Q5 (Most Liquid)"

    return table1


# ── Rolling prediction loop ──────────────────────────────────


def run_rolling_predictions(
    panel: pd.DataFrame,
    features: list[str],
    config: dict,
    oos_months: list[int],
) -> pd.DataFrame:
    """Run ElasticNet rolling-window OOS predictions."""
    seed = config["project"]["seed"]
    all_months = sorted(panel["yyyymm"].unique())
    target_col = config["data"]["target_col"]

    predictions_list = []
    t_start = time.time()

    for i, test_month in enumerate(oos_months):
        # Split
        train_df, val_df, test_df = _split_window(
            panel, test_month, all_months, config
        )

        if len(test_df) < 50:
            logger.warning("Month %d: only %d test stocks — skipping.",
                           test_month, len(test_df))
            continue
        if len(train_df) < 1000:
            logger.warning("Month %d: only %d train rows — skipping.",
                           test_month, len(train_df))
            continue

        # Prepare
        data = _prepare_xy(train_df, val_df, test_df, features, config)

        # Train ElasticNet
        model = ElasticNetPredictor(seed=seed)
        model.fit(data["X_train"], data["y_train"])

        # Predict
        preds = model.predict(data["X_test"])

        # Collect results with liquidity info from test_norm
        test_norm = data["test_norm"]
        month_preds = pd.DataFrame({
            "permno": test_norm["permno"].values,
            "yyyymm": test_norm["yyyymm"].values,
            "y_true": data["y_test"],
            "pred": preds,
            "ret": test_norm["ret"].values if "ret" in test_norm.columns else np.nan,
            "liq_dvol_21d": test_norm["liq_dvol_21d"].values,
            "liq_BidAskSpread": test_norm["liq_BidAskSpread"].values,
            "liq_daily_sigma": test_norm["liq_daily_sigma"].values,
        })
        predictions_list.append(month_preds)

        elapsed = time.time() - t_start
        logger.info("Month %d (%d/%d) — %.0fs elapsed",
                    test_month, i + 1, len(oos_months), elapsed)

    predictions = pd.concat(predictions_list, ignore_index=True)

    # Assign quintiles per month
    def assign_quintile(group):
        valid = group["liq_dvol_21d"].notna()
        group["quintile"] = np.nan
        if valid.sum() >= 50:
            group.loc[valid, "quintile"] = pd.qcut(
                group.loc[valid, "liq_dvol_21d"], 5, labels=[1, 2, 3, 4, 5]
            ).astype(float)
        return group

    predictions = predictions.groupby("yyyymm", group_keys=False).apply(assign_quintile)
    predictions["quintile"] = predictions["quintile"].astype("Int64")

    return predictions


# ── Table 2: OOS R² by Quintile ──────────────────────────────


def compute_table2(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled OOS R² by liquidity quintile (zero benchmark)."""
    rows = []

    for q in range(1, 6):
        mask = predictions["quintile"] == q
        y = predictions.loc[mask, "y_true"].values
        yhat = predictions.loc[mask, "pred"].values
        ss_res = np.sum((y - yhat) ** 2)
        ss_zero = np.sum(y ** 2)
        r2 = 1.0 - ss_res / ss_zero if ss_zero > 0 else np.nan
        rows.append({"Quintile": f"Q{q}", "OOS_R2_pct": r2 * 100})

    # Pooled
    valid = predictions["quintile"].notna()
    y = predictions.loc[valid, "y_true"].values
    yhat = predictions.loc[valid, "pred"].values
    ss_res = np.sum((y - yhat) ** 2)
    ss_zero = np.sum(y ** 2)
    r2_pooled = 1.0 - ss_res / ss_zero if ss_zero > 0 else np.nan
    rows.append({"Quintile": "Pooled", "OOS_R2_pct": r2_pooled * 100})

    return pd.DataFrame(rows)


# ── Table 3: Gross vs Net SR by Quintile ──────────────────────


def compute_table3(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute within-quintile long-short portfolio gross and net SR at multiple AUMs."""
    tc_cfg = config["transaction_costs"]
    spread_scale = tc_cfg["spread_scale"]
    lam = tc_cfg["lambda_market_impact"]
    aum_scenarios = {
        "100M": 100_000_000,
        "500M": 500_000_000,
        "1B":  1_000_000_000,
        "5B":  5_000_000_000,
    }

    all_months = sorted(predictions["yyyymm"].unique())

    # Per-quintile monthly data: gross returns, turnover, and stock-level TC inputs
    quintile_data = {q: {
        "gross": [],
        "turnover": [],
        "spread_arr": [],   # avg half-spread of traded stocks per month
        "impact_arr": [],   # avg sigma/sqrt(ADV) of traded stocks per month
        "n_portfolio": [],  # portfolio size per month
    } for q in range(1, 6)}

    prev_positions = {q: {"long": set(), "short": set()} for q in range(1, 6)}

    for month_idx, month in enumerate(all_months):
        month_df = predictions[predictions["yyyymm"] == month].copy()

        for q in range(1, 6):
            qdf = month_df[month_df["quintile"] == q].copy()
            n = len(qdf)
            if n < 20:
                continue

            # Within-quintile decile sort
            n_bins = max(5, min(10, n // 10))
            qdf["decile"] = pd.qcut(
                qdf["pred"].rank(method="first"), n_bins, labels=False
            ) + 1

            long_stocks = qdf[qdf["decile"] == n_bins]
            short_stocks = qdf[qdf["decile"] == 1]

            if len(long_stocks) == 0 or len(short_stocks) == 0:
                continue

            # Gross return (AUM-independent)
            r_gross = long_stocks["ret"].mean() - short_stocks["ret"].mean()
            quintile_data[q]["gross"].append(r_gross)

            # Turnover (AUM-independent)
            current_long = set(long_stocks["permno"].values)
            current_short = set(short_stocks["permno"].values)
            n_portfolio = len(current_long) + len(current_short)

            if month_idx == 0:
                turnover = 2.0
            else:
                prev_long = prev_positions[q]["long"]
                prev_short = prev_positions[q]["short"]
                entered = (current_long - prev_long) | (current_short - prev_short)
                exited = (prev_long - current_long) | (prev_short - current_short)
                n_prev = len(prev_long) + len(prev_short)
                turnover = (len(entered) + len(exited)) / max(n_prev, 1)

            prev_positions[q]["long"] = current_long
            prev_positions[q]["short"] = current_short
            quintile_data[q]["turnover"].append(turnover)
            quintile_data[q]["n_portfolio"].append(n_portfolio)

            # Stock-level TC components (separate AUM-independent parts)
            portfolio_stocks = pd.concat([long_stocks, short_stocks])
            spread = portfolio_stocks["liq_BidAskSpread"].fillna(
                portfolio_stocks["liq_BidAskSpread"].median()
            )
            sigma = portfolio_stocks["liq_daily_sigma"].fillna(
                portfolio_stocks["liq_daily_sigma"].median()
            )
            adv = portfolio_stocks["liq_dvol_21d"].clip(lower=1e3)

            avg_half_spread = (spread * spread_scale / 2.0).mean()
            avg_impact_base = (sigma / np.sqrt(adv)).mean()  # needs * lam * sqrt(q_i)

            quintile_data[q]["spread_arr"].append(avg_half_spread)
            quintile_data[q]["impact_arr"].append(avg_impact_base)

    # Compute SR for each quintile × AUM
    rows = []
    for q in range(1, 6):
        gross = np.array(quintile_data[q]["gross"])
        turnover = np.array(quintile_data[q]["turnover"])
        spread_arr = np.array(quintile_data[q]["spread_arr"])
        impact_arr = np.array(quintile_data[q]["impact_arr"])
        n_port = np.array(quintile_data[q]["n_portfolio"])
        n_months = len(gross)

        if n_months < 2:
            row = {"Quintile": f"Q{q}", "Gross_SR": np.nan,
                   "Avg_Ret_pct": np.nan, "N_months": 0}
            for label in aum_scenarios:
                row[f"Net_SR_{label}"] = np.nan
                row[f"Avg_TC_{label}_pct"] = np.nan
            rows.append(row)
            continue

        gross_sr = np.mean(gross) / np.std(gross, ddof=1) * np.sqrt(12)

        row = {
            "Quintile": f"Q{q}",
            "Gross_SR": gross_sr,
            "Avg_Ret_pct": np.mean(gross) * 100,
            "N_months": n_months,
        }

        for label, aum in aum_scenarios.items():
            # TC = turnover * (half_spread + lam * sqrt(q_i/ADV) component)
            # q_i = aum / n_portfolio per month
            q_i = aum / np.maximum(n_port, 1)
            monthly_tc = turnover * (spread_arr + lam * np.sqrt(q_i) * impact_arr)
            net = gross - monthly_tc[:len(gross)]

            net_sr = np.mean(net) / np.std(net, ddof=1) * np.sqrt(12)
            row[f"Net_SR_{label}"] = net_sr
            row[f"Avg_TC_{label}_pct"] = np.mean(monthly_tc[:len(gross)]) * 100

        rows.append(row)

    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────


def plot_r2_by_quintile(table2: pd.DataFrame, output_dir: Path):
    """Bar chart of OOS R² by liquidity quintile."""
    # Exclude pooled row for the bar chart
    data = table2[table2["Quintile"] != "Pooled"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(data))
    bars = ax.bar(x, data["OOS_R2_pct"], color="#4C72B0", edgecolor="black", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(data["Quintile"])
    ax.set_xlabel("Liquidity Quintile")
    ax.set_ylabel("OOS R² (%)")
    ax.set_title("Out-of-Sample R² by Liquidity Quintile (ElasticNet)")
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Add pooled reference line
    pooled_r2 = table2.loc[table2["Quintile"] == "Pooled", "OOS_R2_pct"].values[0]
    ax.axhline(y=pooled_r2, color="red", linestyle="--", linewidth=1,
               label=f"Pooled R² = {pooled_r2:.2f}%")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "figure_r2_by_quintile.png", dpi=150)
    plt.close(fig)
    logger.info("Saved figure_r2_by_quintile.png")


def plot_sr_scissors(table3: pd.DataFrame, output_dir: Path):
    """Scissors chart: gross SR vs net SR at multiple AUM levels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(table3))

    # Gross SR (AUM-independent)
    ax.plot(x, table3["Gross_SR"], "o-", color="#4C72B0", linewidth=2.5,
            markersize=8, label="Gross SR", zorder=5)

    # Net SR at each AUM
    net_styles = {
        "Net_SR_100M": ("#66C2A5", "^", "--", "$100M"),
        "Net_SR_500M": ("#DD8452", "s", "--", "$500M"),
        "Net_SR_1B":   ("#E78AC3", "D", ":",  "$1B"),
        "Net_SR_5B":   ("#E5C494", "v", ":",  "$5B"),
    }

    for col, (color, marker, ls, label) in net_styles.items():
        if col in table3.columns:
            ax.plot(x, table3[col], marker=marker, linestyle=ls,
                    color=color, linewidth=1.8, markersize=7,
                    label=f"Net SR ({label})", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(table3["Quintile"])
    ax.set_xlabel("Liquidity Quintile")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title("Gross vs. Net Sharpe Ratio by Liquidity Quintile (ElasticNet)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "figure_sr_scissors.png", dpi=150)
    plt.close(fig)
    logger.info("Saved figure_sr_scissors.png")


# ── Main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Motivation analysis — Section 2 evidence"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use OOS start from 2015 for faster iteration"
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Recompute tables/figures from existing predictions (skip model training)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config()

    if args.quick:
        config["training"]["oos_start"] = 201501
        logger.info("Quick mode: OOS start set to 201501")

    # Load data
    logger.info("Loading panel data...")
    panel = load_panel(config)
    features = get_feature_names(panel)
    logger.info("Panel: %d rows, %d features", len(panel), len(features))

    # Output directory
    output_dir = Path(get_output_dir()) / "motivation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # OOS months
    oos_months = _get_oos_months(panel, config)

    # ── Table 1: Distribution mismatch ──
    logger.info("Computing Table 1: Distribution mismatch...")
    table1 = compute_table1(panel, config, oos_months)
    table1.to_csv(output_dir / "table1_distribution.csv", index=False)
    logger.info("Table 1:\n%s", table1.to_string(index=False))

    # ── Rolling predictions (or load existing) ──
    pred_path = output_dir / "predictions_elasticnet.parquet"

    if args.recompute and pred_path.exists():
        logger.info("Loading existing predictions from %s", pred_path)
        predictions = pd.read_parquet(pred_path)
        logger.info("Loaded predictions: %d rows", len(predictions))
    else:
        logger.info("Running ElasticNet rolling predictions...")
        predictions = run_rolling_predictions(panel, features, config, oos_months)
        predictions.to_parquet(pred_path, index=False)
        logger.info("Saved predictions: %d rows", len(predictions))

    # ── Table 2: OOS R² by quintile ──
    logger.info("Computing Table 2: OOS R² by quintile...")
    table2 = compute_table2(predictions)
    table2.to_csv(output_dir / "table2_oos_r2_quintile.csv", index=False)
    logger.info("Table 2:\n%s", table2.to_string(index=False))

    # ── Table 3: Gross vs net SR by quintile ──
    logger.info("Computing Table 3: SR by quintile...")
    table3 = compute_table3(predictions, config)
    table3.to_csv(output_dir / "table3_sr_quintile_elasticnet.csv", index=False)
    logger.info("Table 3:\n%s", table3.to_string(index=False))

    # ── Figures ──
    logger.info("Generating figures...")
    plot_r2_by_quintile(table2, output_dir)
    plot_sr_scissors(table3, output_dir)

    logger.info("Done. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
