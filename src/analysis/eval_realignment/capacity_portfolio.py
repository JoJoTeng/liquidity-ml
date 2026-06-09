"""Signal-weighted capacity portfolio (note section 4.2).

The note's preferred economic test replaces the quantile long-short with a
continuous, dollar-neutral book that holds each stock in proportion to its
capacity (the spec's own implementability weight w-tilde) and its centered signal:

    theta_it ∝ w̃_it · (r̂_it − rbar_w_t),    rbar_w_t = Σ_i w̃_it·r̂_it / Σ_i w̃_it

Centering on the w-tilde-weighted cross-sectional mean makes the book dollar-neutral
(Σ_i theta_it = 0): no quantile cut, no equal-weighting of microcaps, no long-short
non-additivity. We size the book to unit gross exposure (Σ_i |theta_it| = 1, so AUM
is the gross capital) and build it from both the standard and the weighted r̂ (same
w-tilde) to test whether the prediction-level training improvement survives at the
portfolio level. Gross and net (turnover-based TC) annualized Sharpe and
certainty-equivalent are reported across the AUM grid.

This module only READS the existing formal predictions (no retraining). It reuses
the transaction-cost primitives (prepare_transaction_cost_context + the Frazzini
formula) and the statistics helpers; only the unified signed-book net-return loop and
the certainty-equivalent are new. There is no portfolio-stage cost sort (the section
5.2 device the note moves away from); cost enters via w-tilde and the realized
turnover-based TC.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.formal.common import (
    compute_formal_utility_weights,
    is_proportional_tc_scenario,
    scenario_sizing_aum,
)
from src.evaluation.statistics import (
    bootstrap_sharpe_test,
    newey_west_tstat,
    sharpe_ratio,
)

logger = logging.getLogger(__name__)

# Risk-aversion grid for the certainty-equivalent (monthly returns).
CE_GAMMAS = (1.0, 5.0, 10.0)
MONTHS_PER_YEAR = 12


def certainty_equivalent(returns, gamma: float) -> float:
    """Mean-variance certainty-equivalent on monthly returns: E[r] − 0.5·gamma·Var[r].

    Returns are monthly; multiply by 12 at the call-site for an annual figure.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return np.nan
    return float(np.mean(r) - 0.5 * gamma * np.var(r, ddof=1))


def build_capacity_book(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
    min_names: int = 2,
) -> tuple[pd.DataFrame, dict[int, dict[int, float]]]:
    """Build the dollar-neutral, unit-gross capacity book from one prediction set.

    Per month t: theta_raw = w̃·(r̂ − rbar_w) (Σ theta_raw = 0 by construction),
    normalized to Σ|theta| = 1. Returns (gross_df, positions) where gross_df has
    [yyyymm, ret_gross, n_long, n_short] and positions maps yyyymm -> {permno: theta}.
    Degenerate months (flat signal, or fewer than ``min_names`` names on a side) are
    skipped.
    """
    target = config["data"]["target_col"]
    sub = panel[["permno", "yyyymm", target]].copy()
    sub["w_tilde"] = compute_formal_utility_weights(panel, config, spec)

    merged = (
        predictions[["permno", "yyyymm", "prediction"]]
        .merge(sub, on=["permno", "yyyymm"], how="inner")
        .dropna(subset=["prediction", target, "w_tilde"])
    )

    records: list[dict[str, Any]] = []
    positions: dict[int, dict[int, float]] = {}
    for yyyymm, group in merged.groupby("yyyymm", sort=True):
        w = group["w_tilde"].to_numpy(dtype=np.float64)
        r_hat = group["prediction"].to_numpy(dtype=np.float64)
        r = group[target].to_numpy(dtype=np.float64)
        sum_w = w.sum()
        if sum_w <= 0:
            continue

        rbar_w = float((w * r_hat).sum() / sum_w)
        theta_raw = w * (r_hat - rbar_w)  # Σ theta_raw = 0 by definition of rbar_w
        gross_abs = float(np.abs(theta_raw).sum())
        n_long = int((theta_raw > 0).sum())
        n_short = int((theta_raw < 0).sum())
        if gross_abs <= 1e-12 or n_long < min_names or n_short < min_names:
            continue

        theta = theta_raw / gross_abs  # Σ|theta| = 1
        ret_gross = float((theta * r).sum())
        permnos = group["permno"].to_numpy()
        positions[int(yyyymm)] = {
            int(p): float(t) for p, t in zip(permnos, theta) if abs(t) > 0.0
        }
        records.append(
            {
                "yyyymm": int(yyyymm),
                "ret_gross": ret_gross,
                "n_long": n_long,
                "n_short": n_short,
            }
        )

    gross_df = pd.DataFrame(
        records, columns=["yyyymm", "ret_gross", "n_long", "n_short"]
    )
    return gross_df, positions


def _safe_lookup_value(row, col: str, default: float) -> float:
    """Extract a positive, non-NaN value from a lookup row, else the default."""
    try:
        val = row[col] if col in row.index else default
    except (KeyError, TypeError):
        val = default
    if pd.isna(val) or val <= 0:
        val = default
    return float(val)


def _drift_capacity(
    positions_prev: dict[int, float],
    from_yyyymm: int,
    lookup: pd.DataFrame,
) -> dict[int, float]:
    """Drift signed weights by the holding-period return, then renormalize gross to 1.

    ``ret`` in ``lookup`` is forward-shifted by ``load_panel``, so
    ``lookup[(permno, from_yyyymm)]`` is the return earned between ``from_yyyymm`` and
    the next rebalance. The gross renormalization keeps the carried book at unit gross
    exposure, the unified analogue of the existing per-leg renormalization.
    """
    drifted: dict[int, float] = {}
    gross = 0.0
    for permno, w in positions_prev.items():
        try:
            ret = lookup.loc[(permno, from_yyyymm), "ret"]
            ret = 0.0 if pd.isna(ret) else float(ret)
        except KeyError:
            ret = 0.0
        value = w * (1.0 + ret)
        drifted[permno] = value
        gross += abs(value)
    if gross > 0:
        for permno in drifted:
            drifted[permno] /= gross
    return drifted


def capacity_net_returns(
    gross_df: pd.DataFrame,
    positions: dict[int, dict[int, float]],
    tc_context: dict[str, Any],
    aum_scenario: int | float | str,
) -> pd.DataFrame:
    """Add turnover-based transaction costs and net returns to the gross book.

    TC uses the FULL traded notional Σ|Δθ| (one-way cost on every dollar traded);
    the reported ``turnover`` is the 0.5·Σ|Δθ| diagnostic only and does not enter
    net returns. Market impact (Frazzini) is sized on the dollar trade ``Δθ·AUM``.
    """
    lookup = tc_context["lookup"]
    spread_col = tc_context["spread_col"]
    sigma_col = tc_context["sigma_col"]
    adv_col = tc_context["adv_col"]
    lam = tc_context["lam"]
    aum = scenario_sizing_aum(aum_scenario)
    proportional = is_proportional_tc_scenario(aum_scenario)

    months = sorted(positions.keys())
    tc_series: dict[int, float] = {}
    turnover_series: dict[int, float] = {}
    prev: dict[int, float] = {}

    for i, yyyymm in enumerate(months):
        target = positions[yyyymm]
        drifted = {} if i == 0 else _drift_capacity(prev, months[i - 1], lookup)

        total_tc = 0.0
        raw_trade = 0.0
        for permno in set(target) | set(drifted):
            delta = abs(target.get(permno, 0.0) - drifted.get(permno, 0.0))
            if delta < 1e-12:
                continue
            raw_trade += delta
            delta_q = delta * aum

            row = None
            try:
                row = lookup.loc[(permno, yyyymm)]
            except KeyError:
                if i > 0:
                    try:
                        row = lookup.loc[(permno, months[i - 1])]
                    except KeyError:
                        row = None
            if row is None:
                total_tc += delta * 0.005  # 50bps default
                continue

            spread = _safe_lookup_value(row, spread_col, 0.01)
            half_spread = spread / 2.0
            if proportional:
                market_impact = 0.0
            else:
                sigma = _safe_lookup_value(row, sigma_col, 0.02)
                adv = _safe_lookup_value(row, adv_col, 1e6)
                market_impact = lam * sigma * np.sqrt(delta_q / adv)
            total_tc += delta * (half_spread + market_impact)

        tc_series[yyyymm] = total_tc
        turnover_series[yyyymm] = 0.5 * raw_trade
        prev = target

    out = gross_df.copy()
    out["transaction_cost"] = out["yyyymm"].map(tc_series).fillna(0.0)
    out["turnover"] = out["yyyymm"].map(turnover_series).fillna(0.0)
    out["ret_net"] = out["ret_gross"] - out["transaction_cost"]
    return out


def capacity_metrics_for_predictions(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
    aum_scenario: int | float | str,
    tc_context: dict[str, Any],
) -> tuple[dict | None, pd.DataFrame | None]:
    """Build the capacity book for one prediction set and summarize @ one AUM."""
    gross_df, positions = build_capacity_book(predictions, panel, config, spec)
    if gross_df.empty:
        return None, None

    net = capacity_net_returns(gross_df, positions, tc_context, aum_scenario)
    g = net["ret_gross"].to_numpy(dtype=np.float64)
    n = net["ret_net"].to_numpy(dtype=np.float64)

    row: dict[str, Any] = {
        "gross_sr_annual": sharpe_ratio(g, annualize=True),
        "net_sr_annual": sharpe_ratio(n, annualize=True),
        "gross_sr_monthly": sharpe_ratio(g),
        "net_sr_monthly": sharpe_ratio(n),
        "gross_mean_annual": float(np.nanmean(g)) * MONTHS_PER_YEAR,
        "net_mean_annual": float(np.nanmean(n)) * MONTHS_PER_YEAR,
        "turnover_mean": (
            float(net["turnover"].iloc[1:].mean()) if len(net) > 1 else np.nan
        ),
        "tc_mean_monthly": (
            float(net["transaction_cost"].iloc[1:].mean()) if len(net) > 1 else np.nan
        ),
        "n_months": int(np.isfinite(n).sum()),
    }
    for gamma in CE_GAMMAS:
        row[f"gross_ce_annual_g{int(gamma)}"] = (
            certainty_equivalent(g, gamma) * MONTHS_PER_YEAR
        )
        row[f"net_ce_annual_g{int(gamma)}"] = (
            certainty_equivalent(n, gamma) * MONTHS_PER_YEAR
        )

    series = net.set_index("yyyymm")[
        ["ret_gross", "ret_net", "transaction_cost", "turnover", "n_long", "n_short"]
    ]
    return row, series


def compare_standard_vs_weighted(
    series_std: pd.DataFrame,
    series_wt: pd.DataFrame,
    row_std: dict,
    row_wt: dict,
    config: dict,
) -> dict:
    """Training effect (weighted − standard): bootstrap Sharpe diff, NW t, CE diff."""
    aligned = pd.concat(
        {"std": series_std["ret_net"], "wt": series_wt["ret_net"]}, axis=1
    ).dropna()

    out: dict[str, Any] = {}
    if len(aligned) >= 3:
        bt = bootstrap_sharpe_test(r_i=aligned["wt"], r_n=aligned["std"], config=config)
        out["net_sr_diff"] = bt["difference"]  # SR(weighted) − SR(standard)
        out["net_sr_diff_pval"] = bt["p_value"]
        out["net_sr_diff_reject"] = bt["reject"]
        nw = newey_west_tstat(
            (aligned["wt"] - aligned["std"]).to_numpy(dtype=np.float64),
            lags=config["inference"]["newey_west_lags"],
            config=config,
        )
        out["net_mean_diff_nw_t"] = nw["t_stat"]
    else:
        out.update(
            {
                "net_sr_diff": np.nan,
                "net_sr_diff_pval": np.nan,
                "net_sr_diff_reject": np.nan,
                "net_mean_diff_nw_t": np.nan,
            }
        )

    for gamma in CE_GAMMAS:
        key = f"net_ce_annual_g{int(gamma)}"
        out[f"net_ce_diff_annual_g{int(gamma)}"] = (
            row_wt.get(key, np.nan) - row_std.get(key, np.nan)
        )
    return out


def plot_capacity_net_cumret(
    series_std: pd.DataFrame,
    series_wt: pd.DataFrame,
    out_path: Path,
    model: str,
    aum_label_str: str,
) -> None:
    """Plot cumulative (compounded) net return: standard vs weighted capacity book."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.analysis.motivation import _set_academic_style

    _set_academic_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    for series, label, color in [
        (series_std, "Standard training", "#4C72B0"),
        (series_wt, "Weighted training", "#DD8452"),
    ]:
        ret = series["ret_net"].dropna()
        cum = (1.0 + ret).cumprod() - 1.0
        ax.plot(range(len(cum)), cum.to_numpy(), label=label, color=color, lw=1.4)

    ax.axhline(0, color="black", ls="--", lw=0.5)
    ax.set_ylabel("Cumulative net return")
    ax.set_title(
        f"Capacity-portfolio net return ({model}, {aum_label_str})"
    )
    ax.legend()

    ref = series_wt if len(series_wt) >= len(series_std) else series_std
    months = ref.index.astype(str).tolist()
    tick_every = max(len(months) // 10, 1)
    ticks = range(0, len(months), tick_every)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([months[i] for i in ticks], rotation=45, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
