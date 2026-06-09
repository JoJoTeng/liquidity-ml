"""Deployment-weighted prediction metrics (note section 4.1).

The note argues the headline Q5-Q1 portfolio metric does not measure what
liquidity-weighted training optimizes: the deployment-weighted prediction error
(Eq. 1). Section 4.1 is the literal empirical counterpart and reports

  1. the w-tilde-weighted out-of-sample R^2 (Eq. 2, zero benchmark)

         R^2_w = 1 - sum w~ (r - r_hat)^2 / sum w~ r^2,

     for the standard and weighted predictions, evaluated under the SAME w-tilde
     (the spec's own training weight family); and

  2. the w-tilde-weighted monthly squared-error differential
     wMSE(standard) - wMSE(weighted) with Newey-West t-statistics.

Both metrics are reported per liquidity quintile (Q1-Q5), for the pooled Q4-Q5
liquid subset, and for the full cross-section, mirroring the formalanalysis
``r2_by_quintile`` convention.

This module only READS the existing formal predictions (no retraining). The R^2
reuses ``compute_utility_weighted_r2`` unchanged; the error differential is the
w-tilde-weighted analogue of ``compute_liquid_squared_error_differential`` so the
two tracks cannot silently diverge.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.formal.common import (
    assign_liquidity_quintiles,
    compute_formal_utility_weights,
    formal_weight_label,
    predictions_for_utility_r2,
)
from src.analysis.motivation import compute_utility_weighted_r2
from src.evaluation.statistics import newey_west_tstat

logger = logging.getLogger(__name__)

# Quintile labels (Q5 is most liquid; see assign_liquidity_quintiles).
QUINTILES = (1, 2, 3, 4, 5)
# Pooled liquid (deployable) quintiles.
LIQUID_QUINTILES = (4.0, 5.0)


def _universe_members() -> list[tuple[str, tuple[float, ...] | None]]:
    """Reporting universes: each quintile, the pooled liquid set, and full.

    ``None`` membership means the whole cross-section, including stocks without a
    liquidity-quintile assignment (e.g. months with too few names).
    """
    specs: list[tuple[str, tuple[float, ...] | None]] = [
        (f"Q{q}", (float(q),)) for q in QUINTILES
    ]
    specs.append(("liquid_q4q5", LIQUID_QUINTILES))
    specs.append(("full", None))
    return specs


# Ordered universe labels (Q1..Q5, liquid_q4q5, full).
UNIVERSES = tuple(label for label, _ in _universe_members())


def _panel_with_weights(
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
) -> tuple[pd.DataFrame, str]:
    """Return a (permno, yyyymm, target, liq_quintile, w_tilde) frame.

    ``w_tilde`` is the spec's own mean-one training weight family (the note's
    "same w-tilde that defines the training weights"), recreated exactly as the
    weighted training run did via ``compute_formal_utility_weights``.
    """
    target = config["data"]["target_col"]
    panel_sub = panel[["permno", "yyyymm", target]].copy()
    panel_sub["liq_quintile"] = assign_liquidity_quintiles(panel, config)
    panel_sub["w_tilde"] = compute_formal_utility_weights(panel, config, spec)
    return panel_sub, target


def _usable_obs(eval_df: pd.DataFrame, panel_u: pd.DataFrame) -> int:
    """Count obs actually used by ``compute_utility_weighted_r2`` for a leg."""
    merged = eval_df.merge(
        panel_u[["permno", "yyyymm", "w_tilde"]],
        on=["permno", "yyyymm"],
        how="left",
    )
    return int(merged.dropna(subset=["y_true", "y_pred", "w_tilde"]).shape[0])


def compute_deployment_weighted_r2(
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
) -> pd.DataFrame:
    """w-tilde-weighted zero-benchmark OOS R^2 (Eq. 2) per universe.

    Both the standard and weighted predictions are scored under the SAME w-tilde
    (the spec's training weight family). One row is returned per liquidity
    quintile, for the pooled Q4-Q5 liquid subset, and for the full cross-section.
    The global mean-one w-tilde is kept on every subset; because R^2 is a ratio it
    is invariant to a global rescale, so the subsets are directly comparable.
    """
    panel_sub, target = _panel_with_weights(panel, config, spec)
    weight_label = formal_weight_label(spec, config)

    quint = panel_sub[["permno", "yyyymm", "liq_quintile"]]
    std_eval = predictions_for_utility_r2(preds_standard, panel_sub, target).merge(
        quint, on=["permno", "yyyymm"], how="left"
    )
    wt_eval = predictions_for_utility_r2(preds_weighted, panel_sub, target).merge(
        quint, on=["permno", "yyyymm"], how="left"
    )

    rows = []
    for universe, members in _universe_members():
        if members is None:
            std_u, wt_u, panel_u = std_eval, wt_eval, panel_sub
        else:
            std_u = std_eval[std_eval["liq_quintile"].isin(members)]
            wt_u = wt_eval[wt_eval["liq_quintile"].isin(members)]
            panel_u = panel_sub[panel_sub["liq_quintile"].isin(members)]

        r2_std = compute_utility_weighted_r2(std_u, panel_u)["r2_weighted_zero"]
        r2_wt = compute_utility_weighted_r2(wt_u, panel_u)["r2_weighted_zero"]
        delta = (
            (r2_wt - r2_std) * 100
            if pd.notna(r2_std) and pd.notna(r2_wt)
            else np.nan
        )
        rows.append(
            {
                "universe": universe,
                "weight_label": weight_label,
                "r2_weighted_std_pct": r2_std * 100 if pd.notna(r2_std) else np.nan,
                "r2_weighted_wt_pct": r2_wt * 100 if pd.notna(r2_wt) else np.nan,
                "delta_pct": delta,
                "n_obs": _usable_obs(std_u, panel_u),
            }
        )
    return pd.DataFrame(rows)


def _monthly_quintile_partials(
    preds: pd.DataFrame,
    panel_sub: pd.DataFrame,
    target: str,
) -> dict[str, pd.DataFrame]:
    """Per-month weighted-error partial sums for one prediction set.

    Returns sum(w) and sum(w*err^2) aggregated two ways:
      ``by_month_full`` : per yyyymm over ALL stocks (including those without a
                          quintile), used for the full-universe metric;
      ``by_quintile``   : per (yyyymm, liq_quintile), summed across the requested
                          quintiles for per-quintile and pooled-subset metrics.

    The weighted sum does not skip NaN the way an unweighted ``.mean()`` did, so
    incomplete rows are dropped explicitly before aggregating.
    """
    keep = ["permno", "yyyymm", target, "liq_quintile", "w_tilde"]
    merged = preds.merge(panel_sub[keep], on=["permno", "yyyymm"], how="inner")
    merged = merged.dropna(subset=[target, "prediction", "w_tilde"])

    w = merged["w_tilde"].to_numpy(dtype=np.float64)
    err_sq = (
        merged[target].to_numpy(dtype=np.float64)
        - merged["prediction"].to_numpy(dtype=np.float64)
    ) ** 2
    merged = merged.assign(_sum_w=w, _sum_we=w * err_sq)

    by_month_full = merged.groupby("yyyymm")[["_sum_w", "_sum_we"]].sum()
    by_quintile = (
        merged.dropna(subset=["liq_quintile"])
        .groupby(["yyyymm", "liq_quintile"])[["_sum_w", "_sum_we"]]
        .sum()
    )
    return {"by_month_full": by_month_full, "by_quintile": by_quintile}


def _wmse_from_partials(
    parts: dict[str, pd.DataFrame],
    members: tuple[float, ...] | None,
) -> pd.Series:
    """Monthly weighted MSE = sum(w*err^2) / sum(w) for the selected universe."""
    if members is None:
        agg = parts["by_month_full"]
    else:
        by_q = parts["by_quintile"]
        mask = by_q.index.get_level_values("liq_quintile").isin(members)
        agg = by_q[mask].groupby(level="yyyymm")[["_sum_w", "_sum_we"]].sum()

    denom = agg["_sum_w"].where(agg["_sum_w"] > 0)
    return (agg["_sum_we"] / denom).rename("wmse")


def compute_deployment_weighted_error_differential(
    preds_standard: pd.DataFrame,
    preds_weighted: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
) -> pd.DataFrame:
    """Monthly w-tilde-weighted MSE differential per universe (stacked).

    Returns one stacked frame with a ``universe`` column covering each liquidity
    quintile, the pooled Q4-Q5 liquid subset, and the full cross-section. Sign
    convention matches ``compute_liquid_squared_error_differential``: positive
    means the standard model has larger weighted error, i.e. weighting helps.
    """
    panel_sub, target = _panel_with_weights(panel, config, spec)
    std_parts = _monthly_quintile_partials(preds_standard, panel_sub, target)
    wt_parts = _monthly_quintile_partials(preds_weighted, panel_sub, target)

    frames = []
    for universe, members in _universe_members():
        std_mse = _wmse_from_partials(std_parts, members).rename("wmse_std")
        wt_mse = _wmse_from_partials(wt_parts, members).rename("wmse_wt")
        diff = pd.DataFrame({"wmse_std": std_mse, "wmse_wt": wt_mse})
        diff.index.name = "yyyymm"
        diff = diff.dropna()
        diff["wmse_diff"] = diff["wmse_std"] - diff["wmse_wt"]
        diff["cumulative_wmse_diff"] = diff["wmse_diff"].cumsum()
        diff = diff.reset_index()
        diff.insert(0, "universe", universe)
        frames.append(diff)

    return pd.concat(frames, ignore_index=True)


def summarize_error_differential(diff_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Newey-West summary of the monthly differential, one row per universe."""
    lags = int(config["inference"]["newey_west_lags"])
    rows = []
    for universe in UNIVERSES:
        series = diff_df.loc[diff_df["universe"] == universe, "wmse_diff"]
        nw = newey_west_tstat(series.to_numpy(dtype=np.float64), lags=lags, config=config)
        rows.append(
            {
                "universe": universe,
                "mean_wmse_diff": nw["mean"],
                "std_err": nw["std_err"],
                "t_stat": nw["t_stat"],
                "p_value": nw["p_value"],
                "n_months": nw["n_obs"],
                "newey_west_lags": lags,
            }
        )
    return pd.DataFrame(rows)


def plot_quintile_error_differentials(
    diff_df: pd.DataFrame,
    out_path: Path,
    model: str,
) -> None:
    """Overlay the cumulative w-tilde-weighted MSE differential by quintile.

    ``diff_df`` is the stacked frame from
    ``compute_deployment_weighted_error_differential``; one line per quintile
    (Q1-Q5) is drawn on a shared month axis (illiquid -> liquid colour ramp).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.analysis.motivation import _set_academic_style

    _set_academic_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    months = sorted(diff_df["yyyymm"].unique())
    pos = {m: i for i, m in enumerate(months)}
    cmap = plt.get_cmap("viridis")
    plotted = False
    for i, q in enumerate(QUINTILES):
        sub = diff_df[diff_df["universe"] == f"Q{q}"]
        if sub.empty:
            continue
        ax.plot(
            sub["yyyymm"].map(pos).to_numpy(),
            sub["cumulative_wmse_diff"].to_numpy(),
            label=f"Q{q}",
            color=cmap(i / (len(QUINTILES) - 1)),
            lw=1.3,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return

    ax.axhline(0, color="black", ls="--", lw=0.5)
    ax.set_ylabel(r"Cumulative $\tilde{w}$MSE($M_{std}$) $-$ $\tilde{w}$MSE($M_w$)")
    ax.set_title(
        f"Deployment-weighted squared-error differential by liquidity quintile ({model})"
    )
    ax.legend(title="Liquidity quintile", ncol=5, fontsize=8)

    tick_every = max(len(months) // 10, 1)
    ticks = range(0, len(months), tick_every)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([str(months[i]) for i in ticks], rotation=45, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
