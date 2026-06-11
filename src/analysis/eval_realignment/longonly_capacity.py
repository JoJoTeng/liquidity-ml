"""Capacity-weighted long-only Q5 book from the liquid universe (note section 4.3).

The note calls this "the most defensible implementable object": a long-only book
drops the illiquid short leg — the least realistic part of any long-short backtest
(borrow costs and locates are not modeled at all) — and every remaining position is
one a real fund could hold at the stated AUM.

Construction, per month:
  1. Liquid universe L_t = top-60% of the cross-section by dollar volume (the note
     section 5.1's screen), within month.
  2. Q5 target = names in L_t with prediction at or above b_t, the 80th percentile
     of r_hat within L_t.
  3. Plain book (A): members = Q5 target, weights theta_i = w_tilde_i / sum w_tilde
     over members (the spec's own capacity weight — fixing the section-3
     consistency violation of equal/ME-weighted legs). Fully invested: sum theta = 1,
     theta >= 0.
  4. Hysteresis book (B): cost-scaled membership hysteresis, the parameter-free
     breakeven device for a selection book. Selling held name i to buy the marginal
     entrant earns (b_t − r_hat_i) per month and costs ½spread_i + ½spread_entrant,
     so a held name that left Q5 is kept while

         r_hat_i >= b_t − (½spread_i + c̄_m,t),

     with c̄_m,t the median ½spread over the month's Q5 target (the entrant's toll).
     Entries are ungated; leaving the screen/universe forces the sale; weights are
     re-trued to w_tilde over members every month. As spreads → 0 the buffer
     vanishes and the hysteresis book coincides with the plain book.

The section-43 breakeven gate deliberately does NOT transfer here: a quantile
book's turnover is membership churn at the boundary — names whose alpha is still
large — so an |alpha| >= cost gate would pass exactly the trades that need damping.

Returns/costs reuse the capacity machinery unchanged: ``_drift_capacity`` and
``capacity_net_returns`` are sign-agnostic, and the within-leg re-truing trades are
charged transaction costs automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.eval_realignment.capacity_portfolio import _safe_lookup_value
from src.analysis.formal.common import compute_formal_utility_weights

logger = logging.getLogger(__name__)

SCREEN_PCT = 0.40   # keep within-month DolVol percentile >= 0.40 (top 60%)
Q5_QUANTILE = 0.80  # prediction threshold b_t = 80th percentile within the screen
MIN_NAMES = 5


def compute_longonly_inputs(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
    screen_pct: float = SCREEN_PCT,
) -> dict[int, pd.DataFrame]:
    """Per-month liquid-universe frames (permno, prediction, target, w_tilde).

    The merge mirrors ``build_capacity_book``; the top-(1-screen_pct) DolVol screen
    is applied within month BEFORE any selection, on the raw liquidity helper
    ``liq_{config[liquidity][primary]}``.
    """
    target = config["data"]["target_col"]
    dolvol_col = f"liq_{config['liquidity']['primary']}"
    sub = panel[["permno", "yyyymm", target, dolvol_col]].copy()
    sub["w_tilde"] = compute_formal_utility_weights(panel, config, spec)

    merged = (
        predictions[["permno", "yyyymm", "prediction"]]
        .merge(sub, on=["permno", "yyyymm"], how="inner")
        .dropna(subset=["prediction", target, "w_tilde", dolvol_col])
    )
    merged["dolvol_pct"] = merged.groupby("yyyymm")[dolvol_col].rank(pct=True)
    liquid = merged[merged["dolvol_pct"] >= screen_pct]
    return {int(ym): g for ym, g in liquid.groupby("yyyymm", sort=True)}


def _half_spreads(
    permnos: pd.Series, yyyymm: int, lookup: pd.DataFrame, spread_col: str
) -> np.ndarray:
    out = np.empty(len(permnos), dtype=np.float64)
    for k, permno in enumerate(permnos):
        try:
            spread = _safe_lookup_value(
                lookup.loc[(int(permno), yyyymm)], spread_col, 0.01
            )
        except KeyError:
            spread = 0.01
        out[k] = 0.5 * spread
    return out


def build_longonly_book(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
    tc_context: dict[str, Any],
    hysteresis: bool = False,
    min_names: int = MIN_NAMES,
    screen_pct: float = SCREEN_PCT,
    quantile: float = Q5_QUANTILE,
    inputs: dict[int, pd.DataFrame] | None = None,
    diag_out: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict[int, dict[int, float]]]:
    """Build the long-only capacity-weighted Q5 book (plain or hysteresis).

    Returns (gross_df, positions) in the 42/43 shapes: gross_df has
    [yyyymm, ret_gross, n_long, n_short(=0)]; positions maps yyyymm -> {permno: theta}.
    Degenerate months (< min_names members) are skipped; the last recorded
    membership carries into the next month's hysteresis test. When ``diag_out`` is
    a list and ``hysteresis`` is on, per-month diagnostics are appended to it.
    """
    target = config["data"]["target_col"]
    if inputs is None:
        inputs = compute_longonly_inputs(
            predictions, panel, config, spec, screen_pct=screen_pct
        )
    lookup = tc_context["lookup"]
    spread_col = tc_context["spread_col"]

    records: list[dict[str, Any]] = []
    positions: dict[int, dict[int, float]] = {}
    prev_members: set[int] = set()

    for yyyymm in sorted(inputs.keys()):
        g = inputs[yyyymm]
        b_t = float(g["prediction"].quantile(quantile))
        q5 = g[g["prediction"] >= b_t]
        members = q5

        n_buffer = 0
        n_forced = 0
        buffer_width = np.nan
        if hysteresis and prev_members:
            cm = (
                float(np.median(_half_spreads(q5["permno"], yyyymm, lookup, spread_col)))
                if len(q5)
                else 0.005
            )
            held = g[g["permno"].isin(prev_members) & (g["prediction"] < b_t)]
            if len(held):
                hs = _half_spreads(held["permno"], yyyymm, lookup, spread_col)
                keep = held["prediction"].to_numpy(dtype=np.float64) >= b_t - (hs + cm)
                buffer_hold = held[keep]
                n_buffer = int(len(buffer_hold))
                # Width among the names actually retained, consistent with n_buffer_held.
                buffer_width = (
                    float(np.median((hs + cm)[keep])) if keep.any() else np.nan
                )
                members = pd.concat([q5, buffer_hold])
            # held names absent from this month's liquid frame are force-sold
            n_forced = int(len(prev_members - set(g["permno"].astype(int))))

        if len(members) < min_names:
            continue  # degenerate month; keep last membership for the next test
        w = members["w_tilde"].to_numpy(dtype=np.float64)
        sum_w = w.sum()
        if sum_w <= 0:
            continue

        theta = w / sum_w  # fully invested long-only: sum theta = 1, theta >= 0
        r = members[target].to_numpy(dtype=np.float64)
        ret_gross = float((theta * r).sum())
        permnos = members["permno"].to_numpy()
        positions[int(yyyymm)] = {
            int(p): float(t) for p, t in zip(permnos, theta) if t > 0.0
        }
        records.append(
            {
                "yyyymm": int(yyyymm),
                "ret_gross": ret_gross,
                "n_long": len(positions[int(yyyymm)]),
                "n_short": 0,
            }
        )
        if diag_out is not None and hysteresis:
            diag_out.append(
                {
                    "yyyymm": int(yyyymm),
                    "n_members": len(positions[int(yyyymm)]),
                    "n_q5": int(len(q5)),
                    "n_buffer_held": n_buffer,
                    "n_forced_exits": n_forced,
                    "entry_threshold": b_t,
                    "buffer_width_median": buffer_width,
                }
            )
        prev_members = set(positions[int(yyyymm)])

    gross_df = pd.DataFrame(
        records, columns=["yyyymm", "ret_gross", "n_long", "n_short"]
    )
    return gross_df, positions


def summarize_hysteresis_diag(diag: pd.DataFrame) -> dict[str, float]:
    """Panel-C values per training row prefix (caller stacks std/wt)."""
    if diag.empty:
        return {}
    return {
        "mean_members": float(diag["n_members"].mean()),
        "mean_q5": float(diag["n_q5"].mean()),
        "mean_buffer_held": float(diag["n_buffer_held"].mean()),
        "mean_forced_exits": float(diag["n_forced_exits"].mean()),
        "median_entry_threshold": float(diag["entry_threshold"].median()),
        "median_buffer_width": float(diag["buffer_width_median"].median()),
    }


def plot_longonly_net_cumret(
    series_map: dict[str, pd.DataFrame],
    out_path: Path,
    model: str,
    aum_label_str: str,
) -> None:
    """Cumulative net return, plain vs hysteresis x standard vs weighted.

    Shared calendar axis (union of months) so books with different live months
    stay time-aligned.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.analysis.motivation import _set_academic_style

    _set_academic_style()
    styles = {
        "standard plain": ("#4C72B0", "--"),
        "standard hysteresis": ("#4C72B0", "-"),
        "weighted plain": ("#DD8452", "--"),
        "weighted hysteresis": ("#DD8452", "-"),
    }
    months = sorted({m for s in series_map.values() for m in s.index})
    month_pos = {m: i for i, m in enumerate(months)}

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, series in series_map.items():
        ret = series["ret_net"].dropna()
        if ret.empty:
            continue
        cum = (1.0 + ret).cumprod() - 1.0
        color, ls = styles.get(label, ("#555555", "-"))
        x = [month_pos[m] for m in cum.index]
        ax.plot(x, cum.to_numpy(), label=label, color=color, ls=ls, lw=1.3)

    ax.axhline(0, color="black", ls=":", lw=0.5)
    tick_every = max(len(months) // 10, 1)
    ticks = list(range(0, len(months), tick_every))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(months[i]) for i in ticks], rotation=45, fontsize=8)
    ax.set_ylabel("Cumulative net return")
    ax.set_title(
        f"Long-only capacity-weighted Q5 book, net of costs ({model}, {aum_label_str})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
