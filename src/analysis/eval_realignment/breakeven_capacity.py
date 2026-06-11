"""Breakeven-gated capacity portfolio: a parameter-free per-stock no-trade gate.

The section-4.2 capacity book (capacity_portfolio.py) rebalances fully to its target
theta*_t = w̃·(r̂ − rbar_w) every month, churning on signal noise and paying heavy
transaction costs at scale. This module adds a turnover-reducing execution layer that
gates each name's TRADE by its own breakeven: a trade adds value only if the edge it
captures exceeds the cost of capturing it. Both sides are known, in the same units
(monthly returns):

    alpha_i = r̂_i − rbar_w_t        (the centered signal the book harvests)
    c_i     = ½·spread_i            (the one-way proportional cost the net-return
                                     accounting charges per unit traded)

Gate: trade name i FULLY to its target iff |alpha_i| >= c_i; otherwise hold the
drifted position. With proportional (linear) costs the myopic optimum is bang-bang
— trade fully or not at all — so the binary gate is the closed-form solution of the
one-period trade-or-not problem, not a heuristic. There is no tuning parameter: the
threshold IS the cost, and the gate is AUM-independent (market impact stays in the
realized net-return accounting, as in script 42).

Conventions (mirroring the 4.2 book): names that leave the prediction universe are
closed; degenerate months are skipped with the prior executed book carried into the
next drift; long/short legs are rescaled to +0.5/−0.5 after gating. Caveats, accepted
by design: the test is myopic (cost paid once, alpha accrues per month held — a name
with |alpha| < c can still pay off over H* = c/|alpha| months), and a held name whose
signal flips sign but fails the gate keeps its old direction until |alpha| clears the
cost.

This module reuses capacity_portfolio.py read-only; the 4.2 book and script 42 are
unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.eval_realignment.capacity_portfolio import (
    CE_GAMMAS,
    MONTHS_PER_YEAR,
    _drift_capacity,
    _safe_lookup_value,
    build_capacity_book,
    capacity_net_returns,
    certainty_equivalent,
)
from src.analysis.formal.common import compute_formal_utility_weights
from src.evaluation.statistics import sharpe_ratio

logger = logging.getLogger(__name__)

SPREAD_DEFAULT = 0.01  # lookup-miss fallback, consistent with _safe_lookup_value use


def _returns_by_month(
    panel: pd.DataFrame, config: dict, months: set[int]
) -> dict[int, dict[int, float]]:
    """Per-month {permno: realized target return}, restricted to ``months``."""
    target = config["data"]["target_col"]
    sub = panel.loc[
        panel["yyyymm"].isin(months), ["permno", "yyyymm", target]
    ].dropna(subset=[target])
    return {
        int(ym): dict(zip(g["permno"].astype(int), g[target].astype(float)))
        for ym, g in sub.groupby("yyyymm")
    }


def compute_alpha_maps(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
) -> dict[int, dict[int, float]]:
    """Per-month centered alpha {permno: r̂ − rbar_w}, same merge as the 4.2 book.

    rbar_w is local to ``build_capacity_book``; recomputing it here (identical merge,
    identical pivot) keeps capacity_portfolio.py untouched.
    """
    target = config["data"]["target_col"]
    sub = panel[["permno", "yyyymm", target]].copy()
    sub["w_tilde"] = compute_formal_utility_weights(panel, config, spec)

    merged = (
        predictions[["permno", "yyyymm", "prediction"]]
        .merge(sub, on=["permno", "yyyymm"], how="inner")
        .dropna(subset=["prediction", target, "w_tilde"])
    )

    alpha_maps: dict[int, dict[int, float]] = {}
    for yyyymm, group in merged.groupby("yyyymm", sort=True):
        w = group["w_tilde"].to_numpy(dtype=np.float64)
        r_hat = group["prediction"].to_numpy(dtype=np.float64)
        sum_w = w.sum()
        if sum_w <= 0:
            continue
        rbar_w = float((w * r_hat).sum() / sum_w)
        alpha = r_hat - rbar_w
        alpha_maps[int(yyyymm)] = {
            int(p): float(a)
            for p, a in zip(group["permno"].to_numpy(), alpha)
        }
    return alpha_maps


def _half_spread(
    lookup: pd.DataFrame, permno: int, yyyymm: int, spread_col: str
) -> float:
    """One-way proportional cost ½·spread for one name-month (default on miss)."""
    try:
        spread = _safe_lookup_value(
            lookup.loc[(permno, yyyymm)], spread_col, SPREAD_DEFAULT
        )
    except KeyError:
        spread = SPREAD_DEFAULT
    return 0.5 * spread


def _gate_trades(
    target: dict[int, float],
    drift: dict[int, float],
    alphas: dict[int, float],
    lookup: pd.DataFrame,
    yyyymm: int,
    spread_col: str,
) -> dict[int, float]:
    """Apply the breakeven gate: full trade iff |alpha| >= ½·spread, else hold drift.

    Names absent from the month's target (left the prediction universe) are closed —
    they are excluded here, and ``capacity_net_returns`` charges TC on the close via
    its target-union-drift loop.
    """
    exec_raw: dict[int, float] = {}
    for permno in target:
        alpha = alphas.get(permno)
        if alpha is not None and abs(alpha) >= _half_spread(
            lookup, permno, yyyymm, spread_col
        ):
            new = target[permno]
        else:
            new = drift.get(permno, 0.0)  # hold; a zero position never opens
        if abs(new) > 1e-12:
            exec_raw[permno] = float(new)
    return exec_raw


def _reimpose_unit_gross_neutral(
    positions: dict[int, float], min_names: int
) -> dict[int, float] | None:
    """Scale long/short legs to +0.5 / -0.5 (dollar-neutral, unit gross)."""
    longs = {p: v for p, v in positions.items() if v > 0}
    shorts = {p: v for p, v in positions.items() if v < 0}
    if len(longs) < min_names or len(shorts) < min_names:
        return None
    sum_l = sum(longs.values())
    sum_s = -sum(shorts.values())
    if sum_l <= 1e-12 or sum_s <= 1e-12:
        return None
    out: dict[int, float] = {}
    for p, v in longs.items():
        out[p] = 0.5 * v / sum_l
    for p, v in shorts.items():
        out[p] = 0.5 * v / sum_s  # v < 0 -> stays negative, sums to -0.5
    return out


def build_breakeven_book(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
    tc_context: dict[str, Any],
    min_names: int = 2,
    targets: tuple[pd.DataFrame, dict[int, dict[int, float]]] | None = None,
    alpha_maps: dict[int, dict[int, float]] | None = None,
    ret_by_month: dict[int, dict[int, float]] | None = None,
) -> tuple[pd.DataFrame, dict[int, dict[int, float]]]:
    """Build the breakeven-gated capacity book (sequential, path-dependent).

    ``targets`` / ``alpha_maps`` / ``ret_by_month`` may be precomputed and passed in
    so the script computes them once per prediction set. Degenerate months (a leg
    below ``min_names`` after gating) are skipped; the last executed book carries
    into the next month's drift.
    """
    if targets is None:
        targets = build_capacity_book(
            predictions, panel, config, spec, min_names=min_names
        )
    _, positions_target = targets
    if not positions_target:
        return pd.DataFrame(
            columns=["yyyymm", "ret_gross", "n_long", "n_short"]
        ), {}

    if alpha_maps is None:
        alpha_maps = compute_alpha_maps(predictions, panel, config, spec)
    if ret_by_month is None:
        ret_by_month = _returns_by_month(panel, config, set(positions_target))

    lookup = tc_context["lookup"]
    spread_col = tc_context["spread_col"]

    records: list[dict[str, Any]] = []
    positions: dict[int, dict[int, float]] = {}
    prev_exec: dict[int, float] = {}
    prev_ym: int | None = None

    for yyyymm in sorted(positions_target.keys()):
        target = positions_target[yyyymm]
        alphas = alpha_maps.get(yyyymm, {})
        drift = (
            _drift_capacity(prev_exec, prev_ym, lookup) if prev_ym is not None else {}
        )
        exec_raw = _gate_trades(target, drift, alphas, lookup, yyyymm, spread_col)
        exec_final = _reimpose_unit_gross_neutral(exec_raw, min_names)
        if exec_final is None:
            continue  # degenerate month; keep last executed book for the next drift

        rmap = ret_by_month.get(yyyymm, {})
        ret_gross = float(sum(th * rmap.get(p, 0.0) for p, th in exec_final.items()))
        records.append(
            {
                "yyyymm": int(yyyymm),
                "ret_gross": ret_gross,
                "n_long": int(sum(1 for v in exec_final.values() if v > 0)),
                "n_short": int(sum(1 for v in exec_final.values() if v < 0)),
            }
        )
        positions[int(yyyymm)] = exec_final
        prev_exec = exec_final
        prev_ym = int(yyyymm)

    gross_df = pd.DataFrame(
        records, columns=["yyyymm", "ret_gross", "n_long", "n_short"]
    )
    return gross_df, positions


def gate_diagnostics(
    positions_target: dict[int, dict[int, float]],
    alpha_maps: dict[int, dict[int, float]],
    tc_context: dict[str, Any],
) -> pd.DataFrame:
    """Per-month gate diagnostics on the target book (AUM-independent).

    gross_pass_frac = Σ|theta*|·1[pass] / Σ|theta*| — the share of the target book's
    gross exposure that clears its own breakeven.
    """
    lookup = tc_context["lookup"]
    spread_col = tc_context["spread_col"]
    rows: list[dict[str, Any]] = []
    for yyyymm in sorted(positions_target.keys()):
        target = positions_target[yyyymm]
        alphas = alpha_maps.get(yyyymm, {})
        abs_alpha, half_spreads, passes, gross, gross_pass = [], [], 0, 0.0, 0.0
        for permno, theta in target.items():
            alpha = alphas.get(permno, 0.0)
            hs = _half_spread(lookup, permno, yyyymm, spread_col)
            ok = abs(alpha) >= hs
            abs_alpha.append(abs(alpha))
            half_spreads.append(hs)
            passes += int(ok)
            gross += abs(theta)
            gross_pass += abs(theta) * ok
        n = len(target)
        rows.append(
            {
                "yyyymm": int(yyyymm),
                "n_names": n,
                "n_pass": passes,
                "pass_frac": passes / n if n else np.nan,
                "gross_pass_frac": gross_pass / gross if gross > 0 else np.nan,
                "median_abs_alpha": float(np.median(abs_alpha)) if n else np.nan,
                "median_half_spread": float(np.median(half_spreads)) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def metrics_from_book(
    gross_df: pd.DataFrame,
    positions: dict[int, dict[int, float]],
    tc_context: dict[str, Any],
    aum_scenario: int | float | str,
) -> tuple[dict | None, pd.DataFrame | None]:
    """Net-of-TC metric row + monthly series for any capacity-style book @ one AUM.

    Same metric schema as script 42 (``capacity_metrics_for_predictions``), but takes
    an already-built (gross_df, positions) so full-rebalance and gated books share it.
    """
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


def plot_breakeven_net_cumret(
    series_map: dict[str, pd.DataFrame],
    out_path: Path,
    model: str,
    aum_label_str: str,
) -> None:
    """Cumulative net return, full-rebalance vs breakeven-gated, std vs weighted.

    All series are placed on a shared calendar axis (union of months) so books with
    different live months stay time-aligned.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.analysis.motivation import _set_academic_style

    _set_academic_style()
    styles = {
        "standard full": ("#4C72B0", "--"),
        "standard gated": ("#4C72B0", "-"),
        "weighted full": ("#DD8452", "--"),
        "weighted gated": ("#DD8452", "-"),
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
        f"Breakeven-gated capacity portfolio, net of costs ({model}, {aum_label_str})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
