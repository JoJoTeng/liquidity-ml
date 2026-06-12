"""Implementable mean-variance optimizer (note section 4.5; JKMP 2024, Static tier).

Feeds the existing one-month return predictions into the Static-ML allocator of
Jensen, Kelly, Malamud, and Pedersen (2024): a one-period mean-variance problem
with a quadratic trading-cost penalty and the book inherited from last month,

    max_theta  theta'mu_hat - (gamma/2) theta'Sigma theta
               - 1/2 (theta - theta_drift)' Lambda (theta - theta_drift)
    =>  theta_t = (gamma*Sigma_t + Lambda)^{-1} (mu_hat_t + Lambda theta_drift)

solved in closed form per month (SPD Cholesky). Lambda = 0 gives the cost-blind
Markowitz baseline (memoryless full rebalance); Lambda > 0 trades partially toward
the Markowitz target, more where costs are low — the smooth, risk-aware analogue of
the section-43 gate. theta is in WEALTH-FRACTION units (JKMP's omega): its scale is
a choice variable, so the book is NOT gross-normalized; net/gross exposure are
outputs (leverage diagnostics), and drift does not renormalize.

Design choices (documented in the appendix):
  * Universe: top `top_n` names by dollar volume among names with >= `min_obs`
    valid excess returns in the trailing `window` months (JKMP restrict to the top
    half by market cap with 12-month validity rules).
  * Sigma: Ledoit-Wolf shrinkage on the trailing monthly excess-return window —
    the lightweight alternative to JKMP's daily Barra-style factor model; shared
    by all 2x2 cells each month so estimation noise cancels in the comparisons.
  * Lambda: diagonal quadratic (Kyle-lambda) cost built from the same Frazzini
    inputs as the realized accounting, Lambda_ii = 2*lam*sigma_i*AUM/ADV_i. The
    quadratic shape is what makes the FOC linear (the closed form); realized net
    returns always use the true half-spread + sqrt-impact accounting, so the
    approximation is policed by the evaluation.
  * CE(gamma) is evaluated at the gamma the book was built with (JKMP convention;
    their base case gamma=10, wealth grid up to $100B).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from src.analysis.eval_realignment.capacity_portfolio import (
    CE_GAMMAS,
    MONTHS_PER_YEAR,
    certainty_equivalent,
)
from src.analysis.formal.common import scenario_sizing_aum
from src.evaluation.statistics import sharpe_ratio

logger = logging.getLogger(__name__)

TOP_N = 1000
WINDOW = 60
MIN_OBS = 36
POSITION_EPS = 1e-9


def prepare_mv_inputs(panel: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Wide month x permno matrices of excess returns and dollar volume."""
    target = config["data"]["target_col"]
    dolvol_col = f"liq_{config['liquidity']['primary']}"
    ret_wide = panel.pivot_table(index="yyyymm", columns="permno", values=target)
    dolvol_wide = panel.pivot_table(
        index="yyyymm", columns="permno", values=dolvol_col
    )
    return {"ret_wide": ret_wide.sort_index(), "dolvol_wide": dolvol_wide.sort_index()}


def monthly_universe(
    inputs: dict[str, pd.DataFrame],
    months: list[int],
    top_n: int = TOP_N,
    window: int = WINDOW,
    min_obs: int = MIN_OBS,
) -> dict[int, list[int]]:
    """Per-month universe: top `top_n` by DolVol among names with enough history.

    History counts non-missing excess returns over the `window` months strictly
    before t (forward-shifted return convention: those returns have realized by
    the rebalance date — no lookahead).
    """
    ret_wide = inputs["ret_wide"]
    dolvol_wide = inputs["dolvol_wide"]
    all_months = list(ret_wide.index)
    pos = {m: i for i, m in enumerate(all_months)}

    universes: dict[int, list[int]] = {}
    for t in months:
        i = pos.get(t)
        if i is None or i < window:
            continue
        window_months = all_months[i - window:i]
        counts = ret_wide.loc[window_months].notna().sum()
        if t not in dolvol_wide.index:
            continue
        dolvol = dolvol_wide.loc[t]
        eligible = counts[(counts >= min_obs) & dolvol.reindex(counts.index).notna()]
        if eligible.empty:
            continue
        chosen = (
            dolvol.reindex(eligible.index).sort_values(ascending=False).head(top_n)
        )
        universes[int(t)] = [int(p) for p in chosen.index]
    return universes


def ledoit_wolf_sigma(
    inputs: dict[str, pd.DataFrame],
    t: int,
    permnos: list[int],
    window: int = WINDOW,
) -> np.ndarray:
    """Ledoit-Wolf covariance of the universe's excess returns over [t-window, t-1]."""
    from sklearn.covariance import LedoitWolf

    ret_wide = inputs["ret_wide"]
    all_months = list(ret_wide.index)
    i = all_months.index(t)
    X = ret_wide.loc[all_months[i - window:i], permnos]
    X = X - X.mean(axis=0)
    X = X.fillna(0.0).to_numpy(dtype=np.float64)
    try:
        return LedoitWolf().fit(X).covariance_
    except Exception as exc:  # rank-deficient / degenerate window
        logger.warning("LedoitWolf failed at %s (%s); diagonal fallback", t, exc)
        return np.diag(np.maximum(X.var(axis=0), 1e-8))


def _month_lookup_col(
    lookup: pd.DataFrame,
    yyyymm: int,
    col: str,
    permnos: list[int],
    default: float,
) -> np.ndarray:
    """Vectorized per-month lookup with the _safe_lookup_value default semantics."""
    try:
        month = lookup.xs(yyyymm, level="yyyymm")
        v = month[col].reindex(permnos).to_numpy(dtype=np.float64)
    except (KeyError, ValueError):
        return np.full(len(permnos), default, dtype=np.float64)
    return np.where(np.isfinite(v) & (v > 0), v, default)


def quadratic_lambda(
    permnos: list[int],
    yyyymm: int,
    tc_context: dict[str, Any],
    aum: float,
) -> np.ndarray:
    """Diagonal Kyle-lambda quadratic cost: Lambda_ii = 2*lam*sigma_i*AUM/ADV_i.

    Trading d = dtheta*AUM dollars at linear impact slope lam*sigma_i/ADV_i costs
    dtheta^2 * lam*sigma_i*AUM/ADV_i in return units, i.e. 1/2*Lambda_ii*dtheta^2.
    Same sigma/ADV/lam inputs (and lookup-miss defaults) as the realized accounting.
    """
    sigma = _month_lookup_col(
        tc_context["lookup"], yyyymm, tc_context["sigma_col"], permnos, 0.02
    )
    adv = _month_lookup_col(
        tc_context["lookup"], yyyymm, tc_context["adv_col"], permnos, 1e6
    )
    return 2.0 * tc_context["lam"] * sigma * aum / adv


def solve_static_ml(
    mu: np.ndarray,
    sigma: np.ndarray,
    lam_diag: np.ndarray | None,
    theta_drift: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Closed-form Static-ML solve; lam_diag None or zeros => plain Markowitz."""
    lhs = gamma * sigma
    rhs = mu.copy()
    if lam_diag is not None and np.any(lam_diag > 0):
        lhs = lhs + np.diag(lam_diag)
        rhs = rhs + lam_diag * theta_drift
    try:
        factor = cho_factor(lhs)
    except np.linalg.LinAlgError:
        factor = cho_factor(lhs + 1e-8 * np.eye(lhs.shape[0]))
    return cho_solve(factor, rhs)


def _drift_one_month(
    positions: dict[int, float],
    yyyymm: int,
    lookup: pd.DataFrame,
) -> dict[int, float]:
    """Wealth-fraction drift theta*(1+r)/(1+theta'r) — gross is NOT renormalized."""
    if not positions:
        return {}
    permnos = list(positions.keys())
    w = np.array([positions[p] for p in permnos], dtype=np.float64)
    try:
        month = lookup.xs(yyyymm, level="yyyymm")
        r = month["ret"].reindex(permnos).to_numpy(dtype=np.float64)
        r = np.where(np.isfinite(r), r, 0.0)
    except (KeyError, ValueError):
        r = np.zeros(len(permnos), dtype=np.float64)
    r_p = float(w @ r)
    denom = 1.0 + r_p
    if abs(denom) < 1e-6:
        denom = 1.0
    drifted = w * (1.0 + r) / denom
    return {p: float(v) for p, v in zip(permnos, drifted)}


def _drift_unnormalized(
    positions_prev: dict[int, float],
    from_yyyymm: int,
    to_yyyymm: int,
    lookup: pd.DataFrame,
    all_months: list[int],
) -> dict[int, float]:
    """Compound the wealth-fraction drift over every month in [from, to).

    For consecutive rebalances this is a single step; when a month was skipped
    (degenerate universe) the carried book compounds through each intervening
    month rather than applying one month's return across the whole gap.
    """
    drifted = positions_prev
    for m in all_months:
        if from_yyyymm <= m < to_yyyymm:
            drifted = _drift_one_month(drifted, m, lookup)
    return drifted


def monthly_universe_screen(
    inputs: dict[str, pd.DataFrame],
    months: list[int],
    screen_pct: float = 0.60,
) -> dict[int, list[int]]:
    """Per-month universe: the within-month top `screen_pct` of names by DolVol.

    The factor-covariance universe (matches script 45's section-5.1 screen); no
    return-history requirement — the risk model's median-idio fallback covers
    short-history names.
    """
    dolvol_wide = inputs["dolvol_wide"]
    universes: dict[int, list[int]] = {}
    for t in months:
        if t not in dolvol_wide.index:
            continue
        dolvol = dolvol_wide.loc[t].dropna()
        if dolvol.empty:
            continue
        pct = dolvol.rank(pct=True)
        chosen = dolvol[pct >= 1.0 - screen_pct].sort_values(ascending=False)
        if len(chosen) >= 2:
            universes[int(t)] = [int(p) for p in chosen.index]
    return universes


def rescale_positions_unit_gross(
    gross_df: pd.DataFrame,
    positions: dict[int, dict[int, float]],
) -> tuple[pd.DataFrame, dict[int, dict[int, float]]]:
    """The unit-gross VIEW of a book: theta/Sum|theta| per month, ret_gross rescaled.

    Reports the optimizer's composition at standardized size (comparable with the
    42/43/45 books); months with zero gross are dropped.
    """
    out_positions: dict[int, dict[int, float]] = {}
    records = []
    for _, row in gross_df.iterrows():
        ym = int(row["yyyymm"])
        theta = positions.get(ym, {})
        gross = sum(abs(v) for v in theta.values())
        if gross <= 1e-12:
            continue
        out_positions[ym] = {p: v / gross for p, v in theta.items()}
        rec = row.to_dict()
        rec["ret_gross"] = float(row["ret_gross"]) / gross
        rec["leverage"] = 1.0
        records.append(rec)
    return (
        pd.DataFrame(records, columns=list(gross_df.columns)),
        out_positions,
    )


def run_mv_books(
    preds_by_row: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    config: dict,
    tc_context: dict[str, Any],
    gammas: list[float],
    aums: list[float],
    top_n: int = TOP_N,
    window: int = WINDOW,
    min_obs: int = MIN_OBS,
    cov_model: str = "lw",
    screen_pct: float = 0.60,
    theme_map: dict[str, str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Build all MV books month-outer so Sigma/factorizations are shared.

    cov_model: "lw" (Ledoit-Wolf on the top-N universe; the original path) or
    "factor" (JKMP-style characteristic-factor model on the top-`screen_pct`
    DolVol universe, solved via Woodbury). Returns (books, diag_df); books maps
    key -> (gross_df, positions) with key = (row_type, 'markowitz', gamma, None)
    or (row_type, 'static_ml', gamma, aum). Markowitz positions are
    AUM-independent (Lambda=0): one book per (row, gamma).
    """
    inputs = prepare_mv_inputs(panel, config)
    lookup = tc_context["lookup"]

    # Months where BOTH prediction sets exist; cell universes share the same names.
    pred_maps: dict[str, dict[int, dict[int, float]]] = {}
    for row_type, preds in preds_by_row.items():
        pred_maps[row_type] = {
            int(ym): dict(zip(g["permno"].astype(int), g["prediction"].astype(float)))
            for ym, g in preds.groupby("yyyymm")
        }
    common_months = sorted(set.intersection(*(set(m) for m in pred_maps.values())))

    keys = [
        (row, "markowitz", g, None) for row in pred_maps for g in gammas
    ] + [
        (row, "static_ml", g, aum) for row in pred_maps for g in gammas for aum in aums
    ]
    state: dict[tuple, dict[str, Any]] = {
        k: {"records": [], "positions": {}, "prev": {}, "prev_ym": None} for k in keys
    }
    diag_rows: list[dict] = []
    ret_wide = inputs["ret_wide"]
    all_months = [int(m) for m in ret_wide.index]

    def _priced(p: int, t: int) -> bool:
        for row in pred_maps:
            v = pred_maps[row].get(t, {}).get(p)
            if v is None or not np.isfinite(v):
                return False
        return True

    def _solve_month(t, permnos, solver, n_universe):
        """Shared per-month cell solve/record logic for both covariance paths."""
        r_t = ret_wide.loc[t, permnos].fillna(0.0).to_numpy(dtype=np.float64)
        lam_by_aum = {
            aum: quadratic_lambda(permnos, t, tc_context, aum) for aum in aums
        }
        idx_of = {p: k for k, p in enumerate(permnos)}
        for key in keys:
            row_type, book, gamma, aum = key
            st = state[key]
            mu = np.array(
                [pred_maps[row_type][t][p] for p in permnos], dtype=np.float64
            )
            drift_map = (
                _drift_unnormalized(st["prev"], st["prev_ym"], t, lookup, all_months)
                if st["prev_ym"] is not None
                else {}
            )
            theta_drift = np.zeros(len(permnos))
            for p, w in drift_map.items():
                k = idx_of.get(p)
                if k is not None:
                    theta_drift[k] = w  # leavers are force-sold in the net loop
            lam_diag = lam_by_aum[aum] if book == "static_ml" else None
            theta = solver(mu, lam_diag, theta_drift, gamma)

            positions = {
                int(p): float(v)
                for p, v in zip(permnos, theta)
                if abs(v) > POSITION_EPS
            }
            st["positions"][t] = positions
            st["records"].append(
                {
                    "yyyymm": int(t),
                    "ret_gross": float(theta @ r_t),
                    "n_long": int((theta > POSITION_EPS).sum()),
                    "n_short": int((theta < -POSITION_EPS).sum()),
                    "leverage": float(np.abs(theta).sum()),
                }
            )
            st["prev"] = positions
            st["prev_ym"] = int(t)
        diag_rows.append(
            {"yyyymm": int(t), "n_universe": n_universe, "n_priced": len(permnos)}
        )

    if cov_model == "factor":
        from src.analysis.eval_realignment.factor_covariance import (
            FactorRiskModel,
            load_theme_map,
            woodbury_solve,
        )

        if theme_map is None:
            theme_map = load_theme_map()
        frm = FactorRiskModel(theme_map)
        universes = monthly_universe_screen(inputs, common_months, screen_pct)
        panel_groups = panel.groupby("yyyymm")
        group_keys = set(int(k) for k in panel_groups.groups.keys())

        for t in all_months:
            if t not in group_keys:
                continue
            month_df = panel_groups.get_group(t)
            permnos_m, X_m = frm.exposures(month_df)

            if t in universes and frm.ready:
                pos_of = {int(p): i for i, p in enumerate(permnos_m)}
                permnos = [
                    p for p in universes[t] if _priced(p, t) and p in pos_of
                ]
                if len(permnos) >= 2:
                    Xu = X_m[[pos_of[p] for p in permnos]]
                    omega, d_vec = frm.sigma_parts(np.array(permnos))

                    def solver(mu, lam_diag, theta_drift, gamma,
                               Xu=Xu, omega=omega, d_vec=d_vec):
                        a = gamma * d_vec
                        rhs = mu
                        if lam_diag is not None and np.any(lam_diag > 0):
                            a = a + lam_diag
                            rhs = mu + lam_diag * theta_drift
                        return woodbury_solve(a, Xu, gamma * omega, rhs)

                    _solve_month(t, permnos, solver, len(universes[t]))

            # Fold in month-t factor return AFTER the solve (it realizes at t+1).
            y = month_df["excess_ret"].to_numpy(dtype=np.float64)
            frm.update(X_m, permnos_m, y)
    else:
        universes = monthly_universe(inputs, common_months, top_n, window, min_obs)
        for t in sorted(universes.keys()):
            permnos = [p for p in universes[t] if _priced(p, t)]
            if len(permnos) < 2:
                continue
            sigma = ledoit_wolf_sigma(inputs, t, permnos, window)

            def solver(mu, lam_diag, theta_drift, gamma, sigma=sigma):
                return solve_static_ml(mu, sigma, lam_diag, theta_drift, gamma)

            _solve_month(t, permnos, solver, len(universes[t]))

    books = {
        k: (
            pd.DataFrame(
                st["records"],
                columns=["yyyymm", "ret_gross", "n_long", "n_short", "leverage"],
            ),
            st["positions"],
        )
        for k, st in state.items()
    }
    return books, pd.DataFrame(diag_rows)


def mv_net_returns(
    gross_df: pd.DataFrame,
    positions: dict[int, dict[int, float]],
    tc_context: dict[str, Any],
    aum_scenario: int | float | str,
) -> pd.DataFrame:
    """Realized net returns with the TRUE Frazzini accounting, unnormalized drift.

    Mirrors capacity_net_returns' tau (half-spread + lam*sigma*sqrt(trade/ADV) on
    the full traded notional) but drifts positions in wealth-fraction space
    (theta*(1+r)/(1+theta'r)) instead of renormalizing gross to one.
    """
    lookup = tc_context["lookup"]
    spread_col = tc_context["spread_col"]
    sigma_col = tc_context["sigma_col"]
    adv_col = tc_context["adv_col"]
    lam = tc_context["lam"]
    aum = scenario_sizing_aum(aum_scenario)

    months = sorted(positions.keys())
    all_months = sorted(
        int(m) for m in lookup.index.get_level_values("yyyymm").unique()
    )
    tc_series: dict[int, float] = {}
    turnover_series: dict[int, float] = {}
    prev: dict[int, float] = {}
    prev_ym: int | None = None
    for yyyymm in months:
        target = positions[yyyymm]
        drifted = (
            _drift_unnormalized(prev, prev_ym, yyyymm, lookup, all_months)
            if prev_ym is not None
            else {}
        )
        names = sorted(set(target) | set(drifted))
        delta = np.array(
            [abs(target.get(p, 0.0) - drifted.get(p, 0.0)) for p in names],
            dtype=np.float64,
        )
        traded = delta > 0
        names_traded = [p for p, m in zip(names, traded) if m]
        delta = delta[traded]
        half_spread = (
            _month_lookup_col(lookup, yyyymm, spread_col, names_traded, 0.01) / 2.0
        )
        impact = np.zeros(len(names_traded), dtype=np.float64)
        if aum > 0 and len(names_traded):
            sigma = _month_lookup_col(lookup, yyyymm, sigma_col, names_traded, 0.02)
            adv = _month_lookup_col(lookup, yyyymm, adv_col, names_traded, 1e6)
            impact = lam * sigma * np.sqrt(delta * aum / adv)
        tc_series[yyyymm] = float(np.sum(delta * (half_spread + impact)))
        turnover_series[yyyymm] = 0.5 * float(delta.sum())
        prev = target
        prev_ym = yyyymm

    net = gross_df.copy()
    net["transaction_cost"] = net["yyyymm"].map(tc_series).fillna(0.0)
    net["turnover"] = net["yyyymm"].map(turnover_series).fillna(0.0)
    net["ret_net"] = net["ret_gross"] - net["transaction_cost"]
    return net


def mv_metrics(net: pd.DataFrame, gamma: float) -> tuple[dict, pd.DataFrame]:
    """42-style metric row + the book's own-gamma CE headline + leverage."""
    g = net["ret_gross"].to_numpy(dtype=np.float64)
    n = net["ret_net"].to_numpy(dtype=np.float64)
    row: dict[str, Any] = {
        "gamma": float(gamma),
        "net_ce_own_gamma": certainty_equivalent(n, gamma) * MONTHS_PER_YEAR,
        "gross_ce_own_gamma": certainty_equivalent(g, gamma) * MONTHS_PER_YEAR,
        "gross_sr_annual": sharpe_ratio(g, annualize=True),
        "net_sr_annual": sharpe_ratio(n, annualize=True),
        "gross_mean_annual": float(np.nanmean(g)) * MONTHS_PER_YEAR,
        "net_mean_annual": float(np.nanmean(n)) * MONTHS_PER_YEAR,
        "net_sd_annual": float(np.nanstd(n, ddof=1)) * np.sqrt(MONTHS_PER_YEAR),
        "turnover_mean": (
            float(net["turnover"].iloc[1:].mean()) if len(net) > 1 else np.nan
        ),
        "tc_mean_monthly": (
            float(net["transaction_cost"].iloc[1:].mean()) if len(net) > 1 else np.nan
        ),
        "leverage_mean": float(net["leverage"].mean()),
        "n_months": int(np.isfinite(n).sum()),
    }
    # Track-grid CEs so compare_standard_vs_weighted's CE-diff columns populate.
    for g_ in CE_GAMMAS:
        row[f"net_ce_annual_g{int(g_)}"] = (
            certainty_equivalent(n, g_) * MONTHS_PER_YEAR
        )
        row[f"gross_ce_annual_g{int(g_)}"] = (
            certainty_equivalent(g, g_) * MONTHS_PER_YEAR
        )
    series = net.set_index("yyyymm")[
        ["ret_gross", "ret_net", "transaction_cost", "turnover",
         "n_long", "n_short", "leverage"]
    ]
    return row, series


def plot_implementable_frontier(
    rows: list[dict],
    out_path: Path,
    model: str,
    aum_label_str: str,
) -> None:
    """Net mean vs net SD (annualized), points per gamma, lines per (book x row)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.analysis.motivation import _set_academic_style

    _set_academic_style()
    styles = {
        ("markowitz", "standard"): ("#4C72B0", "--", "o"),
        ("markowitz", "weighted"): ("#DD8452", "--", "o"),
        ("static_ml", "standard"): ("#4C72B0", "-", "s"),
        ("static_ml", "weighted"): ("#DD8452", "-", "s"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    df = pd.DataFrame(rows)
    for (book, row_type), sub in df.groupby(["book", "row_type"]):
        sub = sub.sort_values("net_sd_annual")
        color, ls, marker = styles.get((book, row_type), ("#555555", "-", "o"))
        ax.plot(
            sub["net_sd_annual"], sub["net_mean_annual"],
            color=color, ls=ls, marker=marker, lw=1.2, ms=5,
            label=f"{row_type} {book}",
        )
        for _, r in sub.iterrows():
            ax.annotate(
                f"γ={r['gamma']:g}", (r["net_sd_annual"], r["net_mean_annual"]),
                fontsize=7, xytext=(4, 3), textcoords="offset points",
            )
    ax.axhline(0, color="black", ls=":", lw=0.5)
    ax.set_xlabel("Net return SD (annualized)")
    ax.set_ylabel("Net mean return (annualized)")
    ax.set_title(f"Implementable frontier ({model}, {aum_label_str})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
