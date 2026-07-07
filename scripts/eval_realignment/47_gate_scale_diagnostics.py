"""Gate-scale diagnostics (referee M3): is the gated advantage scale or signal?

The breakeven gate of script 43 triggers on |centred signal| >= half-spread,
so gate composition depends on the cross-sectional SCALE of the forecast.
Weighted training compresses centred signals (median ~31bps vs ~49bps), so a
sceptic can attribute the gated book's advantage to mechanical shrinkage
rather than ranking information. Three checks, all built by transforming the
alpha maps and re-running the pipeline's own ``build_breakeven_book``:

D  -- forecast-panel diagnostics: monthly cross-sectional rank correlation
      between the standard and weighted centred signals, and the monthly
      dispersion ratio sigma(alpha_wt)/sigma(alpha_std).

V1 -- dispersion-matched counterfactual: rescale the STANDARD model's centred
      signals month by month so their cross-sectional dispersion equals the
      weighted model's, and rebuild the gated standard book (1B_matched).
      The full-rebalance target book is scale-invariant (unit gross), so the
      rescaling acts only through the gate. If SR(1B_matched) reproduces
      SR(2B), the adoption margin is pure scale; the surviving gap is the
      information component.

V2 -- scale-invariant gate: trade the top-k fraction of |alpha|/half-spread
      within each month, k FIXED ACROSS MODELS, implemented by scaling each
      month's alphas by 1/q_t(k) (the (1-k) quantile of the ratio), which
      makes the fixed-threshold gate exactly equivalent to the ratio rule.
      Reported at k = the standard model's original mean pass rate and at the
      weighted model's, so composition is controlled at both anchors.

Outputs one tidy CSV: gate_scale_diagnostics.csv under the spec's
eval_realignment analysis directory.

Usage:
    python scripts/eval_realignment/47_gate_scale_diagnostics.py \
        --model xgboost --weight-spec tc_rank_lam3_500m
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.eval_realignment.breakeven_capacity import (  # noqa: E402
    _half_spread,
    _returns_by_month,
    build_breakeven_book,
    compute_alpha_maps,
    metrics_from_book,
)
from src.analysis.eval_realignment.capacity_portfolio import (  # noqa: E402
    build_capacity_book,
)
from src.analysis.eval_realignment.script_utils import (  # noqa: E402
    eval_realignment_output_dirs,
    load_filtered_specs,
)
from src.analysis.formal.common import load_predictions  # noqa: E402
from src.analysis.formal.script_utils import parse_aum_millions  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.loader import load_processed_panel  # noqa: E402
from src.evaluation.statistics import bootstrap_sharpe_test, sharpe_ratio  # noqa: E402
from src.portfolio.construction import prepare_transaction_cost_context  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("47_gate_scale_diagnostics")


def monthly_sigma(alpha_maps: dict[int, dict[int, float]]) -> dict[int, float]:
    return {ym: float(np.std(list(a.values()), ddof=1))
            for ym, a in alpha_maps.items() if len(a) > 2}


def scale_alphas(alpha_maps, factor_by_month):
    out = {}
    for ym, a in alpha_maps.items():
        f = factor_by_month.get(ym)
        if f is None or not np.isfinite(f) or f <= 0:
            out[ym] = dict(a)
        else:
            out[ym] = {p: v * f for p, v in a.items()}
    return out


def ratio_quantile_factors(alpha_maps, targets_positions, tc_context, spread_col, k):
    """1/q_t(k) per month, q_t(k) = (1-k) quantile of |alpha|/half-spread
    over the month's target names — makes the fixed gate a top-k ratio rule."""
    lookup = tc_context["lookup"]
    factors = {}
    for ym, target in targets_positions.items():
        alphas = alpha_maps.get(ym, {})
        ratios = []
        for p in target:
            a = alphas.get(p)
            if a is None:
                continue
            hs = _half_spread(lookup, p, ym, spread_col)
            if hs > 0:
                ratios.append(abs(a) / hs)
        if len(ratios) < 10:
            continue
        q = float(np.quantile(ratios, 1.0 - k))
        if q > 0:
            factors[ym] = 1.0 / q
    return factors


def pass_fraction(alpha_maps, targets_positions, tc_context, spread_col):
    """Mean share of target names whose |alpha| clears the half-spread."""
    lookup = tc_context["lookup"]
    fracs = []
    for ym, target in targets_positions.items():
        alphas = alpha_maps.get(ym, {})
        n, n_pass = 0, 0
        for p in target:
            a = alphas.get(p)
            if a is None:
                continue
            n += 1
            if abs(a) >= _half_spread(lookup, p, ym, spread_col):
                n_pass += 1
        if n:
            fracs.append(n_pass / n)
    return float(np.mean(fracs)) if fracs else np.nan


def book_net_series(gross_df, positions, tc_context, aum):
    row, series = metrics_from_book(gross_df, positions, tc_context, aum)
    if series is None:
        return None, None
    return row, series["ret_net"]  # yyyymm is already the index


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="xgboost")
    ap.add_argument("--weight-spec", default="tc_rank_lam3_500m")
    ap.add_argument("--aum", nargs="*", default=None)
    args = ap.parse_args()

    config = load_config()
    panel = load_processed_panel()
    if isinstance(panel, tuple):
        panel = panel[0]
    aum_scenarios = parse_aum_millions(args.aum, config)

    _, experiment_dir, _ = eval_realignment_output_dirs(config)

    class _A:  # minimal shim for load_filtered_specs' expected args
        model = args.model
        weights = None
        weight_spec = args.weight_spec
    specs = load_filtered_specs(experiment_dir, _A(), logger)
    assert specs, "no spec matched"
    spec = specs[0]
    logger.info("Spec: %s / %s", args.model, spec.get("spec_name", args.weight_spec))

    tc_context = prepare_transaction_cost_context(panel, config)
    spread_col = tc_context["spread_col"]
    preds_std, preds_wt = load_predictions(spec)

    logger.info("Building targets and alpha maps...")
    targets_std = build_capacity_book(preds_std, panel, config, spec)
    targets_wt = build_capacity_book(preds_wt, panel, config, spec)
    alpha_std = compute_alpha_maps(preds_std, panel, config, spec)
    alpha_wt = compute_alpha_maps(preds_wt, panel, config, spec)
    universe = set(targets_std[1]) | set(targets_wt[1])
    ret_by_month = _returns_by_month(panel, config, universe)

    rows = []

    # ── D: forecast-panel diagnostics ─────────────────────────────
    sig_std, sig_wt = monthly_sigma(alpha_std), monthly_sigma(alpha_wt)
    common_m = sorted(set(sig_std) & set(sig_wt))
    disp_ratio = pd.Series({ym: sig_wt[ym] / sig_std[ym] for ym in common_m})
    rank_corrs = []
    for ym in common_m:
        a, b = alpha_std[ym], alpha_wt[ym]
        ps = sorted(set(a) & set(b))
        if len(ps) > 50:
            sa = pd.Series([a[p] for p in ps])
            sb = pd.Series([b[p] for p in ps])
            rank_corrs.append(sa.rank().corr(sb.rank()))
    rc = pd.Series(rank_corrs)
    rows.append({"part": "D_diagnostics", "statistic": "dispersion_ratio_wt_over_std",
                 "mean": disp_ratio.mean(), "p5": disp_ratio.quantile(0.05),
                 "p95": disp_ratio.quantile(0.95), "n_months": len(disp_ratio)})
    rows.append({"part": "D_diagnostics", "statistic": "alpha_rank_correlation",
                 "mean": rc.mean(), "p5": rc.quantile(0.05),
                 "p95": rc.quantile(0.95), "n_months": len(rc)})
    logger.info("[D] dispersion ratio mean %.3f | rank corr mean %.3f",
                disp_ratio.mean(), rc.mean())

    # original pass fractions (anchors for V2's k)
    k_std = pass_fraction(alpha_std, targets_std[1], tc_context, spread_col)
    k_wt = pass_fraction(alpha_wt, targets_wt[1], tc_context, spread_col)
    rows.append({"part": "D_diagnostics", "statistic": "pass_frac_standard", "mean": k_std})
    rows.append({"part": "D_diagnostics", "statistic": "pass_frac_weighted", "mean": k_wt})
    logger.info("[D] pass fractions: std %.3f, wt %.3f", k_std, k_wt)

    # ── reference books: original 1B and 2B ───────────────────────
    logger.info("Building reference gated books...")
    g1B, p1B = build_breakeven_book(preds_std, panel, config, spec, tc_context,
                                    targets=targets_std, alpha_maps=alpha_std,
                                    ret_by_month=ret_by_month)
    g2B, p2B = build_breakeven_book(preds_wt, panel, config, spec, tc_context,
                                    targets=targets_wt, alpha_maps=alpha_wt,
                                    ret_by_month=ret_by_month)

    # ── V1: dispersion-matched standard gate ──────────────────────
    factors = {ym: sig_wt[ym] / sig_std[ym] for ym in common_m}
    alpha_std_matched = scale_alphas(alpha_std, factors)
    gM, pM = build_breakeven_book(preds_std, panel, config, spec, tc_context,
                                  targets=targets_std, alpha_maps=alpha_std_matched,
                                  ret_by_month=ret_by_month)
    match_pass = pass_fraction(alpha_std_matched, targets_std[1], tc_context, spread_col)
    rows.append({"part": "V1_matched", "statistic": "pass_frac_matched_standard",
                 "mean": match_pass})
    logger.info("[V1] matched-standard pass fraction: %.3f (weighted: %.3f)",
                match_pass, k_wt)

    for aum in aum_scenarios:
        label = aum if isinstance(aum, str) else f"{int(aum/1e6)}M" if aum < 1e9 else "1B"
        _, s1B = book_net_series(g1B, p1B, tc_context, aum)
        _, s2B = book_net_series(g2B, p2B, tc_context, aum)
        _, sM = book_net_series(gM, pM, tc_context, aum)
        if s1B is None or s2B is None or sM is None:
            continue
        idx = s1B.index.intersection(s2B.index).intersection(sM.index)
        s1B, s2B, sM = s1B.loc[idx], s2B.loc[idx], sM.loc[idx]
        t_m_1 = bootstrap_sharpe_test(sM, s1B, config=config)
        t_2_m = bootstrap_sharpe_test(s2B, sM, config=config)
        rows.append({
            "part": "V1_matched", "aum": label, "statistic": "gated_books_annualised",
            "sr_1B": sharpe_ratio(s1B, annualize=True),
            "sr_1B_matched": sharpe_ratio(sM, annualize=True),
            "sr_2B": sharpe_ratio(s2B, annualize=True),
            "d_matched_minus_1B": (sharpe_ratio(sM) - sharpe_ratio(s1B)) * np.sqrt(12),
            "p_matched_vs_1B": t_m_1.get("p_value", np.nan),
            "d_2B_minus_matched": (sharpe_ratio(s2B) - sharpe_ratio(sM)) * np.sqrt(12),
            "p_2B_vs_matched": t_2_m.get("p_value", np.nan),
            "n_months": len(idx),
        })
        logger.info("[V1] %s: 1B %.3f | matched %.3f | 2B %.3f | 2B-matched %+.3f (p=%.4f)",
                    label, rows[-1]["sr_1B"], rows[-1]["sr_1B_matched"],
                    rows[-1]["sr_2B"], rows[-1]["d_2B_minus_matched"],
                    rows[-1]["p_2B_vs_matched"])

    # ── V2: scale-invariant top-k ratio gate ──────────────────────
    for k in sorted({round(k_std, 3), round(k_wt, 3)}):
        f_std = ratio_quantile_factors(alpha_std, targets_std[1], tc_context, spread_col, k)
        f_wt = ratio_quantile_factors(alpha_wt, targets_wt[1], tc_context, spread_col, k)
        a_std_k = scale_alphas(alpha_std, f_std)
        a_wt_k = scale_alphas(alpha_wt, f_wt)
        g1k, p1k = build_breakeven_book(preds_std, panel, config, spec, tc_context,
                                        targets=targets_std, alpha_maps=a_std_k,
                                        ret_by_month=ret_by_month)
        g2k, p2k = build_breakeven_book(preds_wt, panel, config, spec, tc_context,
                                        targets=targets_wt, alpha_maps=a_wt_k,
                                        ret_by_month=ret_by_month)
        for aum in aum_scenarios:
            label = aum if isinstance(aum, str) else f"{int(aum/1e6)}M" if aum < 1e9 else "1B"
            _, s1 = book_net_series(g1k, p1k, tc_context, aum)
            _, s2 = book_net_series(g2k, p2k, tc_context, aum)
            if s1 is None or s2 is None:
                continue
            idx = s1.index.intersection(s2.index)
            s1, s2 = s1.loc[idx], s2.loc[idx]
            t = bootstrap_sharpe_test(s2, s1, config=config)
            rows.append({
                "part": "V2_topk_gate", "aum": label, "k": k,
                "statistic": "adoption_under_scale_invariant_gate",
                "sr_1B_k": sharpe_ratio(s1, annualize=True),
                "sr_2B_k": sharpe_ratio(s2, annualize=True),
                "d_annualised": (sharpe_ratio(s2) - sharpe_ratio(s1)) * np.sqrt(12),
                "p_one_sided": t.get("p_value", np.nan),
                "n_months": len(idx),
            })
            logger.info("[V2] k=%.3f %s: 1B_k %.3f | 2B_k %.3f | d %+.3f (p=%.4f)",
                        k, label, rows[-1]["sr_1B_k"], rows[-1]["sr_2B_k"],
                        rows[-1]["d_annualised"], rows[-1]["p_one_sided"])

    out_dir = ROOT / "outputs/eval_realignment/analysis" / args.model / args.weight_spec
    out_path = out_dir / "gate_scale_diagnostics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
