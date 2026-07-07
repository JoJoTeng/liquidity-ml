"""Inference supplement for the two-by-two designs (referee M2/M4).

Computes the contrasts the main pipeline does not persist, from the cached
monthly cell series (no book reconstruction):

Part A -- pairwise studentised Ledoit-Wolf bootstrap tests (identical
machinery and seed as the pipeline, src.evaluation.statistics.bootstrap_
sharpe_test), long-short capacity book and long-only book, per AUM:
    execution   1B vs 1A   (computed inside every pipeline decomposition but
                            never persisted by script 44; the seed makes this
                            an exact recovery, not a new test)
    adoption    2B vs 1B   (what weighted training adds GIVEN the execution
                            rule -- the contrast referee M2(ii) asks for)
    exec|wt     2B vs 2A   (the execution margin under weighted training)

Part B -- factor-alpha differential 2B minus 1B (referee M2(iv)): monthly
(2B - 1B) net series regressed on each factor model, Newey-West t.

Part C -- joint circular-block percentile bootstraps (single resample of
months per draw, statistic recomputed per draw; fixed block length 6, the
calibrated midpoint used throughout the paper; K draws; seed from config):
    C1 AUM monotonicity (referee M4): end-to-end trend, effect at $1b minus
       effect in the proportional scenario, for training/execution/total.
    C2 dose-response contrasts (referee M4), from the 21e cell time series:
       (a) signal-legs minus equal-legs training effect on the full universe
           (the "flip" contrast);
       (b) mean training effect over the ex-ante predicted-positive cells
           (NYSE universe under all three leg schemes, plus signal legs on
           the full universe).
    C3 interaction per AUM (long-short book), as a sign-stability check.

For each joint statistic we report the observed value and the bootstrap
probability that the statistic is <= 0 (one-sided), i.e. a percentile-
bootstrap sign test; these are deliberately simpler than the studentised
pairwise tests and are labelled as such wherever reported.

Usage:
    python scripts/eval_realignment/46_inference_supplement.py \
        --model xgboost --weight-spec tc_rank_lam3_500m
    # add --quick for a K=200 smoke test of the plumbing
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evaluation.statistics import (  # noqa: E402
    bootstrap_sharpe_test,
    factor_alpha,
    sharpe_ratio,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("46_inference_supplement")

AUMS = ["PropTC", "100M", "500M", "1B"]
BLOCK_LENGTH = 6
FACTOR_MODELS = ["capm", "ff3", "ff5", "ff5_mom"]


# ────────────────────────────────────────────────────────────────────
# Loading helpers
# ────────────────────────────────────────────────────────────────────

def load_ls_cells(base: Path, aum: str) -> pd.DataFrame:
    """Long-short capacity cells: 1A/2A from full rebalance, 1B/2B gated."""
    fullr = pd.read_csv(base / f"capacity_portfolio_monthly_{aum}.csv")
    gated = pd.read_csv(base / f"capacity_breakeven_monthly_{aum}.csv")

    def cell(df, rt):
        return df[df.row_type == rt].set_index("yyyymm")["ret_net"]

    out = pd.DataFrame({
        "1A": cell(fullr, "standard"), "2A": cell(fullr, "weighted"),
        "1B": cell(gated, "standard"), "2B": cell(gated, "weighted"),
    }).dropna()
    return out


def load_lo_cells(base: Path, aum: str) -> pd.DataFrame:
    """Long-only cells: plain (A) vs hysteresis (B) x standard/weighted."""
    m = pd.read_csv(base / f"longonly_q5_monthly_{aum}.csv")

    def cell(book, rt):
        sub = m[(m.book == book) & (m.row_type == rt)]
        return sub.set_index("yyyymm")["ret_net"]

    out = pd.DataFrame({
        "1A": cell("plain", "standard"), "2A": cell("plain", "weighted"),
        "1B": cell("hysteresis", "standard"), "2B": cell("hysteresis", "weighted"),
    }).dropna()
    return out


def load_dose_response_cells(formal_base: Path, aum: str) -> dict[str, pd.DataFrame]:
    """21e monthly net returns for cells 1A/2A, per leg x universe."""
    LEGS = {
        "equal": "prediction_quantile",
        "signal": "prediction_quantile_signal_weight",
        "value": "prediction_quantile_value_weight",
    }
    UNIS = ["full", "nyse", "top60"]
    grids: dict[str, pd.DataFrame] = {}
    for leg, folder in LEGS.items():
        for uni in UNIS:
            # The TRUE dollar-neutral book series underlying the 2x2 cells
            # lives in the two_by_two_timeseries workbook, one sheet per cell
            # (the prediction_quantile_timeseries CSV is the standalone
            # per-quantile object and does NOT reproduce the table effects).
            f = (formal_base / folder / "stock_universe" / uni /
                 f"two_by_two_timeseries_{aum}.xlsx")
            piv = pd.DataFrame({
                c: pd.read_excel(f, sheet_name=c).set_index("yyyymm")["net_return"]
                for c in ["1A", "2A"]
            })
            grids[f"{leg}|{uni}"] = piv.dropna()
    return grids


# ────────────────────────────────────────────────────────────────────
# Joint circular-block bootstrap
# ────────────────────────────────────────────────────────────────────

def circular_block_indices(T: int, block: int, K: int, seed: int) -> np.ndarray:
    """K x T index matrix of circular block resamples."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(K, n_blocks))
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]) % T
    return idx.reshape(K, -1)[:, :T]


def joint_bootstrap(stat_fn, T: int, K: int, seed: int) -> tuple[float, float, np.ndarray]:
    """Observed statistic, one-sided p (share of draws <= 0), draws."""
    obs = stat_fn(np.arange(T))
    idx = circular_block_indices(T, BLOCK_LENGTH, K, seed)
    draws = np.array([stat_fn(idx[k]) for k in range(K)])
    p_le0 = (1 + np.sum(draws <= 0)) / (K + 1)
    return obs, p_le0, draws


def sr(x: np.ndarray) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s) if s > 0 else np.nan


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="xgboost")
    ap.add_argument("--weight-spec", default="tc_rank_lam3_500m")
    ap.add_argument("--quick", action="store_true",
                    help="K=200 smoke test (plumbing only; p-values not usable)")
    args = ap.parse_args()

    config = load_config()
    seed = int(config["project"]["seed"])
    K = 200 if args.quick else 5000

    eval_base = ROOT / "outputs/eval_realignment/analysis" / args.model / args.weight_spec
    formal_base = ROOT / "outputs/formalanalysis/analysis" / args.model / args.weight_spec
    out_dir = eval_base
    rows = []

    # ── Part A: pairwise studentised LW tests ─────────────────────
    for book, loader in [("long_short", load_ls_cells), ("long_only", load_lo_cells)]:
        for aum in AUMS:
            cells = loader(eval_base, aum)
            T = len(cells)
            logger.info("[A] %s %s: %d months", book, aum, T)
            for name, hi, lo in [("execution_1B_vs_1A", "1B", "1A"),
                                 ("adoption_2B_vs_1B", "2B", "1B"),
                                 ("exec_given_wt_2B_vs_2A", "2B", "2A")]:
                res = bootstrap_sharpe_test(cells[hi], cells[lo], K=K, config=config)
                rows.append({
                    "part": "A_pairwise_LW", "book": book, "aum": aum,
                    "statistic": name,
                    "sr_diff_monthly": sharpe_ratio(cells[hi]) - sharpe_ratio(cells[lo]),
                    "sr_diff_annualised": (sharpe_ratio(cells[hi]) - sharpe_ratio(cells[lo])) * np.sqrt(12),
                    "p_value_one_sided": res.get("p_value", np.nan),
                    "K": K, "n_months": T,
                })
                logger.info("    %-24s dSR(ann)=%+.3f p=%.4f", name,
                            rows[-1]["sr_diff_annualised"], rows[-1]["p_value_one_sided"])

    # ── Part B: alpha differential 2B - 1B ────────────────────────
    for book, loader in [("long_short", load_ls_cells), ("long_only", load_lo_cells)]:
        for aum in AUMS:
            cells = loader(eval_base, aum)
            diff = (cells["2B"] - cells["1B"]).rename("ret_long_short").reset_index()
            for fm in FACTOR_MODELS:
                try:
                    a = factor_alpha(diff, model=fm, config=config)
                    rows.append({
                        "part": "B_alpha_diff_2B_1B", "book": book, "aum": aum,
                        "statistic": f"alpha_{fm}",
                        "alpha_annualised": a.get("alpha_annual", np.nan),
                        "t_stat": a.get("alpha_tstat", np.nan),
                        "p_value": a.get("alpha_pvalue", np.nan),
                        "n_months": len(diff),
                    })
                except Exception as e:  # factor files may be absent in quick envs
                    logger.warning("    alpha %s %s %s failed: %s", book, aum, fm, e)

    # ── Part C1: AUM monotonicity (long-short book) ───────────────
    panels = {aum: load_ls_cells(eval_base, aum) for aum in AUMS}
    common = panels["PropTC"].index
    for aum in AUMS:
        common = common.intersection(panels[aum].index)
    mats = {aum: panels[aum].loc[common].values for aum in AUMS}  # T x 4 (1A,2A,1B,2B)
    colmap = {c: i for i, c in enumerate(panels["PropTC"].columns)}
    T = len(common)
    logger.info("[C1] AUM monotonicity on %d common months", T)

    def effect(mat_idx, aum, kind):
        m = mats[aum][mat_idx]
        s = {c: sr(m[:, colmap[c]]) for c in colmap}
        if kind == "training":
            return s["2A"] - s["1A"]
        if kind == "execution":
            return s["1B"] - s["1A"]
        return s["2B"] - s["1A"]

    for kind in ["training", "execution", "total"]:
        def trend(idx, kind=kind):
            return (effect(idx, "1B", kind) - effect(idx, "PropTC", kind)) * np.sqrt(12)
        obs, p, _ = joint_bootstrap(trend, T, K, seed)
        rows.append({"part": "C1_aum_trend", "book": "long_short",
                     "statistic": f"{kind}_effect_1B_minus_PropTC_annualised",
                     "observed": obs, "p_boot_le0_one_sided": p, "K": K, "n_months": T})
        logger.info("    trend(%s) = %+.3f, P(<=0) = %.4f", kind, obs, p)

    # ── Part C2: dose-response contrasts (500M grid) ──────────────
    grids = load_dose_response_cells(formal_base, "500M")
    gcommon = None
    for g in grids.values():
        gcommon = g.index if gcommon is None else gcommon.intersection(g.index)
    gmats = {k: v.loc[gcommon].values for k, v in grids.items()}  # T x 2 (1A,2A)
    Tg = len(gcommon)
    logger.info("[C2] dose-response joint bootstrap on %d common months", Tg)

    def cell_training(idx, key):
        m = gmats[key][idx]
        return (sr(m[:, 1]) - sr(m[:, 0])) * np.sqrt(12)

    def contrast_signal_vs_equal_full(idx):
        return cell_training(idx, "signal|full") - cell_training(idx, "equal|full")

    PREDICTED_POSITIVE = ["equal|nyse", "signal|nyse", "value|nyse", "signal|full"]

    def mean_predicted_positive(idx):
        return float(np.mean([cell_training(idx, k) for k in PREDICTED_POSITIVE]))

    for name, fn in [("signal_minus_equal_full_universe", contrast_signal_vs_equal_full),
                     ("mean_training_predicted_positive_cells", mean_predicted_positive)]:
        obs, p, _ = joint_bootstrap(fn, Tg, K, seed)
        rows.append({"part": "C2_dose_response", "book": "sorted",
                     "statistic": name + "_annualised",
                     "observed": obs, "p_boot_le0_one_sided": p, "K": K, "n_months": Tg})
        logger.info("    %s = %+.3f, P(<=0) = %.4f", name, obs, p)

    # ── Part C3: interaction sign-stability per AUM ───────────────
    for aum in AUMS:
        mat = mats[aum]

        def interaction(idx, mat=mat):
            m = mat[idx]
            s = {c: sr(m[:, colmap[c]]) for c in colmap}
            return ((s["2B"] - s["1A"]) - (s["1B"] - s["1A"]) - (s["2A"] - s["1A"])) * np.sqrt(12)
        obs, p, draws = joint_bootstrap(interaction, T, K, seed)
        p_two = 2 * min(p, 1 - p)
        rows.append({"part": "C3_interaction", "book": "long_short", "aum": aum,
                     "statistic": "interaction_annualised",
                     "observed": obs, "p_boot_le0_one_sided": p,
                     "p_boot_two_sided": p_two, "K": K, "n_months": T})
        logger.info("    interaction(%s) = %+.3f, two-sided P = %.4f", aum, obs, p_two)

    out = pd.DataFrame(rows)
    suffix = "_quick" if args.quick else ""
    out_path = out_dir / f"inference_supplement{suffix}.csv"
    out.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(out), out_path)


if __name__ == "__main__":
    main()
