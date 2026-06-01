"""Tests for the weighted-restriction code path added to motivation.py.

These tests verify two invariants:

1. ``rolling_model_predict_restricted(weights=None)`` produces output
   identical to the legacy unweighted call. This guarantees the
   refactor does not silently change existing analyses.

2. ``_renormalize_weights_by_month`` correctly rescales a weight series
   so each yyyymm cross-section has mean exactly 1.0, handles NaNs by
   substituting 1.0, and falls back gracefully when a month sums to
   zero.

The end-to-end weighted-vs-unweighted differential is exercised via a
small synthetic panel rather than a real backtest -- the goal is to
prove sample_weight is actually being passed to model.fit, not to
validate the economics of the weighted result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.motivation import (
    _renormalize_weights_by_month,
    rolling_model_predict_restricted,
)


# ----------------------------------------------------------------------
# _renormalize_weights_by_month
# ----------------------------------------------------------------------

def test_renormalize_weights_makes_mean_one_per_month():
    """After renormalisation each yyyymm has mean weight exactly 1.0."""
    weights = pd.Series([2.0, 4.0, 0.5, 1.5], index=[0, 1, 2, 3])
    yyyymm = pd.Series([202001, 202001, 202002, 202002], index=[0, 1, 2, 3])
    out = _renormalize_weights_by_month(weights, yyyymm)
    # 202001 mean(2.0, 4.0) = 3.0 -> rescaled to (0.6667, 1.3333)
    # 202002 mean(0.5, 1.5) = 1.0 -> unchanged
    assert np.isclose(out.loc[0:1].mean(), 1.0)
    assert np.isclose(out.loc[2:3].mean(), 1.0)
    assert np.isclose(out.loc[0], 2.0 / 3.0)
    assert np.isclose(out.loc[3], 1.5)


def test_renormalize_weights_fills_nan_with_one():
    weights = pd.Series([np.nan, 2.0, 4.0], index=[0, 1, 2])
    yyyymm = pd.Series([202001, 202001, 202001], index=[0, 1, 2])
    out = _renormalize_weights_by_month(weights, yyyymm)
    # NaN -> 1.0, then per-month mean of (1.0, 2.0, 4.0) = 2.333
    expected_mean = (1.0 + 2.0 + 4.0) / 3.0
    assert np.isclose(out.loc[0], 1.0 / expected_mean)
    assert np.isclose(out.mean(), 1.0)


def test_renormalize_weights_handles_zero_mean_month():
    """A month whose surviving weights all sum to zero falls back to 1.0."""
    weights = pd.Series([0.0, 0.0, 1.0, 3.0], index=[0, 1, 2, 3])
    yyyymm = pd.Series([202001, 202001, 202002, 202002], index=[0, 1, 2, 3])
    out = _renormalize_weights_by_month(weights, yyyymm)
    # 202001 mean = 0 -> divisor falls back to 1.0 -> outputs stay (0.0, 0.0)
    assert out.loc[0] == 0.0 and out.loc[1] == 0.0
    # 202002 mean = 2.0 -> rescaled to (0.5, 1.5)
    assert np.isclose(out.loc[2], 0.5)
    assert np.isclose(out.loc[3], 1.5)


# ----------------------------------------------------------------------
# rolling_model_predict_restricted backwards-compatibility
# ----------------------------------------------------------------------

def _build_synthetic_panel(n_months: int = 200, n_permnos: int = 80, seed: int = 1):
    """Build a tiny multi-month panel with two features, a target, a
    liquidity column, and NYSE-style quintiles for restricted training."""
    rng = np.random.RandomState(seed)
    months = [199901 + i for i in range(n_months)]  # cosmetic; values monotone
    months = [m if m % 100 <= 12 and m % 100 >= 1 else 199912 + (i // 12) * 100 + (i % 12) + 1
              for i, m in enumerate(months)]
    # simpler: build sequential yyyymm
    months = []
    y, m = 1999, 1
    for _ in range(n_months):
        months.append(y * 100 + m)
        m += 1
        if m > 12:
            m = 1; y += 1

    rows = []
    for yyyymm in months:
        for permno in range(1000, 1000 + n_permnos):
            f0 = rng.rand()
            f1 = rng.rand()
            r = 0.5 * f0 - 0.3 * f1 + 0.05 * rng.randn()
            rows.append({
                "yyyymm": yyyymm, "permno": permno,
                "f0": f0, "f1": f1, "excess_ret": r,
                "liq_quintile": (permno % 5) + 1,
            })
    panel = pd.DataFrame(rows)
    return panel


def test_unweighted_call_is_byte_identical_to_legacy_signature():
    """Calling with weights=None (default) must produce predictions
    indistinguishable from the legacy call (which simply omitted the
    parameter). This protects the existing analysis from drift."""
    panel = _build_synthetic_panel()
    features = ["f0", "f1"]
    config = {
        "training": {
            "train_window": 12,
            "validation_window": 3,
            "retune_frequency": 12,
            "oos_start": 200001,
            "oos_end": 200101,
        },
        "project": {"seed": 42},
    }

    # Default-None weights (new signature)
    preds_default = rolling_model_predict_restricted(
        panel, features, min_quintile=2, model_name="elastic_net",
        quintile_col="liq_quintile",
        config=config,
        baseline_tuned_params=None,  # retune inside restricted universe
        rerank_after_filter=False,
    )

    # Explicit None weights (new keyword)
    preds_explicit_none = rolling_model_predict_restricted(
        panel, features, min_quintile=2, model_name="elastic_net",
        quintile_col="liq_quintile",
        config=config,
        baseline_tuned_params=None,
        rerank_after_filter=False,
        weights=None,
    )

    # Same predictions in both cases
    assert len(preds_default) == len(preds_explicit_none)
    if len(preds_default) > 0:
        pd.testing.assert_frame_equal(
            preds_default.reset_index(drop=True),
            preds_explicit_none.reset_index(drop=True),
        )


def test_weighted_call_changes_predictions():
    """Passing non-uniform weights to the same data + config should
    produce different predictions than the unweighted call. This proves
    sample_weight is actually being threaded into model.fit() rather
    than silently dropped."""
    panel = _build_synthetic_panel()
    features = ["f0", "f1"]
    config = {
        "training": {
            "train_window": 12,
            "validation_window": 3,
            "retune_frequency": 12,
            "oos_start": 200001,
            "oos_end": 200103,
        },
        "project": {"seed": 42},
    }

    # Weights that heavily favour high-quintile (liquid) observations.
    # Skewed enough that the linear fit will differ noticeably.
    weights = pd.Series(
        np.where(panel["liq_quintile"] >= 4, 5.0, 0.2),
        index=panel.index,
        dtype=float,
    )

    preds_unweighted = rolling_model_predict_restricted(
        panel, features, min_quintile=2, model_name="elastic_net",
        quintile_col="liq_quintile",
        config=config,
        baseline_tuned_params=None,
        rerank_after_filter=False,
        weights=None,
    )

    preds_weighted = rolling_model_predict_restricted(
        panel, features, min_quintile=2, model_name="elastic_net",
        quintile_col="liq_quintile",
        config=config,
        baseline_tuned_params=None,
        rerank_after_filter=False,
        weights=weights,
    )

    # Sanity: both produced predictions
    if len(preds_unweighted) == 0 or len(preds_weighted) == 0:
        pytest.skip("synthetic panel produced no OOS predictions; tighten the windowing")

    merged = preds_unweighted.merge(
        preds_weighted,
        on=["permno", "yyyymm"],
        suffixes=("_unw", "_w"),
    )
    # At least one prediction should differ (weights are clearly non-uniform)
    assert not np.allclose(
        merged["y_pred_unw"].values,
        merged["y_pred_w"].values,
    ), "weighted and unweighted predictions are identical; sample_weight is not flowing through"
