"""Unit tests for the old-format 2x2 capacity deliverables (script 44)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.eval_realignment.two_by_two_tables import (
    CELLS,
    build_table_values,
    format_two_by_two_rows,
    load_cell_series,
    summarize_gate_diag,
    write_timeseries_workbook,
    write_two_by_two_block,
)
from src.evaluation.statistics import sharpe_ratio

CONFIG = {
    "project": {"output_dir": "outputs", "seed": 42},
    "inference": {
        "newey_west_lags": 6,
        "bootstrap_samples": 100,
        "significance_level": 0.05,
        "one_sided_test": True,
        "calibrate_block_length": False,
        "block_grid": [2, 4, 6],
    },
}

AUM = "500M"


def _yyyymm_seq(n, start=200001):
    out, y, m = [], start // 100, start % 100
    for _ in range(n):
        out.append(y * 100 + m)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _monthly_frame(months, ret_net, row_type, tc=0.002):
    ret_net = np.asarray(ret_net, dtype=float)
    return pd.DataFrame({
        "yyyymm": months,
        "row_type": row_type,
        "ret_gross": ret_net + tc,
        "ret_net": ret_net,
        "transaction_cost": tc,
        "turnover": 0.3,
        "n_long": 50,
        "n_short": 50,
    })


def _write_inputs(tmp_path, n_months=36, gate_months=None, seed=0):
    """Write 42/43 monthly CSVs; returns the per-cell net series used."""
    rng = np.random.RandomState(seed)
    months = _yyyymm_seq(n_months)
    base = rng.normal(0.005, 0.02, n_months)
    series = {
        "1A": base,
        "2A": base + 0.002,
        "1B": base + 0.001,
        "2B": base + 0.004,
    }
    full = pd.concat([
        _monthly_frame(months, series["1A"], "standard"),
        _monthly_frame(months, series["2A"], "weighted"),
    ])
    gm = gate_months if gate_months is not None else n_months
    gate = pd.concat([
        _monthly_frame(months[:gm], series["1B"][:gm], "standard"),
        _monthly_frame(months[:gm], series["2B"][:gm], "weighted"),
    ])
    full.to_csv(tmp_path / f"capacity_portfolio_monthly_{AUM}.csv", index=False)
    gate.to_csv(tmp_path / f"capacity_breakeven_monthly_{AUM}.csv", index=False)
    return months, series


def test_load_cell_series_aligns_on_common_months(tmp_path):
    months, _ = _write_inputs(tmp_path, n_months=36, gate_months=30)
    cells = load_cell_series(tmp_path, AUM)
    assert cells is not None
    for cell in CELLS:
        assert list(cells[cell].index) == months[:30]


def test_load_cell_series_missing_input_returns_none(tmp_path):
    _write_inputs(tmp_path)
    (tmp_path / f"capacity_breakeven_monthly_{AUM}.csv").unlink()
    assert load_cell_series(tmp_path, AUM) is None


def test_format_rows_values_and_decomposition_identity(tmp_path):
    _, series = _write_inputs(tmp_path)
    cells = load_cell_series(tmp_path, AUM)
    rows = {r["metric"]: r["value"] for r in
            format_two_by_two_rows(cells, CONFIG, factor_alphas=False)}

    for cell in CELLS:
        expected = sharpe_ratio(np.asarray(series[cell]), annualize=True)
        assert rows[f"SR_net_annualized({cell})"] == pytest.approx(expected)
        assert rows[f"Net return monthly ({cell})"] == pytest.approx(
            float(np.mean(series[cell]))
        )

    training = rows["Net training effect annualized"]
    portfolio = rows["Net portfolio effect annualized"]
    total = rows["Net total effect annualized"]
    assert training == pytest.approx(
        rows["SR_net_annualized(2A)"] - rows["SR_net_annualized(1A)"]
    )
    assert portfolio == pytest.approx(
        rows["SR_net_annualized(1B)"] - rows["SR_net_annualized(1A)"]
    )
    assert rows["Net interaction annualized"] == pytest.approx(
        total - training - portfolio
    )
    assert np.isfinite(rows["LW p-val (training, net)"])


def test_metric_name_parity_with_old_format(tmp_path):
    _write_inputs(tmp_path)
    cells = load_cell_series(tmp_path, AUM)
    names = {r["metric"] for r in
             format_two_by_two_rows(cells, CONFIG, factor_alphas=False)}
    for expected in [
        "Gross return monthly (1A)", "Net return annualized (2B)",
        "TC mean monthly (1B)", "SR_net_annualized(1B)", "SR_gross_monthly(2A)",
        "Net portfolio effect annualized", "Training share annualized (%)",
        "LW p-val (H3, net)", "LW p-val (total, gross)",
        "Turnover (2B)", "Raw trade sum (1A)",
    ]:
        assert expected in names


def test_timeseries_workbook_sheets_and_columns(tmp_path):
    _write_inputs(tmp_path)
    cells = load_cell_series(tmp_path, AUM)
    path = tmp_path / f"two_by_two_timeseries_{AUM}.xlsx"
    write_timeseries_workbook(cells, path)
    sheets = pd.read_excel(path, sheet_name=None)
    assert set(sheets) == set(CELLS)
    for df in sheets.values():
        assert list(df.columns) == [
            "yyyymm", "gross_return", "transaction_cost", "net_return",
            "turnover", "n_long", "n_short",
        ]
        assert len(df) == 36


def test_table_block_panels(tmp_path):
    from openpyxl import Workbook

    _write_inputs(tmp_path)
    cells = load_cell_series(tmp_path, AUM)
    table = build_table_values(cells, CONFIG)
    diag = summarize_gate_diag(pd.DataFrame({
        "row_type": ["standard", "weighted"],
        "yyyymm": [200001, 200001],
        "n_names": [100, 100], "n_pass": [40, 30],
        "pass_frac": [0.4, 0.3], "gross_pass_frac": [0.6, 0.5],
        "median_abs_alpha": [0.005, 0.003], "median_half_spread": [0.0045, 0.0045],
    }))

    wb = Workbook()
    ws = wb.active
    next_row = write_two_by_two_block(ws, table, diag, 1, "Test block", "subtitle")
    assert next_row > 15
    values = {
        str(c.value) for row in ws.iter_rows() for c in row if c.value is not None
    }
    for needle in [
        "Panel A: Net Sharpe Ratios", "Panel B: 2x2 Decomposition",
        "Panel C: Breakeven-Gate Diagnostics", "Full rebalance", "Breakeven gate",
        "Portfolio effect (1B - 1A)", "Mean pass fraction (weighted)",
    ]:
        assert needle in values
    assert diag["weighted_pass_frac"] == pytest.approx(0.3)
