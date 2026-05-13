"""
Prepare paper-style Excel tables from formal portfolio outputs.

This script is intentionally lightweight: it does not recompute portfolios. It
reads the CSV outputs produced by ``21e_formal_portfolio_decomposition.py`` and
writes formatted Excel tables for reporting. Multi-table exports use one
workbook per model, one worksheet per weighted-training spec, and one repeated
subtable per AUM case.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402


MODEL_CHOICES = ["elastic_net", "xgboost", "neural_network"]
DEFAULT_AUMS = [10, 100, 500, 1000]
WEIGHT_SPEC_CHOICES = [
    "dolvol",
    "softmax_rank_lam2",
    "softmax_rank_lam3",
    "tc_10m",
    "tc_100m",
    "tc_500m",
    "tc_1000m",
] + [
    f"tc_rank_lam{lam}_{aum}m"
    for lam in (3,)
    for aum in DEFAULT_AUMS
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare formatted Excel tables from formal portfolio outputs"
    )
    parser.add_argument(
        "--table",
        default="table11",
        choices=["table11", "table12"],
        help=(
            "Report table to export. table11 is within-quintile performance; "
            "table12 is the 2x2 decomposition."
        ),
    )
    parser.add_argument(
        "--model",
        default="elastic_net",
        help=(
            "Model family to export. Use one model, a comma-separated list, "
            "or 'all'."
        ),
    )
    parser.add_argument(
        "--weight-spec",
        default="softmax_rank_lam3",
        help=(
            "Formal weight-spec folder to export. Use one spec, a "
            "comma-separated list, or 'all'."
        ),
    )
    parser.add_argument(
        "--portfolio-mode",
        default="long_short",
        choices=["long_short"],
        help="Portfolio output folder produced by 21e.",
    )
    parser.add_argument(
        "--aum",
        default="500",
        help=(
            "AUM in $M for the net Sharpe column. Use one value, a "
            "comma-separated list, or 'all'. Default is 500."
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "Skip missing 21e output combinations instead of failing. Useful "
            "when exporting all models before every model has been run."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .xlsx path. Default writes under outputs/formalanalysis/tables/.",
    )
    return parser.parse_args()


def _aum_label(aum_millions: int) -> str:
    if aum_millions < 1000:
        return f"{aum_millions}M"
    if aum_millions % 1000 == 0:
        return f"{aum_millions // 1000}B"
    return f"{aum_millions / 1000:.1f}B"


def _parse_selection(value: str, choices: Sequence[str], name: str) -> list[str]:
    if value.lower() == "all":
        return list(choices)

    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected).difference(choices))
    if invalid:
        valid = ", ".join(choices)
        raise ValueError(f"Unknown {name}: {invalid}. Valid values are: {valid}")
    if not selected:
        raise ValueError(f"No {name} selected.")
    return selected


def _parse_aums(value: str) -> list[int]:
    if value.lower() == "all":
        return list(DEFAULT_AUMS)

    out: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(int(raw))
        except ValueError as exc:
            raise ValueError(f"AUM must be an integer in $M, got {raw!r}") from exc
    if not out:
        raise ValueError("No AUM selected.")
    return out


def _spec_title(weight_spec: str) -> str:
    if weight_spec.startswith("tc_rank_lam"):
        token = weight_spec.removeprefix("tc_rank_lam")
        try:
            lam_token, aum_token = token.rsplit("_", 1)
            lam = float(lam_token.replace("m", "-").replace("p", "."))
            aum_m = int(aum_token.removesuffix("m"))
            return f"TC-Rank Weights, lambda={lam:g}, training AUM ${_aum_label(aum_m)}"
        except ValueError:
            return weight_spec

    labels = {
        "dolvol": "Dollar-Volume Weights",
        "softmax_rank_lam2": "Softmax-Rank Weights, lambda=2",
        "softmax_rank_lam3": "Softmax-Rank Weights, lambda=3",
        "tc_10m": "TC Weights, $10M",
        "tc_100m": "TC Weights, $100M",
        "tc_500m": "TC Weights, $500M",
        "tc_1000m": "TC Weights, $1B",
    }
    return labels.get(weight_spec, weight_spec)


def _model_title(model: str) -> str:
    labels = {
        "elastic_net": "ElasticNet",
        "xgboost": "XGBoost",
        "neural_network": "Neural Network",
    }
    return labels.get(model, model)


def _quintile_label(q: str) -> str:
    if q == "Q1":
        return "Q1 (Illiquid)"
    if q == "Q5":
        return "Q5 (Liquid)"
    return q


def _selection_label(
    selected: Sequence[object],
    choices: Sequence[object],
    all_label: str,
    multi_label: str,
) -> str:
    if list(selected) == list(choices):
        return all_label
    if len(selected) == 1:
        value = selected[0]
        if isinstance(value, int):
            return _aum_label(value)
        return str(value)
    return f"{len(selected)}_{multi_label}"


def _set_common_widths(ws) -> None:
    widths = {
        "A": 18,
        "B": 11,
        "C": 11,
        "D": 11,
        "E": 11,
        "F": 11,
        "G": 11,
    }
    for col, width in widths.items():
        ws.column_dimensions[col].width = width


def _set_table12_widths(ws) -> None:
    widths = {
        "A": 30,
        "B": 20,
        "C": 20,
    }
    for col, width in widths.items():
        ws.column_dimensions[col].width = width


def _write_table_block(
    ws,
    table: pd.DataFrame,
    start_row: int,
    title: str,
    subtitle: str | None = None,
) -> int:
    thin = Side(style="thin", color="808080")
    medium = Side(style="medium", color="404040")
    header_fill = PatternFill("solid", fgColor="F2F2F2")

    title_row = start_row
    ws.merge_cells(start_row=title_row, start_column=1, end_row=title_row, end_column=7)
    title_cell = ws.cell(row=title_row, column=1, value=title)
    title_cell.font = Font(bold=True, size=12)
    title_cell.alignment = Alignment(horizontal="center")

    group_row = title_row + 2
    if subtitle:
        subtitle_row = title_row + 1
        ws.merge_cells(
            start_row=subtitle_row,
            start_column=1,
            end_row=subtitle_row,
            end_column=7,
        )
        subtitle_cell = ws.cell(row=subtitle_row, column=1, value=subtitle)
        subtitle_cell.font = Font(italic=True, size=10, color="555555")
        subtitle_cell.alignment = Alignment(horizontal="center")
        group_row = title_row + 3

    for first_col, last_col, label in [
        (2, 3, "Gross SR"),
        (4, 5, "Net SR"),
        (6, 7, "Difference"),
    ]:
        ws.merge_cells(
            start_row=group_row,
            start_column=first_col,
            end_row=group_row,
            end_column=last_col,
        )
        cell = ws.cell(row=group_row, column=first_col, value=label)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
        cell.border = Border(top=medium, bottom=thin)
        cell.fill = header_fill

    header_row = group_row + 1
    headers = ["Quintile", "M^std", "M^w", "M^std", "M^w", "Gross", "Net"]
    for col_idx, value in enumerate(headers, start=1):
        cell = ws.cell(row=header_row, column=col_idx, value=value)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
        cell.border = Border(bottom=medium)
        cell.fill = header_fill

    data_start = header_row + 1
    for row_offset, (_, row) in enumerate(table.iterrows()):
        excel_row = data_start + row_offset
        values = [
            row["Quintile"],
            row["Gross_SR_Mstd"],
            row["Gross_SR_Mw"],
            row["Net_SR_Mstd"],
            row["Net_SR_Mw"],
            row["Diff_Gross"],
            row["Diff_Net"],
        ]
        for col_idx, value in enumerate(values, start=1):
            cell = ws.cell(row=excel_row, column=col_idx, value=value)
            cell.alignment = Alignment(
                horizontal="left" if col_idx == 1 else "center"
            )
            if col_idx > 1:
                cell.number_format = "0.000"

    bottom_row = data_start + len(table) - 1
    for col_idx in range(1, 8):
        ws.cell(row=bottom_row, column=col_idx).border = Border(bottom=medium)

    return bottom_row + 1


def _metric_value(metrics: dict[str, float], name: str, path: Path) -> float:
    if name not in metrics:
        raise ValueError(f"{path} missing metric: {name}")
    return metrics[name]


def load_two_by_two_table(
    analysis_dir: Path,
    model: str,
    weight_spec: str,
    portfolio_mode: str,
    aum_millions: int,
) -> dict[str, float]:
    """Load 21e 2x2 decomposition output and extract Table 12 metrics."""
    spec_dir = analysis_dir / model / weight_spec / portfolio_mode
    path = spec_dir / f"two_by_two_{_aum_label(aum_millions)}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing 21e 2x2 output:\n{path}\nRun 21e before preparing Table 12."
        )

    raw = pd.read_csv(path)
    required_cols = {"metric", "value"}
    missing_cols = required_cols.difference(raw.columns)
    if missing_cols:
        raise ValueError(f"{path} missing columns: {sorted(missing_cols)}")

    metrics = raw.set_index("metric")["value"].to_dict()
    return {
        "SR_net_1A": _metric_value(metrics, "SR_net_annualized(1A)", path),
        "SR_net_1B": _metric_value(metrics, "SR_net_annualized(1B)", path),
        "SR_net_2A": _metric_value(metrics, "SR_net_annualized(2A)", path),
        "SR_net_2B": _metric_value(metrics, "SR_net_annualized(2B)", path),
        "training_effect": _metric_value(metrics, "Net training effect", path),
        "portfolio_effect": _metric_value(metrics, "Net portfolio effect", path),
        "total_effect": _metric_value(metrics, "Net total effect", path),
        "interaction": _metric_value(metrics, "Net interaction", path),
        "training_share_pct": _metric_value(metrics, "Training share (%)", path),
    }


def _write_table12_block(
    ws,
    table: dict[str, float],
    start_row: int,
    title: str,
    subtitle: str | None = None,
) -> int:
    thin = Side(style="thin", color="808080")
    medium = Side(style="medium", color="404040")
    header_fill = PatternFill("solid", fgColor="F2F2F2")

    title_row = start_row
    ws.merge_cells(start_row=title_row, start_column=1, end_row=title_row, end_column=3)
    title_cell = ws.cell(row=title_row, column=1, value=title)
    title_cell.font = Font(bold=True, size=12)
    title_cell.alignment = Alignment(horizontal="center")

    header_row = title_row + 1
    if subtitle:
        subtitle_row = title_row + 1
        ws.merge_cells(
            start_row=subtitle_row,
            start_column=1,
            end_row=subtitle_row,
            end_column=3,
        )
        subtitle_cell = ws.cell(row=subtitle_row, column=1, value=subtitle)
        subtitle_cell.font = Font(italic=True, size=10, color="555555")
        subtitle_cell.alignment = Alignment(horizontal="center")
        header_row = title_row + 2

    for col_idx, value in enumerate(["", "Sort on r_hat", "Sort on r_hat - TC"], start=1):
        cell = ws.cell(row=header_row, column=col_idx, value=value)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
        cell.border = Border(bottom=medium)
        cell.fill = header_fill

    panel_a_row = header_row + 1
    ws.merge_cells(
        start_row=panel_a_row,
        start_column=1,
        end_row=panel_a_row,
        end_column=3,
    )
    cell = ws.cell(row=panel_a_row, column=1, value="Panel A: Net Sharpe Ratios")
    cell.font = Font(bold=True, italic=True)
    cell.border = Border(top=thin)

    net_rows = [
        ("Standard training", "1A", table["SR_net_1A"], "1B", table["SR_net_1B"]),
        ("Weighted training", "2A", table["SR_net_2A"], "2B", table["SR_net_2B"]),
    ]
    row_idx = panel_a_row + 1
    for label, left_cell, left_value, right_cell, right_value in net_rows:
        ws.cell(row=row_idx, column=1, value=label)
        ws.cell(row=row_idx, column=2, value=f"{left_cell}: {left_value:.3f}")
        ws.cell(row=row_idx, column=3, value=f"{right_cell}: {right_value:.3f}")
        for col_idx in [1, 2, 3]:
            ws.cell(row=row_idx, column=col_idx).alignment = Alignment(
                horizontal="left" if col_idx == 1 else "center"
            )
        row_idx += 1

    panel_b_row = row_idx
    ws.merge_cells(
        start_row=panel_b_row,
        start_column=1,
        end_row=panel_b_row,
        end_column=3,
    )
    cell = ws.cell(row=panel_b_row, column=1, value="Panel B: Decomposition")
    cell.font = Font(bold=True, italic=True)
    cell.border = Border(top=medium)

    decomp_rows = [
        ("Training effect (2A - 1A)", table["training_effect"], "0.000"),
        ("Portfolio effect (1B - 1A)", table["portfolio_effect"], "0.000"),
        ("Total effect (2B - 1A)", table["total_effect"], "0.000"),
        ("Interaction", table["interaction"], "0.000"),
        ("Training share (%)", table["training_share_pct"], "0.0"),
    ]
    row_idx = panel_b_row + 1
    for label, value, fmt in decomp_rows:
        ws.cell(row=row_idx, column=1, value=label)
        ws.merge_cells(start_row=row_idx, start_column=2, end_row=row_idx, end_column=3)
        value_cell = ws.cell(row=row_idx, column=2, value=value)
        value_cell.number_format = fmt
        value_cell.alignment = Alignment(horizontal="center")
        row_idx += 1

    bottom_row = row_idx
    for col_idx in range(1, 4):
        ws.cell(row=bottom_row, column=col_idx).border = Border(top=medium)

    return bottom_row + 1


def load_within_quintile_table(
    analysis_dir: Path,
    model: str,
    weight_spec: str,
    portfolio_mode: str,
    aum_millions: int,
) -> pd.DataFrame:
    """Load 21e annualized within-quintile SR outputs and make Table 11 rows."""
    spec_dir = analysis_dir / model / weight_spec / portfolio_mode
    std_path = spec_dir / "table3_sr_quintile_std.csv"
    weighted_path = spec_dir / "table3_sr_quintile_weighted.csv"

    missing = [str(p) for p in [std_path, weighted_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing 21e within-quintile output(s):\n"
            + "\n".join(missing)
            + "\nRun 21e before preparing the Excel table."
        )

    net_col = f"Net_SR_{_aum_label(aum_millions)}"
    std = pd.read_csv(std_path)
    weighted = pd.read_csv(weighted_path)

    for path, frame in [(std_path, std), (weighted_path, weighted)]:
        required = {"Quintile", "Gross_SR", net_col}
        missing_cols = required.difference(frame.columns)
        if missing_cols:
            raise ValueError(f"{path} missing columns: {sorted(missing_cols)}")

    out = std[["Quintile", "Gross_SR", net_col]].rename(
        columns={
            "Gross_SR": "Gross_SR_Mstd",
            net_col: "Net_SR_Mstd",
        }
    )
    out = out.merge(
        weighted[["Quintile", "Gross_SR", net_col]].rename(
            columns={
                "Gross_SR": "Gross_SR_Mw",
                net_col: "Net_SR_Mw",
            }
        ),
        on="Quintile",
        how="inner",
    )
    out["Diff_Gross"] = out["Gross_SR_Mw"] - out["Gross_SR_Mstd"]
    out["Diff_Net"] = out["Net_SR_Mw"] - out["Net_SR_Mstd"]
    out["Quintile"] = out["Quintile"].map(_quintile_label)

    return out[
        [
            "Quintile",
            "Gross_SR_Mstd",
            "Gross_SR_Mw",
            "Net_SR_Mstd",
            "Net_SR_Mw",
            "Diff_Gross",
            "Diff_Net",
        ]
    ]


def write_table11_excel(
    table: pd.DataFrame,
    output_path: Path,
    model: str,
    weight_spec: str,
    portfolio_mode: str,
    aum_millions: int,
) -> None:
    """Write a formatted Table 11 workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Table 11"

    no_border = Border()

    title = (
        "Table 11: Within-Quintile Portfolio Performance "
        f"(${_aum_label(aum_millions)} AUM)"
    )
    subtitle = (
        f"{_model_title(model)} - {_spec_title(weight_spec)} - "
        f"{portfolio_mode.replace('_', '-')}"
    )

    bottom_row = _write_table_block(
        ws=ws,
        table=table,
        start_row=1,
        title=title,
        subtitle=subtitle,
    ) - 1

    note_row = bottom_row + 2
    ws.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=7)
    ws.cell(row=note_row, column=1).value = (
        "Difference columns report weighted training minus standard training. "
        "Sharpe ratios are annualized and read from 21e table3_sr_quintile outputs."
    )
    ws.cell(row=note_row, column=1).font = Font(size=9, color="555555")
    ws.cell(row=note_row, column=1).alignment = Alignment(wrap_text=True)

    _set_common_widths(ws)

    for row_idx in range(1, note_row + 1):
        ws.row_dimensions[row_idx].height = 20

    ws.freeze_panes = "A6"
    ws.sheet_view.showGridLines = False

    # Keep a clean workbook: no default borders outside the table.
    for row in ws.iter_rows(min_row=1, max_row=note_row, min_col=1, max_col=7):
        for cell in row:
            if cell.border == no_border:
                cell.border = Border()

    wb.save(output_path)


def _safe_spec_sheet_title(weight_spec: str) -> str:
    return weight_spec[:31]


def _load_model_spec_tables(
    analysis_dir: Path,
    model: str,
    weight_specs: Sequence[str],
    portfolio_mode: str,
    aums: Sequence[int],
    skip_missing: bool,
) -> tuple[dict[str, list[tuple[int, pd.DataFrame]]], list[str]]:
    tables_by_spec: dict[str, list[tuple[int, pd.DataFrame]]] = {}
    skipped: list[str] = []

    for weight_spec in weight_specs:
        spec_tables: list[tuple[int, pd.DataFrame]] = []
        for aum_millions in aums:
            try:
                table = load_within_quintile_table(
                    analysis_dir=analysis_dir,
                    model=model,
                    weight_spec=weight_spec,
                    portfolio_mode=portfolio_mode,
                    aum_millions=aum_millions,
                )
            except (FileNotFoundError, ValueError) as exc:
                if not skip_missing:
                    raise
                skipped.append(
                    f"{model}/{weight_spec}/{portfolio_mode}/{_aum_label(aum_millions)}: {exc}"
                )
                continue
            spec_tables.append((aum_millions, table))
        if spec_tables:
            tables_by_spec[weight_spec] = spec_tables

    return tables_by_spec, skipped


def _load_model_spec_decomp_tables(
    analysis_dir: Path,
    model: str,
    weight_specs: Sequence[str],
    portfolio_mode: str,
    aums: Sequence[int],
    skip_missing: bool,
) -> tuple[dict[str, list[tuple[int, dict[str, float]]]], list[str]]:
    tables_by_spec: dict[str, list[tuple[int, dict[str, float]]]] = {}
    skipped: list[str] = []

    for weight_spec in weight_specs:
        spec_tables: list[tuple[int, dict[str, float]]] = []
        for aum_millions in aums:
            try:
                table = load_two_by_two_table(
                    analysis_dir=analysis_dir,
                    model=model,
                    weight_spec=weight_spec,
                    portfolio_mode=portfolio_mode,
                    aum_millions=aum_millions,
                )
            except (FileNotFoundError, ValueError) as exc:
                if not skip_missing:
                    raise
                skipped.append(
                    f"{model}/{weight_spec}/{portfolio_mode}/{_aum_label(aum_millions)}: {exc}"
                )
                continue
            spec_tables.append((aum_millions, table))
        if spec_tables:
            tables_by_spec[weight_spec] = spec_tables

    return tables_by_spec, skipped


def write_table11_model_workbook(
    model: str,
    tables_by_spec: dict[str, list[tuple[int, pd.DataFrame]]],
    output_path: Path,
    portfolio_mode: str,
    skipped: Sequence[str] | None = None,
) -> None:
    """Write one model workbook with one worksheet per weighted spec."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    default_ws = wb.active

    for idx, (weight_spec, spec_tables) in enumerate(tables_by_spec.items()):
        ws = default_ws if idx == 0 else wb.create_sheet()
        ws.title = _safe_spec_sheet_title(weight_spec)
        ws.sheet_view.showGridLines = False
        _set_common_widths(ws)

        ws.merge_cells("A1:G1")
        ws["A1"] = (
            f"Within-Quintile Portfolio Performance - {_model_title(model)} - "
            f"{_spec_title(weight_spec)}"
        )
        ws["A1"].font = Font(bold=True, size=14)
        ws["A1"].alignment = Alignment(horizontal="center")

        ws.merge_cells("A2:G2")
        ws["A2"] = (
            f"Portfolio mode: {portfolio_mode.replace('_', '-')}. "
            "Each block reports one AUM case."
        )
        ws["A2"].font = Font(size=9, color="555555")
        ws["A2"].alignment = Alignment(horizontal="center", wrap_text=True)

        next_row = 4
        for aum_millions, table in spec_tables:
            title = (
                f"Within-Quintile Portfolio Performance "
                f"(${_aum_label(aum_millions)} AUM)"
            )
            subtitle = (
                f"{_model_title(model)} - {_spec_title(weight_spec)} - "
                f"{portfolio_mode.replace('_', '-')}"
            )
            next_row = _write_table_block(
                ws=ws,
                table=table,
                start_row=next_row,
                title=title,
                subtitle=subtitle,
            )
            next_row += 2

        for row_idx in range(1, next_row):
            ws.row_dimensions[row_idx].height = 20
        ws.freeze_panes = "A6"

    if skipped:
        ws = wb.create_sheet("Skipped")
        ws["A1"] = "Skipped combinations"
        ws["A1"].font = Font(bold=True)
        ws["A2"] = "These combinations did not have complete 21e table3 outputs."
        for row_idx, message in enumerate(skipped, start=4):
            ws.cell(row=row_idx, column=1, value=message)
        ws.column_dimensions["A"].width = 120

    wb.save(output_path)


def write_table12_excel(
    table: dict[str, float],
    output_path: Path,
    model: str,
    weight_spec: str,
    portfolio_mode: str,
    aum_millions: int,
) -> None:
    """Write a formatted single-AUM Table 12 workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Table 12"
    ws.sheet_view.showGridLines = False
    _set_table12_widths(ws)

    title = f"Table 12: 2x2 Decomposition at ${_aum_label(aum_millions)} AUM"
    subtitle = (
        f"{_model_title(model)} - {_spec_title(weight_spec)} - "
        f"{portfolio_mode.replace('_', '-')}"
    )
    next_row = _write_table12_block(
        ws=ws,
        table=table,
        start_row=1,
        title=title,
        subtitle=subtitle,
    )
    for row_idx in range(1, next_row):
        ws.row_dimensions[row_idx].height = 20

    wb.save(output_path)


def write_table12_model_workbook(
    model: str,
    tables_by_spec: dict[str, list[tuple[int, dict[str, float]]]],
    output_path: Path,
    portfolio_mode: str,
    skipped: Sequence[str] | None = None,
) -> None:
    """Write one model workbook with one 2x2-decomposition sheet per spec."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    default_ws = wb.active

    for idx, (weight_spec, spec_tables) in enumerate(tables_by_spec.items()):
        ws = default_ws if idx == 0 else wb.create_sheet()
        ws.title = _safe_spec_sheet_title(weight_spec)
        ws.sheet_view.showGridLines = False
        _set_table12_widths(ws)

        ws.merge_cells("A1:C1")
        ws["A1"] = (
            f"2x2 Decomposition - {_model_title(model)} - "
            f"{_spec_title(weight_spec)}"
        )
        ws["A1"].font = Font(bold=True, size=14)
        ws["A1"].alignment = Alignment(horizontal="center")

        ws.merge_cells("A2:C2")
        ws["A2"] = (
            f"Portfolio mode: {portfolio_mode.replace('_', '-')}. "
            "Each block reports one AUM case."
        )
        ws["A2"].font = Font(size=9, color="555555")
        ws["A2"].alignment = Alignment(horizontal="center", wrap_text=True)

        next_row = 4
        for aum_millions, table in spec_tables:
            title = f"2x2 Decomposition at ${_aum_label(aum_millions)} AUM"
            next_row = _write_table12_block(
                ws=ws,
                table=table,
                start_row=next_row,
                title=title,
            )
            next_row += 1

        for row_idx in range(1, next_row):
            ws.row_dimensions[row_idx].height = 20
        ws.freeze_panes = "A3"

    if skipped:
        ws = wb.create_sheet("Skipped")
        ws["A1"] = "Skipped combinations"
        ws["A1"].font = Font(bold=True)
        ws["A2"] = "These combinations did not have complete 21e 2x2 outputs."
        for row_idx, message in enumerate(skipped, start=4):
            ws.cell(row=row_idx, column=1, value=message)
        ws.column_dimensions["A"].width = 120

    wb.save(output_path)


def _default_output_path(
    output_dir: Path,
    table_name: str,
    model: str,
    weight_specs: Sequence[str],
    portfolio_mode: str,
    aums: Sequence[int],
) -> Path:
    spec_part = _selection_label(
        weight_specs,
        WEIGHT_SPEC_CHOICES,
        "all_specs",
        "specs",
    )
    aum_part = _selection_label(aums, DEFAULT_AUMS, "all_aums", "aums")
    safe_name = f"{table_name}_{model}_{spec_part}_{portfolio_mode}_{aum_part}.xlsx"
    return output_dir / safe_name


def _resolve_model_output_path(
    output_arg: str | None,
    default_dir: Path,
    table_name: str,
    model: str,
    weight_specs: Sequence[str],
    portfolio_mode: str,
    aums: Sequence[int],
    n_models: int,
) -> Path:
    if not output_arg:
        return _default_output_path(
            output_dir=default_dir,
            table_name=table_name,
            model=model,
            weight_specs=weight_specs,
            portfolio_mode=portfolio_mode,
            aums=aums,
        )

    requested = Path(output_arg)
    if n_models == 1 and requested.suffix == ".xlsx":
        return requested

    output_dir = requested if requested.suffix != ".xlsx" else requested.parent
    return _default_output_path(
        output_dir=output_dir,
        table_name=table_name,
        model=model,
        weight_specs=weight_specs,
        portfolio_mode=portfolio_mode,
        aums=aums,
    )


def main() -> None:
    args = parse_args()
    config = load_config()
    analysis_dir = Path(config["project"]["output_dir"]) / "formalanalysis" / "analysis"
    output_dir = Path(config["project"]["output_dir"]) / "formalanalysis" / "tables"

    models = _parse_selection(args.model, MODEL_CHOICES, "model")
    weight_specs = _parse_selection(
        args.weight_spec,
        WEIGHT_SPEC_CHOICES,
        "weight spec",
    )
    aums = _parse_aums(args.aum)

    single_table = len(models) == len(weight_specs) == len(aums) == 1
    if single_table:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = _default_output_path(
                output_dir=output_dir,
                table_name=args.table,
                model=models[0],
                weight_specs=weight_specs,
                portfolio_mode=args.portfolio_mode,
                aums=aums,
            )
        if args.table == "table11":
            table = load_within_quintile_table(
                analysis_dir=analysis_dir,
                model=models[0],
                weight_spec=weight_specs[0],
                portfolio_mode=args.portfolio_mode,
                aum_millions=aums[0],
            )
            write_table11_excel(
                table=table,
                output_path=output_path,
                model=models[0],
                weight_spec=weight_specs[0],
                portfolio_mode=args.portfolio_mode,
                aum_millions=aums[0],
            )
        else:
            table = load_two_by_two_table(
                analysis_dir=analysis_dir,
                model=models[0],
                weight_spec=weight_specs[0],
                portfolio_mode=args.portfolio_mode,
                aum_millions=aums[0],
            )
            write_table12_excel(
                table=table,
                output_path=output_path,
                model=models[0],
                weight_spec=weight_specs[0],
                portfolio_mode=args.portfolio_mode,
                aum_millions=aums[0],
            )
        print(output_path)
        return

    output_paths: list[Path] = []
    skipped_models: list[str] = []
    for model in models:
        if args.table == "table11":
            tables_by_spec, skipped = _load_model_spec_tables(
                analysis_dir=analysis_dir,
                model=model,
                weight_specs=weight_specs,
                portfolio_mode=args.portfolio_mode,
                aums=aums,
                skip_missing=args.skip_missing,
            )
        else:
            tables_by_spec, skipped = _load_model_spec_decomp_tables(
                analysis_dir=analysis_dir,
                model=model,
                weight_specs=weight_specs,
                portfolio_mode=args.portfolio_mode,
                aums=aums,
                skip_missing=args.skip_missing,
            )
        if not tables_by_spec:
            if args.skip_missing:
                skipped_models.append(model)
                continue
            raise FileNotFoundError(f"No matching 21e outputs were available for {model}.")

        output_path = _resolve_model_output_path(
            output_arg=args.output,
            default_dir=output_dir,
            table_name=args.table,
            model=model,
            weight_specs=weight_specs,
            portfolio_mode=args.portfolio_mode,
            aums=aums,
            n_models=len(models),
        )
        if args.table == "table11":
            write_table11_model_workbook(
                model=model,
                tables_by_spec=tables_by_spec,
                output_path=output_path,
                portfolio_mode=args.portfolio_mode,
                skipped=skipped,
            )
        else:
            write_table12_model_workbook(
                model=model,
                tables_by_spec=tables_by_spec,
                output_path=output_path,
                portfolio_mode=args.portfolio_mode,
                skipped=skipped,
            )
        output_paths.append(output_path)

    if not output_paths:
        skipped = ", ".join(skipped_models)
        raise FileNotFoundError(f"No workbooks were written. Skipped models: {skipped}")

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
