"""Shared helpers for formal experiment analysis."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.motivation import assign_nyse_quintiles
from src.weighting.schemes import compute_weights

logger = logging.getLogger(__name__)

PROPORTIONAL_TC_SCENARIO = "proportional_tc"
PROPORTIONAL_TC_ALIASES = {
    "proportional",
    "proportional_tc",
    "prop_tc",
    "spread_only",
    "spread-only",
}


def is_proportional_tc_scenario(value: object) -> bool:
    """Return True for the AUM-independent spread-only TC reporting scenario."""
    return isinstance(value, str) and value.strip().lower() in PROPORTIONAL_TC_ALIASES


def parse_tc_reporting_scenario(value: object) -> int | float | str:
    """Parse a realized-return TC scenario.

    Numeric values are dollar AUMs. String aliases such as ``"proportional_tc"``
    select the spread-only proportional-cost case with no market impact.
    """
    if is_proportional_tc_scenario(value):
        return PROPORTIONAL_TC_SCENARIO
    if isinstance(value, str):
        token = value.strip().replace("_", "")
        return int(token)
    return value


def scenario_sizing_aum(value: int | float | str) -> float:
    """Return numeric AUM for sizing market impact in a reporting scenario."""
    if is_proportional_tc_scenario(value):
        return 0.0
    return float(value)


def discover_experiments(base_dir: Path) -> list[dict]:
    """Discover completed standard-vs-weighted formal experiment specs."""
    specs = []
    if not base_dir.exists():
        return specs

    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        std_dir = model_dir / "standard"
        if not (std_dir / "predictions.parquet").exists():
            continue

        for wt_dir in sorted(model_dir.iterdir()):
            if wt_dir.name == "standard" or not wt_dir.is_dir():
                continue
            if not (wt_dir / "predictions.parquet").exists():
                continue

            dirname = wt_dir.name
            if dirname == "dolvol":
                weight_family = "dolvol"
                aum_label = None
                softmax_lambda = None
                tc_rank_lambda = None
            elif dirname.startswith("softmax_rank_lam"):
                weight_family = "softmax_rank"
                aum_label = None
                token = dirname.removeprefix("softmax_rank_lam")
                try:
                    softmax_lambda = float(
                        token.replace("m", "-").replace("p", ".")
                    )
                except ValueError:
                    continue
                tc_rank_lambda = None
            elif dirname.startswith("tc_rank_lam"):
                weight_family = "tc_rank"
                aum_label = dirname
                token = dirname.removeprefix("tc_rank_lam")
                try:
                    lam_token, _aum_token = token.rsplit("_", 1)
                    tc_rank_lambda = float(
                        lam_token.replace("m", "-").replace("p", ".")
                    )
                except ValueError:
                    continue
                softmax_lambda = None
            elif dirname.startswith("tc_"):
                weight_family = "tc"
                aum_label = dirname
                softmax_lambda = None
                tc_rank_lambda = None
            else:
                continue

            specs.append(
                {
                    "model": model_name,
                    "weight_family": weight_family,
                    "aum_label": aum_label,
                    "softmax_lambda": softmax_lambda,
                    "tc_rank_lambda": tc_rank_lambda,
                    "weight_spec": dirname,
                    "std_dir": std_dir,
                    "wt_dir": wt_dir,
                    "spec_label": f"{model_name}_{dirname}",
                }
            )

    return specs


def load_predictions(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load standard and weighted prediction files for one formal spec."""
    preds_standard = pd.read_parquet(spec["std_dir"] / "predictions.parquet")
    preds_weighted = pd.read_parquet(spec["wt_dir"] / "predictions.parquet")
    return preds_standard, preds_weighted


def load_importance(
    spec: dict,
    source: str = "shap",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load standard and weighted importance files for one formal spec.

    ``source="shap"`` reads mean absolute SHAP importance files and is the
    default. ``source="native"`` reads the model-native importance files. For
    ElasticNet these are absolute coefficients; for XGBoost these are native
    feature importances; for neural networks these are permutation-style
    importances when available. ``source="naive"`` is accepted as an alias for
    ``source="native"``.
    """
    aliases = {"naive": "native", "native": "native", "shap": "shap"}
    try:
        source_key = aliases[source]
    except KeyError as exc:
        raise ValueError(
            f"Unknown importance source {source!r}. Use 'native' or 'shap'."
        ) from exc

    filename = f"importance_{source_key}.csv"
    importance_standard = pd.read_csv(spec["std_dir"] / filename)
    importance_weighted = pd.read_csv(spec["wt_dir"] / filename)
    return importance_standard, importance_weighted


def assign_liquidity_quintiles(panel: pd.DataFrame, config: dict) -> pd.Series:
    """Assign Q1-Q5 liquidity buckets using configured monthly breakpoints.

    ``liquidity.quintile_breakpoints`` controls the breakpoint universe:
    ``"nyse"`` uses monthly NYSE breakpoints, while ``"full_sample"`` uses the
    full monthly cross-section. In both cases Q1 is most illiquid and Q5 is most
    liquid for the current primary liquidity variable.
    """
    liq_col = f"liq_{config['liquidity']['primary']}"
    breakpoint_universe = config["liquidity"].get("quintile_breakpoints", "nyse")

    if breakpoint_universe == "full_sample":
        return _assign_full_sample_quintiles(panel, sort_col=liq_col, ascending=True)

    if breakpoint_universe != "nyse":
        raise ValueError(
            "liquidity.quintile_breakpoints must be 'nyse' or 'full_sample', "
            f"got {breakpoint_universe!r}"
        )

    if "exchcd" not in panel.columns:
        logger.warning(
            "exchcd column missing; falling back to full-sample quintiles. "
            "Re-run scripts/00_fetch_data.py to get NYSE breakpoints."
        )
        return _assign_full_sample_quintiles(panel, sort_col=liq_col, ascending=True)

    return assign_nyse_quintiles(panel, sort_col=liq_col, ascending=True)


def _assign_full_sample_quintiles(
    panel: pd.DataFrame,
    sort_col: str,
    ascending: bool = True,
) -> pd.Series:
    """Assign Q1-Q5 using the full monthly cross-section as breakpoints."""

    def _per_month(group):
        vals = group[sort_col]
        valid = vals.dropna()
        if len(valid) < 20:
            return pd.Series(np.nan, index=group.index, dtype="float64")

        ranks = vals.rank(method="first", ascending=ascending)
        quintiles = pd.qcut(
            ranks,
            q=5,
            labels=False,
            duplicates="drop",
        )
        out = quintiles.astype(float) + 1
        out[vals.isna()] = np.nan
        return pd.Series(out, index=group.index, dtype="float64")

    results = []
    for _, group in panel.groupby("yyyymm"):
        results.append(_per_month(group))
    if not results:
        return pd.Series(np.nan, index=panel.index, dtype="float64")
    return pd.concat(results).sort_index()


def parse_tc_aum(aum_label: str | None) -> float | None:
    """Parse a TC folder label such as ``tc_500m`` or ``tc_rank_lam3_500m``."""
    if aum_label is None:
        return None
    token = aum_label.rsplit("_", 1)[-1]
    if not token.endswith("m"):
        raise ValueError(f"Could not parse TC AUM label: {aum_label!r}")
    return float(token[:-1]) * 1_000_000


def formal_weight_label(spec: dict, config: dict) -> str:
    """Human-readable label for the utility weights in a formal spec."""
    family = spec["weight_family"]
    if family == "dolvol":
        return "dollar-volume"
    if family == "softmax_rank":
        lam = spec["softmax_lambda"]
        if lam is None:
            raise ValueError("softmax_rank specs must include an explicit lambda")
        return f"softmax-rank(lambda={float(lam):g})"
    if family == "tc":
        return f"TC ({spec['aum_label']})"
    if family == "tc_rank":
        lam = spec["tc_rank_lambda"]
        if lam is None:
            raise ValueError("tc_rank specs must include an explicit lambda")
        aum = parse_tc_aum(spec["aum_label"])
        aum_m = int(aum / 1_000_000) if aum is not None else "NA"
        return f"TC-rank(lambda={float(lam):g}, {aum_m}m)"
    return family


def compute_formal_utility_weights(
    panel: pd.DataFrame,
    config: dict,
    spec: dict,
) -> pd.Series:
    """Recreate the mean-one utility weights used by the weighted training run."""
    cfg = copy.deepcopy(config)
    family = spec["weight_family"]
    aum = None

    if family == "softmax_rank":
        if spec["softmax_lambda"] is None:
            raise ValueError("softmax_rank specs must include an explicit lambda")
        cfg.setdefault("weighting", {})["softmax_rank_lambda"] = spec[
            "softmax_lambda"
        ]
    elif family == "tc":
        aum = parse_tc_aum(spec["aum_label"])
    elif family == "tc_rank":
        if spec["tc_rank_lambda"] is None:
            raise ValueError("tc_rank specs must include an explicit lambda")
        cfg.setdefault("weighting", {})["tc_rank_lambda"] = spec["tc_rank_lambda"]
        aum = parse_tc_aum(spec["aum_label"])

    return compute_weights(panel, scheme=family, config=cfg, aum=aum)


def predictions_for_utility_r2(
    predictions: pd.DataFrame,
    panel_sub: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Convert formal predictions to the y_true/y_pred schema used by R2 helpers."""
    if "prediction" not in predictions.columns:
        raise ValueError("formal predictions must contain a 'prediction' column")

    return (
        predictions[["permno", "yyyymm", "prediction"]]
        .rename(columns={"prediction": "y_pred"})
        .merge(
            panel_sub[["permno", "yyyymm", target]],
            on=["permno", "yyyymm"],
            how="inner",
        )
        .rename(columns={target: "y_true"})
    )


def aum_label(aum: int | float | str) -> str:
    """Format a TC reporting scenario for filenames and column labels."""
    if is_proportional_tc_scenario(aum):
        return "PropTC"
    if aum < 1_000_000_000:
        return f"{int(aum // 1_000_000)}M"
    return f"{int(aum // 1_000_000_000)}B"
