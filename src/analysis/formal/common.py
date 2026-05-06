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
            elif dirname.startswith("tc_"):
                weight_family = "tc"
                aum_label = dirname
                softmax_lambda = None
            else:
                continue

            specs.append(
                {
                    "model": model_name,
                    "weight_family": weight_family,
                    "aum_label": aum_label,
                    "softmax_lambda": softmax_lambda,
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
    """Assign Q1-Q5 liquidity buckets using monthly NYSE breakpoints."""
    liq_col = f"liq_{config['liquidity']['primary']}"
    if "exchcd" not in panel.columns:
        logger.warning(
            "exchcd column missing; falling back to full-sample quintiles. "
            "Re-run scripts/00_fetch_data.py to get NYSE breakpoints."
        )
        quintiles = pd.Series(np.nan, index=panel.index, name="liq_quintile")
        for _, group in panel.groupby("yyyymm"):
            liq = group[liq_col]
            breaks = liq.quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
            breaks[0] = -np.inf
            breaks[-1] = np.inf
            q = pd.cut(
                liq,
                bins=breaks,
                labels=[1, 2, 3, 4, 5],
                include_lowest=True,
            )
            quintiles.loc[group.index] = q.astype(float)
        return quintiles

    return assign_nyse_quintiles(panel, sort_col=liq_col, ascending=True)


def parse_tc_aum(aum_label: str | None) -> float | None:
    """Parse a TC folder label such as ``tc_500m`` into dollars."""
    if aum_label is None:
        return None
    token = aum_label.removeprefix("tc_")
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


def aum_label(aum: int | float) -> str:
    """Format an AUM value for filenames."""
    if aum < 1_000_000_000:
        return f"{int(aum // 1_000_000)}M"
    return f"{int(aum // 1_000_000_000)}B"
