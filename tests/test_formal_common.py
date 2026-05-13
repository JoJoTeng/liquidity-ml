from pathlib import Path

import pytest

from src.analysis.formal.common import (
    discover_experiments,
    formal_weight_label,
    parse_tc_aum,
)


def _touch_prediction(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "predictions.parquet").touch()


def test_parse_tc_aum_supports_level_and_rank_specs():
    assert parse_tc_aum("tc_500m") == pytest.approx(500_000_000)
    assert parse_tc_aum("tc_rank_lam3_500m") == pytest.approx(500_000_000)


def test_discover_experiments_supports_tc_rank_specs(tmp_path):
    model_dir = tmp_path / "elastic_net"
    _touch_prediction(model_dir / "standard")
    _touch_prediction(model_dir / "tc_rank_lam3_500m")

    specs = discover_experiments(tmp_path)

    assert len(specs) == 1
    spec = specs[0]
    assert spec["model"] == "elastic_net"
    assert spec["weight_family"] == "tc_rank"
    assert spec["weight_spec"] == "tc_rank_lam3_500m"
    assert spec["aum_label"] == "tc_rank_lam3_500m"
    assert spec["tc_rank_lambda"] == pytest.approx(3.0)
    assert spec["softmax_lambda"] is None


def test_formal_weight_label_supports_tc_rank_specs():
    spec = {
        "weight_family": "tc_rank",
        "aum_label": "tc_rank_lam3_500m",
        "tc_rank_lambda": 3.0,
    }

    assert formal_weight_label(spec, {}) == "TC-rank(lambda=3, 500m)"
