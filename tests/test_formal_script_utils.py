from argparse import Namespace
from pathlib import Path

from src.analysis.formal.script_utils import (
    config_for_liquidity_breakpoint,
    liquidity_breakpoint_dir,
    load_filtered_specs,
    namespace_liquidity_breakpoints,
    parse_aum_millions,
    resolve_liquidity_breakpoints,
)


class _NullLogger:
    def info(self, *args, **kwargs):
        pass


def _touch_prediction(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "predictions.parquet").touch()


def test_load_filtered_specs_supports_exact_weight_spec(tmp_path):
    model_dir = tmp_path / "elastic_net"
    _touch_prediction(model_dir / "standard")
    _touch_prediction(model_dir / "dolvol")
    _touch_prediction(model_dir / "tc_500m")

    args = Namespace(model="elastic_net", weights=None, weight_spec="tc_500m")

    specs = load_filtered_specs(tmp_path, args, _NullLogger())

    assert len(specs) == 1
    assert specs[0]["model"] == "elastic_net"
    assert specs[0]["weight_spec"] == "tc_500m"


def test_load_filtered_specs_intersects_family_and_exact_spec(tmp_path):
    model_dir = tmp_path / "elastic_net"
    _touch_prediction(model_dir / "standard")
    _touch_prediction(model_dir / "softmax_rank_lam2")
    _touch_prediction(model_dir / "softmax_rank_lam3")
    _touch_prediction(model_dir / "tc_500m")

    args = Namespace(
        model="elastic_net",
        weights="softmax_rank",
        weight_spec="softmax_rank_lam2",
    )

    specs = load_filtered_specs(tmp_path, args, _NullLogger())

    assert [spec["weight_spec"] for spec in specs] == ["softmax_rank_lam2"]


def test_load_filtered_specs_supports_primary_weight_specs(tmp_path):
    model_dir = tmp_path / "elastic_net"
    _touch_prediction(model_dir / "standard")
    for name in [
        "dolvol",
        "softmax_rank_lam2",
        "softmax_rank_lam3",
        "tc_500m",
        "tc_rank_lam3_500m",
        "tc_1000m",
    ]:
        _touch_prediction(model_dir / name)

    args = Namespace(model="elastic_net", weights=None, weight_spec="primary")

    specs = load_filtered_specs(tmp_path, args, _NullLogger())

    assert [spec["weight_spec"] for spec in specs] == [
        "dolvol",
        "softmax_rank_lam2",
        "tc_500m",
        "tc_rank_lam3_500m",
    ]


def test_load_filtered_specs_supports_comma_separated_weight_specs(tmp_path):
    model_dir = tmp_path / "elastic_net"
    _touch_prediction(model_dir / "standard")
    _touch_prediction(model_dir / "dolvol")
    _touch_prediction(model_dir / "softmax_rank_lam2")
    _touch_prediction(model_dir / "tc_500m")

    args = Namespace(
        model="elastic_net",
        weights=None,
        weight_spec="dolvol,tc_500m",
    )

    specs = load_filtered_specs(tmp_path, args, _NullLogger())

    assert [spec["weight_spec"] for spec in specs] == ["dolvol", "tc_500m"]


def test_resolve_liquidity_breakpoints_uses_explicit_modes():
    config = {"liquidity": {"quintile_breakpoints": "full_sample"}}

    assert resolve_liquidity_breakpoints(config, "nyse") == ["nyse"]
    assert resolve_liquidity_breakpoints(config, "full_sample") == ["full_sample"]
    assert resolve_liquidity_breakpoints(config, "both") == ["nyse", "full_sample"]


def test_config_for_liquidity_breakpoint_does_not_mutate_original():
    config = {"liquidity": {"quintile_breakpoints": "nyse"}}

    updated = config_for_liquidity_breakpoint(config, "full_sample")

    assert updated["liquidity"]["quintile_breakpoints"] == "full_sample"
    assert config["liquidity"]["quintile_breakpoints"] == "nyse"


def test_liquidity_breakpoint_dir_uses_explicit_namespace(tmp_path):
    base = tmp_path / "elastic_net" / "dolvol"

    assert namespace_liquidity_breakpoints("nyse") is True
    assert namespace_liquidity_breakpoints("both") is True
    assert (
        liquidity_breakpoint_dir(base, "full_sample", True)
        == base / "liquidity_breakpoints" / "full_sample"
    )


def test_parse_aum_millions_supports_proportional_tc_reporting_case():
    config = {
        "transaction_costs": {
            "aum_scenarios": ["proportional_tc", 100_000_000],
        },
    }

    assert parse_aum_millions(None, config) == ["proportional_tc", 100_000_000]
    assert parse_aum_millions(["proportional_tc", "500"], config) == [
        "proportional_tc",
        500_000_000,
    ]


def test_parse_aum_millions_supports_all_alias():
    config = {
        "transaction_costs": {
            "aum_scenarios": ["proportional_tc", 100_000_000, 500_000_000],
        },
    }

    assert parse_aum_millions(["all"], config) == [
        "proportional_tc",
        100_000_000,
        500_000_000,
    ]
