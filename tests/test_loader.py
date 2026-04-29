import numpy as np
import pandas as pd

from src.data.loader import load_processed_panel, normalize_features


def test_normalize_features_maps_each_month_to_zero_one_interval():
    df = pd.DataFrame(
        {
            "yyyymm": [202001, 202001, 202001, 202002, 202002, 202002],
            "x": [10.0, 20.0, 30.0, 5.0, np.nan, 15.0],
        }
    )

    out = normalize_features(df, ["x"])

    np.testing.assert_allclose(
        out.loc[out["yyyymm"] == 202001, "x"].to_numpy(),
        [0.0, 0.5, 1.0],
    )
    np.testing.assert_allclose(
        out.loc[out["yyyymm"] == 202002, "x"].dropna().to_numpy(),
        [0.0, 1.0],
    )
    assert np.isnan(out.loc[4, "x"])


def test_normalize_features_single_observation_month_uses_neutral_rank():
    df = pd.DataFrame({"yyyymm": [202001, 202001], "x": [np.nan, 7.0]})

    out = normalize_features(df, ["x"])

    assert np.isnan(out.loc[0, "x"])
    assert out.loc[1, "x"] == 0.5


def test_normalize_features_uses_average_tie_ranks():
    df = pd.DataFrame({"yyyymm": [202001, 202001, 202001], "x": [1.0, 1.0, 3.0]})

    out = normalize_features(df, ["x"])

    np.testing.assert_allclose(out["x"].to_numpy(), [0.25, 0.25, 1.0])


def test_load_processed_panel_reads_expected_parquet(tmp_path):
    df = pd.DataFrame({"permno": [1], "yyyymm": [202001], "x": [0.5]})
    df.to_parquet(tmp_path / "processed_panel.parquet", index=False)

    out = load_processed_panel(tmp_path)

    pd.testing.assert_frame_equal(out, df)
