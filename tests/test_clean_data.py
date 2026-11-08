"""Tests for src/clean_data.py: fixed row rules in training mode, every row kept in inference mode, trip features."""
import pandas as pd
import pytest

from src.clean_data import clean_dataframe


def _make_df() -> pd.DataFrame:
    """Five raw trips as text, like load_data returns: r1 and r5 valid, r2 to r4 each break one row rule."""
    return pd.DataFrame({
        "ride_id": ["r1", "r2", "r3", "r4", "r5"],
        "rideable_type": ["classic_bike", "electric_bike", "classic_bike", "classic_bike", "electric_bike"],
        "started_at": ["2026-08-24 08:00:00.000", "2026-08-24 09:00:00.000", "2026-08-25 10:00:00.000",
                       "2026-08-26 11:00:00.000", "2026-08-29 14:00:00.000"],
        "ended_at": ["2026-08-24 08:10:00.000", "2026-08-24 09:05:00.000", "2026-08-25 10:20:00.000",
                     "2026-08-26 14:20:00.000", "2026-08-29 14:05:00.000"],
        "start_station_name": ["Grove St PATH", "Grove St PATH", "Exchange Pl", "Exchange Pl", "Clinton St & 7 St"],
        "start_station_id": ["JC005", "JC005", "JC006", "JC006", "HB303"],
        "end_station_name": ["Clinton St & 7 St", None, "W 21 St & 6 Ave", "Grove St PATH", "Clinton St & 7 St"],
        "end_station_id": ["HB303", None, "5329.10", "JC005", "HB303"],
        "start_lat": ["40.719", "40.719", "40.716", "40.716", "40.745"],
        "start_lng": ["-74.043", "-74.043", "-74.033", "-74.033", "-74.033"],
        "end_lat": ["40.745", None, "40.741", "40.719", "40.745"],
        "end_lng": ["-74.033", None, "-73.994", "-74.043", "-74.033"],
        "member_casual": ["member", "casual", "member", "casual", "casual"],
    })


def test_unit_clean_training_applies_row_rules_and_adds_features() -> None:
    """Training mode keeps only r1 and r5, drops coordinates, keeps labels as text and adds the trip features."""
    out = clean_dataframe(_make_df(), target_column="member_casual")
    assert out["ride_id"].tolist() == ["r1", "r5"]
    assert list(out.index) == [0, 1]
    assert out["member_casual"].tolist() == ["member", "casual"]
    assert out["duration_min"].tolist() == [10.0, 5.0]
    assert out["hour"].tolist() == [8, 14]
    assert out["day_of_week"].tolist() == [0, 5]
    assert out["is_weekend"].tolist() == [0, 1]
    assert out["is_round_trip"].tolist() == [0, 1]
    assert not {"start_lat", "start_lng", "end_lat", "end_lng"} & set(out.columns)


def test_unit_clean_inference_keeps_every_row_and_input_unchanged() -> None:
    """Inference mode keeps all 5 trips (decision 1A) and never modifies the DataFrame it receives."""
    df_raw = _make_df()
    original = df_raw.copy()
    out = clean_dataframe(df_raw)
    assert out["ride_id"].tolist() == ["r1", "r2", "r3", "r4", "r5"]
    assert out["duration_min"].tolist() == [10.0, 5.0, 20.0, 200.0, 5.0]
    assert out["is_round_trip"].tolist() == [0, 0, 0, 0, 1]
    pd.testing.assert_frame_equal(df_raw, original)


def test_integration_clean_training_output_passes_validate() -> None:
    """Training-mode output of clean_dataframe passes validate_dataframe with the target column."""
    from src.validate import validate_dataframe

    out = clean_dataframe(_make_df(), target_column="member_casual")
    required = ["ride_id", "rideable_type", "start_station_id", "end_station_id", "hour",
                "day_of_week", "is_weekend", "duration_min", "is_round_trip", "member_casual"]
    assert validate_dataframe(out, required_columns=required, target_column="member_casual") is True


def test_unit_clean_invalid_timestamp_raises() -> None:
    """A started_at value that is not a timestamp stops cleaning with a ValueError naming the column."""
    df_raw = _make_df()
    df_raw.loc[0, "started_at"] = "not a time"
    with pytest.raises(ValueError, match="started_at"):
        clean_dataframe(df_raw, target_column="member_casual")
