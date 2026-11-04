"""Tests for src/validate.py: valid data passes, each broken rule stops the run with a clear ValueError."""
import pandas as pd
import pytest

from src.validate import validate_dataframe


_REQUIRED = ["ride_id", "start_station_id", "end_station_id", "member_casual"]


def _make_df() -> pd.DataFrame:
    """Four clean trips that pass every rule: unique IDs, valid labels, station IDs present, one name per ID."""
    return pd.DataFrame({
        "ride_id": ["r1", "r2", "r3", "r4"],
        "start_station_id": ["JC005", "HB303", "JC006", "JC005"],
        "start_station_name": ["Grove St PATH", "Clinton St & 7 St", "Exchange Pl", "Grove St PATH"],
        "end_station_id": ["HB303", "JC005", "JC005", "JC006"],
        "end_station_name": ["Clinton St & 7 St", "Grove St PATH", "Grove St PATH", "Exchange Pl"],
        "member_casual": ["member", "casual", "member", "casual"],
    })


def test_unit_validate_clean_data_passes_in_both_modes() -> None:
    """Clean trips return True in training mode (labels checked) and in inference mode (no target), unchanged."""
    df = _make_df()
    assert validate_dataframe(df, _REQUIRED, target_column="member_casual",
                              allowed_target_values=["member", "casual"]) is True
    inference_df = df.drop(columns=["member_casual"])
    assert validate_dataframe(inference_df, _REQUIRED[:-1]) is True
    pd.testing.assert_frame_equal(df, _make_df())


def test_unit_validate_each_broken_rule_raises() -> None:
    """Each rule, broken on its own, raises ValueError with a message naming that rule."""
    labels = {"target_column": "member_casual", "allowed_target_values": ["member", "casual"]}
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(_make_df().drop(columns=["ride_id"]), _REQUIRED, **labels)
    with pytest.raises(ValueError, match="empty"):
        validate_dataframe(_make_df().iloc[0:0], _REQUIRED, **labels)
    duplicated = _make_df().assign(ride_id=["r1", "r1", "r3", "r4"])
    with pytest.raises(ValueError, match="Duplicate ride_id"):
        validate_dataframe(duplicated, _REQUIRED, **labels)
    no_label = _make_df().assign(member_casual=["member", None, "member", "casual"])
    with pytest.raises(ValueError, match="missing values"):
        validate_dataframe(no_label, _REQUIRED, **labels)
    bad_label = _make_df().assign(member_casual=["member", "Member", "member", "casual"])
    with pytest.raises(ValueError, match="Unexpected member_casual values"):
        validate_dataframe(bad_label, _REQUIRED, **labels)
    no_station = _make_df().assign(end_station_id=["HB303", None, "JC005", "JC006"])
    with pytest.raises(ValueError, match="missing a station ID"):
        validate_dataframe(no_station, _REQUIRED, **labels)
    renamed = _make_df().assign(
        start_station_name=["Grove St PATH", "Clinton St & 7 St", "Exchange Pl", "Grove St"])
    with pytest.raises(ValueError, match="more than one name"):
        validate_dataframe(renamed, _REQUIRED, **labels)


def test_integration_validate_accepts_clean_dataframe_output() -> None:
    """Training-mode output of clean_dataframe passes validate_dataframe with the brief's label rules."""
    from src.clean_data import clean_dataframe

    df_raw = pd.DataFrame({
        "ride_id": ["a1", "a2"],
        "rideable_type": ["classic_bike", "electric_bike"],
        "started_at": ["2026-08-24 08:00:00.000", "2026-08-29 14:00:00.000"],
        "ended_at": ["2026-08-24 08:10:00.000", "2026-08-29 14:05:00.000"],
        "start_station_name": ["Grove St PATH", "Clinton St & 7 St"],
        "start_station_id": ["JC005", "HB303"],
        "end_station_name": ["Clinton St & 7 St", "Clinton St & 7 St"],
        "end_station_id": ["HB303", "HB303"],
        "member_casual": ["member", "casual"],
    })
    out = clean_dataframe(df_raw, target_column="member_casual")
    assert validate_dataframe(out, _REQUIRED + ["duration_min", "hour"], target_column="member_casual",
                              allowed_target_values=["member", "casual"]) is True
