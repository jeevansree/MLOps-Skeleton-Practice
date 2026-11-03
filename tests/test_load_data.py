"""Tests for src/load_data.py: reading raw trip files, stacking monthly files, failing fast on bad paths."""
from pathlib import Path

import pandas as pd
import pytest

from src.load_data import load_raw_data


def _write_trips(filepath: Path, n_rows: int, prefix: str) -> Path:
    """Write a tiny trip CSV with n_rows rows (last station ID looks numeric) and return its path."""
    df = pd.DataFrame({
        "ride_id": [f"{prefix}{i}" for i in range(n_rows)],
        "start_station_id": ["HB303"] * (n_rows - 1) + ["5329.10"],
        "member_casual": ["member"] * n_rows,
    })
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath


def test_unit_load_pattern_stacks_files_as_text(tmp_path: Path) -> None:
    """Two matching monthly files stack into 5 rows, sorted by file name, with station IDs kept as text."""
    _write_trips(tmp_path / "JC-202609-citibike-tripdata.csv", n_rows=2, prefix="sep")
    _write_trips(tmp_path / "JC-202608-citibike-tripdata.csv", n_rows=3, prefix="aug")
    _write_trips(tmp_path / "notes.csv", n_rows=4, prefix="x")
    df = load_raw_data(tmp_path, file_pattern="JC-*-citibike-tripdata.csv")
    assert len(df) == 5
    assert df["ride_id"].iloc[0] == "aug0"
    assert list(df.index) == list(range(5))
    assert df["start_station_id"].tolist().count("5329.10") == 2


def test_unit_load_bad_paths_fail_fast(tmp_path: Path) -> None:
    """A missing file, a folder without a pattern, a pattern with no matches and an empty file all raise."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_raw_data(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="folder"):
        load_raw_data(tmp_path)
    with pytest.raises(FileNotFoundError, match="No files matching"):
        load_raw_data(tmp_path, file_pattern="JC-*.csv")
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_raw_data(empty_file)


def test_integration_load_reads_utils_save_csv_output(tmp_path: Path) -> None:
    """A file written by utils.save_csv loads back with the same rows, IDs kept as text."""
    from src import utils

    df = pd.DataFrame({"ride_id": ["a1", "b2"], "start_station_id": ["JC027", "5329.10"]})
    filepath = tmp_path / "raw" / "trips.csv"
    utils.save_csv(df, filepath)
    loaded = load_raw_data(filepath)
    pd.testing.assert_frame_equal(loaded, df)