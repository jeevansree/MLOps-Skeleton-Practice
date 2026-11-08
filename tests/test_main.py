"""Tests for src/main.py: the split, and the whole pipeline run end to end on synthetic files in a temporary folder."""
from pathlib import Path

import pandas as pd
import pytest

from src import main as main_module
from src.main import split_data


_STATIONS = [("JC005", "Grove St PATH"), ("HB303", "Clinton St & 7 St"), ("JC006", "Exchange Pl")]


def _write_trips(filepath: Path, n_trips: int, prefix: str) -> None:
    """Write n raw-format trips: even rows are short weekday member trips, odd rows long weekend casual trips."""
    rows = []
    for i in range(n_trips):
        is_member = i % 2 == 0
        start_id, start_name = _STATIONS[i % 3]
        end_id, end_name = _STATIONS[(i + 1) % 3]
        day = 24 + (i % 5) if is_member else 29 + (i % 2)  # 24-28 Aug 2026 are weekdays, 29-30 the weekend
        hour = 8 if is_member else 14
        minutes = 5 + (i % 5) if is_member else 25 + (i % 15)
        rows.append({
            "ride_id": f"{prefix}{i:04d}",
            "rideable_type": "electric_bike" if i % 3 == 0 else "classic_bike",
            "started_at": f"2026-08-{day:02d} {hour:02d}:00:00.000",
            "ended_at": f"2026-08-{day:02d} {hour:02d}:{minutes:02d}:00.000",
            "start_station_name": start_name, "start_station_id": start_id,
            "end_station_name": end_name, "end_station_id": end_id,
            "start_lat": "40.72", "start_lng": "-74.04", "end_lat": "40.73", "end_lng": "-74.03",
            "member_casual": "member" if is_member else "casual",
        })
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(filepath, index=False)


def _make_project(root: Path, monkeypatch: pytest.MonkeyPatch, threshold: float) -> None:
    """Write raw and inference files under root and point main.py at root, with the given quality-gate threshold."""
    _write_trips(root / "data" / "raw" / "JC-202608-citibike-tripdata.csv", n_trips=120, prefix="tr")
    _write_trips(root / "data" / "inference" / "JC-202607-citibike-tripdata.csv", n_trips=20, prefix="in")
    monkeypatch.setattr(main_module, "PROJECT_ROOT", root)
    monkeypatch.setitem(main_module.SETTINGS["quality_gate"], "threshold", threshold)


def test_unit_split_data_sizes_and_no_shared_rows() -> None:
    """100 rows split 15% test and 15% validation of the full data: 70/15/15, no row in two splits, y aligned."""
    X = pd.DataFrame({"duration_min": range(100)})
    y = pd.Series(["member", "casual"] * 50, name="member_casual")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.15, val_size=0.15,
                                                                random_state=42, stratify=True)
    assert (len(X_train), len(X_val), len(X_test)) == (70, 15, 15)
    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_train.index) & set(X_test.index)
    assert not set(X_val.index) & set(X_test.index)
    assert list(y_val.index) == list(X_val.index)


def test_integration_main_writes_three_artifacts_when_gate_met(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline test: main() writes clean.csv, model.joblib and a ranked predictions.csv of casual July trips."""
    from src import utils

    _make_project(tmp_path, monkeypatch, threshold=0.0)
    main_module.main()
    assert (tmp_path / "data" / "processed" / "clean.csv").is_file()
    model = utils.load_model(tmp_path / "models" / "model.joblib")
    assert list(model.classes_) == ["casual", "member"]
    predictions = pd.read_csv(tmp_path / "reports" / "predictions.csv")
    assert len(predictions) == 10
    assert list(predictions.columns) == ["ride_id", "prediction", "probability", "conversion_target"]
    assert predictions["probability"].is_monotonic_decreasing
    assert predictions["ride_id"].str.startswith("in").all()


def test_integration_main_stops_without_predictions_when_gate_not_met(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Decision 1A: an unreachable gate keeps clean.csv and the model, writes no predictions, and exits with 1."""
    _make_project(tmp_path, monkeypatch, threshold=1.01)
    with pytest.raises(SystemExit) as stopped:
        main_module.main()
    assert stopped.value.code == 1
    assert (tmp_path / "models" / "model.joblib").is_file()
    assert not (tmp_path / "reports" / "predictions.csv").exists()
