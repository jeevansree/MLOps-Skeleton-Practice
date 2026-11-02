"""Tests for src/utils.py: CSV and model read/write helpers."""
from pathlib import Path

import pandas as pd
import pytest

from src.utils import load_csv, load_model, save_csv, save_model


def _make_df() -> pd.DataFrame:
    """Tiny trip-like table: 3 rows, one text ID and one float column."""
    return pd.DataFrame({"ride_id": ["A1", "B2", "C3"], "duration_min": [4.5, 12.0, 30.25]})


def test_unit_save_and_load_csv_roundtrip(tmp_path: Path) -> None:
    """save_csv then load_csv returns the same 3 rows, with no index column added."""
    df = _make_df()
    filepath = tmp_path / "nested" / "trips.csv"
    save_csv(df, filepath)
    loaded = load_csv(filepath)
    assert list(loaded.columns) == ["ride_id", "duration_min"]
    pd.testing.assert_frame_equal(loaded, df)


def test_unit_load_csv_missing_file_raises(tmp_path: Path) -> None:
    """load_csv on a path that does not exist fails fast with a FileNotFoundError naming the file."""
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        load_csv(tmp_path / "missing.csv")


def test_integration_save_and_load_model_same_predictions(tmp_path: Path) -> None:
    """A fitted model saved with save_model and reloaded with load_model predicts exactly the same."""
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame({"duration_min": [3.0, 5.0, 20.0, 25.0], "is_weekend": [0, 0, 1, 1]})
    y = pd.Series(["member", "member", "casual", "casual"])
    model = LogisticRegression().fit(X, y)
    filepath = tmp_path / "models" / "model.joblib"
    save_model(model, filepath)
    reloaded = load_model(filepath)
    assert list(reloaded.predict(X)) == list(model.predict(X))
    assert (reloaded.predict_proba(X) == model.predict_proba(X)).all()
