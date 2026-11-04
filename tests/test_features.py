"""Tests for src/features.py: an unfitted recipe, fail-fast column checks, and safe transforms of unseen values."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.features import get_feature_preprocessor


_CATEGORICAL = ["rideable_type", "hour", "day_of_week"]
_NUMERIC = ["is_weekend", "duration_min", "is_round_trip"]
_STATION = ["start_station_id", "end_station_id"]


def _make_df() -> pd.DataFrame:
    """Six cleaned training trips: 2 bike types, 3 hours, 5 weekdays, 3 stations (15 columns once fitted)."""
    return pd.DataFrame({
        "rideable_type": ["classic_bike", "electric_bike"] * 3,
        "hour": [8, 8, 17, 17, 12, 12],
        "day_of_week": [0, 1, 2, 5, 6, 0],
        "is_weekend": [0, 0, 0, 1, 1, 0],
        "duration_min": [10.0, 5.0, 20.0, 12.0, 8.0, 15.0],
        "is_round_trip": [0, 0, 0, 0, 1, 0],
        "start_station_id": ["JC005", "HB303", "JC006", "JC005", "HB303", "JC006"],
        "end_station_id": ["HB303", "JC005", "JC005", "JC006", "HB303", "HB303"],
    })


def test_unit_features_returns_unfitted_recipe_with_three_groups() -> None:
    """The recipe is an unfitted ColumnTransformer with one named step per brief feature group."""
    preprocessor = get_feature_preprocessor(_CATEGORICAL, _NUMERIC, _STATION)
    assert isinstance(preprocessor, ColumnTransformer)
    assert [name for name, _, _ in preprocessor.transformers] == ["categorical", "numeric", "station"]
    assert not hasattr(preprocessor, "transformers_")


def test_unit_features_bad_column_lists_raise() -> None:
    """An empty group, or station columns that are not exactly [start, end], raise ValueError."""
    with pytest.raises(ValueError, match="categorical_columns"):
        get_feature_preprocessor([], _NUMERIC, _STATION)
    with pytest.raises(ValueError, match="numeric_columns"):
        get_feature_preprocessor(_CATEGORICAL, [], _STATION)
    with pytest.raises(ValueError, match="exactly two"):
        get_feature_preprocessor(_CATEGORICAL, _NUMERIC, ["start_station_id"])


def test_integration_features_fit_on_train_handles_unseen_values() -> None:
    """Fitted on train: 15 columns, and new trips with unseen hours or stations get no missing values."""
    preprocessor = get_feature_preprocessor(_CATEGORICAL, _NUMERIC, _STATION)
    train_out = preprocessor.fit_transform(_make_df())
    assert train_out.shape == (6, 15)
    new_trips = _make_df().iloc[:2].assign(hour=[23, 23], start_station_id=["JC999", "JC998"],
                                           end_station_id=[None, "JC997"])
    new_out = preprocessor.transform(new_trips)
    assert new_out.shape == (2, 15)
    assert not np.isnan(new_out).any()
    assert (new_out[0, -2:] == new_out[1, -2:]).all()
