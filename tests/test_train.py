"""Tests for src/train.py: the only fit in the project, on train data, with preprocessing saved inside the model."""
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train import train_model


def _make_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Twelve cleaned trips with every feature column: weekday short trips are members, weekend long ones casual."""
    X = pd.DataFrame({
        "rideable_type": ["classic_bike", "electric_bike"] * 6,
        "hour": [8, 17, 8, 17, 8, 17, 14, 15, 14, 15, 14, 15],
        "day_of_week": [0, 1, 2, 3, 4, 0, 5, 6, 5, 6, 5, 6],
        "is_weekend": [0] * 6 + [1] * 6,
        "duration_min": [5.0, 6.0, 7.0, 5.0, 6.0, 7.0, 25.0, 30.0, 28.0, 35.0, 22.0, 40.0],
        "is_round_trip": [0] * 6 + [1, 0, 1, 0, 1, 0],
        "start_station_id": ["JC005", "HB303", "JC006"] * 4,
        "end_station_id": ["HB303", "JC006", "JC005"] * 4,
    })
    y = pd.Series(["member"] * 6 + ["casual"] * 6, name="member_casual")
    return X, y


def test_unit_train_returns_fitted_pipeline_with_text_classes() -> None:
    """A simple recipe plus the model come back as one fitted Pipeline, classes sorted as [casual, member]."""
    X, y = _make_xy()
    recipe = ColumnTransformer([("numeric", StandardScaler(), ["is_weekend", "duration_min"])])
    model = train_model(X, y, recipe, problem_type="classification", random_state=42)
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["preprocess", "model"]
    assert list(model.classes_) == ["casual", "member"]
    assert len(model.predict(X)) == 12


def test_unit_train_bad_inputs_raise_before_fitting() -> None:
    """An unknown problem_type, empty data, or X and y of different lengths raise ValueError before any fit."""
    X, y = _make_xy()
    recipe = ColumnTransformer([("numeric", StandardScaler(), ["is_weekend", "duration_min"])])
    with pytest.raises(ValueError, match="problem_type"):
        train_model(X, y, recipe, problem_type="clustering")
    with pytest.raises(ValueError, match="empty"):
        train_model(X.iloc[0:0], y.iloc[0:0], recipe, problem_type="classification")
    with pytest.raises(ValueError, match="same number of rows"):
        train_model(X, y.iloc[:5], recipe, problem_type="classification")
    assert not hasattr(recipe, "transformers_")


def test_integration_train_with_features_recipe_survives_save_and_reload(tmp_path) -> None:
    """features recipe -> train_model -> save -> reload gives the same probabilities (custom class reloads)."""
    from src import utils
    from src.features import get_feature_preprocessor

    X, y = _make_xy()
    recipe = get_feature_preprocessor(["rideable_type", "hour", "day_of_week"],
                                      ["is_weekend", "duration_min", "is_round_trip"],
                                      ["start_station_id", "end_station_id"])
    model = train_model(X, y, recipe, problem_type="classification", random_state=42)
    filepath = tmp_path / "models" / "model.joblib"
    utils.save_model(model, filepath)
    reloaded = utils.load_model(filepath)
    assert (reloaded.predict_proba(X) == model.predict_proba(X)).all()
