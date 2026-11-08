"""Tests for src/infer.py: predictions keep the input index, the threshold flags member-like trips, inputs untouched."""
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.infer import run_inference


def _fit_tiny_model() -> Pipeline:
    """A test-only model on six separable trips: short trips are members, long ones casual."""
    X = pd.DataFrame({"duration_min": [5.0, 6.0, 7.0, 30.0, 35.0, 40.0]})
    y = pd.Series(["member", "member", "member", "casual", "casual", "casual"], name="member_casual")
    return Pipeline([("model", LogisticRegression())]).fit(X, y)


def test_unit_infer_keeps_input_index_and_leaves_input_unchanged() -> None:
    """Default output is one "prediction" per trip with the input's own index; the input is not modified."""
    model = _fit_tiny_model()
    X_infer = pd.DataFrame({"duration_min": [4.0, 38.0, 6.5]}, index=[10, 11, 12])
    original = X_infer.copy()
    out = run_inference(model, X_infer)
    assert list(out.columns) == ["prediction"]
    assert list(out.index) == [10, 11, 12]
    assert out["prediction"].tolist() == ["member", "casual", "member"]
    pd.testing.assert_frame_equal(X_infer, original)


def test_unit_infer_threshold_adds_probability_and_flag() -> None:
    """A conversion_threshold adds probability (P(member)) and a 0/1 flag; empty input or a bad threshold raise."""
    model = _fit_tiny_model()
    X_infer = pd.DataFrame({"duration_min": [4.0, 38.0]}, index=[10, 11])
    out = run_inference(model, X_infer, conversion_threshold=0.5)
    assert list(out.columns) == ["prediction", "probability", "conversion_target"]
    assert out["probability"].iloc[0] > 0.5 > out["probability"].iloc[1]
    assert out["conversion_target"].tolist() == [1, 0]
    with pytest.raises(ValueError, match="empty"):
        run_inference(model, X_infer.iloc[0:0])
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_inference(model, X_infer, conversion_threshold=1.5)


def test_integration_infer_on_model_reloaded_with_utils(tmp_path) -> None:
    """A model saved with utils.save_model and reloaded with utils.load_model scores trips like the original."""
    from src import utils

    model = _fit_tiny_model()
    X_infer = pd.DataFrame({"duration_min": [4.0, 38.0, 6.5]}, index=[10, 11, 12])
    filepath = tmp_path / "models" / "model.joblib"
    utils.save_model(model, filepath)
    reloaded = utils.load_model(filepath)
    pd.testing.assert_frame_equal(run_inference(reloaded, X_infer, include_proba=True),
                                  run_inference(model, X_infer, include_proba=True))
