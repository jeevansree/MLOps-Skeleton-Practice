"""Tests for src/evaluate.py: plain float metrics on unseen data, the right positive class, fail-fast inputs."""
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_model


def _fit_tiny_model() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """A test-only model on six separable trips: short trips are members, long ones casual."""
    X = pd.DataFrame({"duration_min": [5.0, 6.0, 7.0, 30.0, 35.0, 40.0]})
    y = pd.Series(["member", "member", "member", "casual", "casual", "casual"], name="member_casual")
    model = Pipeline([("model", LogisticRegression())]).fit(X, y)
    return model, X, y


def test_unit_evaluate_returns_float_metrics_with_known_roc_auc() -> None:
    """Separable data gives roc_auc exactly 1.0 with member as the positive class; every value is a plain float."""
    model, X, y = _fit_tiny_model()
    metrics = evaluate_model(model, X, y, problem_type="classification")
    assert set(metrics) == {"roc_auc", "accuracy"}
    assert all(type(value) is float for value in metrics.values())
    assert metrics["roc_auc"] == 1.0


def test_unit_evaluate_bad_inputs_raise() -> None:
    """An unknown problem_type, empty data, mismatched lengths, or a single-class y raise ValueError."""
    model, X, y = _fit_tiny_model()
    with pytest.raises(ValueError, match="problem_type"):
        evaluate_model(model, X, y, problem_type="clustering")
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(model, X.iloc[0:0], y.iloc[0:0], problem_type="classification")
    with pytest.raises(ValueError, match="same number of rows"):
        evaluate_model(model, X, y.iloc[:3], problem_type="classification")
    with pytest.raises(ValueError, match="both classes"):
        evaluate_model(model, X.iloc[:3], y.iloc[:3], problem_type="classification")


def test_integration_evaluate_scores_train_model_output() -> None:
    """A Pipeline from train_model is scored on held-out trips, returning the agreed keys and a valid roc_auc."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    from src.train import train_model

    X = pd.DataFrame({"duration_min": [5.0, 6.0, 30.0, 35.0, 7.0, 8.0, 32.0, 38.0, 4.0, 9.0, 31.0, 36.0]})
    y = pd.Series(["member", "member", "casual", "casual"] * 3, name="member_casual")
    recipe = ColumnTransformer([("numeric", StandardScaler(), ["duration_min"])])
    model = train_model(X.iloc[:8], y.iloc[:8], recipe, problem_type="classification")
    metrics = evaluate_model(model, X.iloc[8:], y.iloc[8:], problem_type="classification")
    assert set(metrics) == {"roc_auc", "accuracy"}
    assert 0.0 <= metrics["roc_auc"] <= 1.0
