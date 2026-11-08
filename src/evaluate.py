"""
Score a fitted model on unseen data and return its metrics as plain Python floats.

Inputs:  a fitted Pipeline from train, X_eval and y_eval (the validation split), and the problem type
Outputs: {"roc_auc": ..., "accuracy": ...} for classification, {"rmse": ..., "mae": ...} for regression
Used by: main.py, which compares the result with the quality gate in SETTINGS
Why:     evaluate only measures; the gate decision stays in main.py, and the test split stays sealed
"""
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score, root_mean_squared_error
from sklearn.pipeline import Pipeline


def _check_inputs(X_eval: pd.DataFrame, y_eval: pd.Series, problem_type: str) -> None:
    """Raise ValueError for an unknown problem_type, empty data, mismatched lengths, or a single-class y."""
    if problem_type not in ("classification", "regression"):
        raise ValueError(f"Unknown problem_type '{problem_type}'. Use 'classification' or 'regression'.")
    if len(X_eval) == 0:
        raise ValueError("Evaluation data is empty. Check the validation split size in SETTINGS['split'].")
    if len(X_eval) != len(y_eval):
        raise ValueError(f"X_eval and y_eval must have the same number of rows: {len(X_eval)} vs {len(y_eval)}.")
    if problem_type == "classification" and y_eval.nunique() < 2:
        raise ValueError("Evaluation labels must include both classes for roc_auc. Check the split is stratified.")


def _classification_metrics(model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict[str, float]:
    """roc_auc and accuracy, with model.classes_[1] as the positive class (decision 1A)."""
    positive_class = model.classes_[1]
    probability = model.predict_proba(X_eval)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_eval == positive_class, probability)),
        "accuracy": float(accuracy_score(y_eval, model.predict(X_eval))),
    }


def _regression_metrics(model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict[str, float]:
    """rmse and mae for problem_type="regression" (decision 2A); not used by the Citi Bike classifier."""
    predictions = model.predict(X_eval)
    return {
        "rmse": float(root_mean_squared_error(y_eval, predictions)),
        "mae": float(mean_absolute_error(y_eval, predictions)),
    }


def evaluate_model(model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series,
                   problem_type: str) -> dict[str, float]:
    """
    Score a fitted model on unseen data and return its metrics as plain Python floats.

    Inputs:  model, the fitted Pipeline from train; X_eval and y_eval, the validation split from main.py
             problem_type, "classification" or "regression"
    Outputs: {"roc_auc", "accuracy"} for classification or {"rmse", "mae"} for regression, as floats
    Why:     measures only; main.py applies the quality gate, so this module never decides pass or fail
    """
    # TODO: replace print with logging (Session 10)
    print(f"Evaluating {problem_type} model on {len(X_eval):,} rows")
    _check_inputs(X_eval, y_eval, problem_type)
    if problem_type == "classification":
        return _classification_metrics(model, X_eval, y_eval)
    return _regression_metrics(model, X_eval, y_eval)
