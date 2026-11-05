"""
Fit the feature recipe and the model together as one Pipeline: the only .fit() in the project.

Inputs:  the train split (X_train, y_train), the unfitted recipe from features, the problem type and model settings
Outputs: a fitted Pipeline([("preprocess", recipe), ("model", estimator)]) ready to evaluate, save and reuse
Used by: main.py, after the split; evaluate.py scores it, utils.save_model saves it, infer.py loads it
Why:     fitting preprocessing and model together, on train only, prevents leakage and training-serving skew
"""
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


# Model family per problem type (decision 1A); the brief uses default parameters plus a fixed random_state
_MODELS = {
    "classification": HistGradientBoostingClassifier,
    "regression": HistGradientBoostingRegressor,
}


def _check_inputs(X_train: pd.DataFrame, y_train: pd.Series, problem_type: str) -> None:
    """Raise ValueError for an unknown problem_type, empty training data, or X and y of different lengths."""
    if problem_type not in _MODELS:
        raise ValueError(f"Unknown problem_type '{problem_type}'. Use one of {sorted(_MODELS)}.")
    if len(X_train) == 0:
        raise ValueError("Training data is empty. Check the split sizes and the clean_data row rules.")
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train and y_train must have the same number of rows: {len(X_train)} vs {len(y_train)}.")


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer,
                problem_type: str, model_params: dict | None = None, random_state: int = 42) -> Pipeline:
    """
    Fit the recipe and the model together on the train split only and return one fitted Pipeline.

    Inputs:  X_train, y_train, the train split from main.py; preprocessor, the unfitted recipe from features
             problem_type, "classification" or "regression"; model_params and random_state, from SETTINGS
    Outputs: a fitted Pipeline with steps "preprocess" and "model"; text labels kept, classes sorted
    Why:     the only .fit() in the project, so nothing is learned from validation or test rows
    """
    # TODO: replace print with logging (Session 10)
    print(f"Training {problem_type} model on {len(X_train):,} rows")
    _check_inputs(X_train, y_train, problem_type)
    model = _MODELS[problem_type](random_state=random_state, **(model_params or {}))
    pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", model)])
    pipeline.fit(X_train, y_train)
    return pipeline
