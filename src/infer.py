"""
Score new trips with the saved model: a prediction per trip, plus P(member) and a conversion flag on request.

Inputs:  the fitted Pipeline (loaded with utils.load_model), cleaned and validated trips, an optional threshold
Outputs: a DataFrame with the input's own index: prediction, plus probability and conversion_target if asked
Used by: main.py, which puts ride_id first, sorts by probability and saves reports/predictions.csv
Why:     the same saved recipe and positive class as in evaluate, so scores mean what the gate measured
"""
import pandas as pd
from sklearn.pipeline import Pipeline


def _check_inputs(X_infer: pd.DataFrame, conversion_threshold: float | None) -> None:
    """Raise ValueError if there is nothing to score, or the threshold is not a probability between 0 and 1."""
    if len(X_infer) == 0:
        raise ValueError("Inference data is empty. Check data/inference/ and the casual-only filter in main.py.")
    if conversion_threshold is not None and not 0 <= conversion_threshold <= 1:
        raise ValueError(f"conversion_threshold must be between 0 and 1, got {conversion_threshold}. Check SETTINGS.")


def run_inference(model: Pipeline, X_infer: pd.DataFrame, include_proba: bool = False,
                  conversion_threshold: float | None = None) -> pd.DataFrame:
    """
    Score new trips and return the results aligned to the input's index (never sorted, never re-indexed).

    Inputs:  model, the fitted Pipeline; X_infer, cleaned and validated trips with the training feature columns
             include_proba, add P(member); conversion_threshold, e.g. 0.8 from SETTINGS (also adds probability)
    Outputs: "prediction", plus "probability" and "conversion_target" (1 if probability >= threshold) when asked
    Why:     the same index lets main.py put each ride_id next to its own score, so offers reach the right riders
    """
    # TODO: replace print with logging (Session 10)
    print(f"Scoring {len(X_infer):,} trips")
    _check_inputs(X_infer, conversion_threshold)
    out = pd.DataFrame({"prediction": model.predict(X_infer)}, index=X_infer.index)
    if include_proba or conversion_threshold is not None:
        out["probability"] = model.predict_proba(X_infer)[:, 1]
    if conversion_threshold is not None:
        out["conversion_target"] = (out["probability"] >= conversion_threshold).astype(int)
    return out
