"""
Shared file helpers: read and write CSV files, save and load the fitted model.

Inputs:  file paths (pathlib.Path), pandas DataFrames, fitted model objects
Outputs: DataFrames, model objects, or files on disk (parent folders created as needed)
Used by: main.py, infer.py and the tests
Why:     one place for file I/O, so every artifact is written the same way; no column names or business logic here
"""
from pathlib import Path

import joblib
import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame.

    Inputs:  filepath, path to an existing CSV file
    Outputs: the file's contents as a pandas DataFrame
    Why:     a wrong path fails fast with a clear message instead of a vague error later
    """
    # TODO: replace print with logging (Session 10)
    print(f"Loading CSV: {filepath}")
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"CSV file not found: {filepath}. Check the path in SETTINGS['paths'].")
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Write a DataFrame to a CSV file.

    Inputs:  df, the DataFrame to save
             filepath, path to the CSV file to create
    Why:     a wrong path fails fast with a clear message instead of a vague error later
    """
    # TODO: replace print with logging (Session 10)
    print(f"Saving CSV: {filepath}")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def load_model(filepath: Path) -> object:
    """
    Load a fitted model from a file.

    Inputs:  filepath, path to an existing model file
    Outputs: the loaded model object
    Why:     a wrong path fails fast with a clear message instead of a vague error later
    """
    # TODO: replace print with logging (Session 10)
    print(f"Loading model: {filepath}")
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Model file not found: {filepath}. Check the path in SETTINGS['paths'].")
    return joblib.load(filepath)


def save_model(model: object, filepath: Path) -> None:
    """
    Save a fitted model (the whole Pipeline, preprocessing included) to a file.

    Inputs:  model, the fitted model to save; filepath, where to write it
    Outputs: None; the joblib file on disk, parent folders created as needed
    Why:     saving preprocessing and model together prevents training-serving skew at inference
    """
    # TODO: replace print with logging (Session 10)
    print(f"Saving model: {filepath}")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
