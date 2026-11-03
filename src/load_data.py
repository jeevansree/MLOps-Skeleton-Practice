"""
Read raw Citi Bike trip files into one DataFrame, exactly as they are on disk.

Inputs:  a CSV file path, or a folder plus a file pattern (e.g. "JC-*-citibike-tripdata.csv")
Outputs: one raw DataFrame, every column read as text, files stacked in sorted name order
Used by: main.py (training files in data/raw/, inference file in data/inference/)
Why:     one fail-fast entry point for data; read only, so every run starts from the same raw files
"""
from pathlib import Path

import pandas as pd


def _read_one_file(filepath: Path) -> pd.DataFrame:
    """Read one CSV with every column as text; raise ValueError if it holds no data."""
    try:
        df = pd.read_csv(filepath, dtype=str)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Raw data file is empty: {filepath}. Check the file downloaded completely.") from error
    if df.empty:
        raise ValueError(f"Raw data file has a header but is otherwise empty: {filepath}. Check the download.")
    return df


def load_raw_data(raw_data_path: Path, file_pattern: str | None = None) -> pd.DataFrame:
    """
    Read raw trip data from one CSV file, or from every file in a folder that matches a pattern.

    Inputs:  raw_data_path, a CSV file (file_pattern=None) or a folder (file_pattern given)
             file_pattern, a glob such as "JC-*-citibike-tripdata.csv", from SETTINGS
    Outputs: one DataFrame, every column as text, files stacked in sorted name order with a fresh index
    Why:     a missing, misplaced or empty input stops the run here, before cleaning and training start
    """
    # TODO: replace print with logging (Session 10)
    print(f"Loading raw data: {raw_data_path} (pattern: {file_pattern})")
    raw_data_path = Path(raw_data_path)
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data path not found: {raw_data_path}. Check SETTINGS['paths'].")
    if file_pattern is None:
        if raw_data_path.is_dir():
            raise ValueError(f"Raw data path is a folder: {raw_data_path}. Pass file_pattern or one CSV file.")
        return _read_one_file(raw_data_path)
    if not raw_data_path.is_dir():
        raise ValueError(f"file_pattern needs a folder, got a file: {raw_data_path}. Check SETTINGS['paths'].")
    files = sorted(raw_data_path.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {file_pattern} in {raw_data_path}. Check folder and pattern.")
    frames = [_read_one_file(filepath) for filepath in files]
    return pd.concat(frames, ignore_index=True)