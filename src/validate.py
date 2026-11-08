"""
Check cleaned trips against the team brief's data rules; return True or stop the run with a clear ValueError.

Inputs:  a cleaned DataFrame from clean_data, the required columns, and the target column in training mode
Outputs: True when every rule holds; otherwise a ValueError naming the broken rule and what to check
Used by: main.py, after clean_data, for training data and again for inference data
Why:     a broken monthly export stops here, before it trains or scores a model; this module never fixes data
"""
import pandas as pd


def _check_shape(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise ValueError if required columns are missing, the data is empty, or ride_id repeats."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Check clean_data output and SETTINGS.")
    if df.empty:
        raise ValueError("Input data is empty. Check the raw files and the clean_data row rules.")
    if "ride_id" in df.columns and df["ride_id"].duplicated().any():
        n_duplicates = int(df["ride_id"].duplicated().sum())
        raise ValueError(f"Duplicate ride_id values found: {n_duplicates}. Check a file was not loaded twice.")


def _check_target(df: pd.DataFrame, target_column: str, allowed_target_values: list | None) -> None:
    """Training mode only: raise ValueError if labels are missing or hold values outside the allowed list."""
    if df[target_column].isna().any():
        n_missing = int(df[target_column].isna().sum())
        raise ValueError(f"{target_column} has missing values in {n_missing} rows. Check the raw export.")
    if allowed_target_values is not None:
        unexpected = sorted(set(df[target_column].unique()) - set(allowed_target_values))
        if unexpected:
            raise ValueError(f"Unexpected {target_column} values: {unexpected}. Allowed: {allowed_target_values}.")


def _check_stations(df: pd.DataFrame, max_missing_station_share: float) -> None:
    """Raise ValueError if too many trips miss a station ID, or one station ID carries more than one name."""
    missing_share = (df["start_station_id"].isna() | df["end_station_id"].isna()).mean()
    if missing_share >= max_missing_station_share:
        limit = f"{max_missing_station_share:.0%}"
        raise ValueError(f"{missing_share:.1%} of trips are missing a station ID (limit {limit}). Check the raw data.")
    pairs = pd.concat([
        df[["start_station_id", "start_station_name"]].set_axis(["station_id", "name"], axis=1),
        df[["end_station_id", "end_station_name"]].set_axis(["station_id", "name"], axis=1),
    ]).dropna().drop_duplicates()
    names_per_id = pairs.groupby("station_id")["name"].nunique()
    renamed = names_per_id[names_per_id > 1].index.tolist()
    if renamed:
        raise ValueError(f"Station IDs with more than one name: {renamed[:5]}. Check for renamed or reused stations.")


def validate_dataframe(df: pd.DataFrame, required_columns: list[str], target_column: str | None = None,
                       allowed_target_values: list | None = None,
                       max_missing_station_share: float = 0.01) -> bool:
    """
    Check a cleaned DataFrame against the team brief's rules; validate only, never fix.

    Inputs:  df, cleaned trips from clean_data (never modified); required_columns, columns that must exist
             target_column, the label in training mode, None in inference mode
             allowed_target_values (e.g. ["member", "casual"]) and max_missing_station_share (e.g. 0.01), from SETTINGS
    Outputs: True when every rule holds
    Why:     raises ValueError naming the broken rule, so a bad export stops before training or scoring
    """
    # TODO: replace print with logging (Session 10)
    print(f"Validating {len(df):,} rows ({'training' if target_column else 'inference'} mode)")
    columns_to_check = list(required_columns)
    if target_column is not None and target_column not in columns_to_check:
        columns_to_check.append(target_column)
    _check_shape(df, columns_to_check)
    if target_column is not None:
        _check_target(df, target_column, allowed_target_values)
    _check_stations(df, max_missing_station_share)
    return True
