"""
Clean raw Citi Bike trips with fixed row rules and add the trip features the model uses.

Inputs:  the raw DataFrame from load_data (every column as text); target_column given means training mode
Outputs: a cleaned copy with dates parsed, coordinates dropped and five trip features added
Used by: main.py, after load_data and before validate, for training and inference data
Why:     fixed rules only, nothing learned from the data, so cleaning can run before the split without leakage
"""
import pandas as pd


def _parse_times(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert timestamp text columns of a copied DataFrame to dates; raise ValueError if any cannot be read."""
    for column in columns:
        try:
            df[column] = pd.to_datetime(df[column], format="ISO8601")
        except (ValueError, TypeError) as error:
            raise ValueError(f"Column '{column}' has invalid timestamps. Check the raw file.") from error
    return df


def _apply_row_rules(df: pd.DataFrame, prefixes: tuple[str, ...], min_duration_min: float,
                     max_duration_min: float) -> pd.DataFrame:
    """Training mode only: keep trips with both station IDs, inside the service area, lasting min to max minutes."""
    n_raw = len(df)
    df = df[df["start_station_id"].notna() & df["end_station_id"].notna()]
    in_area = df["start_station_id"].str.startswith(prefixes) & df["end_station_id"].str.startswith(prefixes)
    df = df[in_area]
    df = df[df["duration_min"].between(min_duration_min, max_duration_min)]
    # TODO: replace print with logging (Session 10)
    print(f"Row rules kept {len(df):,} of {n_raw:,} trips")
    return df.reset_index(drop=True)


def _add_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add duration_min, hour, day_of_week, is_weekend and is_round_trip from each trip's own values."""
    df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    df["hour"] = df["started_at"].dt.hour
    df["day_of_week"] = df["started_at"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_round_trip"] = (df["start_station_id"] == df["end_station_id"]).astype(int)
    return df


def clean_dataframe(df_raw: pd.DataFrame, target_column: str | None = None,
                    valid_station_prefixes: tuple[str, ...] = ("JC", "HB"),
                    min_duration_min: float = 1, max_duration_min: float = 180) -> pd.DataFrame:
    """
    Clean raw trips and add trip features; training mode filters rows, inference mode keeps every row.

    Inputs:  df_raw, raw trips from load_data (never modified)
             target_column, the label column in training mode; None means inference mode
             valid_station_prefixes, min_duration_min, max_duration_min: row-rule settings from SETTINGS
    Outputs: a cleaned copy with dates parsed, coordinates dropped and trip features added
    Why:     the same cleaning for training and inference prevents training-serving skew
    """
    # TODO: replace print with logging (Session 10)
    print(f"Cleaning {len(df_raw):,} trips ({'training' if target_column else 'inference'} mode)")
    df = df_raw.copy()
    df = _parse_times(df, ["started_at", "ended_at"])
    df = df.drop(columns=["start_lat", "start_lng", "end_lat", "end_lng"], errors="ignore")
    df = _add_trip_features(df)
    if target_column is not None:
        df = _apply_row_rules(df, valid_station_prefixes, min_duration_min, max_duration_min)
    return df
