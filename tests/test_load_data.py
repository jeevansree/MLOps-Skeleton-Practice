"""Tests for src/load_data.py: reading raw trip files, stacking monthly files, failing fast on bad paths."""
from pathlib import Path

import pandas as pd
import pytest

from src.load_data import load_raw_data


def _write_trips(filepath: Path, n_rows: int, prefix: str) -> Path:
    """Write a tiny trip CSV with n_rows rows (last station ID looks numeric) and return its path."""
    df = pd.DataFrame({
        "ride_id": [f"{prefix}{i}" for i in range(n_rows)],
        "start_station_id": ["HB303"] * (n_rows - 1) + ["5329.10"],
        "member_casual": ["member"] * n_rows,
    })
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath