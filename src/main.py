"""
Run the whole Citi Bike member-lookalike pipeline with one command: python -m src.main

Inputs:  the raw monthly trip files (data/raw/) and the inference month (data/inference/), paths in SETTINGS
Outputs: data/processed/clean.csv, models/model.joblib, and reports/predictions.csv only when the gate is met
Used by: the team now and CI later; every critical value lives in SETTINGS (moves to config.yaml in Session 10)
Why:     orchestration only; each step lives in its own tested module, and the quality-gate decision is made here
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import clean_data, evaluate, features, infer, load_data, train, utils, validate


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Every critical value lives here (TEAM_BRIEF.md and README section 1); moves to config.yaml in Session 10
SETTINGS = {
    "paths": {
        "raw_data": "data/raw",
        "inference_data": "data/inference",
        "file_pattern": "JC-*-citibike-tripdata.csv",
        "processed_data": "data/processed/clean.csv",
        "model": "models/model.joblib",
        "predictions": "reports/predictions.csv",
    },
    "problem": {"target_column": "member_casual", "problem_type": "classification", "id_column": "ride_id"},
    "cleaning": {"valid_station_prefixes": ("JC", "HB"), "min_duration_min": 1, "max_duration_min": 180},
    "validation": {
        "required_columns": ["ride_id", "rideable_type", "started_at", "ended_at", "start_station_id",
                             "start_station_name", "end_station_id", "end_station_name"],
        "allowed_target_values": ["member", "casual"],
        "max_missing_station_share": 0.01,
    },
    "split": {"test_size": 0.15, "val_size": 0.15, "random_state": 42, "stratify": True},
    "features": {
        "categorical_columns": ["rideable_type", "hour", "day_of_week"],
        "numeric_columns": ["is_weekend", "duration_min", "is_round_trip"],
        "station_columns": ["start_station_id", "end_station_id"],
        "pagerank_alpha": 0.85,
    },
    "model": {},  # TEAM_BRIEF: HistGradientBoostingClassifier with default parameters
    "quality_gate": {"metric_key": "roc_auc", "comparison": ">=", "threshold": 0.68},  # README quality gate
    "inference": {"conversion_threshold": 0.8, "score_only_value": "casual"},  # README cutoff; brief decision 3C
}


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float,
               random_state: int, stratify: bool) -> tuple:
    """
    Split into train, validation and test before anything is fitted; both shares are of the full dataset.

    Inputs:  X and y; test_size and val_size as shares of ALL rows; random_state; stratify for classification
    Outputs: X_train, X_val, X_test, y_train, y_val, y_test (original index kept, so rows stay traceable)
    Why:     the split happens before any fit, so validation and test rows never shape the model (no leakage)
    """
    n_test = round(test_size * len(X))
    n_val = round(val_size * len(X))
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=n_test, random_state=random_state, stratify=y if stratify else None)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=n_val, random_state=random_state, stratify=y_rest if stratify else None)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _gate_met(metrics: dict[str, float], gate: dict) -> bool:
    """Compare the validation metric with the README quality gate; raise if the key or comparison is unknown."""
    if gate["metric_key"] not in metrics:
        raise KeyError(f"Metric '{gate['metric_key']}' not in evaluate output {sorted(metrics)}. Check SETTINGS.")
    value = metrics[gate["metric_key"]]
    if gate["comparison"] == ">=":
        return value >= gate["threshold"]
    if gate["comparison"] == "<=":
        return value <= gate["threshold"]
    raise ValueError(f"Unknown gate comparison '{gate['comparison']}'. Use '>=' or '<='.")


def _load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Steps 2-3: load, clean (training mode), validate and save clean.csv; return X (no target, no ID) and y."""
    paths, problem = SETTINGS["paths"], SETTINGS["problem"]
    target = problem["target_column"]
    df_raw = load_data.load_raw_data(PROJECT_ROOT / paths["raw_data"], file_pattern=paths["file_pattern"])
    df_clean = clean_data.clean_dataframe(df_raw, target_column=target, **SETTINGS["cleaning"])
    validate.validate_dataframe(df_clean, target_column=target, **SETTINGS["validation"])
    utils.save_csv(df_clean, PROJECT_ROOT / paths["processed_data"])
    X = df_clean.drop(columns=[target, problem["id_column"]])
    y = df_clean[target]
    return X, y


def _score_and_save_predictions(model: object) -> int:
    """Step 9: load, clean (inference mode) and validate the new month; score casual trips; save a ranked list."""
    paths, problem, inference = SETTINGS["paths"], SETTINGS["problem"], SETTINGS["inference"]
    df_raw = load_data.load_raw_data(PROJECT_ROOT / paths["inference_data"], file_pattern=paths["file_pattern"])
    df_new = clean_data.clean_dataframe(df_raw, **SETTINGS["cleaning"])
    validate.validate_dataframe(df_new, **SETTINGS["validation"])
    df_new = df_new[df_new[problem["target_column"]] == inference["score_only_value"]]  # brief decision 3C
    scores = infer.run_inference(model, df_new, conversion_threshold=inference["conversion_threshold"])
    scores.insert(0, problem["id_column"], df_new[problem["id_column"]])
    scores = scores.sort_values("probability", ascending=False)  # infer decision 2A: ranking happens here
    utils.save_csv(scores, PROJECT_ROOT / paths["predictions"])
    return len(scores)


def main() -> None:
    """
    Run the pipeline end to end with one command: python -m src.main

    Why: numbered steps, every value from SETTINGS, and no list reaches marketing unless the quality gate is met
    """
    # 1. SETTINGS paths are relative; the helpers resolve them against PROJECT_ROOT
    # 2-3. Load, clean, validate, save clean.csv; X without target and ID, y as text labels
    X, y = _load_training_data()
    # 4. Split before anything is fitted; X_test and y_test stay sealed (no model selection here)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, **SETTINGS["split"])
    # 5. Fail fast: every configured feature column must exist
    feature_settings = SETTINGS["features"]
    columns = (feature_settings["categorical_columns"] + feature_settings["numeric_columns"]
               + feature_settings["station_columns"])
    missing = sorted(set(columns) - set(X_train.columns))
    if missing:
        raise KeyError(f"Feature columns missing after cleaning: {missing}. Check SETTINGS['features'].")
    # 6. Build the unfitted recipe; train on the train split only
    problem = SETTINGS["problem"]
    recipe = features.get_feature_preprocessor(**feature_settings)
    model = train.train_model(X_train, y_train, recipe, problem["problem_type"], model_params=SETTINGS["model"],
                              random_state=SETTINGS["split"]["random_state"])
    # 7. Evaluate on validation only; report the quality gate as met or not met
    metrics = evaluate.evaluate_model(model, X_val, y_val, problem["problem_type"])
    gate = SETTINGS["quality_gate"]
    gate_met = _gate_met(metrics, gate)
    # TODO: replace print with logging (Session 10)
    print(f"Validation metrics: {metrics}")
    print(f"Quality gate ({gate['metric_key']} {gate['comparison']} {gate['threshold']}): "
          f"{'MET' if gate_met else 'NOT MET'}")
    # 8. Save the model, also when the gate is not met, so it can be inspected
    utils.save_model(model, PROJECT_ROOT / SETTINGS["paths"]["model"])
    if not gate_met:
        print("QUALITY GATE NOT MET: no predictions written; marketing keeps broad promotion (README fallback)")
        raise SystemExit(1)
    # 9-10. Score the new month, save the ranked list, report the artifacts
    n_scored = _score_and_save_predictions(model)
    for key in ("processed_data", "model", "predictions"):
        print(f"Saved: {PROJECT_ROOT / SETTINGS['paths'][key]}")
    print(f"Scored {n_scored:,} casual trips")


if __name__ == "__main__":
    main()
