# Team brief

## Business and data
Business case: README.md, section 1 (merged into dev on 2026-11-02)
Use case: Jersey City marketing manager decides who gets a membership offer using a monthly ranked list of member-like casual trips (from README, Solution)
Notebook: citibike_member_classifier_mlops.ipynb
Target: member_casual (member = 1, casual = 0) | Problem type: classification | ID column: ride_id

## Features, model and evaluation
Feature groups: categorical: rideable_type, hour, day_of_week -> OneHotEncoder(handle_unknown="ignore") (few values; an unseen value in a new month is ignored, not a crash)
Feature groups: numeric: is_weekend, duration_min, is_round_trip -> StandardScaler (as in the notebook; harmless for trees, allows a linear model swap)
Feature groups: station: start_station_id, end_station_id -> PageRank fitted on train (unseen station -> median), then StandardScaler (station centrality, README Solution)
Excluded columns: ride_id (ID, kept next to predictions); started_at, ended_at (exact times, replaced by hour and day); start_station_name, end_station_name (duplicate the IDs); start_lat, start_lng, end_lat, end_lng (privacy, dropped in clean_data); member_casual (target)
Model: HistGradientBoostingClassifier with default parameters, random_state 42
Metric: ROC AUC, key in evaluate_model output: "roc_auc", quality gate: >= 0.68 (from README, Goal and success metrics)
Split: test 0.15 and validation 0.15, both of the full dataset; stratified; random_state 42

## Rules, decisions and owners
Validation rules: required columns present; data not empty; ride_id unique; member_casual only "member" or "casual" with no missing values (training mode); under 1% of trips missing a station ID; one name per station ID (README Data and risks)
README follow-ups: time-based split -> Jeevan; row count vs previous month and new-station checks -> Jeevan; baseline conversion rate from Lyft -> Jeevan
features.py signature: get_feature_preprocessor(categorical_columns, numeric_columns, station_columns), settings per group: station PageRank alpha=0.85, unseen station -> median
Agreed extra parameters: load_raw_data(file_pattern=None); clean_dataframe(valid_station_prefixes=("JC", "HB"), min_duration_min=1, max_duration_min=180); validate_dataframe(allowed_target_values=None, max_missing_station_share=0.01); run_inference(conversion_threshold=None)
utils.py helpers: load_csv, save_csv, save_model, load_model
Environment: networkx (station PageRank)
Decisions: split -> random stratified train/validation/test, gate on validation (README check 1C)
Decisions: labels -> member_casual kept as strings; model classes sort to [casual, member], so probability = P(member)
Decisions: conversion cut -> run_inference adds "conversion_target" when conversion_threshold=0.8 is passed (2B)
Decisions: inference data -> JC-202607 file in data/inference/, casual trips only, filtered in main.py; earlier than training months, so no forward-drift test (3C)
Decisions: privacy -> clean_data drops coordinate columns (5B)
Decisions: trip features -> hour, day_of_week, is_weekend, is_round_trip, duration_min computed in clean_data as fixed row rules (1A)
Decisions: PageRank -> one transformer class in features.py, a recorded exception to plain functions (2A)
Owners (solo): all modules -> Jeevan
