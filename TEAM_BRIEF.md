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