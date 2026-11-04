"""
Build the unfitted preprocessing recipe for the three feature groups agreed in TEAM_BRIEF.md.

Inputs:  column lists for the categorical, numeric and station groups (from SETTINGS), and the PageRank alpha
Outputs: an unfitted ColumnTransformer; train.py fits it on the train split only, inside the model Pipeline
Used by: main.py builds it, train.py fits it; it is saved inside model.joblib and reused by infer.py
Why:     one recipe for training and inference prevents leakage and training-serving skew
"""
import networkx as nx
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class StationPageRank(BaseEstimator, TransformerMixin):
    """
    Score each trip's start and end station by PageRank centrality, learned from training trips only.

    Inputs:  a DataFrame with exactly two columns, [start station ID, end station ID]
    Outputs: two float columns, the start and end station scores
    Why:     fit learns the station graph from train only (no leakage); unseen or missing stations get the median
    """

    def __init__(self, alpha: float = 0.85) -> None:
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y=None) -> "StationPageRank":
        """Build the weighted start-to-end station graph from training trips and store each station's score."""
        start, end = X.columns
        edges = X.dropna().groupby([start, end]).size().reset_index(name="weight")
        graph = nx.DiGraph()
        graph.add_weighted_edges_from(edges.itertuples(index=False, name=None))
        scores = pd.Series(nx.pagerank(graph, alpha=self.alpha, weight="weight"))
        self.scores_ = scores * len(scores)
        self.fallback_ = float(self.scores_.median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Look up each station's score; stations unseen in training, or missing, get the median score."""
        looked_up = pd.DataFrame({column: X[column].map(self.scores_) for column in X.columns})
        return looked_up.fillna(self.fallback_).astype(float)


def _check_columns(categorical_columns: list[str], numeric_columns: list[str], station_columns: list[str]) -> None:
    """Raise ValueError if a feature group is empty or station_columns is not exactly [start, end] (decision 1A)."""
    if not categorical_columns:
        raise ValueError("categorical_columns is empty. Check SETTINGS['features'].")
    if not numeric_columns:
        raise ValueError("numeric_columns is empty. Check SETTINGS['features'].")
    if len(station_columns) != 2:
        raise ValueError(f"station_columns must be exactly two columns [start, end], got {station_columns}.")


def get_feature_preprocessor(categorical_columns: list[str], numeric_columns: list[str],
                             station_columns: list[str], pagerank_alpha: float = 0.85) -> ColumnTransformer:
    """
    Build the unfitted recipe: one-hot categoricals, scaled numerics, scaled station PageRank.

    Inputs:  categorical_columns, e.g. rideable_type, hour, day_of_week; numeric_columns, e.g. duration_min
             station_columns, exactly [start, end] station IDs; pagerank_alpha, the brief's 0.85
    Outputs: an unfitted ColumnTransformer with steps "categorical", "numeric" and "station"
    Why:     train.py fits it on the train split only, so nothing is learned from validation or test data
    """
    # TODO: replace print with logging (Session 10)
    print("Building the feature recipe (unfitted)")
    _check_columns(categorical_columns, numeric_columns, station_columns)
    station_pipeline = Pipeline([
        ("pagerank", StationPageRank(alpha=pagerank_alpha)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        # few values per column; an unseen value in a new month is ignored, not a crash
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        # as in the notebook: harmless for trees, and allows a swap to a linear model
        ("numeric", StandardScaler(), numeric_columns),
        # station centrality from the training graph, then scaled
        ("station", station_pipeline, station_columns),
    ])


# Recommendations for later (not built now):
# - Refit PageRank monthly on a rolling window, so newly opened stations stop falling back to the median
# - Try sin/cos encoding of hour instead of one-hot, so 23:00 sits next to 00:00 for the model
# - Group rarely used stations before scoring if new months bring many unseen station IDs
