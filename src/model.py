"""ML model training, evaluation, and network analysis for Transit Ridership Optimization."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


NUMERICAL_FEATURES = [
    "lag_1m", "lag_3m", "lag_12m",
    "rolling_mean_3m", "rolling_mean_6m", "rolling_mean_12m",
    "yoy_change", "month", "quarter", "year",
]

TARGET = "ridership"


def prepare_model_data(df):
    """Prepare feature matrix and target vector for forecasting."""
    df = df.copy()

    # Select available features
    available_features = [c for c in NUMERICAL_FEATURES if c in df.columns]

    # Drop rows with NaN in features (from lagging)
    df = df.dropna(subset=available_features + [TARGET])

    X = df[available_features].copy()
    y = df[TARGET].copy()

    # Fill any remaining missing numerical values with median
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    # Remove any remaining NaN rows
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y, {}, available_features


def train_models(X, y, random_state=42):
    """Train multiple forecasting models using temporal split."""
    # Temporal split: last 12 samples as test (preserving time order)
    n_test = min(12, len(X) // 5)
    if n_test < 2:
        n_test = max(2, int(len(X) * 0.2))

    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=random_state, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state, n_jobs=-1,
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        if name == "Ridge Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Clip negative predictions to zero
        y_pred = np.maximum(y_pred, 0)

        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
            "MAPE": np.mean(
                np.abs((y_test - y_pred) / y_test.replace(0, np.nan).dropna())
            ) * 100 if (y_test != 0).any() else 0,
        }
        trained_models[name] = model

    return trained_models, results, scaler, X_test, y_test


def build_transit_network(stops_df):
    """Build a transit network graph from stops data using NetworkX."""
    try:
        import networkx as nx
    except ImportError:
        return None

    G = nx.Graph()

    # Add stops as nodes
    for _, row in stops_df.iterrows():
        stop_id = row.get("stop_id", row.name)
        attrs = {}
        if "latitude" in row and pd.notna(row["latitude"]):
            attrs["latitude"] = row["latitude"]
        if "longitude" in row and pd.notna(row["longitude"]):
            attrs["longitude"] = row["longitude"]
        if "route_name" in row:
            attrs["route_name"] = str(row["route_name"])
        stop_name = row.get("stop_name", row.get("description", str(stop_id)))
        attrs["name"] = stop_name
        G.add_node(stop_id, **attrs)

    # Add edges between consecutive stops on the same route
    if "route_name" in stops_df.columns:
        for route, group in stops_df.groupby("route_name"):
            stop_ids = group.index.tolist()
            if "stop_id" in group.columns:
                stop_ids = group["stop_id"].tolist()
            for i in range(len(stop_ids) - 1):
                G.add_edge(stop_ids[i], stop_ids[i + 1], route=str(route))

    return G


def get_network_stats(G):
    """Compute network statistics from the transit graph."""
    import networkx as nx

    if G is None or len(G.nodes) == 0:
        return {
            "node_count": 0, "edge_count": 0,
            "avg_degree": 0, "top_bottleneck_stops": [],
        }

    stats = {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "avg_degree": np.mean([d for _, d in G.degree()]) if len(G) > 0 else 0,
    }

    # Degree centrality
    degree_cent = nx.degree_centrality(G)

    # Betweenness centrality (on a sample for large graphs)
    if len(G) > 500:
        betweenness = nx.betweenness_centrality(G, k=min(100, len(G)))
    else:
        betweenness = nx.betweenness_centrality(G)

    # Top bottleneck stops by betweenness centrality
    top_bottlenecks = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    stats["top_bottleneck_stops"] = [
        {
            "stop_id": stop,
            "name": G.nodes[stop].get("name", str(stop)),
            "betweenness": score,
            "degree": degree_cent.get(stop, 0),
        }
        for stop, score in top_bottlenecks
    ]

    stats["degree_centrality"] = degree_cent
    stats["betweenness_centrality"] = betweenness

    return stats


def get_feature_importance(model, feature_names, model_name="XGBoost"):
    """Extract feature importance from tree-based models."""
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        return importance
    return pd.DataFrame()


def save_model(model, scaler, label_encoders, feature_names, model_dir):
    """Save trained model and preprocessing artifacts."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.joblib"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))


def load_model(model_dir):
    """Load trained model and preprocessing artifacts."""
    model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
    return model, scaler, label_encoders, feature_names
