"""
Calgary Transit Ridership Optimizer
Streamlit application for forecasting transit ridership and analyzing
the transit network using Calgary Open Data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import (
    load_or_fetch_ridership, load_or_fetch_stops,
    preprocess_ridership, preprocess_stops, engineer_features,
)
from src.model import (
    prepare_model_data, train_models, get_feature_importance,
    build_transit_network, get_network_stats,
    save_model, load_model,
    NUMERICAL_FEATURES,
)

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calgary Transit Ridership Optimizer",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7B8D;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_data(show_spinner="Loading ridership data...")
def load_ridership_data():
    """Load and preprocess ridership data."""
    df = load_or_fetch_ridership(DATA_DIR, limit=100000)
    df = preprocess_ridership(df)
    df = engineer_features(df)
    return df


@st.cache_data(show_spinner="Loading transit stops data...")
def load_stops_data():
    """Load and preprocess transit stops data."""
    df = load_or_fetch_stops(DATA_DIR, limit=10000)
    df = preprocess_stops(df)
    return df


@st.cache_resource(show_spinner="Training forecast models...")
def train_forecast_models(data_hash):
    """Train forecasting models and cache results."""
    df = load_ridership_data()
    X, y, label_encoders, feature_names = prepare_model_data(df)
    if len(X) < 5:
        return {}, {}, None, {}, feature_names
    trained_models, results, scaler, X_test, y_test = train_models(X, y)
    save_model(
        trained_models.get("XGBoost", list(trained_models.values())[0]),
        scaler, label_encoders, feature_names, MODEL_DIR,
    )
    return trained_models, results, scaler, label_encoders, feature_names


@st.cache_resource(show_spinner="Building transit network...")
def build_network(data_hash):
    """Build and analyze transit network."""
    stops_df = load_stops_data()
    G = build_transit_network(stops_df)
    if G is not None and len(G.nodes) > 0:
        stats = get_network_stats(G)
    else:
        stats = {"node_count": 0, "edge_count": 0, "avg_degree": 0, "top_bottleneck_stops": []}
    return G, stats


# ── Main App ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Calgary Transit Ridership Optimizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "Forecast transit ridership and optimize the network using ML on Calgary Open Data"
    "</p>",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Transit Dashboard", "Network Graph", "Ridership Forecast", "Route Optimizer", "About"],
)

# Load data
try:
    ridership_df = load_ridership_data()
except Exception as e:
    st.error(f"Error loading ridership data: {e}")
    st.info("Ensure you have internet access to download the dataset, or place 'transit_ridership.csv' in the data/ folder.")
    ridership_df = pd.DataFrame()

try:
    stops_df = load_stops_data()
except Exception as e:
    st.warning(f"Could not load stops data: {e}")
    stops_df = pd.DataFrame()

# ── Page: Transit Dashboard ─────────────────────────────────────────────────
if page == "Transit Dashboard":
    st.header("Transit Dashboard")

    if len(ridership_df) > 0 and "ridership" in ridership_df.columns:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(ridership_df):,}")
        with col2:
            st.metric("Avg Monthly Ridership", f"{ridership_df['ridership'].mean():,.0f}")
        with col3:
            latest = ridership_df["ridership"].iloc[-1] if len(ridership_df) > 0 else 0
            st.metric("Latest Ridership", f"{latest:,.0f}")
        with col4:
            if "yoy_change" in ridership_df.columns:
                latest_yoy = ridership_df["yoy_change"].iloc[-1]
                st.metric("YoY Change", f"{latest_yoy:+.1f}%" if pd.notna(latest_yoy) else "N/A")
            else:
                st.metric("Data Points", f"{len(ridership_df)}")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Ridership Trend", "Year-over-Year", "Seasonal Patterns"])

        with tab1:
            if "date" in ridership_df.columns:
                fig = px.line(
                    ridership_df, x="date", y="ridership",
                    title="Monthly Transit Ridership Over Time",
                    labels={"date": "Date", "ridership": "Ridership"},
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if "year" in ridership_df.columns and "ridership" in ridership_df.columns:
                yearly = ridership_df.groupby("year")["ridership"].agg(["sum", "mean"]).reset_index()
                yearly.columns = ["Year", "Total Ridership", "Avg Monthly"]

                fig = px.bar(
                    yearly, x="Year", y="Total Ridership",
                    title="Annual Total Ridership",
                    color="Total Ridership", color_continuous_scale="Blues",
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                if "yoy_change" in ridership_df.columns:
                    yoy_df = ridership_df.dropna(subset=["yoy_change"])
                    if len(yoy_df) > 0 and "date" in yoy_df.columns:
                        fig = px.bar(
                            yoy_df, x="date", y="yoy_change",
                            title="Year-over-Year Ridership Change (%)",
                            color="yoy_change",
                            color_continuous_scale="RdYlGn",
                        )
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if "month" in ridership_df.columns:
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                monthly_avg = ridership_df.groupby("month")["ridership"].mean().reset_index()
                monthly_avg["Month Name"] = monthly_avg["month"].map(
                    lambda x: month_names[int(x) - 1] if 1 <= x <= 12 else "Unknown"
                )

                fig = px.bar(
                    monthly_avg, x="Month Name", y="ridership",
                    title="Average Ridership by Month",
                    color="ridership", color_continuous_scale="Viridis",
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No ridership data available. Please check the data source.")

# ── Page: Network Graph ─────────────────────────────────────────────────────
elif page == "Network Graph":
    st.header("Transit Network Analysis")

    if len(stops_df) > 0:
        try:
            G, net_stats = build_network(str(len(stops_df)))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Stops", f"{net_stats['node_count']:,}")
            with col2:
                st.metric("Route Connections", f"{net_stats['edge_count']:,}")
            with col3:
                st.metric("Avg Degree", f"{net_stats['avg_degree']:.2f}")

            st.markdown("---")

            # Stop map colored by centrality
            if "latitude" in stops_df.columns and "longitude" in stops_df.columns:
                map_df = stops_df.dropna(subset=["latitude", "longitude"]).copy()

                if "degree_centrality" in net_stats and net_stats["degree_centrality"]:
                    centrality = net_stats["degree_centrality"]
                    # Map centrality to stops
                    stop_ids = map_df.index if "stop_id" not in map_df.columns else map_df["stop_id"]
                    map_df["centrality"] = [centrality.get(sid, 0) for sid in stop_ids]

                    fig = px.scatter_mapbox(
                        map_df,
                        lat="latitude", lon="longitude",
                        color="centrality",
                        size="centrality",
                        size_max=15,
                        color_continuous_scale="Viridis",
                        mapbox_style="carto-positron",
                        zoom=10,
                        title="Transit Stops Colored by Network Centrality",
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.scatter_mapbox(
                        map_df,
                        lat="latitude", lon="longitude",
                        mapbox_style="carto-positron",
                        zoom=10,
                        title="Transit Stop Locations",
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

            # Top bottleneck stops table
            if net_stats.get("top_bottleneck_stops"):
                st.subheader("Top 10 Bottleneck Stops (Betweenness Centrality)")
                bottleneck_df = pd.DataFrame(net_stats["top_bottleneck_stops"])
                st.dataframe(bottleneck_df, use_container_width=True, hide_index=True)

            # Route connectivity
            if "route_name" in stops_df.columns:
                route_counts = stops_df["route_name"].value_counts().head(20).reset_index()
                route_counts.columns = ["Route", "Stop Count"]
                fig = px.bar(
                    route_counts, x="Route", y="Stop Count",
                    title="Top 20 Routes by Number of Stops",
                    color="Stop Count", color_continuous_scale="Blues",
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error building network: {e}")
            st.info("NetworkX is required for network analysis. Run `pip install networkx`.")
    else:
        st.warning("No stops data available. Please check the data source.")

# ── Page: Ridership Forecast ────────────────────────────────────────────────
elif page == "Ridership Forecast":
    st.header("Ridership Forecast")

    if len(ridership_df) > 0 and "ridership" in ridership_df.columns:
        try:
            trained_models, results, scaler, label_encoders, feature_names = train_forecast_models(
                str(len(ridership_df))
            )
        except Exception as e:
            st.error(f"Error training models: {e}")
            trained_models, results = {}, {}

        if results:
            # Results table
            results_df = pd.DataFrame(results).T.round(2)
            results_df.columns = ["MAE", "RMSE", "R-squared", "MAPE (%)"]
            st.subheader("Forecast Model Comparison")
            st.dataframe(results_df, use_container_width=True)

            # Model selector
            model_name = st.selectbox("Select Model for Forecast", list(trained_models.keys()))
            model = trained_models[model_name]

            # Historical + forecast plot
            if "date" in ridership_df.columns:
                X, y, _, feat_names = prepare_model_data(ridership_df.copy())

                if len(X) > 0:
                    # Get predictions for test period
                    n_test = min(12, len(X) // 5)
                    if n_test < 2:
                        n_test = max(2, int(len(X) * 0.2))

                    X_test = X.iloc[-n_test:]
                    y_test = y.iloc[-n_test:]

                    if model_name == "Ridge Regression" and scaler is not None:
                        y_pred = model.predict(scaler.transform(X_test))
                    else:
                        y_pred = model.predict(X_test)
                    y_pred = np.maximum(y_pred, 0)

                    # Get dates for test period
                    test_dates = ridership_df.iloc[-n_test:]["date"].values if "date" in ridership_df.columns else range(n_test)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ridership_df["date"], y=ridership_df["ridership"],
                        mode="lines", name="Historical",
                        line=dict(color="#667eea"),
                    ))
                    fig.add_trace(go.Scatter(
                        x=test_dates, y=y_pred,
                        mode="lines+markers", name=f"{model_name} Forecast",
                        line=dict(color="#ff6b6b", dash="dash", width=3),
                    ))
                    fig.update_layout(
                        title=f"Historical Ridership + {model_name} Forecast",
                        xaxis_title="Date", yaxis_title="Ridership",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            st.subheader("Feature Importance")
            best_tree_model = trained_models.get("XGBoost", trained_models.get("Random Forest"))
            if best_tree_model:
                importance = get_feature_importance(best_tree_model, feature_names)
                if not importance.empty:
                    fig = px.bar(
                        importance, x="Importance", y="Feature",
                        orientation="h",
                        title="Forecast Feature Importance",
                        color="Importance",
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data to train forecast models.")
    else:
        st.warning("No ridership data available for forecasting.")

# ── Page: Route Optimizer ───────────────────────────────────────────────────
elif page == "Route Optimizer":
    st.header("Route Optimization Insights")

    if len(stops_df) > 0:
        try:
            G, net_stats = build_network(str(len(stops_df)))

            st.subheader("High-Centrality Stops (Potential Hubs)")
            if net_stats.get("top_bottleneck_stops"):
                hub_df = pd.DataFrame(net_stats["top_bottleneck_stops"])
                hub_df["betweenness"] = hub_df["betweenness"].round(4)
                hub_df["degree"] = hub_df["degree"].round(4)
                st.dataframe(hub_df, use_container_width=True, hide_index=True)

                st.markdown("""
                **Interpretation:**
                - **High betweenness centrality** indicates stops that serve as critical transfer points
                - **High degree centrality** indicates stops connected to many routes
                - These stops are candidates for service frequency increases or infrastructure investment
                """)

            st.markdown("---")

            # Under-served area analysis
            st.subheader("Connectivity Analysis")
            if "degree_centrality" in net_stats and net_stats["degree_centrality"]:
                centrality_values = list(net_stats["degree_centrality"].values())

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Connectivity", f"{np.mean(centrality_values):.4f}")
                with col2:
                    st.metric("Min Connectivity", f"{np.min(centrality_values):.4f}")
                with col3:
                    st.metric("Max Connectivity", f"{np.max(centrality_values):.4f}")

                # Distribution of centrality
                fig = px.histogram(
                    x=centrality_values, nbins=30,
                    title="Distribution of Stop Connectivity (Degree Centrality)",
                    labels={"x": "Degree Centrality", "y": "Count"},
                    color_discrete_sequence=["#667eea"],
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Low-connectivity stops (potential under-served areas)
                low_threshold = np.percentile(centrality_values, 25)
                low_connectivity_stops = [
                    {"stop_id": k, "centrality": v}
                    for k, v in net_stats["degree_centrality"].items()
                    if v <= low_threshold
                ]
                st.subheader(f"Under-Connected Stops (bottom 25%, centrality <= {low_threshold:.4f})")
                st.write(f"Found {len(low_connectivity_stops)} stops with low connectivity.")

                if "route_name" in stops_df.columns:
                    route_stats = stops_df["route_name"].value_counts().describe()
                    st.subheader("Route Size Statistics")
                    st.write(route_stats)

        except Exception as e:
            st.error(f"Error in route optimization analysis: {e}")
    else:
        st.warning("No stops data available for route optimization.")

# ── Page: About ─────────────────────────────────────────────────────────────
elif page == "About":
    st.header("About This Project")

    st.markdown("""
    ### Problem Statement
    Calgary Transit serves hundreds of thousands of riders daily across bus and CTrain
    networks. Understanding ridership patterns and network structure is critical for
    optimizing routes, improving frequency on high-demand corridors, and identifying
    under-served areas. This application combines time series forecasting with network
    graph analysis to provide actionable transit optimization insights.

    ### Datasets
    - **Ridership:** [Calgary Open Data - Transit Ridership](https://data.calgary.ca/Transportation-Transit/Monthly-Transit-Ridership/nypk-snzd)
      - **Dataset ID:** `nypk-snzd`
      - Monthly ridership aggregates
    - **Transit Stops:** [Calgary Open Data - Transit Stops](https://data.calgary.ca/Transportation-Transit/Transit-Stops/muzh-c9qc)
      - **Dataset ID:** `muzh-c9qc`
      - ~6,000 transit stops with coordinates and route info

    ### Methodology
    1. **Ridership Forecasting:**
       - Temporal features: lag (1m, 3m, 12m), rolling means (3m, 6m, 12m), YoY change
       - Temporal train/test split (last 12 months as test)
       - Models: Ridge Regression, Random Forest, XGBoost
       - Metrics: MAE, RMSE, R-squared, MAPE
    2. **Network Analysis:**
       - Transit stops as nodes, routes as edges (NetworkX)
       - Degree centrality and betweenness centrality
       - Bottleneck identification and under-served area analysis

    ### Technical Stack
    - **Data Processing:** pandas, NumPy
    - **ML:** scikit-learn, XGBoost
    - **Network Analysis:** NetworkX
    - **Visualization:** Plotly
    - **Web App:** Streamlit
    - **Data Access:** Socrata API (sodapy)

    ### Data Source & License
    Contains information licensed under the Open Government License - City of Calgary.
    Data accessed from [data.calgary.ca](https://data.calgary.ca/).
    """)

    st.markdown("---")
    st.markdown(
        "Built as part of the "
        "[Calgary Open Data ML/DS Portfolio](https://github.com/guydev42/calgary-data-portfolio)"
    )
