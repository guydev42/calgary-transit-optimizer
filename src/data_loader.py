"""Data loading and preprocessing for Calgary Transit Ridership and Stops datasets."""

import os
import logging
import pandas as pd
import numpy as np
from sodapy import Socrata

logger = logging.getLogger(__name__)

RIDERSHIP_ID = "nypk-snzd"
STOPS_ID = "muzh-c9qc"
DOMAIN = "data.calgary.ca"


def fetch_ridership_data(limit=100000):
    """Fetch transit ridership data from Calgary Open Data API."""
    logger.info("Fetching ridership data from Socrata API (dataset %s)...", RIDERSHIP_ID)
    try:
        client = Socrata(DOMAIN, None, timeout=60)
        results = client.get(RIDERSHIP_ID, limit=limit)
        client.close()
        logger.info("Fetched %d ridership records from API.", len(results))
        df = pd.DataFrame.from_records(results)
        return df
    except Exception as exc:
        logger.error("Failed to fetch ridership data from Socrata API: %s", exc)
        raise


def fetch_stops_data(limit=10000):
    """Fetch transit stops data from Calgary Open Data API."""
    logger.info("Fetching stops data from Socrata API (dataset %s)...", STOPS_ID)
    try:
        client = Socrata(DOMAIN, None, timeout=60)
        results = client.get(STOPS_ID, limit=limit)
        client.close()
        logger.info("Fetched %d stop records from API.", len(results))
        df = pd.DataFrame.from_records(results)
        return df
    except Exception as exc:
        logger.error("Failed to fetch stops data from Socrata API: %s", exc)
        raise


def load_or_fetch_ridership(data_dir, limit=100000, force_refresh=False):
    """Load ridership data from local CSV or fetch from API."""
    csv_path = os.path.join(data_dir, "transit_ridership.csv")
    if os.path.exists(csv_path) and not force_refresh:
        logger.info("Loading cached ridership data from %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info("Loaded %d records from cache.", len(df))
        return df

    try:
        df = fetch_ridership_data(limit=limit)
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Cached %d ridership records to %s", len(df), csv_path)
    except Exception as exc:
        logger.error("API fetch failed: %s", exc)
        if os.path.exists(csv_path):
            logger.warning("Falling back to cached ridership data.")
            return pd.read_csv(csv_path, low_memory=False)
        raise
    return df


def load_or_fetch_stops(data_dir, limit=10000, force_refresh=False):
    """Load stops data from local CSV or fetch from API."""
    csv_path = os.path.join(data_dir, "transit_stops.csv")
    if os.path.exists(csv_path) and not force_refresh:
        logger.info("Loading cached stops data from %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info("Loaded %d records from cache.", len(df))
        return df

    try:
        df = fetch_stops_data(limit=limit)
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Cached %d stop records to %s", len(df), csv_path)
    except Exception as exc:
        logger.error("API fetch failed: %s", exc)
        if os.path.exists(csv_path):
            logger.warning("Falling back to cached stops data.")
            return pd.read_csv(csv_path, low_memory=False)
        raise
    return df


def preprocess_ridership(df):
    """Clean and preprocess ridership data."""
    df = df.copy()

    # Identify date/time columns and ridership value columns
    # Try common column patterns from the Socrata dataset
    date_cols = [c for c in df.columns if "date" in c.lower() or "month" in c.lower() or "year" in c.lower()]
    numeric_cols = [c for c in df.columns if c not in date_cols]

    # Convert potential ridership columns to numeric
    for col in df.columns:
        if col not in date_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Try to create a datetime index from available date info
    if "year" in df.columns and "month" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        df = df.dropna(subset=["year", "month"])
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

    # Identify ridership column (look for common names)
    ridership_candidates = [c for c in df.columns if "rider" in c.lower() or "passenger" in c.lower() or "boarding" in c.lower()]
    if ridership_candidates:
        df["ridership"] = pd.to_numeric(df[ridership_candidates[0]], errors="coerce")
    else:
        # Use first numeric column that isn't year/month
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["year", "month"]]
        if num_cols:
            df["ridership"] = df[num_cols[0]]

    # Drop rows without ridership data
    if "ridership" in df.columns:
        df = df.dropna(subset=["ridership"])
        df = df[df["ridership"] > 0]

    # Sort by date
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # Quarter indicator
    if "month" in df.columns:
        df["quarter"] = ((df["month"] - 1) // 3) + 1

    return df


def preprocess_stops(df):
    """Clean and preprocess transit stops data."""
    df = df.copy()

    # Parse latitude/longitude
    for col in ["latitude", "longitude", "lat", "lon", "stop_lat", "stop_lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardize lat/lon column names
    if "stop_lat" in df.columns and "latitude" not in df.columns:
        df["latitude"] = df["stop_lat"]
    if "stop_lon" in df.columns and "longitude" not in df.columns:
        df["longitude"] = df["stop_lon"]
    if "lat" in df.columns and "latitude" not in df.columns:
        df["latitude"] = df["lat"]
    if "lon" in df.columns and "longitude" not in df.columns:
        df["longitude"] = df["lon"]

    # Extract route name if available
    route_cols = [c for c in df.columns if "route" in c.lower()]
    if route_cols and "route_name" not in df.columns:
        df["route_name"] = df[route_cols[0]].astype(str)

    # Drop rows without coordinates
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.dropna(subset=["latitude", "longitude"])

    return df


def engineer_features(df):
    """Create time-series features for ridership forecasting."""
    df = df.copy()

    if "ridership" not in df.columns:
        return df

    # Lag features
    df["lag_1m"] = df["ridership"].shift(1)
    df["lag_3m"] = df["ridership"].shift(3)
    df["lag_12m"] = df["ridership"].shift(12)

    # Rolling means
    df["rolling_mean_3m"] = df["ridership"].rolling(window=3, min_periods=1).mean()
    df["rolling_mean_6m"] = df["ridership"].rolling(window=6, min_periods=1).mean()
    df["rolling_mean_12m"] = df["ridership"].rolling(window=12, min_periods=1).mean()

    # Year-over-year percent change
    df["yoy_change"] = df["ridership"].pct_change(periods=12) * 100

    # Month and quarter are already created in preprocess_ridership

    return df
