<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Calgary%20Transit%20Ridership%20Optimizer&fontSize=35&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Graph%20network%20analysis%20%2B%20demand%20forecasting%20for%207%2C700%2B%20stops&descSize=16&descAlignY=55&descColor=c8ddf0" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/status-complete-2ea44f?style=for-the-badge" />
  <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/NetworkX-graphs-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Prophet-forecasting-0467DF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/streamlit-dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

---

## Overview

**Problem** -- Calgary Transit serves hundreds of thousands of riders daily across a network of 7,700+ stops, and planners need accurate ridership forecasts to allocate resources and adjust service frequency. Additionally, understanding network structure is essential for identifying critical transfer points and under-served areas.

**Solution** -- This project combines time-series demand forecasting using XGBoost and Prophet with graph network analysis powered by NetworkX to model the transit network topology, compute centrality metrics for every stop, and predict monthly ridership volumes with high accuracy.

**Impact** -- Achieves an R-squared of 0.80 with 8% MAPE on ridership forecasting, while graph analysis identifies bottleneck stops and connectivity gaps that inform service optimization decisions.

---

## Results

| Metric | Value |
|--------|-------|
| Best model | XGBoost |
| R-squared | **0.80** |
| MAPE | 8% |
| Transit stops analyzed | 7,700+ |

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Calgary Open     │────▶│  Time-Series      │────▶│  XGBoost /        │
│  Data (Socrata)   │     │  Feature Eng.     │     │  Prophet          │
│  - Ridership      │     │  - Lag features   │     │  Forecasting      │
│  - Transit stops  │     │  - Rolling means  │     └────────┬─────────┘
└──────────────────┘     └──────────────────┘              │
        │                                                    │
        │                ┌──────────────────┐     ┌────────▼─────────┐
        └───────────────▶│  NetworkX         │────▶│  Streamlit        │
                         │  Graph Analysis   │     │  Dashboard        │
                         │  - Centrality     │     └──────────────────┘
                         │  - Bottlenecks    │
                         └──────────────────┘
```

---

<details>
<summary><strong>Project structure</strong></summary>

```
project_13_transit_ridership_optimizer/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md
├── data/                   # Cached CSV data
├── models/                 # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching & feature engineering
    └── model.py            # Forecasting models & graph analysis
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/calgary-transit-optimizer.git
cd calgary-transit-optimizer

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Dataset | Source | Records | Key fields |
|---------|--------|---------|------------|
| Monthly ridership | Calgary Open Data | Multi-year | Route, month, total boardings |
| Transit stops | Calgary Open Data | 7,700+ | Stop ID, latitude, longitude, routes served |

---

## Tech stack

<p align="center">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat-square" />
  <img src="https://img.shields.io/badge/Prophet-0467DF?style=flat-square" />
  <img src="https://img.shields.io/badge/NetworkX-graphs-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-API-blue?style=flat-square" />
</p>

---

## Methodology

1. **Data collection** -- Fetched monthly ridership volumes and transit stop location data from Calgary Open Data via Socrata API.
2. **Time-series feature engineering** -- Created lag features (1, 3, and 12-month), rolling mean windows, and year-over-year change metrics to capture seasonal patterns and trends.
3. **Demand forecasting** -- Trained Ridge Regression, Random Forest, and XGBoost regressors for ridership prediction, with XGBoost achieving 0.80 R-squared and 8% MAPE. Prophet was used for long-horizon seasonal decomposition and trend forecasting.
4. **Graph network analysis** -- Built a transit network graph with NetworkX using 7,700+ stops as nodes and shared routes as edges. Computed degree centrality and betweenness centrality to identify critical transfer hubs and connectivity bottlenecks.
5. **Optimization insights** -- Identified under-served areas through connectivity analysis and flagged high-centrality stops where service disruptions would have the greatest network-wide impact.
6. **Dashboard** -- Built a Streamlit application with ridership forecast visualizations, network topology maps, and stop-level centrality rankings.

---

## Acknowledgements

Data provided by the [City of Calgary Open Data Portal](https://data.calgary.ca/). This project was developed as part of a municipal data analytics portfolio.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>

<p align="center">
  Built by <a href="https://github.com/guydev42">Ola K.</a>
</p>
