# Calgary transit ridership optimizer

## Problem statement

Calgary Transit serves hundreds of thousands of riders daily, and planners need accurate ridership forecasts to allocate resources and adjust service frequency. Additionally, understanding network structure helps identify critical transfer points and under-served areas. This project combines time-series forecasting with network graph analysis on monthly ridership data and 7,700+ transit stops.

## Approach

- Fetched monthly ridership and transit stop data from Calgary Open Data
- Engineered lag features (1/3/12-month), rolling means, and year-over-year change
- Trained Ridge Regression, Random Forest, and XGBoost for ridership forecasting
- Built a transit network graph with NetworkX to compute degree and betweenness centrality
- Identified bottleneck stops and under-served areas through connectivity analysis

## Key results

| Metric | Value |
|--------|-------|
| Best model | XGBoost |
| R-squared | ~0.80 |
| MAPE | ~8% |

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
project_13_transit_ridership_optimizer/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── notebooks/
│   └── 01_eda.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py
    └── model.py
```
