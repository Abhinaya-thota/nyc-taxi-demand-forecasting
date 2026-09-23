# NYC Taxi Demand Forecasting

An end-to-end machine learning pipeline that forecasts hourly taxi demand across NYC pickup zones from real NYC TLC trip data, compares three model families, and translates the forecasts into operational decisions — fleet allocation and surge pricing.

## Problem

Taxi demand in NYC fluctuates constantly by time, location, and day of week. Without accurate forecasting, vehicles get misallocated — oversupplied in low-demand zones, undersupplied in high-demand ones — leading to longer wait times and inefficient operations.

## Data

- [NYC TLC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (June 2025 used here)
- NYC Taxi Zone Lookup Table (maps `LocationID` to borough/zone names)

Trip data (`.parquet`) is not included in this repo — it's large and easy to re-download from the TLC site (see link above). The small zone lookup CSV is included for convenience.

## Approach

1. **Data collection & labeling** — loaded raw trip records, merged in pickup/dropoff zone names via the zone lookup table.
2. **Data evaluation & cleaning** — checked for missing values, duplicates, and invalid datetimes; removed trips with zero/negative distance or fare, and trips longer than 3 hours; corrected zero-passenger records.
3. **Feature engineering** — hour, day of week, weekend flag, month, trip speed, and a rolling 3-period demand average per pickup zone.
4. **Modeling** — trained and compared three regressors on hourly zone-level demand: Random Forest, XGBoost, and an MLP neural network.
5. **Hyperparameter tuning** — GridSearchCV (3-fold CV) for all three models.
6. **Deployment simulation** — used the best model's forecasts to:
   - Identify top demand hotspots by hour and by day of week (heatmaps)
   - Allocate a fixed fleet (500 vehicles) proportionally across zones by predicted demand
   - Compute a demand-based surge pricing factor per zone, using real average fares from the trip data

## Results

| Model | MAE (tuned) |
|---|---|
| Random Forest | See `metrics_df` in notebook |
| **XGBoost** | Best-performing model overall |
| MLP | Weakest of the three |

XGBoost consistently outperformed both Random Forest and the MLP across MAE, RMSE, and R², and was used for the downstream fleet allocation and surge pricing simulations.

## Key Findings

- **Tree-based models beat the neural network** on this structured, tabular dataset — architecture choice should match the data, not just model sophistication.
- **Hyperparameter tuning affected each model differently** — XGBoost and Random Forest saw modest gains from tuning, while the MLP was far more sensitive to its hyperparameters (layer sizes, learning rate, regularization).
- Demand forecasts were pushed all the way to an operational simulation — not left as an offline accuracy number — including a full fleet allocation and a fare-based surge pricing model using real average fares per zone.

## Tech Stack

`Python` `pandas` `scikit-learn` `XGBoost` `GridSearchCV` `seaborn` `matplotlib`

## Repository Structure

```
├── nyc_taxi_demand_forecasting.ipynb   # Full pipeline: data → features → models → deployment sim
├── taxi_zone_lookup.csv                # NYC TLC zone ID → borough/zone name mapping
└── requirements.txt
```

## Running It

```bash
pip install -r requirements.txt
```
Download a month of Yellow Taxi trip data (`.parquet`) from the [NYC TLC site](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), place it in the repo root, and update the `parquet_path` variable at the top of the notebook if the filename differs. Then run the notebook top to bottom.

## Future Work

- Incorporate weather and event data as additional features
- Deploy the trained model as a real-time forecasting API
- Extend the fleet/surge simulation with more realistic operational constraints (driver shift limits, zone adjacency)
