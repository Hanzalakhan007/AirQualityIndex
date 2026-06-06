# Karachi AirWatch: AQI Forecasting System

Karachi AirWatch is an end-to-end air-quality forecasting system for Karachi, Pakistan. It uses OpenWeather air-pollution data, derives a continuous AQI target from PM2.5, builds time-series features, trains multiple forecasting models, and serves the final Ridge Regression model through a Streamlit dashboard and Flask API.

**Live dashboard:** https://hanzalakhan-aqi.streamlit.app/

The production system provides:

- latest observed AQI from the feature store;
- today's AQI confirmation prediction;
- next 3 days AQI forecast;
- model leaderboard and evaluation metrics;
- SHAP/LIME explainability outputs;
- automated hourly feature refresh and daily model training.

## What is included

- `app/dashboard.py`: modular Streamlit dashboard
- `app/api.py`: Flask prediction API with `/health` and `/predict`
- `config/settings.py`: centralized paths and environment variables
- `src/`: data collection, feature engineering, training, explainability
- `scripts/`: simple wrappers for fetch, feature build, training, EDA, and full pipeline runs
- `.github/workflows/`: training, feature, CI, and deployment templates

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and set your OpenWeather and MongoDB settings.
4. Run the pipeline:

```bash
python scripts/fetch_raw_data.py
python scripts/feature_pipeline.py
python scripts/training_pipeline.py
```

5. Start the dashboard:

```bash
streamlit run app.py
```

6. Start the API:

```bash
python -m flask --app app.api run
```

## Project structure

```text
.github/workflows/      GitHub Actions pipelines
.streamlit/config.toml  Streamlit theme and server settings
app/                    Dashboard and API entrypoints
config/                 Shared settings
data/                   Raw and processed AQI data
models/                 Trained model artifacts
reports/                EDA charts and explainability outputs
scripts/                Friendly task wrappers
src/                    Core data science code
```

## Model Summary

- Production model: Ridge Regression
- Production RMSE: 29.13 AQI points
- Fit status: OK
- Evaluation: chronological holdout split over the engineered historical dataset

## Operational Notes

- The dashboard and API read features and model artifacts from MongoDB at runtime.
- The hourly workflow refreshes features only; it does not retrain the model.
- The daily workflow retrains and evaluates the models.
- `verify_prediction.py` provides a lightweight smoke test for CI.
