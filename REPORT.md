# Final Project Report: Air Quality Index Predictor

## 1. Project Overview

This project delivers an end-to-end AQI forecasting system for Karachi using a serverless-style workflow built around GitHub Actions, MongoDB Atlas, Streamlit, and Flask. The system collects historical air-quality data, engineers model-ready features, retrains forecasting models automatically, stores artifacts in a cloud-backed feature store and model registry, and serves predictions through a live dashboard and API.

The final product is framed as a **next 3 days AQI prediction system**. The dashboard also shows **today's predicted AQI** separately as a confirmation signal, but the forecast product itself is presented as a three-day outlook.

## 2. Objectives Achieved

The implemented system covers the core project goals:

- Automated feature pipeline for air-quality data collection and feature engineering
- Historical backfill through a rolling two-year rebuild window
- Automated daily model training and evaluation
- Cloud-backed feature store and model registry using MongoDB Atlas
- Multiple forecasting models, including a deep learning model
- Streamlit dashboard for interactive prediction and monitoring
- Flask API for programmatic prediction access
- SHAP and LIME explainability outputs
- GitHub Actions workflows for CI, hourly feature refresh, daily training, and deployment verification
- Hazard alert panel in the dashboard for risky forecast conditions

One intentionally deferred enhancement remains outside this report's completion scope:

- dedicated arbitrary date-range backfill pipeline

## 3. System Architecture

The system is organized into four layers:

1. Data ingestion
   Raw historical air-quality data is fetched from OpenWeather, with Open-Meteo used as a fallback when necessary.

2. Feature engineering and storage
   The raw hourly observations are transformed into model-ready features and stored in MongoDB Atlas. Local CSV artifacts are also refreshed as a fallback when MongoDB is unavailable.

3. Training and model registry
   Forecasting models are trained on engineered features, evaluated on chronological holdout data, saved locally, and uploaded to MongoDB GridFS plus a registry collection.

4. Prediction serving
   The Streamlit dashboard and Flask API load the latest features, scaler, and models from MongoDB at runtime, with local artifact fallback support when cloud access is unavailable.

## 4. Data Collection and Feature Pipeline

The feature pipeline is implemented in `src/features/build_features.py` and wrapped by `scripts/feature_pipeline.py` and `scripts/run_hourly_pipeline.py`.

### Data sources

- OpenWeather Air Pollution History API
- Open-Meteo air-quality fallback

### Engineered features

The pipeline generates:

- pollutant values: `co`, `no`, `no2`, `o3`, `so2`, `pm2_5`, `pm10`, `nh3`
- time-based features: `hour`, `day_of_week`, `month`, `is_weekend`
- lag features: `aqi_lag_1`, `aqi_lag_24`, `aqi_lag_48`, `aqi_lag_72`
- rolling features: `aqi_rolling_24`, `pm25_rolling_24`, `pm10_rolling_24`, `co_rolling_24`
- derived trend feature: `aqi_change_rate`

### Historical backfill behavior

The current implementation automatically rebuilds approximately the last two years of historical AQI data every time the feature pipeline runs. This provides a robust rolling backfill and refresh strategy, although it is not yet parameterized by an arbitrary user-specified date range.

### Storage behavior

The pipeline:

- updates local fallback files in `data/raw/` and `data/processed/`
- upserts engineered features into the MongoDB `aqi_features` collection
- maintains a MongoDB lease document to prevent duplicate hourly runs within the same time slot

## 5. Model Training Pipeline

The training pipeline is implemented in `src/models/train_model.py` and wrapped by `scripts/training_pipeline.py`.

### Training target

The model predicts:

- `target_day_0`: today's AQI confirmation signal
- `target_day_1`: next day forecast
- `target_day_2`: second day forecast
- `target_day_3`: third day forecast

This structure supports the final dashboard/API design where today's prediction is displayed separately while the official forecast covers the next three days.

### Models used

The project includes:

- Ridge Regression
- Random Forest Regressor
- XGBoost wrapped in `MultiOutputRegressor`
- PyTorch MLP regressor
- Persistence baseline for comparison during training

### Preprocessing

- chronological train/test split
- `StandardScaler` fit on the training data
- shared feature alignment for both training and inference

### Evaluation metrics

The training pipeline computes:

- RMSE
- MAE
- R²
- train/test RMSE gap
- overfit heuristic
- selection score used to pick the best available model

### Current checked-in metrics snapshot

The current `reports/model_metrics.csv` snapshot in the repository reports:

| Model | RMSE | MAE | R² | Fit Status | Best |
|---|---:|---:|---:|---|---|
| Ridge Regression | 29.67 | 21.97 | 0.33 | ok | yes |
| Random Forest | 46.10 | 33.14 | -0.62 | overfit-risk | no |
| XGBoost | 54.24 | 38.32 | -1.25 | overfit-risk | no |

This checked-in CSV currently reflects the classical models only. PyTorch MLP support has been integrated into the training and inference pipeline and is available in the deployed dashboard/model registry, but the local CSV snapshot has not yet been regenerated in this repository state to include its latest benchmark row.

## 6. Model Registry and Storage Management

The model registry uses MongoDB Atlas with:

- `model_registry` for metadata
- `model_artifacts.files` and `model_artifacts.chunks` via GridFS for model binaries

Each training run stores:

- Ridge model
- Random Forest model
- XGBoost model
- PyTorch model
- scaler artifact

### Storage reliability improvement

During development, the hourly pipeline failed when MongoDB Atlas exceeded the free-tier storage cap. To resolve this, the project was updated with:

- model registry retention controls via `MONGO_MODEL_REGISTRY_MAX_VERSIONS`
- registry pruning helpers in `src/model_registry.py`
- cleanup script `scripts/prune_model_registry.py`
- clearer quota-aware failure messages

This change stabilized the pipeline and prevented unbounded artifact growth.

## 7. Automation and CI/CD

GitHub Actions now drive the project automation in a way that matches the project brief:

### Hourly Feature Pipeline

File: `.github/workflows/feature_pipeline.yml`

- runs on a scheduled cron
- refreshes the feature store every hour
- no longer retrains models hourly

### Daily Training Pipeline

File: `.github/workflows/training_pipeline.yml`

- runs once per day
- trains models from the latest feature store snapshot
- uploads refreshed artifacts to the model registry

### Continuous Integration

File: `.github/workflows/ci.yml`

- repository checks
- smoke validation of prediction artifacts

### Deployment Verification

File: `.github/workflows/deploy.yml`

- verifies the deployed Streamlit app after pushes to `main`
- optionally verifies a deployed Flask API health endpoint
- supports Streamlit Cloud health-endpoint redirects

## 8. Web Application and API

### Streamlit dashboard

The dashboard is implemented in `app/dashboard.py`.

Main capabilities:

- choose the best available model or a specific model
- adjust pollutant inputs with sliders
- view current observed AQI
- view today's predicted AQI as a confirmation signal
- view the next 3 forecast days
- inspect observed-versus-forecast charts
- download the forecast report as CSV
- see hazardous or unhealthy alert banners when forecast risk crosses alert thresholds

### Flask API

The API is implemented in `app/api.py`.

Endpoints:

- `/health`
- `/predict`

The prediction response includes:

- selected model
- today's confirmation AQI
- full internal prediction array
- separated next-three-day forecast values
- alert information
- model leaderboard metadata

## 9. Explainability and EDA

### EDA

EDA outputs are stored in `reports/` and include:

- AQI distribution
- AQI time series
- correlation heatmap
- PM2.5 to AQI relationship
- markdown summary report

Key findings from `reports/eda_summary.md`:

- latest daily average AQI: 66.9
- dataset average daily AQI: 112.4
- highest observed daily average AQI: 402.6
- most common AQI band: Moderate
- PM2.5 shows the strongest link to AQI movement with correlation 0.91

### Explainability

Explainability is implemented in `src/models/explain_model.py` using:

- SHAP summary bar plot
- SHAP impact plot
- LIME explanation for a single XGBoost day-1 prediction

Generated outputs are stored in `reports/explainability/`.

## 10. Deployment

The dashboard is designed for Streamlit Cloud deployment using `app.py` as the entrypoint. MongoDB Atlas provides the hosted feature store and model registry.

Deployment support includes:

- Streamlit app secret configuration
- MongoDB-backed runtime loading
- deployment verification workflow
- optional API health verification

The deployment documentation is stored in `DEPLOYMENT.md`.

## 11. Challenges and Resolutions

### MongoDB Atlas free-tier quota

Problem:

- repeated model uploads filled the 512 MB free-tier quota
- writes were blocked, causing the hourly pipeline to fail

Resolution:

- cleaned old registry artifacts
- added registry pruning and retention controls
- separated hourly feature refresh from daily training

### Dashboard and deployment synchronization

Problem:

- newly added models and UI changes did not immediately appear in the live app

Resolution:

- verified MongoDB model-registry contents
- rebooted Streamlit deployment
- added deployment verification workflow

### Forecast wording mismatch

Problem:

- the product originally rendered as a 4-day forecast even though the project brief required 3-day forecasting

Resolution:

- today's prediction was preserved as a confirmation signal
- the visible forecast section was realigned to the next 3 days only

## 12. Limitations

- The repository's checked-in metrics snapshot does not yet show a refreshed PyTorch benchmark row.
- Historical backfill is currently implemented as a fixed rolling two-year rebuild rather than a user-selectable arbitrary date-range backfill script.
- MongoDB Atlas is used instead of Hopsworks or Vertex AI for the feature store/model registry layer.
- The deep learning implementation uses PyTorch rather than TensorFlow.

## 13. Final Outcome

The final project is a working AQI forecasting platform with:

- automated data refresh
- automated retraining
- multiple forecasting models
- deep-learning support
- cloud-backed artifact management
- dashboard and API prediction serving
- explainability outputs
- deployment verification
- operational fixes for quota and workflow stability

The system is suitable as a complete end-to-end ML project submission and demonstrates practical ML engineering beyond model training alone, including feature operations, registry management, deployment checks, and user-facing monitoring.

## 14. Future Work

- add dedicated arbitrary date-range backfill command
- refresh and persist a new metrics CSV that includes the PyTorch model row
- optionally add richer alert severity controls and notification channels
- optionally deploy the Flask API as a public hosted service alongside the dashboard
