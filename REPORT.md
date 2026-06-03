# Project Report: Karachi AirWatch AQI Forecasting System

## 1. Executive Summary

**Karachi AirWatch** is an end-to-end machine learning and MLOps project for monitoring and forecasting air quality in Karachi, Pakistan. The system collects air pollution data, converts pollutant readings into a continuous AQI target, engineers time-series features, trains multiple forecasting models, stores production artifacts in MongoDB Atlas, and presents results through a deployed Streamlit dashboard.

The final system is designed as a **next 3 days AQI prediction platform**. The dashboard also displays **today's predicted AQI** as a confirmation signal so the model's current behavior can be compared with the latest observed AQI, but the official forecast scope remains the next three days.

Current production best model: **Ridge Regression**  
Current production RMSE: **29.63 AQI points**  
Deployment: **Streamlit Cloud dashboard with MongoDB Atlas feature store and model registry**

---

## 2. Technical Objective

The primary objective was to build a production-style AQI forecasting system rather than a simple notebook experiment.

The system was designed to:

- collect current and historical air pollution data for Karachi;
- convert OpenWeather pollutant readings into a continuous AQI target suitable for regression;
- build model-ready time-series features from historical observations;
- train and compare multiple machine learning models;
- select the best model using real evaluation metrics and fit-status checks;
- deploy a professional dashboard for current AQI, prediction, charts, alerts, and health guidance;
- automate feature refresh, daily training, CI checks, and deployment verification using GitHub Actions;
- store production features and model artifacts in MongoDB Atlas.

---

## 3. Technology Stack

| Area | Technology | Purpose |
|---|---|---|
| Dashboard | Streamlit | Interactive AQI dashboard and visual forecasting interface |
| API | Flask | Health and prediction API endpoints |
| Data Source | OpenWeather Air Pollution API | Historical and current pollutant data |
| Fallback Data Source | Open-Meteo Air Quality API | Backup air-quality source if OpenWeather chunks fail |
| Data Processing | pandas, NumPy | Data cleaning, feature engineering, aggregation |
| Machine Learning | scikit-learn, XGBoost, PyTorch | Model training and comparison |
| Model Serialization | joblib, PyTorch state dict | Saving local model artifacts |
| Database | MongoDB Atlas | Feature store and model registry metadata |
| Artifact Storage | MongoDB GridFS | Model and scaler storage |
| Automation | GitHub Actions | Hourly feature pipeline, daily training, CI, deployment checks |
| Explainability | SHAP, LIME | Model interpretation and feature impact analysis |

---

## 4. Data Source and AQI Calculation

### 4.1 OpenWeather Data Used

The project uses the **OpenWeather Air Pollution API**, specifically:

- current air pollution endpoint;
- historical air pollution endpoint.

The historical endpoint used in the project is:

```text
/data/2.5/air_pollution/history
```

This is different from OpenWeather's paid historical weather API. The project does not rely on paid historical weather data. It uses OpenWeather's air pollution history endpoint, which provides historical pollutant readings.

### 4.2 How Two Years of Data Are Fetched

The feature pipeline calculates a rolling two-year window:

```python
end_time = int(time.time())
start_time = end_time - (2 * 365 * 24 * 60 * 60)
```

Instead of requesting all two years in one large API call, the system fetches the data in **30-day chunks**:

```python
fetch_historical_aqi_chunks(start_time, end_time, chunk_days=30)
```

This avoids API response timeouts and makes the pipeline more reliable.

For one 30-day chunk:

```text
30 days * 24 hours = approximately 720 hourly records
```

For two years:

```text
around 730 days / 30 days = about 25 API chunks
```

This is why the hourly pipeline logs repeatedly show messages such as:

```text
Fetched 720 OpenWeather rows; running total: ...
```

### 4.3 OpenWeather AQI Scale vs Project AQI

OpenWeather provides `main.aqi` on a **1 to 5 scale**:

| OpenWeather AQI | Meaning |
|---:|---|
| 1 | Good |
| 2 | Fair |
| 3 | Moderate |
| 4 | Poor |
| 5 | Very Poor |

This scale is useful for classification, but it is too coarse for regression forecasting.

The project therefore derives a continuous US-style AQI value from **PM2.5 concentration** using AQI breakpoint interpolation:

```python
pm25_to_us_aqi(pm25_ugm3)
```

This is why the dashboard shows values such as `60.3`, `71.8`, or `95.0` instead of only `1`, `2`, `3`, `4`, or `5`.

Technical explanation:

> OpenWeather gives a 1-5 AQI category, but my model needs a continuous numerical AQI target. So I derive a US-style AQI from PM2.5 concentration using breakpoint interpolation, while still keeping OpenWeather pollutant values as input features.

---

## 5. Algorithms and Techniques

### 5.1 Algorithms Used

| Algorithm | Role | Why It Was Used |
|---|---|---|
| Ridge Regression | Production lead model | Stable, regularized linear model that achieved the best reliable RMSE and OK fit status |
| PyTorch MLP | Deep learning comparison model | Tests whether a neural network can learn non-linear pollutant patterns |
| Random Forest | Tree ensemble benchmark | Captures non-linear relationships but showed overfit risk in final evaluation |
| XGBoost | Boosted tree benchmark | Strong benchmark model and useful for explainability, but overfit in this dataset |
| Persistence Baseline | Sanity baseline | Checks whether ML models beat a simple "future similar to current" assumption |

### 5.2 Feature Engineering Techniques

The system uses time-series feature engineering to convert raw hourly pollution records into model-ready features.

Feature groups include:

- pollutant features: `co`, `no`, `no2`, `o3`, `so2`, `pm2_5`, `pm10`, `nh3`;
- time features: `hour`, `day_of_week`, `month`, `is_weekend`;
- AQI lag features: `aqi_lag_1`, `aqi_lag_24`, `aqi_lag_48`, `aqi_lag_72`;
- rolling features: `aqi_rolling_24`, `pm25_rolling_24`, `pm10_rolling_24`, `co_rolling_24`;
- trend feature: `aqi_change_rate`.

Why these are useful:

> AQI is a time-series problem. Recent AQI values, previous-day behavior, rolling averages, and pollutant trends help the model understand short-term and daily air-quality patterns.

---

## 6. System Architecture

The project is organized into five main layers.

### 6.1 Data Ingestion Layer

Main file:

```text
src/data_collection/fetch_data.py
```

Responsibilities:

- fetch historical OpenWeather air pollution data;
- fetch current OpenWeather air pollution data;
- use Open-Meteo as a fallback if needed;
- convert raw pollutant JSON into structured rows;
- derive continuous US-style AQI from PM2.5.

### 6.2 Feature Engineering Layer

Main file:

```text
src/features/build_features.py
```

Responsibilities:

- fetch approximately two years of historical data in chunks;
- append current OpenWeather observation;
- build time, lag, rolling, and trend features;
- create local fallback CSV files;
- upload engineered feature rows to MongoDB Atlas.

### 6.3 Model Training Layer

Main file:

```text
src/models/train_model.py
```

Responsibilities:

- load feature data from MongoDB or local fallback;
- build target columns for today plus next three days;
- train Ridge Regression, Random Forest, XGBoost, and PyTorch MLP;
- evaluate models using RMSE, MAE, R2, train RMSE, RMSE gap, and fit status;
- save local artifacts;
- upload accepted artifacts to MongoDB model registry.

### 6.4 Inference Layer

Main file:

```text
src/inference.py
```

Responsibilities:

- load latest production model and scaler;
- load latest feature row;
- generate today confirmation prediction and next 3-day forecast;
- build leaderboard data for the dashboard;
- support MongoDB and local fallback loading.

### 6.5 Serving Layer

Main files:

```text
app/dashboard.py
app/api.py
app.py
```

Responsibilities:

- Streamlit dashboard for user-facing visualization;
- Flask API for health and prediction endpoints;
- app entrypoint for Streamlit Cloud deployment.

---

## 7. MLOps Pipelines

### 7.1 Hourly Feature Pipeline

Workflow file:

```text
.github/workflows/feature_pipeline.yml
```

Script:

```text
scripts/run_hourly_pipeline.py
```

Purpose:

- refresh latest feature data;
- fetch current OpenWeather air pollution observation;
- rebuild current engineered features;
- upload latest features to MongoDB;
- keep the dashboard's current observed AQI fresh.

Important point:

> The hourly pipeline does not retrain the model. It updates the feature store only.

### 7.2 Daily Training Pipeline

Workflow file:

```text
.github/workflows/training_pipeline.yml
```

Script:

```text
scripts/training_pipeline.py
```

Purpose:

- retrain models once per day;
- evaluate all models;
- update metrics and model registry;
- promote the best safe model.

Important point:

> Daily retraining does not guarantee daily metric changes because one new day is very small compared with the two-year historical dataset.

### 7.3 Continuous Integration Pipeline

Workflow file:

```text
.github/workflows/ci.yml
```

Purpose:

- linting and code quality checks;
- import/smoke validation;
- helps prevent broken code from reaching production.

### 7.4 Deployment Verification Pipeline

Workflow file:

```text
.github/workflows/deploy.yml
```

Purpose:

- verifies the Streamlit dashboard is reachable after deployment;
- checks deployment configuration;
- optionally checks API health.

---

## 8. Model Training and Evaluation

### 8.1 Prediction Targets

The training pipeline creates four targets:

| Target | Meaning |
|---|---|
| `target_day_0` | Today's AQI confirmation prediction |
| `target_day_1` | Tomorrow's AQI forecast |
| `target_day_2` | Day 2 forecast |
| `target_day_3` | Day 3 forecast |

The dashboard displays `target_day_0` separately as **Predicted AQI Today**, while the main forecast section focuses on the next three days.

### 8.2 Evaluation Method

The project uses chronological train/test splitting instead of random splitting. This is important because AQI is time-series data.

Why chronological split:

> In real forecasting, the model predicts the future using the past. A chronological split better simulates real-world forecasting than random train/test splitting.

### 8.3 Metrics Used

| Metric | Meaning |
|---|---|
| RMSE | Average error magnitude with stronger penalty for large errors |
| MAE | Average absolute error |
| R2 Score | How much variance the model explains |
| Train RMSE | Error on training data |
| RMSE Gap | Difference between test and train behavior |
| Fit Status | Health label such as OK, overfit-risk, or underfit-risk |
| Selection Score | Final score used for production model selection |

---

## 9. Current Model Performance

The current production metrics are stored in:

```text
reports/model_metrics.csv
```

Current leaderboard:

| Model | RMSE | MAE | R2 Score | Train RMSE | Fit Status | Production Best |
|---|---:|---:|---:|---:|---|---|
| Ridge Regression | 29.6349 | 22.0005 | 0.3335 | 41.4758 | ok | Yes |
| PyTorch MLP | 31.0649 | 23.4490 | 0.2683 | 43.5648 | ok | No |
| Random Forest | 46.0114 | 33.2238 | -0.6077 | 20.0857 | overfit-risk | No |
| XGBoost | 54.7130 | 38.6964 | -1.2782 | 29.2853 | overfit-risk | No |

### 9.1 Why Ridge Regression Is Best

Ridge Regression is selected because it has:

- the best reliable RMSE;
- an `ok` fit status;
- better generalization than Random Forest and XGBoost;
- more stable behavior on the historical holdout set.

The more complex tree models performed worse because they showed signs of overfitting. This means they learned training patterns too strongly but did not generalize well enough to the test period.

### 9.2 Why Metrics May Not Change Every Day

The dataset contains around two years of hourly observations. One new day adds only around 24 new rows. Compared with thousands of rows, that is a very small change.

Therefore, daily retraining may produce the same or almost identical metrics.

Technical explanation:

> Metrics are not hardcoded. They are calculated by the training pipeline, but because the dataset is large and the train/test split remains almost the same, daily retraining may not visibly change the final leaderboard.

---

## 10. Dashboard and API

### 10.1 Streamlit Dashboard

Main file:

```text
app/dashboard.py
```

Dashboard features:

- current observed AQI from the feature store;
- today's predicted AQI as a confirmation signal;
- next 3-day AQI outlook;
- production model card;
- model leaderboard;
- observed vs forecast chart;
- pollutant trend charts;
- AQI health guidance;
- hazardous alert panel;
- SHAP/LIME explainability outputs;
- stored data reload control.

### 10.2 Flask API

Main file:

```text
app/api.py
```

Endpoints:

- `/health`
- `/predict`

The API allows prediction access outside the dashboard and supports deployment health checks.

---

## 11. Explainability and EDA

### 11.1 Exploratory Data Analysis

EDA outputs are stored in:

```text
reports/
```

Important outputs:

- AQI distribution plot;
- AQI time series;
- correlation heatmap;
- PM2.5 vs AQI relationship;
- EDA summary markdown.

### 11.2 SHAP and LIME

Explainability files are generated by:

```text
src/models/explain_model.py
```

Outputs are stored in:

```text
reports/explainability/
```

Why XGBoost appears in explainability:

> XGBoost is part of the benchmark model set and is convenient for SHAP/LIME-style feature impact analysis. The production model remains Ridge Regression.

---

## 12. Deployment

The project is deployed as a Streamlit Cloud application.

Deployment components:

- Streamlit Cloud for dashboard hosting;
- MongoDB Atlas for feature store and model registry;
- GitHub Actions for automation;
- GitHub repository as the deployment source;
- environment variables/secrets for API keys and database credentials.

The deployment is verified through GitHub Actions using:

```text
.github/workflows/deploy.yml
```

---

## 13. Challenges and Technical Resolutions

### 13.1 OpenWeather AQI Scale Confusion

Problem:

- OpenWeather provides AQI as a 1-5 category, but the dashboard shows continuous values like 60.3 or 71.8.

Resolution:

- implemented `pm25_to_us_aqi()` to derive continuous AQI from PM2.5 concentration using AQI breakpoints.

### 13.2 Fetching Two Years of Data

Problem:

- fetching two years in one request would be unreliable and could timeout.

Resolution:

- the pipeline fetches 30-day chunks and combines them into a rolling two-year dataset.

### 13.3 MongoDB Connectivity and IP Allowlist

Problem:

- MongoDB Atlas failed when the IP allowlist entry was temporary or when network selection timed out.

Resolution:

- restored permanent MongoDB network access;
- retained local fallback files;
- kept MongoDB timeouts and clear failure messages.

### 13.4 MongoDB Free-Tier Storage Pressure

Problem:

- repeated model artifact uploads could consume MongoDB free-tier storage.

Resolution:

- separated hourly feature refresh from daily training;
- added model registry retention and pruning support.

### 13.5 Forecast Wording Mismatch

Problem:

- early wording could make the system look like a 4-day forecast.

Resolution:

- current-day prediction is labeled as a confirmation signal;
- project forecast is consistently described as next 3 days.

### 13.6 Experimental Features Worsened Metrics

Problem:

- weather and stronger trend feature experiments did not improve production metrics.

Resolution:

- reverted temporary experiments;
- kept the stable final feature set;
- removed temporary experiment folders from the final project structure.

---

## 14. Final Results

The final system is complete, operational, and ready for demonstration.

Completed outcomes:

- real OpenWeather data ingestion;
- rolling two-year feature rebuild;
- current AQI feature-store update;
- continuous AQI derivation from PM2.5;
- time-series feature engineering;
- multiple ML models and deep learning model;
- Ridge Regression selected as best production model;
- MongoDB Atlas feature store and model registry;
- Streamlit Cloud dashboard;
- Flask API support;
- SHAP/LIME explainability;
- EDA outputs;
- hourly feature automation;
- daily training automation;
- deployment verification workflow;
- final report and Word deliverable generated.

---

## 15. Limitations

The project is strong, but it is not perfect. Important limitations:

- AQI is currently derived mainly from PM2.5 rather than full pollutant sub-index aggregation.
- OpenWeather's built-in AQI is only 1-5, so continuous AQI is calculated by the project.
- GitHub Actions schedules are best-effort and may not run at exact clock times.
- Weather feature experiments were not kept because they did not improve RMSE.
- Independent validation from more Karachi monitoring stations would improve real-world confidence.
- Arbitrary date-range backfill is intentionally deferred as future work.

---

## 16. Future Work

Possible improvements:

- calculate pollutant-wise AQI sub-indexes for PM2.5, PM10, O3, NO2, SO2, and CO;
- add more independent data sources for validation;
- add model drift monitoring;
- add confidence intervals for forecasts;
- improve Random Forest/XGBoost regularization through careful ablation;
- add email/SMS health alerts;
- add a dedicated arbitrary date-range backfill script;
- add a model-version comparison page in the dashboard.

---

## 17. Project Summary

Karachi AirWatch is a complete AQI forecasting system for Karachi. It uses OpenWeather air pollution data, derives continuous AQI from PM2.5, engineers time-series features, trains multiple machine learning models, selects Ridge Regression as the best stable model, stores production artifacts in MongoDB Atlas, and serves predictions through a deployed Streamlit dashboard and Flask API. GitHub Actions automates hourly feature refresh, daily model training, CI checks, and deployment verification.

The project demonstrates not only machine learning but also practical MLOps: feature storage, model registry, automation, deployment, monitoring, explainability, and operational debugging.

---

## 18. Project Metadata

**Project:** Karachi AirWatch AQI Forecasting System  
**Final production model:** Ridge Regression  
**Final production RMSE:** 29.63 AQI points
