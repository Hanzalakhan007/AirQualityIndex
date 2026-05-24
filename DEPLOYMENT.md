# Deployment

## Streamlit Cloud

1. Push the repository to GitHub.
2. In Streamlit Cloud, create a new app and point it to `app.py`.
3. Add secrets for:
   - `OPENWEATHER_API_KEY`
   - `MONGO_URI`, `MONGO_DB_NAME`, `MONGO_FEATURES_COLLECTION`, `MONGO_MODEL_REGISTRY_COLLECTION`, `MONGO_MODEL_BUCKET`
   - `AQI_CITY`, `AQI_LAT`, `AQI_LON`, `AQI_TIMEZONE`
4. Deploy and verify the dashboard loads predictions.

Streamlit Cloud automatically redeploys the dashboard when `main` changes. The GitHub Actions deployment workflow now verifies that the production dashboard URL responds after each push.

## API deployment

The API lives in `app/api.py` and can be deployed separately with any WSGI-compatible host.

Useful routes:

- `/health`
- `/predict?model=Ridge%20Regression`
- `/predict?model=XGBoost&pm2_5=75&pm10=120`

## GitHub Actions

- `feature_pipeline.yml`: refreshes feature data every hour
- `training_pipeline.yml`: retrains the models daily from the latest feature store snapshot
- `ci.yml`: lint + prediction smoke test
- `deploy.yml`: verifies the deployed Streamlit dashboard and optional API health endpoint after pushes to `main`

## Deployment verification secrets

Set these GitHub Actions secrets to activate the deployment verification workflow:

- `STREAMLIT_APP_URL`: public URL of your Streamlit Cloud app
- `API_HEALTHCHECK_URL`: optional `/health` URL for a deployed Flask API
