# Deployment

## Streamlit Cloud

1. Push the repository to GitHub.
2. In Streamlit Cloud, create a new app and point it to `app.py`.
3. Add secrets for:
   - `OPENWEATHER_API_KEY`
   - `HOPSWORKS_API_KEY` if you use Hopsworks remotely
   - `AQI_CITY`, `AQI_LAT`, `AQI_LON`, `AQI_TIMEZONE`
4. Deploy and verify the dashboard loads predictions.

## API deployment

The API lives in `app/api.py` and can be deployed separately with any WSGI-compatible host.

Useful routes:

- `/health`
- `/predict?model=Ridge%20Regression`
- `/predict?model=XGBoost&pm2_5=75&pm10=120`

## GitHub Actions

- `feature_pipeline.yml`: fetches new data and builds features
- `training_pipeline.yml`: retrains the models
- `ci.yml`: lint + prediction smoke test
- `deploy.yml`: deployment placeholder for your preferred hosting setup
