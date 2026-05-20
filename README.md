# Air Quality Index Predictor

This project predicts the **next 3 days of AQI** from historical pollution data, engineered time features, and multiple ML models. It uses MongoDB as both the feature store and model registry, plus shared config, an API entrypoint, Streamlit theming, wrapper scripts, and deployment docs.

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

## Notes

- The dashboard and API read features and model artifacts from MongoDB at runtime.
- `verify_prediction.py` is included as a lightweight smoke test for CI.
- Existing tracked data and model files are preserved.
