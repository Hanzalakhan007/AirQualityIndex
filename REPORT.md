# Project Update Report

This version of `AirQualityIndex` was upgraded by comparing it against the reference `aqi-predictor-main` project and filling the main gaps without replacing the original Hopsworks workflow.

## Added or improved

- Centralized configuration in `config/settings.py`
- Modular dashboard entrypoint in `app/dashboard.py`
- Flask API in `app/api.py`
- Streamlit theme config in `.streamlit/config.toml`
- Wrapper scripts in `scripts/`
- CI smoke test in `verify_prediction.py`
- Richer setup and deployment documentation
- Additional GitHub workflow templates

## Preserved

- Existing `src/` pipeline structure
- Existing trained model files in `models/`
- Existing feature dataset and explainability outputs
- Existing scheduled training and feature workflows
