# Setup

## 1. Environment

```bash
python -m venv venv
```

Windows:

```bash
.\venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure environment variables

Copy `.env.example` to `.env` and set:

- `OPENWEATHER_API_KEY`
- `MONGO_URI`, `MONGO_DB_NAME`, and the MongoDB collection/bucket names for the feature store and model registry
- `AQI_CITY`, `AQI_LAT`, `AQI_LON`, `AQI_TIMEZONE` if you want another location

## 4. Run the pipeline

```bash
python scripts/master_pipeline.py
```

Or step by step:

```bash
python scripts/fetch_raw_data.py
python scripts/feature_pipeline.py
python scripts/training_pipeline.py
python scripts/eda.py
python src/models/explain_model.py
```

## 5. Run the app

```bash
streamlit run app.py
```

## 6. Run the API

```bash
python -m flask --app app.api run
```
