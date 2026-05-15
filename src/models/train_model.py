import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

load_dotenv()

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("Starting Machine Learning Pipeline...")

FEATURE_COLUMNS = [
    "us_aqi",
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "pm2_5",
    "pm10",
    "nh3",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "aqi_lag_1",
    "aqi_lag_24",
    "aqi_rolling_24",
    "pm25_rolling_24",
    "pm10_rolling_24",
    "co_rolling_24",
    "aqi_change_rate",
]


def load_training_data() -> pd.DataFrame:
    """Load processed features from Hopsworks with a local CSV fallback."""
    print("Connecting to Hopsworks Feature Store...")
    try:
        import hopsworks

        project = hopsworks.login()
        feature_store = project.get_feature_store()
        print(f"Fetching features from Feature Group '{FEATURE_GROUP_NAME}'...")
        feature_group = feature_store.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        dataframe = feature_group.read()
        dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)
        return dataframe
    except Exception as exc:
        print(f"Could not read from Hopsworks Feature Store: {exc}")
        print("Falling back to local CSV...")
        return pd.read_csv("data/processed/features.csv")


def get_target_series(dataframe: pd.DataFrame) -> pd.Series:
    """Prefer US AQI and fall back to a rough OpenWeather conversion when needed."""
    if "us_aqi" in dataframe.columns:
        series = pd.to_numeric(dataframe["us_aqi"], errors="coerce")
        if not series.dropna().empty:
            return series
    base = pd.to_numeric(dataframe["aqi"], errors="coerce")
    return base * 50.0


dataframe = load_training_data()
dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

aqi_target = get_target_series(dataframe)
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24)
dataframe["target_day_1"] = aqi_target.shift(-1).rolling(window=indexer).mean()
dataframe["target_day_2"] = aqi_target.shift(-25).rolling(window=indexer).mean()
dataframe["target_day_3"] = aqi_target.shift(-49).rolling(window=indexer).mean()
dataframe = dataframe.dropna().reset_index(drop=True)

available_features = [column for column in FEATURE_COLUMNS if column in dataframe.columns]
X = dataframe[available_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
y = dataframe[["target_day_1", "target_day_2", "target_day_3"]]

train_size = int(len(dataframe) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X_train.shape[1], "models/nn_input_dim.pkl")

print("\n[1/4] Training Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_preds = ridge.predict(X_test_scaled)
joblib.dump(ridge, "models/ridge_model.pkl")

print("[2/4] Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)
joblib.dump(rf, "models/rf_model.pkl")

print("[3/4] Training XGBoost (Multi-Output)...")
xgb_base = xgb.XGBRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)
xgb_model = MultiOutputRegressor(xgb_base)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)
joblib.dump(xgb_model, "models/xgb_model.pkl")

print("[4/4] Training Custom PyTorch Neural Network...")
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)


class AQIPredictorNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, inputs):
        outputs = self.relu(self.fc1(inputs))
        outputs = self.relu(self.fc2(outputs))
        return self.fc3(outputs)


nn_model = AQIPredictorNN(X_train_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.01)

for _ in range(100):
    optimizer.zero_grad()
    outputs = nn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

nn_model.eval()
with torch.no_grad():
    nn_preds = nn_model(X_test_tensor).numpy()

torch.save(nn_model.state_dict(), "models/pytorch_model.pth")

print("\n" + "=" * 40)
print("MODEL EVALUATION RESULTS (Test Set Average across 3 Days)")
print("=" * 40)


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name}:\n  Avg RMSE: {rmse:.4f} | Avg MAE: {mae:.4f} | Avg R2: {r2:.4f}\n")
    return {"rmse": rmse, "mae": mae, "r2": r2}


results = {
    "Ridge Regression": evaluate("Ridge Regression", y_test, ridge_preds),
    "Random Forest": evaluate("Random Forest", y_test, rf_preds),
    "XGBoost": evaluate("XGBoost", y_test, xgb_preds),
    "PyTorch Deep Learning": evaluate("PyTorch NN", y_test, nn_preds),
}

best_model_name = min(results, key=lambda key: results[key]["rmse"])
print(f"BEST MODEL: {best_model_name}")

pd.DataFrame(
    [
        {
            "Model": model_name,
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "R2_Score": metrics["r2"],
            "Is_Best_Model": model_name == best_model_name,
        }
        for model_name, metrics in results.items()
    ]
).to_csv("reports/model_metrics.csv", index=False)

print("\n" + "=" * 40)
print("UPLOADING TO HOPSWORKS MODEL REGISTRY")
print("=" * 40)


def build_registry_metrics(model_name: str) -> dict[str, float | str | int]:
    metrics = dict(results[model_name])
    metrics["feature_count"] = len(available_features)
    return metrics


try:
    import hopsworks

    project = hopsworks.login()
    model_registry = project.get_model_registry()

    xgb_hw_model = model_registry.python.create_model(
        name="aqi_xgboost_model",
        metrics=build_registry_metrics("XGBoost"),
        description="XGBoost model predicting next 3 days AQI",
    )
    xgb_hw_model.save("models/xgb_model.pkl")

    pytorch_hw_model = model_registry.python.create_model(
        name="aqi_pytorch_model",
        metrics=build_registry_metrics("PyTorch Deep Learning"),
        description="PyTorch neural network predicting next 3 days AQI",
    )
    pytorch_hw_model.save("models/pytorch_model.pth")

    rf_hw_model = model_registry.python.create_model(
        name="aqi_rf_model",
        metrics=build_registry_metrics("Random Forest"),
        description="Random Forest model predicting next 3 days AQI",
    )
    rf_hw_model.save("models/rf_model.pkl")

    ridge_hw_model = model_registry.python.create_model(
        name="aqi_ridge_model",
        metrics=build_registry_metrics("Ridge Regression"),
        description="Ridge Regression model predicting next 3 days AQI",
    )
    ridge_hw_model.save("models/ridge_model.pkl")

    scaler_hw_model = model_registry.python.create_model(
        name="aqi_scaler",
        description="StandardScaler for AQI features",
    )
    scaler_hw_model.save("models/scaler.pkl")

    print("Successfully uploaded all models and scaler to Hopsworks Registry!")
except Exception as exc:
    print(f"Failed to upload models to Hopsworks: {exc}")
