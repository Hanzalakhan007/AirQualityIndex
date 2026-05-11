import pandas as pd
import numpy as np
import os
import joblib
import hopsworks
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ML Models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# PyTorch (Deep Learning)
import torch
import torch.nn as nn
import torch.optim as optim

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("Starting Machine Learning Pipeline...")

# 1. Load Processed Data
print("Connecting to Hopsworks Feature Store...")
try:
    project = hopsworks.login()
    fs = project.get_feature_store()
    print("Fetching features from Feature Group 'aqi_features'...")
    fg = fs.get_feature_group('aqi_features', version=1)
    df = fg.read()
    # Sort chronologically as Hopsworks doesn't guarantee order
    df = df.sort_values('timestamp').reset_index(drop=True)
except Exception as e:
    print(f"Could not read from Hopsworks Feature Store: {e}")
    print("Falling back to local CSV...")
    df = pd.read_csv('data/processed/features.csv')

# 2. Forecasting Target (What are we predicting?)
# We want to predict the *average* AQI for Day 1, Day 2, and Day 3
# Using forward rolling windows of 24 hours.
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24)
df['target_day_1'] = df['aqi'].shift(-1).rolling(window=indexer).mean()
df['target_day_2'] = df['aqi'].shift(-25).rolling(window=indexer).mean()
df['target_day_3'] = df['aqi'].shift(-49).rolling(window=indexer).mean()

# Drop rows that don't have enough future data to calculate the 3-day targets
df = df.dropna()

# 3. Select Features
features = ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 
            'hour', 'day_of_week', 'month', 'is_weekend', 'aqi_lag_1', 'aqi_lag_24', 'aqi_rolling_24']

X = df[features]
y = df[['target_day_1', 'target_day_2', 'target_day_3']]

# 4. Train-Test Split (Chronological)
train_size = int(len(df) * 0.8) # 80% past data
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl') # Save scaler for dashboard

# ==========================================
# MODEL 1: RIDGE REGRESSION (Linear Math)
# ==========================================
print("\n[1/4] Training Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_preds = ridge.predict(X_test_scaled)
joblib.dump(ridge, 'models/ridge_model.pkl')

# ==========================================
# MODEL 2: RANDOM FOREST (Ensemble of Trees)
# ==========================================
print("[2/4] Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)
joblib.dump(rf, 'models/rf_model.pkl')

# ==========================================
# MODEL 3: XGBOOST (Gradient Boosting)
# ==========================================
print("[3/4] Training XGBoost (Multi-Output)...")
from sklearn.multioutput import MultiOutputRegressor
# XGBoost requires wrapping for scikit-learn API multi-output in older versions, 
# and it's safer for compatibility to use MultiOutputRegressor.
xgb_base = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model = MultiOutputRegressor(xgb_base)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)
joblib.dump(xgb_model, 'models/xgb_model.pkl')

# ==========================================
# MODEL 4: PYTORCH NEURAL NETWORK (Deep Learning)
# ==========================================
print("[4/4] Training Custom PyTorch Neural Network...")
# Neural networks need data formatted as "Tensors"
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32) # Shape: (N, 3)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Define a Feed-Forward Neural Network Architecture
class AQIPredictorNN(nn.Module):
    def __init__(self, input_dim):
        super(AQIPredictorNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        # Output 3 continuous predictions (Day 1, Day 2, Day 3)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

nn_model = AQIPredictorNN(X_train_tensor.shape[1])
criterion = nn.MSELoss() 
optimizer = optim.Adam(nn_model.parameters(), lr=0.01)

# Training Loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad() 
    outputs = nn_model(X_train_tensor) 
    loss = criterion(outputs, y_train_tensor) 
    loss.backward() 
    optimizer.step() 

# Get Predictions
nn_model.eval()
with torch.no_grad():
    nn_preds = nn_model(X_test_tensor).numpy()
    
torch.save(nn_model.state_dict(), 'models/pytorch_model.pth')

# ==========================================
# PHASE 6: MODEL EVALUATION
# ==========================================
print("\n" + "="*40)
print("🏆 MODEL EVALUATION RESULTS (Test Set Average across 3 Days) 🏆")
print("="*40)

def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # Average across 3 targets
    mae = mean_absolute_error(y_true, y_pred) 
    r2 = r2_score(y_true, y_pred) 
    print(f"{name}:\n  Avg RMSE: {rmse:.4f} | Avg MAE: {mae:.4f} | Avg R²: {r2:.4f}\n")
    return r2

results = {}
results['Ridge'] = evaluate("Ridge Regression", y_test, ridge_preds)
results['Random Forest'] = evaluate("Random Forest", y_test, rf_preds)
results['XGBoost'] = evaluate("XGBoost", y_test, xgb_preds)
results['PyTorch NN'] = evaluate("PyTorch NN", y_test, nn_preds)

best_model_name = max(results, key=results.get)
print(f"🌟 BEST MODEL: {best_model_name} 🌟")

pd.DataFrame({
    'Model': list(results.keys()),
    'R2_Score': list(results.values())
}).to_csv('reports/model_metrics.csv', index=False)

# ==========================================
# PHASE 7: HOPSWORKS MODEL REGISTRY
# ==========================================
print("\n" + "="*40)
print("☁️ UPLOADING TO HOPSWORKS MODEL REGISTRY ☁️")
print("="*40)

try:
    project = hopsworks.login()
    mr = project.get_model_registry()
    
    print("Uploading XGBoost model...")
    xgb_hw_model = mr.python.create_model(
        name="aqi_xgboost_model",
        metrics={"r2_avg": results['XGBoost']},
        description="XGBoost model predicting Next 3 Days AQI"
    )
    xgb_hw_model.save('models/xgb_model.pkl')
    
    print("Uploading PyTorch model...")
    pytorch_hw_model = mr.python.create_model(
        name="aqi_pytorch_model",
        metrics={"r2_avg": results['PyTorch NN']},
        description="PyTorch NN predicting Next 3 Days AQI"
    )
    pytorch_hw_model.save('models/pytorch_model.pth')

    print("Uploading Random Forest model...")
    rf_hw_model = mr.python.create_model(
        name="aqi_rf_model",
        metrics={"r2_avg": results['Random Forest']},
        description="Random Forest model predicting Next 3 Days AQI"
    )
    rf_hw_model.save('models/rf_model.pkl')

    print("Uploading Ridge model...")
    ridge_hw_model = mr.python.create_model(
        name="aqi_ridge_model",
        metrics={"r2_avg": results['Ridge']},
        description="Ridge Regression model predicting Next 3 Days AQI"
    )
    ridge_hw_model.save('models/ridge_model.pkl')

    print("Uploading Scaler...")
    scaler_hw_model = mr.python.create_model(
        name="aqi_scaler",
        description="StandardScaler for AQI features"
    )
    scaler_hw_model.save('models/scaler.pkl')
    
    print("Successfully uploaded all models and scaler to Hopsworks Registry!")
except Exception as e:
    print(f"Failed to upload models to Hopsworks: {e}")


