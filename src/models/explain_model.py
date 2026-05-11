import pandas as pd
import joblib
import shap
import lime
import lime.lime_tabular
import numpy as np
import os
import matplotlib.pyplot as plt

os.makedirs('reports/explainability', exist_ok=True)
print("Starting Model Explainability (SHAP & LIME)...")

# 1. Load the Best Model (XGBoost), Scaler, and Data
xgb_model = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')

df = pd.read_csv('data/processed/features.csv')
features = ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 
            'hour', 'day_of_week', 'month', 'is_weekend', 'aqi_lag_1', 'aqi_lag_24', 'aqi_rolling_24']

# We need a small sample for SHAP/LIME because analyzing all 16,000 rows mathematically takes a while.
# Let's take the most recent 500 hours of the dataset as our explanation test case.
X_sample = df[features].tail(500)
X_sample_scaled = scaler.transform(X_sample)

# ==========================================
# 1. SHAP (SHapley Additive exPlanations)
# ==========================================
print("Generating SHAP Explanations (Focusing on Day 1 Forecast)...")
# Since we use a MultiOutputRegressor, we extract the first underlying estimator 
# to explain the Day 1 predictions.
xgb_day1_model = xgb_model.estimators_[0]

explainer = shap.TreeExplainer(xgb_day1_model)
shap_values = explainer.shap_values(X_sample_scaled)

# 1a. SHAP Summary Plot (Bar Chart format)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Day 1 Forecast)")
plt.savefig('reports/explainability/shap_summary_bar.png', bbox_inches='tight')
plt.close()

# 1b. SHAP Impact Plot (shows positive vs negative influence per feature)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Feature Impact (Day 1 Forecast)")
plt.savefig('reports/explainability/shap_impact.png', bbox_inches='tight')
plt.close()

# ==========================================
# 2. LIME (Local Interpretable Model-agnostic Explanations)
# ==========================================
print("Generating LIME Explanation for a single prediction (Day 1)...")

# Fit the LIME explainer on a random sample of the training data
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=scaler.transform(df[features].tail(2000).values), # Pass subset of data
    feature_names=features,
    mode='regression',
    random_state=42
)

instance_idx = -1
instance_scaled = X_sample_scaled[instance_idx]
instance_raw = X_sample.iloc[instance_idx]

# For LIME, the predict_fn needs to output a 1D array of predictions for the instances
def predict_day1(x):
    # Predict all 3 days, but return only Day 1 (the first column)
    return xgb_model.predict(x)[:, 0]

# Ask LIME to explain *this specific row* for Day 1
exp = explainer_lime.explain_instance(
    data_row=instance_scaled, 
    predict_fn=predict_day1, 
    num_features=5 
)

# Save LIME plot
fig = exp.as_pyplot_figure()
plt.title(f"LIME: Why did the model predict this Day 1 AQI?")
fig.savefig('reports/explainability/lime_single_prediction.png', bbox_inches='tight')
plt.close()

print("Explainability complete! Charts saved in 'reports/explainability' folder.")
