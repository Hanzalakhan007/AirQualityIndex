from __future__ import annotations

import os
import sys
from pathlib import Path

import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inference import load_feature_data, load_models
from src.schema import FEATURE_COLUMNS

os.makedirs("reports/explainability", exist_ok=True)
print("Starting Model Explainability (SHAP & LIME)...")


def align_feature_frame(dataframe: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Match the explainability frame to the scaler's fitted feature order."""
    aligned = dataframe.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            if column == "aqi" and "us_aqi" in aligned.columns:
                aligned[column] = aligned["us_aqi"]
            else:
                aligned[column] = 0.0
    return aligned.loc[:, expected_columns]


# 1. Load the XGBoost model, scaler, and processed features from MongoDB.
models, scaler, _ = load_models()
if scaler is None:
    raise RuntimeError("Scaler could not be loaded from the MongoDB model registry.")
if "XGBoost" not in models:
    raise RuntimeError("XGBoost model could not be loaded from the MongoDB model registry.")

xgb_model = models["XGBoost"]
df = load_feature_data()

expected_features = list(getattr(scaler, "feature_names_in_", FEATURE_COLUMNS))
X_sample = align_feature_frame(df.tail(500), expected_features)
X_sample_scaled = scaler.transform(X_sample)

# ==========================================
# 1. SHAP (SHapley Additive exPlanations)
# ==========================================
print("Generating SHAP Explanations (focusing on the Day 1 XGBoost forecast)...")

# MultiOutputRegressor stores one estimator per forecast day.
xgb_day1_model = xgb_model.estimators_[0]
explainer = shap.TreeExplainer(xgb_day1_model)
shap_values = explainer.shap_values(X_sample_scaled)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Day 1 XGBoost Forecast)")
plt.savefig("reports/explainability/shap_summary_bar.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Feature Impact (Day 1 XGBoost Forecast)")
plt.savefig("reports/explainability/shap_impact.png", bbox_inches="tight")
plt.close()

# ==========================================
# 2. LIME (Local Interpretable Model-agnostic Explanations)
# ==========================================
print("Generating LIME explanation for a single Day 1 forecast...")

training_sample = align_feature_frame(df.tail(2000), expected_features)
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=scaler.transform(training_sample),
    feature_names=expected_features,
    mode="regression",
    random_state=42,
)

instance_idx = -1
instance_scaled = X_sample_scaled[instance_idx]


def predict_day1(values):
    """Return only the Day 1 prediction for LIME explanations."""
    return xgb_model.predict(values)[:, 0]


exp = explainer_lime.explain_instance(
    data_row=instance_scaled,
    predict_fn=predict_day1,
    num_features=5,
)

fig = exp.as_pyplot_figure()
plt.title("LIME: Why did XGBoost predict this Day 1 AQI?")
fig.savefig("reports/explainability/lime_single_prediction.png", bbox_inches="tight")
plt.close()

print("Explainability complete! Charts saved in 'reports/explainability' folder.")
