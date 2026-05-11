import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

# SETUP INITIAL CONFIG (Must be the first command)
st.set_page_config(page_title="AQI Prediction System", page_icon="🌍", layout="wide")

# CSS Styling to make the dashboard look modern and premium
st.markdown("""
<style>
    .big-font {
        font-size:40px !important;
        font-weight: 800;
        color: #1F2937;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E5E7EB;
    }
    h1, h2, h3 {
        color: #111827;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. LOAD MODELS & SCALERS (Cached for performance)
# ----------------------------------------------------
@st.cache_resource
def load_models():
    # Define PyTorch NN Architecture inside or load it
    class AQIPredictorNN(nn.Module):
        def __init__(self, input_dim):
            super(AQIPredictorNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 3)
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            
    try:
        import hopsworks
        project = hopsworks.login()
        mr = project.get_model_registry()
        
        # 1. XGBoost
        try:
            print("Downloading XGBoost model...")
            xgb_hw = mr.get_model(name="aqi_xgboost_model", version=2)
            xgb_dir = xgb_hw.download()
            xgb_model = joblib.load(os.path.join(xgb_dir, 'xgb_model.pkl'))
        except Exception as e:
            print(f"XGBoost load failed: {e}")
            xgb_model = None
        
        # 2. Random Forest
        try:
            print("Downloading Random Forest model...")
            rf_hw = mr.get_model(name="aqi_rf_model", version=1)
            rf_dir = rf_hw.download()
            rf_model = joblib.load(os.path.join(rf_dir, 'rf_model.pkl'))
        except Exception as e:
            print(f"RF load failed: {e}")
            rf_model = None

        # 3. Ridge
        try:
            print("Downloading Ridge model...")
            ridge_hw = mr.get_model(name="aqi_ridge_model", version=1)
            ridge_dir = ridge_hw.download()
            ridge_model = joblib.load(os.path.join(ridge_dir, 'ridge_model.pkl'))
        except Exception as e:
            print(f"Ridge load failed: {e}")
            ridge_model = None

        # 4. PyTorch
        try:
            print("Downloading PyTorch model...")
            pytorch_hw = mr.get_model(name="aqi_pytorch_model", version=1)
            pytorch_dir = pytorch_hw.download()
            pytorch_model = AQIPredictorNN(16)
            pytorch_model.load_state_dict(torch.load(os.path.join(pytorch_dir, 'pytorch_model.pth')))
            pytorch_model.eval()
        except Exception as e:
            print(f"PyTorch load failed: {e}")
            pytorch_model = None

        # 5. Scaler
        try:
            print("Downloading Scaler...")
            scaler_hw = mr.get_model(name="aqi_scaler", version=1)
            scaler_dir = scaler_hw.download()
            scaler = joblib.load(os.path.join(scaler_dir, 'scaler.pkl'))
        except Exception as e:
            print(f"Scaler load failed: {e}")
            scaler = None
        
    except Exception as e:
        print(f"Hopsworks connection error: {e}")
        # Local fallback
        xgb_model, rf_model, ridge_model, pytorch_model, scaler = None, None, None, None, None

    # Final check: If everything is None, try local fallback
    if xgb_model is None:
        try:
            xgb_model = joblib.load('models/xgb_model.pkl')
            rf_model = joblib.load('models/rf_model.pkl')
            ridge_model = joblib.load('models/ridge_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            pytorch_model = AQIPredictorNN(16)
            pytorch_model.load_state_dict(torch.load('models/pytorch_model.pth'))
            pytorch_model.eval()
        except:
            pass
            
    return xgb_model, rf_model, ridge_model, pytorch_model, scaler

xgb_model, rf_model, ridge_model, pytorch_model, scaler = load_models()

# If models are still missing (first time setup), show a message
if scaler is None or xgb_model is None:
    st.warning("🔄 System is still initializing! Models are currently being uploaded from the GitHub Pipeline to the Cloud. Please wait 2-3 minutes and refresh.")
    st.info("💡 You can check the progress in your GitHub Actions 'Daily Training Pipeline'. Once it's green, the dashboard will go live.")
    st.stop()

# ----------------------------------------------------
# 2. LOAD HISTORICAL DATA
# ----------------------------------------------------
@st.cache_data
def load_data():
    try:
        import hopsworks
        project = hopsworks.login()
        fs = project.get_feature_store()
        
        print("Fetching data from Hopsworks Feature Store...")
        fg = fs.get_feature_group('aqi_features', version=1)
        df = fg.read()
        df = df.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"Hopsworks Feature Store error: {e}. Falling back to local data.")
        df = pd.read_csv('data/processed/features.csv')
    return df

df = load_data()
latest = df.iloc[-1] # The absolute latest available data point

# ----------------------------------------------------
# 3. SIDEBAR (User Inputs)
# ----------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3203/3203875.png", width=60)
st.sidebar.header("🌍 Live Pollutant Values")
st.sidebar.write("Simulate current readings to predict the **Next 3 Days' AQI**.")

co = st.sidebar.slider("Carbon Monoxide (CO)", 0.0, 5000.0, float(latest['co']))
no2 = st.sidebar.slider("Nitrogen Dioxide (NO2)", 0.0, 200.0, float(latest['no2']))
o3 = st.sidebar.slider("Ozone (O3)", 0.0, 300.0, float(latest['o3']))
pm2_5 = st.sidebar.slider("PM2.5", 0.0, 500.0, float(latest['pm2_5']))
pm10 = st.sidebar.slider("PM10", 0.0, 500.0, float(latest['pm10']))
nh3 = st.sidebar.slider("Ammonia (NH3)", 0.0, 100.0, float(latest['nh3']))

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("Choose AI Model", [
    "XGBoost (Recommended)", 
    "Random Forest", 
    "Ridge Regression", 
    "PyTorch Deep Learning"
])

# ----------------------------------------------------
# 4. MAIN DASHBOARD UI
# ----------------------------------------------------
st.title("🌱 End-to-End Air Quality Index Forecaster")
st.write("This dashboard leverages trained Machine Learning models to forecast the Air Quality Index for the **next 3 days**.")

# Combine user inputs with actual background features to create an ML-ready row
input_features = pd.DataFrame({
    'aqi': [latest['aqi']], # current aqi
    'co': [co],
    'no': [latest['no']], # we keep un-adjustable features same as latest
    'no2': [no2],
    'o3': [o3],
    'so2': [latest['so2']],
    'pm2_5': [pm2_5],
    'pm10': [pm10],
    'nh3': [nh3],
    'hour': [12], # fixed to noon for simulation
    'day_of_week': [latest['day_of_week']],
    'month': [latest['month']],
    'is_weekend': [latest['is_weekend']],
    'aqi_lag_1': [latest['aqi_lag_1']],
    'aqi_lag_24': [latest['aqi_lag_24']],
    'aqi_rolling_24': [latest['aqi_rolling_24']]
})

# Reorder columns to exactly match how the model was trained
features_order = ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 
                  'hour', 'day_of_week', 'month', 'is_weekend', 'aqi_lag_1', 'aqi_lag_24', 'aqi_rolling_24']
input_features = input_features[features_order]

# Scale features exactly how they were scaled in training
input_scaled = scaler.transform(input_features)

# Make Prediction
if model_choice == "XGBoost (Recommended)":
    if xgb_model is None:
        st.error("XGBoost model is still loading. Please wait or try another model.")
        st.stop()
    pred_aqi = xgb_model.predict(input_scaled)[0]
elif model_choice == "Random Forest":
    if rf_model is None:
        st.error("Random Forest model is still loading. Please wait or try another model.")
        st.stop()
    pred_aqi = rf_model.predict(input_scaled)[0]
elif model_choice == "Ridge Regression":
    if ridge_model is None:
        st.error("Ridge model is still loading. Please wait or try another model.")
        st.stop()
    pred_aqi = ridge_model.predict(input_scaled)[0]
else: # PyTorch
    if pytorch_model is None:
        st.error("PyTorch model is still loading or not available. Please wait or try another model.")
        st.stop()
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        # Ensure it's on CPU and detached before converting to numpy
        pred_aqi = pytorch_model(input_tensor).cpu().detach().numpy()[0]

# Prevent negative predictions just in case
pred_day1 = max(1.0, pred_aqi[0])
pred_day2 = max(1.0, pred_aqi[1])
pred_day3 = max(1.0, pred_aqi[2])

def get_status_color(pred):
    if pred < 2.5: return "Good 🟢", "#10B981"
    elif pred < 3.5: return "Moderate 🟡", "#F59E0B"
    elif pred < 4.5: return "Poor 🟠", "#F97316"
    else: return "Severe 🔴", "#EF4444"

from datetime import datetime, timedelta
date1 = (datetime.now() + timedelta(days=1)).strftime("%b %d")
date2 = (datetime.now() + timedelta(days=2)).strftime("%b %d")
date3 = (datetime.now() + timedelta(days=3)).strftime("%b %d")

# Display Current AQI First
st.markdown('<div class="metric-card" style="width: 300px; margin: 0 auto; margin-bottom: 20px;"><p>📊 Current AQI</p><p class="big-font">{}</p></div>'.format(int(latest['aqi'])), unsafe_allow_html=True)

# Metrics Display
st.subheader("🔮 3-Day Forecast")
col1, col2, col3 = st.columns(3)

status1, color1 = get_status_color(pred_day1)
with col1:
    st.markdown(f'<div class="metric-card" style="border: 2px solid #8B5CF6;"><p>📅 {date1}</p><p class="big-font">{pred_day1:.2f}</p><p style="color:{color1}; font-weight:bold;">{status1}</p></div>', unsafe_allow_html=True)

status2, color2 = get_status_color(pred_day2)
with col2:
    st.markdown(f'<div class="metric-card"><p>📅 {date2}</p><p class="big-font">{pred_day2:.2f}</p><p style="color:{color2}; font-weight:bold;">{status2}</p></div>', unsafe_allow_html=True)

status3, color3 = get_status_color(pred_day3)
with col3:
    st.markdown(f'<div class="metric-card"><p>📅 {date3}</p><p class="big-font">{pred_day3:.2f}</p><p style="color:{color3}; font-weight:bold;">{status3}</p></div>', unsafe_allow_html=True)

st.markdown("<br><hr><br>", unsafe_allow_html=True)

# ----------------------------------------------------
# 5. CHARTS AND EXPLAINABILITY
# ----------------------------------------------------
st.subheader("📈 Historical 14-Day Trend")
st.write("Visualizing the recent spikes and flow of pollution levels (Daily Average).")
# Convert to datetime and resample to daily mean for a cleaner chart
df['timestamp'] = pd.to_datetime(df['timestamp'])
chart_data = df.set_index('timestamp')['aqi'].resample('D').mean().tail(14)
st.line_chart(chart_data, use_container_width=True)

st.markdown("<hr><br>", unsafe_allow_html=True)

st.subheader("🧠 Model Explainability (SHAP)")
st.write("We don't do Black-Box AI! Here is exactly how the XGBoost model makes its decisions:")

col_img1, col_img2 = st.columns(2)
with col_img1:
    st.markdown("**1. Global Feature Importance**")
    st.write("Which pollutants matter the most overall?")
    if os.path.exists("reports/explainability/shap_summary_bar.png"):
        st.image("reports/explainability/shap_summary_bar.png", use_container_width=True)
    else:
        st.warning("Generate SHAP plots using `explain_model.py` first.")

with col_img2:
    st.markdown("**2. Feature Impact Direction**")
    st.write("Does increasing PM2.5 make AQI better or worse?")
    if os.path.exists("reports/explainability/shap_impact.png"):
        st.image("reports/explainability/shap_impact.png", use_container_width=True)
