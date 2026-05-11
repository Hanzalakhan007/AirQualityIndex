import pandas as pd
import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

print("Starting Feature Engineering...")

def build_features(input_path, output_path):
    # 1. Load Data
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort chronologically, just in case, so our lag features shift correctly
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Time-Based Features
    # Why? A model doesn't intuitively know it's winter or 3 PM from a single timestamp string.
    # By extracting these, we help the model learn "seasonality" and "daily routines".
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 3. Lag Features
    # Why? Pollution right now is highly dependent on pollution just an hour ago. Air doesn't clear instantly!
    # 'shift(1)' looks at the row before it (1 hour ago).
    df['aqi_lag_1'] = df['aqi'].shift(1)
    df['aqi_lag_24'] = df['aqi'].shift(24)
    
    # 4. Rolling Averages
    # Why? What was the average AQI over the last 24 hours?
    # This smooths out random single-hour spikes and gives the model a sense of the broader "trend" for that day.
    df['aqi_rolling_24'] = df['aqi'].rolling(window=24).mean()
    
    # 5. Drop Missing Values
    # Shifting 24 rows forward means the first 24 rows in our dataset now have "NaN" (Missing)
    # values because we don't know the data 24 hours before they started. We drop them.
    df = df.dropna()
    
    # Save the processed features locally
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # --- Hopsworks Integration ---
    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    print("Uploading features to Feature Group 'aqi_features'...")
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="AQI dataset with engineered features"
    )
    fg.insert(df)
    print("Successfully uploaded features to Hopsworks!")
    
    print("Feature Engineering Complete!")
    print(f"Final dataset shape: {df.shape}")
    print("New Features created:")
    print("- hour, day_of_week, month, is_weekend")
    print("- aqi_lag_1, aqi_lag_24")
    print("- aqi_rolling_24")
    print(f"Saved ML-ready Data to: {output_path}")

if __name__ == "__main__":
    input_file = 'data/raw/historical_aqi.csv'
    output_file = 'data/processed/features.csv'
    build_features(input_file, output_file)
