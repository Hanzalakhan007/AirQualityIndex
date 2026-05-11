import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create reports directory if not exists
os.makedirs('reports', exist_ok=True)

print("Starting Exploratory Data Analysis (EDA)...")

# 1. Load Data
df = pd.read_csv('data/raw/historical_aqi.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Data Loaded! Shape: {df.shape}")

# 2. Missing Values Analysis
missing = df.isnull().sum()
print("\nMissing Values per Feature:")
print(missing)

# 3. AQI Distribution (Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(df['aqi'], bins=5, kde=False, discrete=True, color='skyblue')
plt.title('Distribution of AQI Levels (1: Good -> 5: Very Poor)')
plt.xlabel('AQI Index')
plt.ylabel('Frequency (Hours)')
plt.grid(axis='y', alpha=0.3)
plt.savefig('reports/aqi_distribution.png')
plt.close()
print("Saved AQI Distribution Plot.")

# 4. Correlation Analysis (Heatmap)
plt.figure(figsize=(12, 8))
# Only correlate numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Air Pollutants')
plt.savefig('reports/correlation_heatmap.png')
plt.close()
print("Saved Correlation Heatmap Plot.")

# 5. Outlier Detection (Boxplots)
plt.figure(figsize=(12, 6))
pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
sns.boxplot(data=df[pollutants], orient='h', palette='Set2')
plt.title('Outliers in Pollutant Concentrations (Log Scale)')
plt.xscale('log') # Log scale because CO is usually much higher in magnitude
plt.xlabel('Concentration (μg/m3)')
plt.tight_layout()
plt.savefig('reports/pollutants_boxplot_logscale.png')
plt.close()
print("Saved Pollutants Boxplot.")

# 6. Trend Analysis (Time Series)
# Resample to daily average of AQI to see clear trends over time
df.set_index('timestamp', inplace=True)
# Select only numeric columns for mean calculation
numeric_df = df.select_dtypes(include=['number'])
daily_aqi = numeric_df['aqi'].resample('D').mean()

plt.figure(figsize=(15, 5))
plt.plot(daily_aqi.index, daily_aqi.values, label='Daily Avg AQI', color='orange')
plt.title('Daily Average AQI Trend Over Time')
plt.xlabel('Date')
plt.ylabel('AQI Level (Average)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('reports/aqi_time_series.png')
plt.close()
print("Saved Time Series Plot.")

print("\nEDA Completed! Check the 'reports/' folder for the generated plots.")
