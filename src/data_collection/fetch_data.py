import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LAT", 40.7128)
LON = os.getenv("LON", -74.0060)

def fetch_historical_aqi(lat, lon, start_date, end_date):
    """
    Fetches historical AQI data from OpenWeather API between start_date and end_date.
    Dates should be Unix timestamps.
    """
    print(f"Fetching data for Lat: {lat}, Lon: {lon}...")
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_date}&end={end_date}&appid={API_KEY}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return data.get('list', [])
    else:
        print(f"Error fetching data: {response.status_code}")
        print(response.text)
        return None

def process_and_save_data(raw_data, output_path):
    """
    Processes the raw JSON data into a clean Pandas DataFrame and saves it as CSV.
    """
    if not raw_data:
        print("No data to process.")
        return
        
    records = []
    
    for item in raw_data:
        # The 'dt' field is a Unix timestamp, let's convert it to a readable date
        timestamp = datetime.fromtimestamp(item['dt'])
        
        # AQI is in the 'main' dictionary (1 = Good, 5 = Very Poor)
        aqi = item['main']['aqi']
        
        # Pollutant components are in the 'components' dictionary
        components = item['components']
        
        # Flatten the data specifically so it fits in a table row
        record = {
            'timestamp': timestamp,
            'aqi': aqi,
            'co': components.get('co', 0),
            'no': components.get('no', 0),
            'no2': components.get('no2', 0),
            'o3': components.get('o3', 0),
            'so2': components.get('so2', 0),
            'pm2_5': components.get('pm2_5', 0),
            'pm10': components.get('pm10', 0),
            'nh3': components.get('nh3', 0)
        }
        records.append(record)
        
    # Convert list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(records)
    
    # Save to our raw data directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} records to {output_path}!")

if __name__ == "__main__":
    if not API_KEY or API_KEY == "your_openweather_api_key_here":
        print("Error: API Key is missing or invalid. Please check your .env file.")
    else:
        import time
        end_time = int(time.time())
        # Let's get roughly 2 years of data for ML training (gives us solid seasonal variations)
        start_time = end_time - (2 * 365 * 24 * 60 * 60)
        
        raw_api_data = fetch_historical_aqi(LAT, LON, start_time, end_time)
        if raw_api_data:
            process_and_save_data(raw_api_data, "data/raw/historical_aqi.csv")
