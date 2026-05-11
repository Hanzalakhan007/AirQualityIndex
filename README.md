# Air Quality Index (AQI) Prediction System

This is an end-to-end Machine Learning project to predict the Air Quality Index based on pollution and weather data. It is structured using industry best practices for Data Science, separating raw data from notebooks and Python source scripts.

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Environment Variables:
   - Rename `.env.example` to `.env`.
   - Add your OpenWeather API key inside the `.env` file.
