"""Smoke test for local prediction artifacts."""
from __future__ import annotations

from src.inference import get_current_aqi, get_default_model_name, get_latest_feature_row, predict_next_days


if __name__ == "__main__":
    latest_row = get_latest_feature_row()
    forecast = predict_next_days()
    current_aqi, _ = get_current_aqi()
    assert forecast["predictions"], "No predictions were returned."
    assert len(forecast["predictions"]) == 3, "Expected exactly three forecast values."
    assert latest_row.get("pm2_5") is not None, "Latest row is missing PM2.5."
    assert forecast["model_name"] == get_default_model_name(), "Default model selection did not match the leaderboard."
    assert current_aqi is None or current_aqi > 0, "Current AQI normalization failed."
    print("Prediction smoke test passed.")
