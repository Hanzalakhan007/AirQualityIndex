"""Smoke test for local prediction artifacts."""
from __future__ import annotations

from src.inference import get_default_model_name, get_latest_feature_row, predict_next_days


if __name__ == "__main__":
    latest_row = get_latest_feature_row()
    forecast = predict_next_days()
    assert forecast["predictions"], "No predictions were returned."
    assert len(forecast["predictions"]) == 4, "Expected today's confirmation value plus the next 3 forecast days."
    assert latest_row.get("pm2_5") is not None, "Latest row is missing PM2.5."
    assert forecast["model_name"] == get_default_model_name(), "Default model selection did not match the leaderboard."
    assert forecast["today_aqi"] is None or forecast["today_aqi"] > 0, "Today's AQI prediction was invalid."
    print("Prediction smoke test passed.")
