"""Flask API for AQI prediction."""
from __future__ import annotations

from flask import Flask, jsonify, request

from src.inference import (
    SLIDER_FEATURES,
    aqi_level_and_color,
    alert_days,
    get_available_model_options,
    predict_next_days,
)

app = Flask(__name__)


def parse_overrides() -> dict[str, float]:
    overrides: dict[str, float] = {}
    for feature_name in SLIDER_FEATURES:
        raw_value = request.args.get(feature_name)
        if raw_value is None:
            continue
        overrides[feature_name] = float(raw_value)
    return overrides


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict")
def predict():
    model_name = request.args.get("model", "Best Available")
    if model_name not in get_available_model_options():
        return jsonify({"error": f"Unsupported model '{model_name}'."}), 400

    try:
        forecast = predict_next_days(None if model_name == "Best Available" else model_name, parse_overrides())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

    return (
        jsonify(
            {
                "model": forecast["model_name"],
                "current_aqi": forecast["today_aqi"],
                "current_source": "Model Predicted Today",
                "current_category": aqi_level_and_color(forecast["today_aqi"])[0],
                "forecast_dates": forecast["forecast_dates"],
                "predictions": forecast["predictions"],
                "alerts": alert_days(forecast["predictions"]),
                "leaderboard": forecast["leaderboard"],
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
