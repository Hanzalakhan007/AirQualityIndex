"""Streamlit dashboard for AQI forecasting."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import DEFAULT_CITY, TIMEZONE
from src.inference import (
    SLIDER_FEATURES,
    alert_days,
    aqi_level_and_color,
    build_forecast_curve,
    clear_caches,
    get_available_model_names,
    get_available_model_options,
    get_backend_status,
    get_current_aqi,
    get_default_model_name,
    get_latest_feature_row,
    get_model_leaderboard,
    get_recent_daily_history,
    health_recommendation,
    predict_next_days,
)


AUTHOR_NAME = "Hanzala Abbas Khan"
POLLUTANT_LABELS = {
    "co": "CO",
    "no2": "NO2",
    "o3": "O3",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "nh3": "NH3",
}
AQI_BANDS = [
    (0, 50, "rgba(46, 204, 113, 0.10)"),
    (51, 100, "rgba(241, 196, 15, 0.10)"),
    (101, 150, "rgba(230, 126, 34, 0.10)"),
    (151, 200, "rgba(231, 76, 60, 0.10)"),
    (201, 500, "rgba(127, 29, 29, 0.12)"),
]


def format_timestamp(value: object) -> str:
    if value is None:
        return "Unknown time"
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%b %d, %Y %I:%M %p")
    except ValueError:
        return str(value)


def inject_styles() -> None:
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            .stApp {
                font-family: 'Manrope', sans-serif !important;
                color: #edf4f8;
                background:
                    radial-gradient(circle at top left, rgba(89, 195, 195, 0.12), transparent 28%),
                    linear-gradient(160deg, #07121b 0%, #0d1d2a 50%, #122634 100%);
            }
            .stSidebar {
                background: linear-gradient(180deg, rgba(240, 248, 255, 0.97) 0%, rgba(230, 239, 246, 0.97) 100%);
            }
            .stSidebar * {
                color: #122433 !important;
            }
            .hero-card {
                background: linear-gradient(135deg, rgba(10, 21, 31, 0.98) 0%, rgba(14, 30, 45, 0.94) 100%);
                border: 1px solid rgba(125, 211, 252, 0.16);
                border-radius: 28px;
                padding: 2.1rem 2.2rem;
                box-shadow: 0 24px 58px rgba(0, 0, 0, 0.25);
                margin-bottom: 1rem;
            }
            .eyebrow {
                color: #7dd3fc;
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.78rem;
                font-weight: 800;
            }
            .hero-title {
                font-size: 3.2rem;
                font-weight: 800;
                line-height: 1.04;
                margin: 0.35rem 0 0.55rem 0;
            }
            .hero-copy {
                color: #a8bccb;
                line-height: 1.65;
                max-width: 56rem;
            }
            .signature {
                color: #d9e5ec;
                margin-top: 0.9rem;
                font-size: 0.94rem;
                font-weight: 600;
            }
            .card, .forecast-card {
                background: rgba(11, 24, 37, 0.86);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 22px;
                box-shadow: 0 18px 34px rgba(0, 0, 0, 0.18);
            }
            .card {
                padding: 1.2rem 1.3rem;
                height: 100%;
            }
            .forecast-card {
                padding: 1.15rem 1rem;
                min-height: 170px;
            }
            .label {
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.74rem;
                color: #9db1c1;
                font-weight: 800;
                margin-bottom: 0.45rem;
            }
            .value {
                font-size: 2.45rem;
                font-weight: 800;
                line-height: 1;
                margin-bottom: 0.3rem;
            }
            .subtext {
                color: #c6d3dc;
                font-size: 0.92rem;
                line-height: 1.55;
            }
            .section-title {
                font-size: 1.22rem;
                font-weight: 800;
                margin: 0.95rem 0 0.8rem 0;
            }
            .callout {
                border-left: 5px solid #f6bd60;
                background: rgba(246, 189, 96, 0.10);
                border-radius: 18px;
                padding: 1rem 1.15rem;
                color: #f5deb3;
                margin-bottom: 1rem;
            }
            .alert-panel {
                border-radius: 22px;
                padding: 1rem 1.15rem;
                margin: 0.35rem 0 1rem 0;
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 16px 30px rgba(0, 0, 0, 0.16);
            }
            .alert-panel.hazardous {
                background: linear-gradient(135deg, rgba(126, 0, 35, 0.92) 0%, rgba(83, 0, 23, 0.94) 100%);
                border-left: 6px solid #ff8fab;
                color: #ffe3ea;
            }
            .alert-panel.unhealthy {
                background: linear-gradient(135deg, rgba(196, 59, 36, 0.92) 0%, rgba(143, 33, 20, 0.94) 100%);
                border-left: 6px solid #ffd6a5;
                color: #fff1de;
            }
            .alert-title {
                font-size: 1rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }
            .alert-copy {
                font-size: 0.98rem;
                line-height: 1.6;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_aqi_band_shapes() -> list[dict[str, object]]:
    return [
        {
            "type": "rect",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": low,
            "y1": high,
            "fillcolor": color,
            "line": {"width": 0},
            "layer": "below",
        }
        for low, high, color in AQI_BANDS
    ]


def render_metric_card(label: str, value: str, subtext: str, color: str) -> str:
    return f"""
        <div class="card">
            <div class="label">{label}</div>
            <div class="value" style="color:{color};">{value}</div>
            <div class="subtext">{subtext}</div>
        </div>
    """


def render_forecast_cards(predictions: list[float], forecast_dates: list[str]) -> None:
    columns = st.columns(4)
    labels = ["Today", "Tomorrow", "Day 3", "Day 4"]
    for index, column in enumerate(columns):
        level, color = aqi_level_and_color(predictions[index])
        with column:
            st.markdown(
                f"""
                <div class="forecast-card" style="border-top:4px solid {color};">
                    <div class="label">{labels[index]}</div>
                    <div style="font-size:0.98rem; color:#dde8ef; font-weight:700;">{forecast_dates[index]}</div>
                    <div style="font-size:2.3rem; font-weight:800; color:{color}; margin:0.8rem 0 0.25rem 0;">{predictions[index]:.1f}</div>
                    <div style="font-weight:700; color:{color};">{level}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_main_chart(history_df: pd.DataFrame, predictions: list[float]) -> None:
    forecast_curve = build_forecast_curve(predictions)
    daily_forecast = (
        forecast_curve.assign(date=forecast_curve["timestamp"].dt.floor("D"))
        .groupby("date", as_index=False)["aqi_predicted"]
        .mean()
    )

    figure = go.Figure()
    if not history_df.empty:
        figure.add_trace(
            go.Scatter(
                x=pd.to_datetime(history_df["date"]),
                y=history_df["aqi_display"],
                mode="lines+markers",
                name="Observed AQI",
                line={"width": 3, "color": "#7dd3fc"},
                marker={"size": 7, "color": "#edf4f8"},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=daily_forecast["date"],
            y=daily_forecast["aqi_predicted"],
            mode="lines+markers",
            name="Forecast AQI",
            line={"width": 3, "color": "#f6bd60", "dash": "dash"},
            marker={"size": 8, "color": "#f6bd60"},
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7, 18, 28, 0.40)",
        font={"color": "#edf4f8"},
        height=400,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Date",
        yaxis_title="AQI",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        shapes=add_aqi_band_shapes(),
        yaxis={"range": [0, max(220, max(predictions) + 35)]},
    )
    st.plotly_chart(figure, width="stretch")


def render_downloads(predictions: list[float], forecast_dates: list[str]) -> None:
    report_df = pd.DataFrame(
        {
            "Day": ["Today", "Tomorrow", "Day 3", "Day 4"],
            "Date": forecast_dates,
            "Predicted AQI": predictions,
            "Category": [aqi_level_and_color(value)[0] for value in predictions],
            "Health Recommendation": [health_recommendation(value) for value in predictions],
        }
    )
    st.dataframe(report_df, width="stretch", hide_index=True)
    st.download_button(
        "Download forecast CSV",
        report_df.to_csv(index=False).encode("utf-8"),
        "aqi_forecast_report.csv",
        "text/csv",
        use_container_width=True,
    )


def render_alert_panel(predictions: list[float]) -> None:
    alerts = alert_days(predictions)
    if not alerts:
        return

    hazardous_alerts = [item for item in alerts if item["level"] == "Hazardous"]
    panel_type = "hazardous" if hazardous_alerts else "unhealthy"
    headline = "Hazardous AQI Alert" if hazardous_alerts else "Air Quality Alert"
    days_text = ", ".join(f"Day {item['day']} ({item['aqi']:.1f})" for item in alerts)
    guidance = (
        "Avoid outdoor activity, keep windows closed, and use protective masks if travel is necessary."
        if hazardous_alerts
        else "Sensitive groups should reduce outdoor exposure and monitor conditions closely."
    )
    st.markdown(
        f"""
        <div class="alert-panel {panel_type}">
            <div class="alert-title">{headline}</div>
            <div class="alert-copy">
                Forecast risk detected for {days_text}. {guidance}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="AQI Forecast Studio", layout="wide")
    inject_styles()

    try:
        latest_row = get_latest_feature_row()
    except Exception as exc:
        st.error(f"Unable to load feature data: {exc}")
        st.stop()

    backend_status = get_backend_status()
    available_models = get_available_model_names()
    model_options = get_available_model_options()

    with st.sidebar:
        st.markdown("## Forecast Controls")
        model_name = st.selectbox("Model strategy", model_options, index=0)
        if st.button("Generate forecast", width="stretch"):
            clear_caches()
            st.rerun()
        if st.button("Reload data", width="stretch"):
            clear_caches()
            st.rerun()

        st.markdown("### Pollutant Inputs")
        overrides: dict[str, float] = {}
        slider_limits = {
            "co": (0.0, 5000.0),
            "no2": (0.0, 250.0),
            "o3": (0.0, 350.0),
            "pm2_5": (0.0, 500.0),
            "pm10": (0.0, 500.0),
            "nh3": (0.0, 100.0),
        }
        for feature_name in SLIDER_FEATURES:
            minimum, maximum = slider_limits[feature_name]
            current_value = float(latest_row.get(feature_name, minimum))
            current_value = min(maximum, max(minimum, current_value))
            overrides[feature_name] = st.slider(
                POLLUTANT_LABELS[feature_name],
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(current_value),
            )

        st.markdown("---")
        st.markdown("### Status")
        if backend_status["mongo_available"]:
            st.success("MongoDB connected")
        else:
            st.warning("Using local fallback data")
        st.caption(f"Timezone: {TIMEZONE}")
        st.caption(f"Models: {', '.join(available_models) if available_models else 'Local fallback only'}")

    selected_model = None if model_name == "Best Available" else model_name
    try:
        forecast = predict_next_days(selected_model, overrides)
    except Exception as exc:
        st.error(f"Unable to generate forecast: {exc}")
        st.stop()

    predictions = forecast["predictions"]
    forecast_dates = forecast["forecast_dates"]
    predicted_today_aqi = forecast["today_aqi"]
    best_model = forecast["model_name"] or get_default_model_name()
    leaderboard = forecast["leaderboard"] or get_model_leaderboard()
    history_df = get_recent_daily_history()
    current_observed_aqi, current_source = get_current_aqi()

    observed_level, observed_color = aqi_level_and_color(current_observed_aqi)
    forecast_level, forecast_color = aqi_level_and_color(predicted_today_aqi)
    delta_text = "Current and forecast are close"
    if current_observed_aqi is not None and predicted_today_aqi is not None:
        delta = predicted_today_aqi - current_observed_aqi
        if delta > 5:
            delta_text = f"Forecast is {delta:.1f} AQI above current"
        elif delta < -5:
            delta_text = f"Forecast is {abs(delta):.1f} AQI below current"

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">AQI Forecast Studio</div>
            <div class="hero-title">{DEFAULT_CITY} Air Quality Predictor</div>
            <div class="hero-copy">
                A clean forecast view focused on what matters most: current AQI, today’s predicted AQI,
                the next 4 days, and a quick health decision guide.
            </div>
            <div class="signature">Designed by {AUTHOR_NAME}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not backend_status["mongo_available"]:
        fallback_note = backend_status.get("local_feature_timestamp")
        fallback_text = f" Latest local feature timestamp: {format_timestamp(fallback_note)}." if fallback_note else ""
        st.markdown(
            f'<div class="callout"><b>Fallback mode:</b> MongoDB is unavailable, so the dashboard is using local artifacts.{fallback_text}</div>',
            unsafe_allow_html=True,
        )

    render_alert_panel(predictions)

    top_cols = st.columns(3)
    with top_cols[0]:
        observed_text = f"{current_observed_aqi:.1f}" if current_observed_aqi is not None else "N/A"
        observed_subtext = f"{observed_level} · Source: {current_source}"
        st.markdown(render_metric_card("Current Observed AQI", observed_text, observed_subtext, observed_color), unsafe_allow_html=True)
    with top_cols[1]:
        st.markdown(
            render_metric_card("Predicted AQI Today", f"{predicted_today_aqi:.1f}", f"{forecast_level} · Model forecast", forecast_color),
            unsafe_allow_html=True,
        )
    with top_cols[2]:
        reliability = leaderboard[0] if leaderboard else None
        reliability_text = f"RMSE {float(reliability['rmse']):.2f}" if reliability else "Metrics unavailable"
        st.markdown(render_metric_card("Forecast Signal", f"{sum(predictions) / len(predictions):.1f}", f"{delta_text} · {reliability_text}", "#59c3c3"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">4-Day Forecast</div>', unsafe_allow_html=True)
    render_forecast_cards(predictions, forecast_dates)

    st.markdown('<div class="section-title">Observed vs Forecast</div>', unsafe_allow_html=True)
    render_main_chart(history_df, predictions)

    lower_left, lower_right = st.columns([1.2, 1.0])
    with lower_left:
        st.markdown('<div class="section-title">Health Guidance</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card">
                <div class="label">What this means</div>
                <div class="subtext" style="font-size:1rem;">{health_recommendation(predicted_today_aqi)}</div>
                <div class="subtext" style="margin-top:0.8rem;">
                    Best model: <b>{best_model}</b><br>
                    Feature snapshot: <b>{format_timestamp(latest_row.get("timestamp"))}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with lower_right:
        st.markdown('<div class="section-title">Download Report</div>', unsafe_allow_html=True)
        render_downloads(predictions, forecast_dates)

    st.markdown("---")
    st.caption(f"Rendered on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {AUTHOR_NAME}")


if __name__ == "__main__":
    main()
