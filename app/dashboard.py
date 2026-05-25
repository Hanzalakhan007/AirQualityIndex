"""Streamlit dashboard for AQI forecasting."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import TIMEZONE
from src.inference import (
    SLIDER_FEATURES,
    alert_days,
    aqi_level_and_color,
    build_forecast_curve,
    clear_caches,
    explainability_images,
    get_available_model_names,
    get_available_model_options,
    get_backend_status,
    get_current_observation,
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
    (0, 50, "rgba(63, 197, 123, 0.10)"),
    (51, 100, "rgba(247, 201, 72, 0.11)"),
    (101, 150, "rgba(255, 145, 77, 0.12)"),
    (151, 200, "rgba(245, 84, 84, 0.12)"),
    (201, 500, "rgba(121, 27, 27, 0.12)"),
]


def format_timestamp(value: object) -> str:
    if value is None:
        return "Unknown time"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo(TIMEZONE))
        localized = parsed.astimezone(ZoneInfo(TIMEZONE))
        return localized.strftime("%b %d, %Y %I:%M %p")
    except ValueError:
        return str(value)


def inject_styles() -> None:
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Instrument+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-0: #071017;
                --bg-1: #0c1822;
                --bg-2: #111f2c;
                --bg-3: #162636;
                --line: rgba(198, 214, 231, 0.12);
                --muted: #9fb2c4;
                --text: #e8eff6;
                --cyan: #7dd8ff;
                --aqua: #53d7c2;
                --gold: #f4d35e;
                --amber: #ff9f43;
                --good: #56d486;
                --danger: #ff6b6b;
            }

            .stApp {
                font-family: 'Instrument Sans', sans-serif !important;
                color: var(--text);
                background:
                    radial-gradient(circle at 12% 18%, rgba(83, 215, 194, 0.18), transparent 26%),
                    radial-gradient(circle at 88% 8%, rgba(125, 216, 255, 0.12), transparent 24%),
                    linear-gradient(180deg, #060c12 0%, #0a131c 35%, #0d1823 100%);
            }

            .main .block-container {
                padding-top: 2.2rem;
                padding-bottom: 2.5rem;
                max-width: 1380px;
            }

            .stSidebar {
                background:
                    linear-gradient(180deg, rgba(16, 28, 39, 0.98) 0%, rgba(21, 35, 49, 0.98) 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.07);
            }

            .stSidebar * {
                color: #dce7f2 !important;
            }

            .stSidebar [data-testid="stMarkdownContainer"] p {
                color: #c7d5e2 !important;
            }

            [data-testid="stSidebar"] .stButton > button,
            [data-testid="stSidebar"] .stDownloadButton > button,
            .stButton > button {
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.10);
                background: linear-gradient(180deg, rgba(16, 27, 39, 0.96), rgba(18, 31, 45, 0.96));
                color: #edf5ff;
                font-weight: 700;
                transition: all 0.22s ease;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
            }

            .stButton > button:hover,
            [data-testid="stSidebar"] .stButton > button:hover {
                border-color: rgba(125, 216, 255, 0.35);
                transform: translateY(-1px);
                box-shadow: 0 14px 30px rgba(0, 0, 0, 0.24);
            }

            .stSelectbox label,
            .stSlider label {
                font-weight: 600;
                color: #dce7f2 !important;
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div {
                background: rgba(22, 36, 50, 0.86) !important;
                border-radius: 14px !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
            }

            .hero-shell {
                position: relative;
                overflow: hidden;
                border-radius: 34px;
                padding: 2.3rem 2.4rem 2rem 2.4rem;
                background:
                    radial-gradient(circle at top right, rgba(125, 216, 255, 0.14), transparent 28%),
                    linear-gradient(140deg, rgba(10, 20, 31, 0.98) 0%, rgba(15, 27, 39, 0.95) 58%, rgba(16, 34, 50, 0.92) 100%);
                border: 1px solid rgba(155, 194, 230, 0.12);
                box-shadow: 0 28px 68px rgba(0, 0, 0, 0.28);
                margin-bottom: 1.1rem;
            }

            .hero-shell:before {
                content: "";
                position: absolute;
                inset: 0;
                background:
                    linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.03), transparent);
                pointer-events: none;
            }

            .eyebrow {
                color: var(--cyan);
                text-transform: uppercase;
                letter-spacing: 0.24em;
                font-size: 0.74rem;
                font-weight: 800;
            }

            .hero-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 4rem;
                line-height: 0.98;
                font-weight: 700;
                margin: 0.45rem 0 0.7rem 0;
                color: #edf4fb;
            }

            .hero-copy {
                color: #b9c8d6;
                max-width: 48rem;
                line-height: 1.75;
                font-size: 1.02rem;
            }

            .hero-meta {
                display: inline-flex;
                gap: 0.6rem;
                align-items: center;
                margin-top: 1rem;
                padding: 0.55rem 0.85rem;
                border-radius: 999px;
                background: rgba(17, 33, 48, 0.82);
                border: 1px solid rgba(255, 255, 255, 0.08);
                color: #d9e6f0;
                font-size: 0.9rem;
            }

            .sidebar-panel,
            .metric-card,
            .feature-card,
            .panel-card,
            .forecast-card {
                border-radius: 24px;
                background: linear-gradient(180deg, rgba(16, 29, 42, 0.96), rgba(18, 33, 47, 0.92));
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 18px 38px rgba(0, 0, 0, 0.2);
            }

            .sidebar-panel {
                padding: 1.05rem 1rem;
                margin: 0.6rem 0 1rem 0;
            }

            .metric-card,
            .panel-card {
                padding: 1.25rem 1.2rem;
            }

            .feature-card {
                padding: 1.1rem 1rem;
            }

            .metric-label,
            .section-kicker,
            .forecast-label {
                text-transform: uppercase;
                letter-spacing: 0.16em;
                font-size: 0.72rem;
                color: #90a5b8;
                font-weight: 800;
            }

            .metric-value {
                margin-top: 0.65rem;
                font-family: 'Space Grotesk', sans-serif;
                font-size: 2.8rem;
                line-height: 1;
                font-weight: 700;
            }

            .metric-subtext,
            .panel-copy {
                margin-top: 0.45rem;
                color: #c9d6e0;
                line-height: 1.6;
                font-size: 0.96rem;
            }

            .feature-card.observed-hero {
                padding: 1.55rem 1.5rem;
                min-height: 280px;
                background:
                    radial-gradient(circle at top left, rgba(244, 211, 94, 0.16), transparent 30%),
                    linear-gradient(160deg, rgba(21, 33, 49, 0.98), rgba(20, 30, 45, 0.95));
            }

            .observed-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.05rem;
                font-weight: 700;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                color: #ffe96d;
            }

            .observed-aqi {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 5.4rem;
                line-height: 0.95;
                margin: 1rem 0 0.65rem 0;
                font-weight: 700;
            }

            .observed-band {
                font-size: 1.3rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .section-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 2.05rem;
                font-weight: 700;
                margin: 1.35rem 0 0.3rem 0;
                color: #edf4fb;
            }

            .section-copy {
                color: #98adbf;
                margin-bottom: 0.95rem;
                line-height: 1.7;
            }

            .forecast-card {
                padding: 1.15rem 1rem;
                min-height: 190px;
            }

            .forecast-value {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 2.55rem;
                font-weight: 700;
                line-height: 1;
                margin: 0.85rem 0 0.3rem 0;
            }

            .alert-panel {
                border-radius: 24px;
                padding: 1rem 1.15rem;
                margin: 0.4rem 0 1rem 0;
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 14px 32px rgba(0, 0, 0, 0.18);
            }

            .alert-panel.hazardous {
                background: linear-gradient(135deg, rgba(118, 0, 29, 0.95), rgba(76, 0, 18, 0.96));
                border-left: 6px solid #ff96af;
                color: #ffe4ea;
            }

            .alert-panel.unhealthy {
                background: linear-gradient(135deg, rgba(182, 58, 28, 0.95), rgba(122, 34, 17, 0.96));
                border-left: 6px solid #ffcb8f;
                color: #fff2df;
            }

            .alert-panel.sensitive {
                background: linear-gradient(135deg, rgba(171, 118, 9, 0.93), rgba(120, 83, 8, 0.96));
                border-left: 6px solid #fff07a;
                color: #fff9d9;
            }

            .alert-title {
                font-size: 1rem;
                font-weight: 800;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .alert-copy {
                font-size: 0.98rem;
                line-height: 1.6;
            }

            .guide-row {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 0.9rem;
                color: #dce7f1;
            }

            .guide-dot {
                width: 12px;
                height: 12px;
                border-radius: 999px;
                box-shadow: 0 0 0 5px rgba(255, 255, 255, 0.03);
                flex-shrink: 0;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                margin-top: 0.5rem;
                padding: 0.45rem 0.75rem;
                border-radius: 999px;
                background: rgba(20, 41, 57, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.08);
                color: #dce8f4;
                font-size: 0.9rem;
                font-weight: 600;
            }

            .tab-note {
                color: #9db2c5;
                line-height: 1.75;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.35rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                margin-bottom: 1rem;
            }

            .stTabs [data-baseweb="tab"] {
                background: transparent;
                color: #b6c5d4;
                border-radius: 14px 14px 0 0;
                padding: 0.65rem 0.9rem;
                font-weight: 700;
            }

            .stTabs [aria-selected="true"] {
                color: #f0f7ff !important;
                box-shadow: inset 0 -3px 0 var(--cyan);
            }

            [data-testid="stDataFrame"] {
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                overflow: hidden;
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


def render_sidebar_panel(title: str, icon: str, body: str) -> None:
    heading = f"{icon} {title}".strip()
    st.markdown(
        f"""
        <div class="sidebar-panel">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.08rem; font-weight:700; margin-bottom:0.5rem;">
                {heading}
            </div>
            <div class="panel-copy" style="font-size:0.95rem;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str, color: str) -> str:
    return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
    """


def append_live_aqi_to_history(
    history_df: pd.DataFrame,
    current_observed_aqi: float | None,
    current_observed_timestamp: object | None,
    current_pm25: float | None = None,
    current_pm10: float | None = None,
) -> pd.DataFrame:
    """Extend daily charts with the live OpenWeather AQI and pollutants."""
    if current_observed_aqi is None or current_observed_timestamp is None:
        return history_df

    try:
        current_date = pd.to_datetime(current_observed_timestamp).tz_localize(None).floor("D")
    except Exception:
        return history_df

    live_row = pd.DataFrame(
        [
            {
                "date": current_date,
                "aqi_display": current_observed_aqi,
                "pm2_5": current_pm25,
                "pm10": current_pm10,
            }
        ]
    )
    if history_df.empty:
        return live_row

    updated = history_df.copy()
    updated["date"] = pd.to_datetime(updated["date"]).dt.tz_localize(None)
    updated = updated[updated["date"].dt.floor("D") != current_date]
    return pd.concat([updated, live_row], ignore_index=True).sort_values("date")


def render_observed_hero(
    current_observed_aqi: float | None,
    observed_level: str,
    observed_color: str,
    current_source: str,
    latest_timestamp: object,
    pipeline_timestamp: object | None = None,
    source_timestamp_label: str = "Latest source timestamp",
) -> str:
    observed_text = f"{current_observed_aqi:.1f}" if current_observed_aqi is not None else "N/A"
    pipeline_text = (
        f"Latest hourly pipeline: <b>{format_timestamp(pipeline_timestamp)} {TIMEZONE}</b><br>"
        if pipeline_timestamp is not None
        else ""
    )
    return f"""
        <div class="feature-card observed-hero">
            <div class="observed-title">Current Observed AQI</div>
            <div class="observed-aqi" style="color:{observed_color};">{observed_text}</div>
            <div class="observed-band" style="color:{observed_color};">{observed_level}</div>
            <div class="panel-copy" style="margin-top:1rem;">
                Live reading source: <b>{current_source}</b><br>
                {pipeline_text}
                {source_timestamp_label}: <b>{format_timestamp(latest_timestamp)} {TIMEZONE}</b>
            </div>
            <div class="status-pill">Karachi stream | {TIMEZONE}</div>
        </div>
    """


def render_forecast_cards(predictions: list[float], forecast_dates: list[str]) -> None:
    labels = ["Tomorrow", "Day 2", "Day 3"]
    columns = st.columns(3)
    for index, column in enumerate(columns):
        level, color = aqi_level_and_color(predictions[index])
        with column:
            st.markdown(
                f"""
                <div class="forecast-card" style="border-top:4px solid {color};">
                    <div class="forecast-label">{labels[index]}</div>
                    <div style="font-size:1rem; color:#dbe7f1; font-weight:700; margin-top:0.45rem;">{forecast_dates[index]}</div>
                    <div class="forecast-value" style="color:{color};">{predictions[index]:.1f}</div>
                    <div style="font-weight:700; color:{color}; font-size:1rem;">{level}</div>
                    <div class="panel-copy" style="font-size:0.9rem;">
                        {health_recommendation(predictions[index])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_forecast_chart(history_df: pd.DataFrame, predictions: list[float]) -> None:
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
                line={"width": 3, "color": "#7dd8ff"},
                marker={"size": 7, "color": "#edf4fb"},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=daily_forecast["date"],
            y=daily_forecast["aqi_predicted"],
            mode="lines+markers",
            name="3-Day Forecast",
            line={"width": 3, "color": "#f4d35e", "dash": "dash"},
            marker={"size": 8, "color": "#f4d35e"},
        )
    )
    chart_dates = []
    if not history_df.empty:
        chart_dates.extend(pd.to_datetime(history_df["date"]).dt.tz_localize(None).tolist())
    if not daily_forecast.empty:
        chart_dates.extend(pd.to_datetime(daily_forecast["date"]).dt.tz_localize(None).tolist())
    xaxis_config = {"title": "Date"}
    if chart_dates:
        xaxis_config["range"] = [min(chart_dates), max(chart_dates) + pd.Timedelta(hours=12)]

    max_chart_value = max(predictions[1:4] or predictions or [0.0])
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11, 22, 32, 0.85)",
        font={"color": "#edf4fb"},
        height=420,
        margin={"l": 16, "r": 16, "t": 18, "b": 18},
        xaxis=xaxis_config,
        yaxis_title="AQI",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        shapes=add_aqi_band_shapes(),
        yaxis={"range": [0, max(220, max_chart_value + 35)]},
    )
    st.plotly_chart(figure, width="stretch")


def render_history_chart(history_df: pd.DataFrame) -> None:
    if history_df.empty:
        st.info("Historical overview is not available yet.")
        return

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=pd.to_datetime(history_df["date"]),
            y=history_df["pm2_5"],
            mode="lines",
            name="PM2.5",
            line={"width": 3, "color": "#ff9f43"},
            fill="tozeroy",
            fillcolor="rgba(255, 159, 67, 0.10)",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=pd.to_datetime(history_df["date"]),
            y=history_df["pm10"],
            mode="lines",
            name="PM10",
            line={"width": 3, "color": "#53d7c2"},
            fill="tozeroy",
            fillcolor="rgba(83, 215, 194, 0.08)",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=pd.to_datetime(history_df["date"]),
            y=history_df["aqi_display"],
            mode="lines+markers",
            name="AQI",
            line={"width": 2.5, "color": "#7dd8ff"},
            marker={"size": 6},
            yaxis="y2",
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11, 22, 32, 0.85)",
        font={"color": "#edf4fb"},
        height=420,
        margin={"l": 16, "r": 16, "t": 18, "b": 18},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_title="Date",
        yaxis={"title": "Pollutants"},
        yaxis2={"title": "AQI", "overlaying": "y", "side": "right"},
    )
    st.plotly_chart(figure, width="stretch")


def build_report_df(
    today_prediction: float | None,
    today_date: str | None,
    predictions: list[float],
    forecast_dates: list[str],
) -> pd.DataFrame:
    timeline_values = [today_prediction, *predictions]
    return pd.DataFrame(
        {
            "Window": ["Today confirmation", "Forecast day 1", "Forecast day 2", "Forecast day 3"],
            "Date": [today_date, *forecast_dates],
            "Predicted AQI": timeline_values,
            "Category": [aqi_level_and_color(value)[0] for value in timeline_values],
            "Health Recommendation": [health_recommendation(value) for value in timeline_values],
        }
    )


def render_alert_panel(predictions: list[float]) -> None:
    alerts = alert_days(predictions)
    if not alerts:
        return

    levels = {item["level"] for item in alerts}
    if "Hazardous" in levels:
        panel_type = "hazardous"
        headline = "Hazardous AQI Alert"
        guidance = "Avoid outdoor activity, keep windows closed, and use a mask if travel is unavoidable."
    elif "Unhealthy" in levels:
        panel_type = "unhealthy"
        headline = "Severe Air Quality Alert"
        guidance = "Limit time outdoors, especially for children, seniors, and anyone with respiratory sensitivity."
    else:
        panel_type = "sensitive"
        headline = "Health Precaution"
        guidance = "Sensitive groups should consider reducing prolonged outdoor exertion."

    days_text = ", ".join(f"{item['day']} ({item['aqi']:.1f})" for item in alerts)
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


def render_health_guide() -> None:
    guide_rows = [
        ("#56d486", "Good (0-50): Air quality is satisfactory and outdoor activity is generally safe."),
        ("#f4d35e", "Moderate (51-100): Sensitive groups should reduce prolonged outdoor exertion."),
        ("#ff9f43", "Sensitive (101-150): Children, seniors, and sensitive groups should limit outdoor time."),
        ("#ff6b6b", "Unhealthy (151-200): Reduce outdoor exposure and use a mask when necessary."),
        ("#791b1b", "Hazardous (201+): Avoid prolonged outdoor activity and stay indoors when possible."),
    ]
    for color, text in guide_rows:
        st.markdown(
            f"""
            <div class="guide-row">
                <span class="guide-dot" style="background:{color};"></span>
                <span>{text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="AQI Forecast Studio", layout="wide", initial_sidebar_state="expanded")
    inject_styles()

    try:
        latest_row = get_latest_feature_row()
    except Exception as exc:
        st.error(f"Unable to load feature data: {exc}")
        st.stop()

    backend_status = get_backend_status()
    available_models = get_available_model_names()
    model_options = get_available_model_options()
    now_local = datetime.now(ZoneInfo(TIMEZONE))

    with st.sidebar:
        st.markdown(
            """
            <div style="margin-top:0.25rem; margin-bottom:1rem;">
                <div class="eyebrow">Real-Time AQI Forecasting</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; line-height:1.04; margin-top:0.4rem;">
                    Karachi
                    <div style="color:#7dd8ff;">AirWatch</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_sidebar_panel(
            "About",
            "",
            "Real-time Karachi air-quality monitoring and next 3 days forecasting powered by a production-ready ensemble of ML models.",
        )

        st.markdown("### Forecast Controls")
        model_name = st.selectbox("Model strategy", model_options, index=0)
        if st.button("Refresh Live Data", width="stretch"):
            clear_caches()
            st.rerun()
        st.caption("Reloads the latest available feature store snapshot, model artifacts, and forecast output.")

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

        mongo_status = "Active" if backend_status["mongo_available"] else "Fallback mode"
        pipeline_timestamp = backend_status.get("pipeline_last_success_at") or backend_status.get("pipeline_last_completed_slot")
        latest_source_timestamp = latest_row.get("timestamp")
        source_timestamp_label = "Latest source timestamp"
        try:
            source_dt = pd.to_datetime(latest_source_timestamp)
            if getattr(source_dt, "tzinfo", None) is None:
                source_dt = source_dt.tz_localize(ZoneInfo(TIMEZONE))
            else:
                source_dt = source_dt.tz_convert(ZoneInfo(TIMEZONE))
            if source_dt.date() != now_local.date():
                source_timestamp_label = "Latest available source timestamp"
        except Exception:
            pass

        render_sidebar_panel(
            "System Status",
            "Control",
            f"Status: <b>{mongo_status}</b><br>"
            f"Timezone: <b>{TIMEZONE}</b><br>"
            + (
                f"Latest hourly pipeline: <b>{format_timestamp(pipeline_timestamp)} {TIMEZONE}</b><br>"
                if pipeline_timestamp
                else ""
            )
            + f"{source_timestamp_label}: <b>{format_timestamp(latest_source_timestamp)} {TIMEZONE}</b>",
        )
        render_sidebar_panel(
            "Technical Specs",
            "Bars",
            "Feature cadence: hourly<br>"
            "Training cadence: daily<br>"
            f"Models online: <b>{', '.join(available_models) if available_models else 'Local fallback only'}</b>",
        )
        render_sidebar_panel(
            "Data Sources",
            "Globe",
            "Primary: OpenWeather historical air-pollution API<br>"
            "Fallback: Open-Meteo air-quality feed<br>"
            "Registry: MongoDB Atlas feature store and model registry",
        )
        st.caption(f"Developed by {AUTHOR_NAME}")

    selected_model = None if model_name == "Best Available" else model_name
    try:
        forecast = predict_next_days(selected_model, overrides)
    except Exception as exc:
        st.error(f"Unable to generate forecast: {exc}")
        st.stop()

    predictions = forecast["predictions"]
    predicted_today_aqi = forecast["today_aqi"]
    today_date = forecast["today_date"]
    next_three_day_predictions = forecast["next_three_day_predictions"]
    next_three_day_dates = forecast["next_three_day_dates"]
    best_model = forecast["model_name"] or get_default_model_name()
    leaderboard = forecast["leaderboard"] or get_model_leaderboard()
    history_df = get_recent_daily_history()
    current_observation = get_current_observation()
    current_observed_aqi = current_observation["aqi"]
    current_source = current_observation["source"]
    current_observed_timestamp = current_observation["timestamp"]
    chart_history_df = append_live_aqi_to_history(
        history_df,
        current_observed_aqi,
        current_observed_timestamp,
        current_observation.get("pm2_5"),
        current_observation.get("pm10"),
    )
    explainability_assets = explainability_images()

    observed_level, observed_color = aqi_level_and_color(current_observed_aqi)
    forecast_level, forecast_color = aqi_level_and_color(predicted_today_aqi)
    delta_text = "Current and forecast are closely aligned."
    if current_observed_aqi is not None and predicted_today_aqi is not None:
        delta = predicted_today_aqi - current_observed_aqi
        if delta > 5:
            delta_text = f"Forecast is {delta:.1f} AQI above current conditions."
        elif delta < -5:
            delta_text = f"Forecast is {abs(delta):.1f} AQI below current conditions."

    reliability = leaderboard[0] if leaderboard else None
    reliability_text = f"RMSE {float(reliability['rmse']):.2f}" if reliability else "Metrics unavailable"
    forecast_signal = sum(next_three_day_predictions) / len(next_three_day_predictions) if next_three_day_predictions else 0.0
    report_df = build_report_df(predicted_today_aqi, today_date, next_three_day_predictions, next_three_day_dates)
    observed_timestamp_label = (
        "Current source timestamp"
        if current_source.startswith("Current ")
        else source_timestamp_label
    )

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="eyebrow">Karachi AQI Forecasting</div>
            <div class="hero-title">Air Quality, tuned for clarity.</div>
            <div class="hero-copy">
                A cinematic monitoring dashboard for Karachi that blends live observed AQI, today's model confirmation,
                and a polished 72-hour outlook powered by production ML pipelines.
            </div>
            <div class="hero-meta">Best model live: {best_model}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="status-pill">Deployment mode: '
        + ("MongoDB connected" if backend_status["mongo_available"] else "Local fallback active")
        + "</div>",
        unsafe_allow_html=True,
    )

    if not backend_status["mongo_available"]:
        fallback_note = backend_status.get("local_feature_timestamp")
        fallback_text = f"Latest local feature timestamp: {format_timestamp(fallback_note)}." if fallback_note else ""
        st.markdown(
            f"""
            <div class="panel-card" style="margin-top:0.95rem; border-left:5px solid #f4d35e;">
                <div class="metric-label">Fallback Mode</div>
                <div class="panel-copy">MongoDB is unavailable, so the dashboard is serving local artifacts. {fallback_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_alert_panel(predictions)

    headline_left, headline_right = st.columns([1.45, 1.0], gap="large")
    with headline_left:
        st.markdown(
            render_observed_hero(
                current_observed_aqi,
                observed_level,
                observed_color,
                current_source,
                current_observed_timestamp or latest_row.get("timestamp"),
                pipeline_timestamp=pipeline_timestamp,
                source_timestamp_label=observed_timestamp_label,
            ),
            unsafe_allow_html=True,
        )
    with headline_right:
        st.markdown(
            render_metric_card(
                "Predicted AQI Today",
                f"{predicted_today_aqi:.1f}",
                f"{forecast_level} | Confirmation signal",
                forecast_color,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            render_metric_card(
                "3-Day Outlook",
                f"{forecast_signal:.1f}",
                f"Average AQI across the next 3 days. {delta_text} | {reliability_text}",
                "#53d7c2",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            render_metric_card(
                "Production Lead",
                best_model,
                f"Top available model in the latest leaderboard. {reliability_text}.",
                "#7dd8ff",
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">3-Day Forecast Outlook</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Today\'s model prediction is shown above for confirmation. '
        'The forecast cards below cover the next 3 days only.</div>',
        unsafe_allow_html=True,
    )
    render_forecast_cards(next_three_day_predictions, next_three_day_dates)

    st.markdown('<div class="section-title">Observed vs Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A compact trend view showing recent daily AQI behavior against the next 72-hour forecast window.</div>',
        unsafe_allow_html=True,
    )
    render_forecast_chart(chart_history_df, predictions)

    tab_forecast, tab_history, tab_models, tab_health = st.tabs(
        ["Detailed Report", "Historical Overview", "Model Insights", "Health Guidance"]
    )

    with tab_forecast:
        left_col, right_col = st.columns([1.2, 1.0], gap="large")
        with left_col:
            st.markdown(
                """
                <div class="panel-card">
                    <div class="metric-label">Forecast Manifest</div>
                    <div class="panel-copy">
                        This export combines today's model confirmation with the next 3 forecast days and their AQI risk categories.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(report_df, width="stretch", hide_index=True)
        with right_col:
            st.markdown(
                """
                <div class="panel-card">
                    <div class="metric-label">Download Pack</div>
                    <div class="panel-copy">
                        Export a lightweight forecast report for presentation, reporting, or monitoring handoff.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download 3-day forecast CSV",
                report_df.to_csv(index=False).encode("utf-8"),
                "aqi_3_day_forecast_report.csv",
                "text/csv",
                use_container_width=True,
            )
            st.markdown(
                render_metric_card(
                    "Health Snapshot",
                    forecast_level,
                    health_recommendation(predicted_today_aqi),
                    forecast_color,
                ),
                unsafe_allow_html=True,
            )

    with tab_history:
        stat_cols = st.columns(4)
        if not chart_history_df.empty:
            avg_aqi = chart_history_df["aqi_display"].mean()
            avg_pm25 = history_df["pm2_5"].mean() if not history_df.empty else 0.0
            avg_pm10 = history_df["pm10"].mean() if not history_df.empty else 0.0
            peak_aqi = chart_history_df["aqi_display"].max()
        else:
            avg_aqi = avg_pm25 = avg_pm10 = peak_aqi = 0.0
        history_stats = [
            ("Avg AQI", f"{avg_aqi:.1f}", "Recent daily average"),
            ("Peak AQI", f"{peak_aqi:.1f}", "Recent daily maximum"),
            ("Avg PM2.5", f"{avg_pm25:.1f}", "Recent particulate load"),
            ("Avg PM10", f"{avg_pm10:.1f}", "Recent coarse particulate load"),
        ]
        for column, (label, value, subtext) in zip(stat_cols, history_stats):
            with column:
                st.markdown(render_metric_card(label, value, subtext, "#f4d35e"), unsafe_allow_html=True)
        render_history_chart(chart_history_df)

    with tab_models:
        insight_left, insight_right = st.columns([1.0, 1.2], gap="large")
        with insight_left:
            st.markdown(
                render_metric_card(
                    "Best Model Selection",
                    best_model,
                    f"Selected from the live leaderboard. {reliability_text}.",
                    "#7dd8ff",
                ),
                unsafe_allow_html=True,
            )
            leaderboard_df = pd.DataFrame(leaderboard)
            if not leaderboard_df.empty:
                leaderboard_df = leaderboard_df.rename(
                    columns={
                        "model": "Model",
                        "rmse": "RMSE",
                        "mae": "MAE",
                        "r2": "R2",
                        "fit_status": "Fit Status",
                        "selection_score": "Selection Score",
                    }
                )
                st.dataframe(leaderboard_df, width="stretch", hide_index=True)
            else:
                st.info("No leaderboard metadata is available.")
        with insight_right:
            st.markdown(
                """
                <div class="panel-card">
                    <div class="metric-label">Model and Explainability Assets</div>
                    <div class="panel-copy">
                        SHAP and LIME outputs are surfaced below when they are available in the reports directory.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if explainability_assets:
                image_cols = st.columns(2)
                for index, (name, path) in enumerate(explainability_assets.items()):
                    with image_cols[index % 2]:
                        st.image(str(path), caption=name.replace("_", " ").title(), use_container_width=True)
            else:
                st.info("Explainability images are not available in this deployment.")

    with tab_health:
        left_col, right_col = st.columns([0.95, 1.15], gap="large")
        with left_col:
            st.markdown(
                """
                <div class="panel-card">
                    <div class="metric-label">Comprehensive Health Guidance</div>
                    <div class="panel-copy">
                        AQI risk is translated into practical guidance for public safety, especially for sensitive groups.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_health_guide()
        with right_col:
            st.markdown(
                render_metric_card(
                    "Current Recommendation",
                    forecast_level,
                    health_recommendation(predicted_today_aqi),
                    forecast_color,
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="panel-card">
                    <div class="metric-label">Operational Notes</div>
                    <div class="panel-copy">
                        Alerts are generated from the forecast horizon, not only from the currently observed reading.
                        This helps the dashboard surface tomorrow's risk before conditions visibly peak.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption(f"Rendered on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {AUTHOR_NAME}")


if __name__ == "__main__":
    main()
