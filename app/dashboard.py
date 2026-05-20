"""Streamlit dashboard for AQI forecasting."""
from __future__ import annotations

from datetime import datetime

import numpy as np
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
    explainability_images,
    get_available_model_names,
    get_available_model_options,
    get_current_aqi,
    get_default_model_name,
    get_latest_feature_row,
    get_model_leaderboard,
    get_recent_daily_history,
    health_recommendation,
    predict_next_days,
)


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
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
        <style>
            .stApp {
                font-family: 'Plus Jakarta Sans', sans-serif !important;
                background:
                    radial-gradient(circle at top left, rgba(93, 173, 226, 0.12), transparent 35%),
                    radial-gradient(circle at 85% 15%, rgba(69, 179, 157, 0.12), transparent 25%),
                    linear-gradient(160deg, #07111b 0%, #0b1624 55%, #111d2c 100%);
                color: #e6edf3;
            }
            .hero-card {
                background: linear-gradient(135deg, rgba(26, 35, 50, 0.95) 0%, rgba(13, 17, 23, 0.98) 100%);
                border: 1px solid rgba(93, 173, 226, 0.28);
                border-radius: 22px;
                padding: 2.4rem;
                text-align: center;
                box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
                margin-bottom: 1rem;
            }
            .forecast-card, .insight-box, .precautions-box {
                background: rgba(20, 28, 41, 0.88);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 16px;
                box-shadow: 0 12px 26px rgba(0, 0, 0, 0.25);
            }
            .forecast-card {
                padding: 1.4rem;
                text-align: center;
                min-height: 182px;
            }
            .insight-box {
                padding: 1.3rem;
                height: 100%;
            }
            .precautions-box {
                padding: 1.2rem 1.4rem;
                margin-bottom: 1.6rem;
                border-left: 6px solid #5dade2;
            }
            .hero-aqi {
                font-size: 5rem;
                font-weight: 800;
                line-height: 1;
                margin: 0.8rem 0;
            }
            .hero-label {
                font-size: 1.1rem;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                font-weight: 700;
            }
            .metric-title {
                font-size: 0.82rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #8b949e;
                margin-bottom: 0.35rem;
            }
            .muted-note {
                color: #9fb0c4;
                font-size: 0.92rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_cards(predictions: list[float], forecast_dates: list[str]) -> None:
    labels = ["Tomorrow", "Day After", "Next Day"]
    columns = st.columns(3)
    for index, column in enumerate(columns):
        level, color = aqi_level_and_color(predictions[index])
        with column:
            st.markdown(
                f"""
                <div class="forecast-card" style="border-top: 4px solid {color};">
                    <div class="metric-title">{labels[index]}</div>
                    <div style="font-size: 1rem; color: #dce6f3; font-weight: 600;">{forecast_dates[index]}</div>
                    <div style="font-size: 2.4rem; color: {color}; font-weight: 800; margin: 0.8rem 0 0.3rem;">
                        {predictions[index]:.1f}
                    </div>
                    <div style="color: {color}; font-weight: 700;">{level}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_forecast_chart(predictions: list[float]) -> None:
    curve = build_forecast_curve(predictions)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=curve["timestamp"],
            y=curve["aqi_predicted"],
            mode="lines",
            name="Best Forecast",
            line={"width": 4, "color": "#7dd3fc"},
            fill="tozeroy",
            fillcolor="rgba(125, 211, 252, 0.12)",
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 29, 44, 0.55)",
        font={"color": "#e6edf3"},
        height=380,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Time",
        yaxis_title="AQI",
    )
    st.plotly_chart(figure, use_container_width=True)


def render_model_comparison_chart(model_forecasts: dict[str, list[float]]) -> None:
    if not model_forecasts:
        st.info("Model comparison is unavailable until model forecasts are loaded.")
        return

    palette = ["#7dd3fc", "#5dade2", "#45b39d", "#f5b041", "#ec7063"]
    figure = go.Figure()
    for index, (model_name, predictions) in enumerate(model_forecasts.items()):
        curve = build_forecast_curve(predictions)
        if curve.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=curve["timestamp"],
                y=curve["aqi_predicted"],
                mode="lines",
                name=model_name,
                line={"width": 3 if index == 0 else 2, "color": palette[index % len(palette)], "dash": "solid" if index == 0 else "dot"},
            )
        )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 29, 44, 0.55)",
        font={"color": "#e6edf3"},
        height=390,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Time",
        yaxis_title="AQI",
        legend_title_text="Model",
    )
    st.plotly_chart(figure, use_container_width=True)


def render_history_tab(history_df: pd.DataFrame) -> None:
    aqi_column = "aqi_display" if "aqi_display" in history_df.columns else "us_aqi"
    date_column = "date" if "date" in history_df.columns else history_df.columns[0]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=pd.to_datetime(history_df[date_column]),
            y=history_df[aqi_column],
            mode="lines+markers",
            line={"width": 3, "color": "#5dade2"},
            marker={"size": 8, "color": "#ffffff", "line": {"width": 2, "color": "#5dade2"}},
            fill="tozeroy",
            fillcolor="rgba(93, 173, 226, 0.15)",
            name="Observed AQI",
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 29, 44, 0.55)",
        font={"color": "#e6edf3"},
        height=380,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Date",
        yaxis_title="AQI",
    )
    st.plotly_chart(figure, use_container_width=True)

    cols = st.columns(4)
    stats = [
        ("Average AQI", history_df[aqi_column].mean()),
        ("Peak AQI", history_df[aqi_column].max()),
        ("Average PM2.5", history_df["pm2_5"].mean() if "pm2_5" in history_df.columns else None),
        ("Average PM10", history_df["pm10"].mean() if "pm10" in history_df.columns else None),
    ]
    for col, (label, value) in zip(cols, stats):
        with col:
            value_text = "—" if value is None or pd.isna(value) else f"{value:.1f}"
            st.markdown(
                f"""
                <div class="insight-box">
                    <div class="metric-title">{label}</div>
                    <div style="font-size: 2rem; font-weight: 800; color: #ffffff;">{value_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_insights_tab(leaderboard: list[dict], predictions: list[float], current_aqi: float | None) -> None:
    left, right = st.columns(2)
    with left:
        best = leaderboard[0] if leaderboard else None
        st.markdown("### Model Leaderboard")
        if best:
            rows = "".join(
                [
                    f"<div style='margin-bottom:0.45rem;'><b>{row['model']}</b>: RMSE {row['rmse']:.2f} | MAE {row['mae']:.2f} | R2 {row['r2']:.3f}</div>"
                    for row in leaderboard
                ]
            )
            st.markdown(
                f"""
                <div class="insight-box">
                    <div class="metric-title">Best Available Model</div>
                    <div style="font-size: 1.45rem; font-weight: 700; color: #ffffff; margin: 0.45rem 0;">
                        {best['model']}
                    </div>
                    <div class="muted-note">Automatically selected from your latest MongoDB model registry entries.</div>
                    <div style="margin-top: 0.9rem; color: #dce6f3;">{rows}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Model metrics are not available yet.")

    with right:
        alerts = alert_days(predictions)
        avg_prediction = float(np.mean(predictions))
        trend_text = "Worsening" if current_aqi is not None and predictions[0] > current_aqi else "Stable / Improving"
        alerts_text = "".join(
            [
                f"<div style='margin-bottom:0.4rem;'><b>Day {alert['day']}</b>: {alert['aqi']:.1f} AQI ({alert['level']})</div>"
                for alert in alerts
            ]
        ) or "<div>No unhealthy forecast days detected.</div>"
        st.markdown(
            f"""
            <div class="insight-box">
                <div class="metric-title">Forecast Analysis</div>
                <div style="font-size: 1rem; color: #dce6f3; line-height: 1.7;">
                    <div><b>3-Day Average:</b> {avg_prediction:.1f}</div>
                    <div><b>3-Day Peak:</b> {max(predictions):.1f}</div>
                    <div><b>Trend Signal:</b> {trend_text}</div>
                    <div style="margin-top: 0.75rem;"><b>Alerts:</b></div>
                    {alerts_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_explainability() -> None:
    images = explainability_images()
    left, right = st.columns(2)

    with left:
        st.markdown("### Global Feature Importance")
        if "shap_summary_bar" in images:
            st.image(str(images["shap_summary_bar"]), use_container_width=True)
        else:
            st.info("Run `python src/models/explain_model.py` to generate the SHAP summary plot.")

    with right:
        st.markdown("### Feature Impact Direction")
        if "shap_impact" in images:
            st.image(str(images["shap_impact"]), use_container_width=True)
        else:
            st.info("Run `python src/models/explain_model.py` to generate the SHAP impact plot.")


def render_data_insights_tab(
    history_df: pd.DataFrame,
    leaderboard: list[dict[str, object]],
    current_aqi: float | None,
    predictions: list[float],
    model_forecasts: dict[str, list[float]],
) -> None:
    left, right = st.columns(2)
    history_column = "aqi_display" if "aqi_display" in history_df.columns else "us_aqi"
    history_values = pd.to_numeric(history_df.get(history_column, pd.Series(dtype=float)), errors="coerce").dropna()
    forecast_values = pd.Series(predictions, dtype=float).dropna()

    with left:
        st.markdown("### Historical Overview")
        if history_values.empty:
            st.info("Historical AQI data is not available yet.")
        else:
            trend = "Worsening" if current_aqi is not None and history_values.iloc[-1] > history_values.mean() else "Stable / Improving"
            st.markdown(
                f"""
                <div class="insight-box">
                    <div><b>Average AQI:</b> {history_values.mean():.1f}</div>
                    <div><b>Peak AQI:</b> {history_values.max():.1f}</div>
                    <div><b>Latest vs Average:</b> {trend}</div>
                    <div><b>Observed Days:</b> {len(history_values)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("### Forecast Analysis")
        if forecast_values.empty:
            st.info("Forecast data is not available yet.")
        else:
            outlook = aqi_level_and_color(float(forecast_values.mean()))[0]
            st.markdown(
                f"""
                <div class="insight-box">
                    <div><b>3-Day Average:</b> {forecast_values.mean():.1f}</div>
                    <div><b>Projected Peak:</b> {forecast_values.max():.1f}</div>
                    <div><b>Projected Minimum:</b> {forecast_values.min():.1f}</div>
                    <div><b>Forecast Outlook:</b> {outlook}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Model Performance Benchmarks")
    if leaderboard:
        benchmark_df = pd.DataFrame(
            [
                {
                    "Model": row["model"],
                    "RMSE": round(float(row["rmse"]), 2),
                    "MAE": round(float(row["mae"]), 2),
                    "R2": round(float(row["r2"]), 3),
                    "Status": "Active" if index == 0 else "Candidate",
                }
                for index, row in enumerate(leaderboard)
            ]
        )
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
    else:
        st.info("Benchmark metrics are not available yet.")

    st.markdown("### Forecast Comparison")
    render_model_comparison_chart(model_forecasts)


def render_health_guidance_tab(predictions: list[float], current_aqi: float | None) -> None:
    current_level, _ = aqi_level_and_color(current_aqi)
    alerts = alert_days(predictions)
    st.markdown("### AQI Health Guide")
    st.markdown(
        """
        - `0-50`: Air quality is satisfactory and outdoor activity is generally safe.
        - `51-100`: Sensitive groups should reduce prolonged outdoor exertion.
        - `101-150`: Children, seniors, and sensitive groups should limit time outdoors.
        - `151-200`: Outdoor exposure should be reduced and a mask is recommended.
        - `201+`: Avoid prolonged outdoor activity and stay indoors when possible.
        """
    )
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="metric-title">Current Status</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #ffffff;">{current_level}</div>
            <div class="muted-note" style="margin-top: 0.5rem;">
                {'No unhealthy days are forecast right now.' if not alerts else f'{len(alerts)} forecast day(s) currently exceed the unhealthy threshold.'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")
    inject_styles()

    try:
        latest_row = get_latest_feature_row()
    except Exception as exc:
        st.error(f"Unable to load feature data: {exc}")
        st.stop()

    available_models = get_available_model_names()
    model_options = get_available_model_options()

    with st.sidebar:
        st.header("Forecast Controls")
        model_name = st.selectbox("Model Strategy", model_options, index=0)
        if st.button("Generate Forecast", use_container_width=True):
            clear_caches()
            st.rerun()
        if st.button("Reload cached data", use_container_width=True):
            clear_caches()
            st.rerun()

        st.markdown("### Pollutant Inputs")
        overrides = {}
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
                feature_name.replace("_", ".").upper(),
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(current_value),
            )

        st.markdown("---")
        st.markdown("### About")
        st.write("Real-time air quality monitoring and 3-day AQI forecasting for Karachi with automated MongoDB-backed model delivery.")
        st.markdown("---")
        st.markdown("### System Status")
        st.info("Active")
        st.markdown("---")
        st.markdown("### Technical Specs")
        st.caption("Forecast horizon: next 3 days")
        st.caption(f"Models: {', '.join(available_models) if available_models else 'Ridge Regression, Random Forest, XGBoost'}")
        st.caption(f"Timezone: {TIMEZONE}")
        st.markdown("---")
        st.markdown("### Data Sources")
        st.caption("Primary history: OpenWeather Air Pollution API")
        st.caption("Live fallback snapshot: Open-Meteo Air Quality API")
        if available_models and len(available_models) < 3:
            st.warning(
                "Some models are hidden because they are not available in the current deployment. "
                f"Available now: {', '.join(available_models)}."
            )

    selected_model = None if model_name == "Best Available" else model_name
    try:
        forecast = predict_next_days(selected_model, overrides)
    except Exception as exc:
        st.error(f"Unable to generate forecast for the selected model: {exc}")
        st.stop()
    predictions = forecast["predictions"]
    forecast_dates = forecast["forecast_dates"]
    current_aqi, current_label = get_current_aqi()
    level, color = aqi_level_and_color(current_aqi)
    leaderboard = forecast["leaderboard"] or get_model_leaderboard()
    best_model = forecast["model_name"] or get_default_model_name()
    history_df = get_recent_daily_history()
    available_model_names = [str(row["model"]) for row in leaderboard][:3] if leaderboard else []
    model_forecasts: dict[str, list[float]] = {}
    for candidate_model in available_model_names:
        try:
            candidate_forecast = predict_next_days(candidate_model, overrides)
            model_forecasts[candidate_model] = candidate_forecast["predictions"]
        except Exception:
            continue

    st.title("Air Quality Forecast")
    st.markdown(f"**{DEFAULT_CITY}** · Real AQI scale (0-500) · MongoDB-backed automation · Timezone: `{TIMEZONE}`")

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label" style="color:{color};">{current_label}</div>
            <div class="hero-aqi" style="color:{color};">{f"{current_aqi:.1f}" if current_aqi is not None else "—"}</div>
            <div class="hero-label" style="color:{color}; opacity:0.92;">{level}</div>
            <div style="margin-top: 0.9rem; color: #aeb4be;">Latest record: {format_timestamp(latest_row.get("timestamp"))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="precautions-box" style="border-left-color:{color};">
            <div class="metric-title">Health Guidance</div>
            <div style="font-size: 1.05rem; color: #e6edf3;">{health_recommendation(current_aqi)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("3-Day Forecast Outlook")
    render_forecast_cards(predictions, forecast_dates)

    st.markdown("---")
    st.subheader("72-Hour Forecast Trend")
    render_forecast_chart(predictions)

    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            <div class="insight-box">
                <div class="metric-title">Production Model</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #ffffff;">{best_model}</div>
                <div class="muted-note" style="margin-top: 0.5rem;">The dashboard uses the strongest available model from your latest registry metrics unless you override it in the sidebar.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        rmse = leaderboard[0]["rmse"] if leaderboard else None
        rmse_text = f"Best RMSE {rmse:.2f}" if rmse is not None and rmse < 999999 else "Metrics unavailable"
        st.markdown(
            f"""
            <div class="insight-box">
                <div class="metric-title">Performance Signal</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #ffffff;">{rmse_text}</div>
                <div class="muted-note" style="margin-top: 0.5rem;">Model comparison is kept close to the reference project, but now uses your MongoDB-driven deployment flow.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    tab_report, tab_history, tab_data, tab_health, tab_explain = st.tabs(
        ["Detailed Report", "Historical Overview", "Data Insights", "Health Guidance", "Explainability"]
    )

    with tab_report:
        curve_df = build_forecast_curve(predictions)
        report_df = pd.DataFrame(
            {
                "Day": ["Tomorrow", "Day After", "Next Day"],
                "Date": forecast_dates,
                "Predicted AQI": predictions,
                "Category": [aqi_level_and_color(value)[0] for value in predictions],
                "Health Recommendation": [health_recommendation(value) for value in predictions],
            }
        )
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download forecast CSV",
            report_df.to_csv(index=False).encode("utf-8"),
            "aqi_forecast_report.csv",
            "text/csv",
        )
        if not curve_df.empty:
            hourly_export = curve_df.rename(columns={"timestamp": "Forecast Time", "aqi_predicted": "Predicted AQI", "day_label": "Forecast Day"})
            st.download_button(
                "Download 72-hour curve CSV",
                hourly_export.to_csv(index=False).encode("utf-8"),
                "aqi_forecast_curve.csv",
                "text/csv",
            )

    with tab_history:
        render_history_tab(history_df)

    with tab_data:
        render_data_insights_tab(history_df, leaderboard, current_aqi, predictions, model_forecasts)

    with tab_health:
        render_health_guidance_tab(predictions, current_aqi)

    with tab_explain:
        render_insights_tab(leaderboard, predictions, current_aqi)
        render_explainability()

    st.markdown("---")
    st.caption(f"Rendered on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using the latest available model and feature data.")


if __name__ == "__main__":
    main()
