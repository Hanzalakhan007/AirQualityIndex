"""Generate balanced EDA charts for AQI reporting."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPORTS_DIR = Path("reports")
RAW_DATA_PATH = Path("data/raw/historical_aqi.csv")
REPORTS_DIR.mkdir(exist_ok=True)


AQI_BANDS = [
    ("Good", 0, 50, "#2ecc71"),
    ("Moderate", 51, 100, "#f1c40f"),
    ("Unhealthy for Sensitive Groups", 101, 150, "#e67e22"),
    ("Unhealthy", 151, 200, "#e74c3c"),
    ("Very Unhealthy", 201, 300, "#8e44ad"),
    ("Hazardous", 301, 500, "#7f1d1d"),
]
POLLUTANTS = ["pm2_5", "pm10", "o3", "no2", "so2", "co", "nh3", "no"]
POLLUTANT_LABELS = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "o3": "Ozone",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",
    "nh3": "NH3",
    "no": "NO",
}
WEEKDAY_LABELS = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}


def aqi_category(value: float) -> str:
    for label, low, high, _color in AQI_BANDS:
        if low <= value <= high:
            return label
    return AQI_BANDS[-1][0]


def load_data() -> pd.DataFrame:
    dataframe = pd.read_csv(RAW_DATA_PATH)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    dataframe["aqi_display"] = pd.to_numeric(
        dataframe["us_aqi"] if "us_aqi" in dataframe.columns else dataframe["aqi"],
        errors="coerce",
    )
    dataframe = dataframe.dropna(subset=["aqi_display"]).sort_values("timestamp").reset_index(drop=True)
    dataframe["aqi_category"] = dataframe["aqi_display"].apply(aqi_category)
    return dataframe


def save_aqi_trend_chart(dataframe: pd.DataFrame) -> None:
    daily = (
        dataframe.set_index("timestamp")["aqi_display"]
        .resample("D")
        .mean()
        .reset_index(name="daily_aqi")
    )
    daily["rolling_7d"] = daily["daily_aqi"].rolling(window=7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(15, 6))
    for label, low, high, color in AQI_BANDS:
        ax.axhspan(low, high, color=color, alpha=0.08)

    ax.plot(daily["timestamp"], daily["daily_aqi"], color="#1f77b4", linewidth=2, label="Daily average AQI")
    ax.plot(daily["timestamp"], daily["rolling_7d"], color="#0f172a", linewidth=2.5, label="7-day average")

    ax.set_title("Karachi AQI Trend", fontsize=18, pad=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "aqi_time_series.png", dpi=160)
    plt.close(fig)


def save_category_mix_chart(dataframe: pd.DataFrame) -> pd.DataFrame:
    ordered_labels = [label for label, *_rest in AQI_BANDS]
    colors = {label: color for label, *_rest, color in AQI_BANDS}

    mix = (
        dataframe["aqi_category"]
        .value_counts(normalize=True)
        .reindex(ordered_labels, fill_value=0)
        .mul(100)
        .reset_index()
    )
    mix.columns = ["category", "share_pct"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=mix,
        x="share_pct",
        y="category",
        hue="category",
        palette=[colors[label] for label in mix["category"]],
        legend=False,
        ax=ax,
    )
    ax.set_title("How Often Each AQI Band Appears", fontsize=18, pad=14)
    ax.set_xlabel("Share of observed hours (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(5, mix["share_pct"].max() * 1.15))
    for index, value in enumerate(mix["share_pct"]):
        ax.text(value + 0.4, index, f"{value:.1f}%", va="center")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "aqi_distribution.png", dpi=160)
    plt.close(fig)
    return mix


def save_pm25_relationship_chart(dataframe: pd.DataFrame) -> pd.DataFrame:
    corr_frame = dataframe[["aqi_display"] + POLLUTANTS].corr(numeric_only=True)
    drivers = (
        corr_frame["aqi_display"]
        .drop(labels=["aqi_display"])
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .rename(index=POLLUTANT_LABELS)
        .reset_index()
    )
    drivers.columns = ["pollutant", "correlation"]

    sample_size = min(len(dataframe), 2500)
    sample = dataframe[["pm2_5", "aqi_display"]].dropna().sample(sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.regplot(
        data=sample,
        x="pm2_5",
        y="aqi_display",
        scatter_kws={"alpha": 0.18, "color": "#5dade2", "s": 24},
        line_kws={"color": "#d62828", "linewidth": 2.5},
        ax=ax,
    )
    ax.set_title("PM2.5 and AQI Move Together", fontsize=18, pad=14)
    ax.set_xlabel("PM2.5 concentration (ug/m3)")
    ax.set_ylabel("AQI")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pm25_aqi_relationship.png", dpi=160)
    plt.close(fig)

    return drivers


def save_aqi_pattern_heatmap(dataframe: pd.DataFrame) -> pd.DataFrame:
    pattern_frame = dataframe.copy()
    pattern_frame["weekday"] = pattern_frame["timestamp"].dt.dayofweek
    pattern_frame["hour"] = pattern_frame["timestamp"].dt.hour

    heatmap_data = (
        pattern_frame.groupby(["weekday", "hour"])["aqi_display"]
        .mean()
        .reset_index()
        .pivot(index="weekday", columns="hour", values="aqi_display")
        .reindex(index=list(WEEKDAY_LABELS.keys()))
    )
    heatmap_data.index = [WEEKDAY_LABELS[index] for index in heatmap_data.index]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        linewidths=0.2,
        cbar_kws={"label": "Average AQI"},
        ax=ax,
    )
    ax.set_title("When AQI Usually Gets Better or Worse", fontsize=18, pad=14)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)
    return heatmap_data


def save_summary(
    dataframe: pd.DataFrame,
    category_mix: pd.DataFrame,
    drivers: pd.DataFrame,
    aqi_pattern: pd.DataFrame,
) -> None:
    latest_day = dataframe["timestamp"].max().strftime("%Y-%m-%d")
    daily = dataframe.set_index("timestamp")["aqi_display"].resample("D").mean()
    latest_daily_aqi = float(daily.iloc[-1])
    average_daily_aqi = float(daily.mean())
    peak_daily_aqi = float(daily.max())
    most_common_band = str(category_mix.sort_values("share_pct", ascending=False).iloc[0]["category"])
    lead_pollutant = str(drivers.iloc[0]["pollutant"])
    strongest_corr = float(drivers.iloc[0]["correlation"])
    busiest_slot = aqi_pattern.stack().idxmax()
    cleanest_slot = aqi_pattern.stack().idxmin()

    summary = f"""# Karachi AQI EDA Summary

- Latest day in dataset: {latest_day}
- Latest daily average AQI: {latest_daily_aqi:.1f}
- Average daily AQI across the dataset: {average_daily_aqi:.1f}
- Highest daily average AQI observed: {peak_daily_aqi:.1f}
- Most common AQI band: {most_common_band}
- Pollutant most closely linked with AQI changes: {lead_pollutant} (correlation {strongest_corr:.2f})
- Highest average AQI time slot: {busiest_slot[0]} around {int(busiest_slot[1]):02d}:00
- Lowest average AQI time slot: {cleanest_slot[0]} around {int(cleanest_slot[1]):02d}:00

Chart mix:
- Simple: AQI trend and AQI band breakdown
- More analytical: PM2.5-to-AQI relationship and AQI weekday-hour pattern
"""
    (REPORTS_DIR / "eda_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid")
    print("Starting Exploratory Data Analysis (EDA)...")

    dataframe = load_data()
    print(f"Data Loaded! Shape: {dataframe.shape}")

    missing = dataframe.isnull().sum()
    print("\nMissing Values per Feature:")
    print(missing)

    save_aqi_trend_chart(dataframe)
    print("Saved AQI trend chart.")

    category_mix = save_category_mix_chart(dataframe)
    print("Saved AQI category mix chart.")

    drivers = save_pm25_relationship_chart(dataframe)
    print("Saved PM2.5 relationship chart.")

    aqi_pattern = save_aqi_pattern_heatmap(dataframe)
    print("Saved AQI pattern heatmap.")

    save_summary(dataframe, category_mix, drivers, aqi_pattern)
    print("Saved plain-language EDA summary.")

    print("\nEDA Completed! Check the 'reports/' folder for the new charts and summary.")


if __name__ == "__main__":
    main()
