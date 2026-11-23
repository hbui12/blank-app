"""Interactive Streamlit view of historical 2025 Random Forest predictions.

Place this beside streamlit_app.py, and keep the supplied merged CSV at
  data/M5_Forecasts_with_Actual_Demand_2025.csv

The original 41-feature model testing screen in streamlit_app.py is unchanged.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = (
    Path(__file__).resolve().parent
    / "data" / "M5_Forecasts_with_Actual_Demand_2025.csv"
)


@st.cache_data(show_spinner="Loading 2025 actual demand and predictions...")
def load_forecasts():
    df = pd.read_csv(DATA_PATH, parse_dates=["Hour_End_EST", "Forecast_Origin_EST"])
    required = {
        "Zone", "Hour_End_EST", "Forecast_Origin_EST", "Forecast_MW",
        "Actual_MW", "Error_Margin_MW", "Lower_Bound_MW", "Upper_Bound_MW",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    for name in ["Forecast_MW", "Actual_MW", "Lower_Bound_MW", "Upper_Bound_MW", "Error_Margin_MW"]:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.loc[df["Hour_End_EST"].dt.year == 2025].copy()
    df = df.dropna(subset=["Hour_End_EST", "Zone", "Forecast_MW", "Actual_MW"])
    df["Error_MW"] = df["Forecast_MW"] - df["Actual_MW"]
    df["Absolute_Error_MW"] = df["Error_MW"].abs()
    return df.sort_values(["Zone", "Hour_End_EST"]).reset_index(drop=True)


def get_mae_wape(data):
    if data.empty:
        return float("nan"), float("nan")
    mae = data["Absolute_Error_MW"].mean()
    denominator = data["Actual_MW"].abs().sum()
    wape = 100 * data["Absolute_Error_MW"].sum() / denominator if denominator else float("nan")
    return mae, wape


def render_forecast_dashboard():
    st.subheader("2025 Regional Electricity Demand: Actual vs. Predicted")
    st.caption(
        "Explore saved 24-hour-ahead Random Forest predictions and observed demand. "
        "Changing the zone, date, or hour filters historical results; it does not rerun the model."
    )

    if not DATA_PATH.is_file():
        st.error("Missing data/M5_Forecasts_with_Actual_Demand_2025.csv in the deployed repository.")
        return
    try:
        df = load_forecasts()
    except Exception as error:
        st.error(f"Could not load the combined forecast dataset: {error}")
        return
    if df.empty:
        st.warning("No matched 2025 actual and predicted records were found.")
        return

    st.sidebar.header("Explore 2025 Predictions")
    zones = sorted(df["Zone"].unique().tolist())
    zone = st.sidebar.selectbox(
        "Electricity zone", zones,
        index=zones.index("Toronto") if "Toronto" in zones else 0,
        key="saved2025_zone",
    )
    zone_df = df.loc[df["Zone"] == zone].copy()
    available_days = zone_df["Hour_End_EST"].dt.date
    day = st.sidebar.date_input(
        "Forecast date (2025)",
        value=available_days.max(),
        min_value=available_days.min(),
        max_value=available_days.max(),
        key="saved2025_day",
    )
    day_df = zone_df.loc[available_days == day].copy()
    if day_df.empty:
        st.info("There are no matched observations for this date. Choose another date.")
        return
    
# Available hours for the selected date
hours = (
    day_df["Hour_End_EST"]
    .dropna()
    .sort_values()
    .drop_duplicates()
    .tolist()
)

if not hours:
    st.warning("No predictions are available for this date.")
    return

# Reset hour selection when the zone or date changes
hour = st.sidebar.selectbox(
    "Hour ending (EST)",
    options=hours,
    format_func=lambda value: value.strftime("%I:%M %p"),
    key=f"saved2025_hour_{zone}_{day.isoformat()}"
)

# Find the matching prediction
matching_rows = day_df.loc[
    day_df["Hour_End_EST"] == hour
]

if matching_rows.empty:
    st.warning(
        "No forecast is available for the selected hour. "
        "Please choose another hour."
    )
    return

point = matching_rows.iloc[0]


    st.markdown(
        f"**Zone:** {zone}  |  **Date:** {day:%B %d, %Y}  |  "
        f"**Hour ending:** {hour:%I:%M %p} EST"
    )
    a, b, c, d = st.columns(4)
    a.metric("Predicted demand", f"{point['Forecast_MW']:,.1f} MW")
    b.metric("Actual demand", f"{point['Actual_MW']:,.1f} MW")
    c.metric("Prediction error", f"{point['Error_MW']:+,.1f} MW")
    _, day_wape = get_mae_wape(day_df)
    d.metric("Selected-day WAPE", f"{day_wape:.2f}%")
    st.caption(
        "Prediction error = predicted minus actual. Positive means overprediction; "
        "negative means underprediction. WAPE sums absolute errors divided by "
        "total actual demand for the selected date."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=day_df["Hour_End_EST"], y=day_df["Lower_Bound_MW"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=day_df["Hour_End_EST"], y=day_df["Upper_Bound_MW"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(56,189,248,0.17)", name="Preliminary forecast range",
        hovertemplate="Upper saved bound: %{y:,.1f} MW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=day_df["Hour_End_EST"], y=day_df["Forecast_MW"],
        mode="lines+markers", name="Random Forest prediction",
        line=dict(color="#2563EB", width=3), marker=dict(size=5),
        hovertemplate="Predicted: %{y:,.1f} MW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=day_df["Hour_End_EST"], y=day_df["Actual_MW"],
        mode="lines+markers", name="Actual electricity demand",
        line=dict(color="#0D9488", width=2.5), marker=dict(size=5),
        hovertemplate="Actual: %{y:,.1f} MW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[hour, hour], y=[point["Forecast_MW"], point["Actual_MW"]],
        mode="markers", name="Selected hour",
        marker=dict(color="#0B2A50", size=12, symbol="circle-open", line=dict(width=2)),
        hovertemplate="Selected hour: %{y:,.1f} MW<extra></extra>",
    ))
    fig.update_layout(
        title=f"{zone} — demand on {day:%B %d, %Y}",
        xaxis_title="Hour ending (EST)", yaxis_title="Electricity demand (MW)",
        hovermode="x unified", legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=65, b=80),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(color="#173354"), title_font=dict(color="#0B2A50", size=19),
        xaxis=dict(showgrid=True, gridcolor="#E6EFFA", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#E6EFFA", zeroline=False),
        hoverlabel=dict(bgcolor="#EDF5FF", font_color="#173354"),
    )
    st.plotly_chart(fig, use_container_width=True)

    day_mae, _ = get_mae_wape(day_df)
    zone_mae, zone_wape = get_mae_wape(zone_df)
    st.markdown("#### Accuracy summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected-day MAE", f"{day_mae:,.1f} MW")
    m2.metric(f"{zone} · 2025 MAE", f"{zone_mae:,.1f} MW")
    m3.metric(f"{zone} · 2025 WAPE", f"{zone_wape:.2f}%")
    st.caption(
        f"Selected forecast was issued at {point['Forecast_Origin_EST']:%Y-%m-%d %I:%M %p} EST. "
        "All accuracy values use matched historical test observations. "
        "Shaded bounds are preliminary and are NOT validated prediction intervals."
    )

    with st.expander("Compare prediction difficulty across all 10 zones"):
        rows = []
        for zone_name, part in df.groupby("Zone"):
            mae, wape = get_mae_wape(part)
            rows.append({"Zone": zone_name, "MAE (MW)": round(mae, 2),
                         "WAPE (%)": round(wape, 2), "Hours evaluated": len(part)})
        comparison = pd.DataFrame(rows).sort_values("WAPE (%)", ascending=False)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.caption("Higher WAPE means a higher error relative to actual demand; MAE depends on regional load scale.")

    with st.expander("View and download hourly results"):
        display = day_df[[
            "Hour_End_EST", "Forecast_Origin_EST", "Actual_MW", "Forecast_MW",
            "Error_MW", "Absolute_Error_MW", "Lower_Bound_MW", "Upper_Bound_MW",
        ]].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button(
            "Download selected day's actual and predicted demand",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name=f"{zone}_{day}_actual_vs_predicted.csv",
            mime="text/csv", key="saved2025_download",
        )
