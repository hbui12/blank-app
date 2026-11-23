"""2025 saved Random Forest electricity-demand forecasts for Streamlit.

Place this file next to streamlit_app.py, and place the supplied CSV at
  data/M5_Combined_Forecasts_2025.csv
Then add `from forecast_dashboard import render_forecast_dashboard` and
`render_forecast_dashboard()` to streamlit_app.py after set_page_config.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "data" / "M5_Combined_Forecasts_2025.csv"


@st.cache_data(show_spinner="Loading saved 2025 forecasts...")
def load_forecasts() -> pd.DataFrame:
    """Read precomputed 24-hour-ahead forecasts (not 41-feature model inputs)."""
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Hour_End_EST", "Forecast_Origin_EST"],
    )
    expected = {
        "Zone", "Hour_End_EST", "Forecast_Origin_EST", "Forecast_MW",
        "Error_Margin_MW", "Lower_Bound_MW", "Upper_Bound_MW",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")
    return df.sort_values(["Zone", "Hour_End_EST"]).reset_index(drop=True)


def render_forecast_dashboard() -> None:
    """Add live-updating sidebar zone/date/hour selections and forecast chart."""
    st.subheader("2025 Regional Electricity Demand Predictions")
    st.caption(
        "Explore saved Random Forest predictions made 24 hours ahead. "
        "The filters display existing forecasts; they do not rerun the model."
    )

    if not DATA_PATH.is_file():
        st.error(
            "Forecast data file not found. Add "
            "data/M5_Combined_Forecasts_2025.csv to your GitHub repository."
        )
        st.code(str(DATA_PATH))
        return

    try:
        df = load_forecasts()
    except Exception as exc:
        st.error(f"Could not load the saved forecasts: {exc}")
        return

    # The file includes a final 2026-01-01 00:00 hour-ending timestamp;
    # keep the interactive demonstration focused on dates in 2025.
    df = df.loc[df["Hour_End_EST"].dt.year == 2025]
    if df.empty:
        st.warning("No 2025 forecast rows were found.")
        return

    st.sidebar.header("Explore 2025 Predictions")
    zones = sorted(df["Zone"].dropna().unique().tolist())
    zone = st.sidebar.selectbox("Electricity zone", zones, key="saved2025_zone")
    zone_df = df.loc[df["Zone"] == zone]

    available_days = zone_df["Hour_End_EST"].dt.date
    chosen_day = st.sidebar.date_input(
        "Forecast date (2025)",
        value=available_days.max(),
        min_value=available_days.min(),
        max_value=available_days.max(),
        key="saved2025_day",
    )
    day_df = zone_df.loc[available_days == chosen_day].copy()
    if day_df.empty:
        st.warning("No predictions for that day. Choose another date.")
        return

    hours = day_df["Hour_End_EST"].tolist()
    chosen_hour = st.sidebar.selectbox(
        "Hour ending (EST)",
        options=hours,
        format_func=lambda hour: hour.strftime("%I:%M %p"),
        key="saved2025_hour",
    )
    point = day_df.loc[day_df["Hour_End_EST"] == chosen_hour].iloc[0]

    st.markdown(f"**Zone:** {zone}  |  **Date:** {chosen_day:%B %d, %Y}  |  **Hour ending:** {chosen_hour:%I:%M %p} EST")

    a, b, c = st.columns(3)
    a.metric("Predicted demand", f"{point['Forecast_MW']:,.1f} MW")
    b.metric("Lower bound", f"{point['Lower_Bound_MW']:,.1f} MW")
    c.metric("Upper bound", f"{point['Upper_Bound_MW']:,.1f} MW")

    st.caption(
        f"Forecast origin: {point['Forecast_Origin_EST']:%Y-%m-%d %I:%M %p} EST "
        f"(24 hours before the predicted hour). "
        f"Preliminary margin for {zone}: ±{point['Error_Margin_MW']:,.1f} MW."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=day_df["Hour_End_EST"], y=day_df["Lower_Bound_MW"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=day_df["Hour_End_EST"], y=day_df["Upper_Bound_MW"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(100, 116, 139, 0.20)", name="Preliminary range",
        hovertemplate="Upper bound: %{y:,.1f} MW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=day_df["Hour_End_EST"], y=day_df["Forecast_MW"],
        mode="lines+markers", name="Predicted demand",
        hovertemplate="Hour ending: %{x|%I:%M %p}<br>Prediction: %{y:,.1f} MW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[chosen_hour], y=[point["Forecast_MW"]],
        mode="markers", marker=dict(size=15, symbol="circle-open", line=dict(width=3)),
        name="Selected hour",
        hovertemplate="Selected: %{y:,.1f} MW<extra></extra>",
    ))
    fig.update_layout(
        title=f"{zone} — hourly forecast for {chosen_day:%B %d, %Y}",
        xaxis_title="Hour ending (EST)", yaxis_title="Electricity demand (MW)",
        hovermode="x unified", legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=65, b=65),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The lower and upper bounds are saved forecast ± error-margin values. "
        "They are preliminary ranges, not validated prediction intervals. "
        "This CSV contains predictions but no actual demand values, so it cannot "
        "show actual-versus-predicted accuracy by itself."
    )

    with st.expander("View hourly results and download this date"):
        display = day_df[[
            "Hour_End_EST", "Forecast_Origin_EST", "Forecast_MW",
            "Lower_Bound_MW", "Upper_Bound_MW",
        ]].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button(
            "Download selected day's predictions",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name=f"{zone}_{chosen_day}_forecasts.csv",
            mime="text/csv",
            key="saved2025_download",
        )
