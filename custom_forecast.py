"""Extra Streamlit page: Custom Demand Prediction.

Works alongside existing forecast_dashboard.py and model-testing code.

Two supported input formats:
* M4 analysis-ready observations: the exact 41 predictor columns plus Zone and Hour_End_EST.
* Simple hourly history: zone demand, Ontario demand, market demand and zone weather.

The raw-history feature builder is an experimental extension. Verify its
hour/date and preprocessing conventions against the original M4 pipeline
before treating future predictions as operational forecasts.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
LAGS = (24, 48, 168)
ZONE_NAMES = (
    "Northwest", "Northeast", "Ottawa", "East", "Toronto",
    "Essa", "Bruce", "Southwest", "Niagara", "West"
)
RAW_REQUIRED = [
    "Zone", "Hour_End_EST", "Zone_Demand_MW", "Ontario_Demand_MW",
    "Market_Demand_MW", "Temperature_C", "Humidity_Pct", "Wind_kmh"
]


@st.cache_resource
def load_saved_random_forest():
    """Load only locally packaged, trusted model artifacts."""
    model = joblib.load(ARTIFACTS / "random_forest_41_compressed.joblib")
    with (ARTIFACTS / "feature_names.json").open(encoding="utf-8") as fh:
        feature_names = json.load(fh)["feature_names"]
    if len(feature_names) != 41 or len(set(feature_names)) != 41:
        raise ValueError("The saved feature list must contain 41 unique columns.")
    if model.n_features_in_ != len(feature_names):
        raise ValueError("Saved model does not match the 41-feature input list.")
    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != feature_names:
            raise ValueError("Model training feature order differs from feature_names.json.")
    return model, feature_names


def _read_dates(data):
    data = data.copy()
    data["Hour_End_EST"] = pd.to_datetime(data["Hour_End_EST"], errors="coerce")
    if data["Hour_End_EST"].isna().any():
        raise ValueError("Some Hour_End_EST values are invalid dates/times.")
    if data["Hour_End_EST"].dt.tz is not None:
        raise ValueError("Use the same timezone-naive EST timestamp convention as M4.")
    data["Zone"] = data["Zone"].astype(str).str.strip()
    if not data["Zone"].isin(ZONE_NAMES).all():
        raise ValueError("Only the 10 IESO zones included in the trained model are supported.")
    if data.duplicated(["Zone", "Hour_End_EST"]).any():
        raise ValueError("Duplicate zone + hour records found; one row per zone-hour is required.")
    return data


def _make_raw_features(raw_zone, target_hours, zone, feature_names):
    """Generate lagged inputs from *prior* observations only.

    No weather imputation is attempted: the source values at the three lag
    timestamps must exist. This avoids inventing the M4 training imputation.
    """
    lookup = raw_zone.set_index("Hour_End_EST")
    records = []
    missing_hours = []
    for target in target_hours:
        past_times = [target - pd.Timedelta(hours=h) for h in LAGS]
        if any(ts not in lookup.index for ts in past_times):
            missing_hours.append(target)
            continue

        past = [lookup.loc[ts] for ts in past_times]
        numeric_cols = RAW_REQUIRED[2:]
        if any(pd.isna(row[numeric_cols]).any() for row in past):
            missing_hours.append(target)
            continue

        # In hourly-ending conventions, 00:00 represents the previous day's
        # 24th hour. Confirm this encoding against your original M4 pipeline.
        calendar_day = target - pd.Timedelta(microseconds=1)
        hour_num = 24 if target.hour == 0 else target.hour
        record = {
            "Hour_End_EST": target,
            "Zone": zone,
            "Hour": hour_num,
            "DayOfWeek": calendar_day.dayofweek,
            "Month": calendar_day.month,
            "IsWeekend": int(calendar_day.dayofweek >= 5),
        }
        for lag, old in zip(LAGS, past):
            record[f"Demand_Lag_{lag}h"] = old["Zone_Demand_MW"]
            record[f"Ontario_Lag_{lag}h"] = old["Ontario_Demand_MW"]
            record[f"Market_Lag_{lag}h"] = old["Market_Demand_MW"]
            for prefix, src in (
                ("Temp", "Temperature_C"),
                ("Humidity", "Humidity_Pct"),
                ("Wind", "Wind_kmh"),
            ):
                record[f"{prefix}_Lag_{lag}h"] = old[src]
                record[f"{prefix}_Lag_{lag}h_Missing"] = 0
        for zone_name in ZONE_NAMES:
            record[f"Zone_{zone_name}"] = int(zone_name == zone)
        records.append(record)

    if missing_hours:
        preview = ", ".join(ts.strftime("%m-%d %H:%M") for ts in missing_hours[:5])
        raise ValueError(
            f"Cannot generate the full 24-hour day: {len(missing_hours)} target "
            f"hours lack complete observations at 24, 48 or 168 hours prior "
            f"(first missing: {preview}). Upload more complete history."
        )
    feature_frame = pd.DataFrame(records)
    missing_features = [f for f in feature_names if f not in feature_frame.columns]
    if missing_features:
        raise ValueError(f"Cannot build expected model feature(s): {missing_features}")
    x = feature_frame[feature_names].astype("float32")
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Non-finite model inputs found.")
    return feature_frame, x


def _plot_output(results, zone, actual_available):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["Hour_End_EST"], y=results["Predicted_Demand_MW"],
        name="Random Forest prediction", mode="lines+markers",
        line=dict(color="#2563EB", width=3),
    ))
    if actual_available:
        fig.add_trace(go.Scatter(
            x=results["Hour_End_EST"], y=results["Actual_Demand_MW"],
            name="Actual electricity demand", mode="lines+markers",
            line=dict(color="#EA580C", width=2),
        ))
    fig.update_layout(
        title=f"{zone} — 24-hour demand prediction",
        xaxis_title="Hour ending (EST)",
        yaxis_title="Electricity demand (MW)",
        hovermode="x unified", font=dict(color="#0B2A50"),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=15, r=15, t=65, b=65),
    )
    fig.update_yaxes(gridcolor="#E6EFFA", zeroline=False)
    fig.update_xaxes(gridcolor="#E6EFFA")
    st.plotly_chart(fig, use_container_width=True)


def _show_results(model, x, timestamps, zone, actual=None):
    predictions = model.predict(x.astype("float32"))
    results = pd.DataFrame({
        "Hour_End_EST": pd.to_datetime(timestamps).to_numpy(),
        "Zone": zone,
        "Predicted_Demand_MW": predictions,
    })
    actual_available = actual is not None and len(actual) == len(results)
    if actual_available:
        vals = pd.to_numeric(pd.Series(actual).reset_index(drop=True), errors="coerce")
        if vals.notna().all():
            results["Actual_Demand_MW"] = vals
        else:
            actual_available = False

    a, b = st.columns(2)
    a.metric("Predicted hourly observations", len(results))
    b.metric("Average predicted demand", f"{np.mean(predictions):,.1f} MW")
    if actual_available:
        error = (results["Actual_Demand_MW"] - results["Predicted_Demand_MW"]).abs()
        denom = results["Actual_Demand_MW"].abs().sum()
        a, b = st.columns(2)
        a.metric("MAE for selected sample", f"{error.mean():,.2f} MW")
        b.metric("WAPE for selected sample", f"{100 * error.sum() / denom:.2f}%" if denom else "N/A")
    _plot_output(results, zone, actual_available)
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.download_button(
        "Download predicted values (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"{zone.lower()}_custom_demand_predictions.csv",
        mime="text/csv",
    )


def _analysis_ready_mode(model, feature_names):
    st.write("Upload your M4 analysis-ready CSV containing the 41 model features, Zone and Hour_End_EST.")
    st.caption("This verified input format uses the same feature definitions as the model's training dataset.")
    file = st.file_uploader("Upload analysis-ready CSV", type="csv", key="custom_ready_csv")
    if file is None:
        st.info("Start with an M4 sample or export a subset from your M5 notebook.")
        return
    try:
        raw = _read_dates(pd.read_csv(file))
        missing = [c for c in feature_names if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing original training features: {missing}")
        zone = st.selectbox("Ontario zone", sorted(raw["Zone"].unique()), key="custom_ready_zone")
        subset = raw[raw["Zone"] == zone].copy()
        dates = sorted(subset["Hour_End_EST"].dt.date.unique())
        day = st.selectbox("Forecast target date", dates, index=len(dates)-1, key="custom_ready_date")
        subset = subset[subset["Hour_End_EST"].dt.date == day].sort_values("Hour_End_EST")
        if len(subset) != 24 or subset["Hour_End_EST"].dt.hour.nunique() != 24:
            st.warning(f"The selected day has {len(subset)} rows; choose a complete 24-hour day.")
            return
        x = subset[feature_names].apply(pd.to_numeric, errors="raise").astype("float32")
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError("Input features have missing or non-finite values.")
        st.success("24 hours and 41 exact training features verified.")
        request_id = (file.name, file.size, zone, str(day))
        if st.button("Generate 24-hour predictions", type="primary", key="custom_ready_go"):
            st.session_state["custom_ready_request"] = request_id
        if st.session_state.get("custom_ready_request") == request_id:
            actual_col = next((c for c in ("Actual_Demand_MW", "Demand_MW", "Actual_MW") if c in subset), None)
            actual = subset[actual_col].to_numpy() if actual_col else None
            _show_results(model, x, subset["Hour_End_EST"], zone, actual)
    except Exception as exc:
        st.error(f"Unable to process this file: {exc}")


def _raw_history_mode(model, feature_names):
    st.write("Upload hourly historical demand and zone weather. The app builds the 41 lagged model features.")
    st.warning(
        "Experimental feature-preparation workflow: use the same data definitions, "
        "units, zone-to-weather mapping and hourly-ending conventions as M4. "
        "Validate predictions against the original M4 pipeline before relying on them."
    )
    with st.expander("Required raw CSV columns and format"):
        st.code(",".join(RAW_REQUIRED), language="text")
        st.caption(
            "Each row is one zone-hour. Zone_Demand_MW is that zone's demand; "
            "Ontario_Demand_MW and Market_Demand_MW are the same provincial/market "
            "series used in M4. Weather should be observed temperature (°C), "
            "relative humidity (%), and wind (same units as M4). "
            "Hour_End_EST must follow your training data's timestamp convention."
        )
        st.download_button(
            "Download empty CSV template",
            data=(",".join(RAW_REQUIRED)+"\n").encode("utf-8"),
            file_name="hourly_history_template.csv", mime="text/csv",
            key="custom_template",
        )
    file = st.file_uploader("Upload hourly history CSV", type="csv", key="custom_raw_csv")
    if file is None:
        return
    try:
        df = pd.read_csv(file)
        missing = [c for c in RAW_REQUIRED if c not in df]
        if missing:
            raise ValueError(f"Missing raw columns: {missing}")
        df = _read_dates(df)
        for col in RAW_REQUIRED[2:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if ((df["Humidity_Pct"].dropna() < 0) | (df["Humidity_Pct"].dropna() > 100)).any():
            raise ValueError("Humidity_Pct must be between 0 and 100.")
        zone = st.selectbox("Ontario zone", sorted(df["Zone"].unique()), key="custom_raw_zone")
        zone_df = df[df["Zone"] == zone].copy().sort_values("Hour_End_EST")
        min_date = (zone_df["Hour_End_EST"].min() + pd.Timedelta(hours=168)).date()
        max_date = (zone_df["Hour_End_EST"].max() + pd.Timedelta(hours=24)).date()
        if min_date > max_date:
            raise ValueError("At least one week of historical data is needed.")
        date = st.date_input(
            "Prediction target date",
            value=max_date, min_value=min_date, max_value=max_date,
            key="custom_raw_date",
        )
        times = pd.date_range(pd.Timestamp(date), periods=24, freq="h")
        prepared, x = _make_raw_features(zone_df, times, zone, feature_names)
        st.success("Created the 41 required inputs for all 24 target hours.")
        st.caption(
            "Each row uses only observations from 24, 48 and 168 hours before its target hour. "
            "A complete day's predictions represent separate hourly forecast origins, "
            "not a single forecast issued at one moment."
        )
        if date.year > 2025:
            st.warning("This prediction date is beyond the 2025 test period; performance is unvalidated.")
        with st.expander("Preview generated model features"):
            st.dataframe(prepared.head(), hide_index=True, use_container_width=True)
        request_id = (file.name, file.size, zone, str(date))
        if st.button("Generate 24-hour predictions", type="primary", key="custom_raw_go"):
            st.session_state["custom_raw_request"] = request_id
        if st.session_state.get("custom_raw_request") == request_id:
            observed = zone_df.set_index("Hour_End_EST")["Zone_Demand_MW"]
            actual = observed.reindex(times)
            if actual.notna().all():
                _show_results(model, x, times, zone, actual.to_numpy())
            else:
                _show_results(model, x, times, zone)
    except Exception as exc:
        st.error(f"Unable to generate predictions: {exc}")


def render_custom_forecast():
    st.header("Custom Demand Prediction")
    st.write(
        "Use the existing 41-feature Random Forest to generate hourly predictions "
        "from uploaded historical observations. This is a prototype, not a "
        "validated live forecasting service."
    )
    try:
        model, feature_names = load_saved_random_forest()
    except Exception as exc:
        st.error(f"Model artifacts could not be loaded: {exc}")
        return

    method = st.radio(
        "Choose input format",
        ("M4 analysis-ready data (recommended first test)",
         "Simple hourly history (experimental automatic feature preparation)"),
        key="custom_method",
    )
    if method.startswith("M4"):
        _analysis_ready_mode(model, feature_names)
    else:
        _raw_history_mode(model, feature_names)
