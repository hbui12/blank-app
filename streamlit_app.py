"""Ontario Regional Demand Intelligence — Streamlit app.

Two views:
1. Explore saved historical forecasts (existing forecast_dashboard.py).
2. Test the trained Random Forest model with an uploaded 41-feature CSV.

The test chart visualizes model outputs. Actual-demand accuracy is only shown
if the uploaded CSV also contains matching observed demand values.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast_dashboard import render_forecast_dashboard
from theme_styles import apply_blue_theme

st.set_page_config(
    page_title="Ontario Regional Demand Intelligence",
    layout="wide",
)
apply_blue_theme()

st.title("Regional Demand Forecasting")
st.write(
    "Interactive decision-support tool for 24-hour-ahead electricity demand "
    "prediction across Ontario's 10 IESO zones."
)

view = st.sidebar.radio(
    "Select view",
    ["Explore 2025 predictions", "Test the Random Forest model"],
)

if view == "Explore 2025 predictions":
    st.info(
        "Explore saved 2025 Random Forest predictions. Selecting a zone, "
        "date, or hour updates the display automatically."
    )
    render_forecast_dashboard()

else:
    st.info(
        "Model testing: upload historical observations containing the 41 "
        "features used to train the Random Forest. A five-row sample checks "
        "that the model runs, but is not sufficient to establish accuracy."
    )

    ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

    @st.cache_resource
    def load_model():
        model_path = ARTIFACTS / "random_forest_41_compressed.joblib"
        features_path = ARTIFACTS / "feature_names.json"
        # Load only the team's trusted model from the repository.
        trained_model = joblib.load(model_path)
        with features_path.open("r", encoding="utf-8") as f:
            feature_names = json.load(f)["feature_names"]
        if len(feature_names) != 41 or trained_model.n_features_in_ != 41:
            raise ValueError("Expected a 41-feature model and feature list.")
        if hasattr(trained_model, "feature_names_in_"):
            if list(trained_model.feature_names_in_) != feature_names:
                raise ValueError("Feature names or order do not match the model.")
        return trained_model, feature_names

    try:
        model, feature_names = load_model()
        st.success("Random Forest model loaded successfully!")
    except Exception as exc:
        st.error(f"Model loading error: {exc}")
        st.stop()

    st.subheader("1. Upload historical input data")
    uploaded_file = st.file_uploader(
        "Upload sample_inputs.csv (41 features)", type=["csv"]
    )

    if uploaded_file is None:
        st.info("Upload your sample_inputs.csv to begin model testing.")
    else:
        data = pd.read_csv(uploaded_file)
        n_rows, n_columns = data.shape
        left, right = st.columns(2)
        left.metric("Input observations", n_rows)
        right.metric("Uploaded columns", n_columns)

        missing = [name for name in feature_names if name not in data.columns]
        if missing:
            st.error(f"Missing required model features: {missing}")
            st.stop()
        extra = [name for name in data.columns if name not in feature_names]
        if extra:
            st.caption(
                "Additional columns are kept for chart labels or evaluation, "
                "but are not used as model inputs: " + ", ".join(extra)
            )

        X_input = data[feature_names].apply(pd.to_numeric, errors="coerce")
        if X_input.empty:
            st.error("The CSV contains no observations.")
            st.stop()
        if not np.isfinite(X_input.to_numpy(dtype=float)).all():
            st.error("Some model inputs are missing or not finite numbers.")
            st.stop()

        with st.expander("Preview the 41 model inputs"):
            st.dataframe(X_input.head(10), use_container_width=True, hide_index=True)
        st.success("All 41 required model features verified!")

        st.subheader("2. Generate and visualize predictions")
        if st.button("Generate prediction", type="primary"):
            try:
                prediction_values = model.predict(X_input)
                st.session_state["latest_test_results"] = pd.DataFrame(
                    {"Observation": np.arange(1, n_rows + 1),
                     "Predicted_Demand_MW": prediction_values}
                )
                st.session_state["latest_test_upload"] = uploaded_file.getvalue()
            except Exception as exc:
                st.error(f"Prediction error: {exc}")
                st.stop()

        results = st.session_state.get("latest_test_results")
        same_upload = st.session_state.get("latest_test_upload") == uploaded_file.getvalue()

        if results is not None and same_upload:
            predictions = results["Predicted_Demand_MW"].to_numpy(dtype=float)
            if len(predictions) != len(data):
                st.error("Previous predictions do not match the current input rows.")
                st.stop()

            # Derive labels from existing model inputs; no invented dates or zones.
            zone_cols = [c for c in feature_names if c.startswith("Zone_")]
            zone_flags = X_input[zone_cols].eq(1) if zone_cols else pd.DataFrame(index=data.index)
            if zone_cols:
                valid_zones = zone_flags.sum(axis=1).eq(1)
                zone_labels = zone_flags.idxmax(axis=1).str.replace("Zone_", "", regex=False)
                zone_labels = zone_labels.where(valid_zones, "Unspecified")
            else:
                zone_labels = pd.Series(["Unspecified"] * n_rows, index=data.index)
            zone_labels = zone_labels.reset_index(drop=True)

            display = results.copy()
            display["Zone"] = zone_labels
            if "Hour" in data.columns:
                display["Hour"] = X_input["Hour"].to_numpy()
            if "Hour_End_EST" in data.columns:
                display["Hour_End_EST"] = data["Hour_End_EST"].to_numpy()

            actual_col = next(
                (c for c in ("Actual_Demand_MW", "Demand_MW", "Actual_MW") if c in data.columns),
                None,
            )
            if actual_col:
                display["Actual_Demand_MW"] = pd.to_numeric(
                    data[actual_col], errors="coerce"
                ).to_numpy()

            zone_values = display["Zone"].unique().tolist()
            zone_name = zone_values[0] if len(zone_values) == 1 else "Multiple zones"
            a, b, c = st.columns(3)
            a.metric("Observations predicted", f"{len(predictions):,}")
            b.metric("Zone(s)", zone_name)
            c.metric("Average predicted demand", f"{predictions.mean():,.1f} MW")

            # A temporal line is appropriate only when timestamps or one zone's
            # non-repeated hour labels are available; otherwise use bars.
            graph_data = display.copy()
            x_col = "Observation"
            x_title = "Input observation"
            temporal = False
            if "Hour_End_EST" in display.columns and len(zone_values) == 1:
                parsed_times = pd.to_datetime(display["Hour_End_EST"], errors="coerce")
                if parsed_times.notna().all() and parsed_times.is_unique:
                    graph_data["TargetTime"] = parsed_times
                    graph_data = graph_data.sort_values("TargetTime")
                    x_col = "TargetTime"
                    x_title = "Target date and hour"
                    temporal = True
            if not temporal and len(zone_values) == 1 and "Hour" in graph_data.columns:
                hrs = graph_data["Hour"]
                if hrs.notna().all() and hrs.between(1, 24).all() and hrs.is_unique:
                    graph_data = graph_data.sort_values("Hour")
                    x_col = "Hour"
                    x_title = "Hour ending (1–24)"
                    temporal = True

            fig = go.Figure()
            plot_type = go.Scatter if temporal else go.Bar
            if temporal:
                fig.add_trace(go.Scatter(
                    x=graph_data[x_col], y=graph_data["Predicted_Demand_MW"],
                    mode="lines+markers", name="Random Forest prediction",
                    line=dict(color="#2563EB", width=3),
                    marker=dict(color="#0D9488", size=9),
                    hovertemplate="%{x}<br>Predicted: %{y:,.1f} MW<extra></extra>",
                ))
            else:
                fig.add_trace(go.Bar(
                    x=graph_data[x_col].astype(str),
                    y=graph_data["Predicted_Demand_MW"],
                    name="Random Forest prediction", marker_color="#2563EB",
                    hovertemplate="Observation %{x}<br>Predicted: %{y:,.1f} MW<extra></extra>",
                ))

            if "Actual_Demand_MW" in graph_data.columns:
                actual = graph_data["Actual_Demand_MW"]
                if actual.notna().any():
                    if temporal:
                        fig.add_trace(go.Scatter(
                            x=graph_data[x_col], y=actual,
                            mode="lines+markers", name="Actual demand",
                            line=dict(color="#EA580C", width=3),
                            marker=dict(size=7),
                            hovertemplate="%{x}<br>Actual: %{y:,.1f} MW<extra></extra>",
                        ))
                    else:
                        fig.add_trace(go.Bar(
                            x=graph_data[x_col].astype(str), y=actual,
                            name="Actual demand", marker_color="#EA580C",
                        ))

            fig.update_layout(
                title=f"{zone_name} — model test predictions",
                xaxis_title=x_title,
                yaxis_title="Electricity demand (MW)",
                hovermode="x unified", barmode="group",
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                font=dict(color="#173354"),
                title_font=dict(color="#0B2A50"),
                xaxis=dict(showgrid=True, gridcolor="#E6EFFA"),
                yaxis=dict(showgrid=True, gridcolor="#E6EFFA", zeroline=False),
                legend=dict(orientation="h", y=-0.25),
                margin=dict(l=20, r=20, t=65, b=65),
            )
            st.plotly_chart(fig, use_container_width=True)
            if not actual_col:
                st.caption(
                    "This chart shows model outputs only. The uploaded file "
                    "does not include actual demand, so it cannot measure accuracy. "
                    "It shows only the hours present in the uploaded sample."
                )
            elif "Actual_Demand_MW" in display.columns:
                mask = display["Actual_Demand_MW"].notna()
                if mask.any():
                    errors = (display.loc[mask, "Actual_Demand_MW"] -
                              display.loc[mask, "Predicted_Demand_MW"]).abs()
                    mae = errors.mean()
                    denom = display.loc[mask, "Actual_Demand_MW"].abs().sum()
                    wape = 100 * errors.sum() / denom if denom > 0 else np.nan
                    acc_a, acc_b = st.columns(2)
                    acc_a.metric("MAE (uploaded rows)", f"{mae:,.2f} MW")
                    acc_b.metric("WAPE (uploaded rows)",
                                 f"{wape:.2f}%" if np.isfinite(wape) else "N/A")
                    st.caption(
                        "Accuracy is calculated only for the uploaded rows with "
                        "actual demand. Small samples do not establish overall model reliability."
                    )

            # Optional comparison to the team's saved Colab reference outputs.
            expected_path = ARTIFACTS / "sample_predictions.csv"
            if expected_path.exists():
                expected = pd.read_csv(expected_path)
                if "Expected_Prediction_MW" in expected.columns:
                    expected_values = expected["Expected_Prediction_MW"].to_numpy()
                    if len(expected_values) == len(predictions):
                        if np.allclose(predictions, expected_values, rtol=0, atol=1e-6):
                            st.success("Predictions match the saved Google Colab outputs!")
                        else:
                            st.warning("Predictions differ from saved Google Colab outputs.")
                    else:
                        st.info("Colab reference file has a different row count; comparison skipped.")

            with st.expander("View detailed prediction table", expanded=False):
                st.dataframe(display, use_container_width=True, hide_index=True)

            st.download_button(
                "Download test predictions",
                data=display.to_csv(index=False).encode("utf-8"),
                file_name="random_forest_test_predictions.csv",
                mime="text/csv",
            )
