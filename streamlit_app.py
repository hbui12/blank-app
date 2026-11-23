"""Ontario Regional Demand Intelligence – Streamlit app.

Two views:
  1. Explore saved 2025 Random Forest predictions interactively.
  2. Test the trained Random Forest with an uploaded 41-feature CSV.

Expected repo structure:
  streamlit_app.py
  forecast_dashboard.py
  requirements.txt
  data/M5_Combined_Forecasts_2025.csv
  artifacts/random_forest_41_compressed.joblib  # user's existing file
  artifacts/feature_names.json                  # user's existing file
  artifacts/sample_predictions.csv              # optional
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from forecast_dashboard import render_forecast_dashboard

st.set_page_config(
    page_title="Ontario Regional Demand Intelligence",
    layout="wide",
)

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
        "Model test: upload a historical dataset containing the exact 41 "
        "features used to train your team's Random Forest."
    )

    ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

    @st.cache_resource
    def load_model():
        model_path = ARTIFACTS / "random_forest_41_compressed.joblib"
        features_path = ARTIFACTS / "feature_names.json"

        # Only load the model saved by your team. Never accept arbitrary
        # uploaded joblib/pickle files from visitors.
        trained_model = joblib.load(model_path)
        with features_path.open("r", encoding="utf-8") as file:
            feature_names = json.load(file)["feature_names"]

        if len(feature_names) != 41:
            raise ValueError("Expected 41 input features.")
        if trained_model.n_features_in_ != 41:
            raise ValueError("Model does not have 41 features.")
        if hasattr(trained_model, "feature_names_in_"):
            if list(trained_model.feature_names_in_) != feature_names:
                raise ValueError("Feature names or order do not match.")
        return trained_model, feature_names

    try:
        model, feature_names = load_model()
        st.success("Random Forest model loaded successfully!")
    except Exception as error:
        st.error(f"Model loading error: {error}")
        st.stop()

    st.subheader("1. Upload historical input data")
    uploaded_file = st.file_uploader(
        "Upload sample_inputs.csv (41 features)", type=["csv"]
    )

    if uploaded_file is None:
        st.info("Upload your sample_inputs.csv file to begin testing.")
    else:
        data = pd.read_csv(uploaded_file)
        st.write("Number of rows:", len(data))
        st.write("Number of columns:", len(data.columns))
        st.dataframe(data.head())

        missing = [col for col in feature_names if col not in data.columns]
        unexpected = [col for col in data.columns if col not in feature_names]
        if missing or unexpected:
            st.error(
                f"Input columns do not match. Missing: {missing}. "
                f"Unexpected: {unexpected}"
            )
            st.stop()

        X_input = data[feature_names].apply(pd.to_numeric, errors="coerce")
        if X_input.empty:
            st.error("The CSV has no rows.")
            st.stop()
        if X_input.isna().any().any():
            st.error("Some inputs are missing or not numeric. Check your CSV.")
            st.stop()

        st.success("All 41 model features verified!")
        st.subheader("2. Generate demand predictions")

        if st.button("Generate prediction", type="primary"):
            predictions = model.predict(X_input)
            results = pd.DataFrame({
                "Observation": range(1, len(predictions) + 1),
                "Predicted_Demand_MW": predictions,
            })
            st.session_state["latest_test_results"] = results
            st.session_state["latest_test_upload"] = uploaded_file.getvalue()

        results = st.session_state.get("latest_test_results")
        if results is not None and st.session_state.get("latest_test_upload") == uploaded_file.getvalue():
            predictions = results["Predicted_Demand_MW"].to_numpy()
            st.success("Predictions generated successfully!")
            st.metric("First historical test prediction", f"{predictions[0]:,.2f} MW")
            st.dataframe(results, use_container_width=True, hide_index=True)

            expected_path = ARTIFACTS / "sample_predictions.csv"
            if expected_path.exists():
                expected = pd.read_csv(expected_path)
                if "Expected_Prediction_MW" in expected.columns:
                    expected_values = expected["Expected_Prediction_MW"].to_numpy()
                    if len(expected_values) == len(predictions):
                        if np.allclose(predictions, expected_values, rtol=0, atol=1e-6):
                            st.success("All predictions match the saved Google Colab results!")
                        else:
                            st.warning("Predictions differ from saved Google Colab results.")
                    else:
                        st.info("Expected predictions row count differs; comparison skipped.")

            st.download_button(
                "Download predictions",
                data=results.to_csv(index=False),
                file_name="hydro_one_predictions.csv",
                mime="text/csv",
            )
