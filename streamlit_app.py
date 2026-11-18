
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# -----------------------------------
# 1. Application setup
# -----------------------------------
st.set_page_config(
    page_title="Ontario Regional Demand Intelligence",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Ontario Regional Demand Intelligence")

st.write(
    "Interactive decision-support prototype for "
    "24-hour-ahead electricity demand prediction "
    "across Ontario's 10 IESO zones."
)

st.info(
    "Current stage: Testing the trained Random Forest "
    "using historical observations."
)

# -----------------------------------
# Zone and Time Selection
# -----------------------------------
from datetime import date, time, timedelta

st.sidebar.header("Prediction Settings")

# All 10 Ontario IESO zones
zones = [
    "Northwest",
    "Northeast",
    "Ottawa",
    "East",
    "Toronto",
    "Essa",
    "Bruce",
    "Southwest",
    "Niagara",
    "West"
]

selected_zone = st.sidebar.selectbox(
    "Select Ontario electricity zone",
    zones,
    index=4
)

# Select the target date
target_date = st.sidebar.date_input(
    "Select prediction date",
    value=date.today() + timedelta(days=1)
)

# Select the target hour
target_time = st.sidebar.time_input(
    "Select prediction hour",
    value=time(14, 0),
    step=3600
)

# Display selections
st.subheader("Prediction Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Selected Zone", selected_zone)

with col2:
    st.metric("Target Date", str(target_date))

with col3:
    st.metric(
        "Target Hour",
        target_time.strftime("%I:%M %p")
    )

st.warning(
    "The zone and time controls are currently a "
    "preview. Predictions still use the 41 features "
    "in the uploaded CSV, not these selections."
)

# -----------------------------------
# 2. Load the saved Random Forest
# -----------------------------------
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

@st.cache_resource
def load_model():
    model_path = (
        ARTIFACTS / "random_forest_41_compressed.joblib"
    )

    features_path = ARTIFACTS / "feature_names.json"

    # Load only our team's trusted model
    trained_model = joblib.load(model_path)

    with open(features_path, "r") as f:
        feature_names = json.load(f)["feature_names"]

    # Confirm the model structure
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

# -----------------------------------
# 3. Upload historical input data
# -----------------------------------
st.subheader("1. Upload Input Data")

uploaded_file = st.file_uploader(
    "Upload sample_inputs.csv",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Number of rows:", len(data))
    st.write("Number of columns:", len(data.columns))

    st.dataframe(data.head())

    # Verify the expected input columns
    missing = [
        col for col in feature_names
        if col not in data.columns
    ]

    unexpected = [
        col for col in data.columns
        if col not in feature_names
    ]

    if missing or unexpected:
        st.error(
            f"Input columns do not match. "
            f"Missing: {missing}. "
            f"Unexpected: {unexpected}"
        )
        st.stop()

    # Use exactly the training feature order
    X_input = data[feature_names].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if X_input.isna().any().any():
        st.error(
            "Some inputs are missing or not numeric. "
            "Please check your CSV."
        )
        st.stop()

    st.success("All 41 model features verified!")

    # -----------------------------------
    # 4. Generate predictions
    # -----------------------------------
    st.subheader("2. Generate Demand Predictions")

    if st.button("Generate Prediction", type="primary"):

        predictions = model.predict(X_input)

        results = pd.DataFrame({
            "Observation": range(1, len(predictions) + 1),
            "Predicted_Demand_MW": predictions
        })

        st.success("Predictions generated successfully!")

        st.metric(
            "First historical test prediction",
            f"{predictions[0]:,.2f} MW"
        )

        st.dataframe(results)

        # -----------------------------------
        # 5. Compare with expected predictions
        # -----------------------------------
        expected_path = (
            ARTIFACTS / "sample_predictions.csv"
        )

        if expected_path.exists():

            expected = pd.read_csv(expected_path)
            expected_values = expected[
                "Expected_Prediction_MW"
            ].to_numpy()

            if len(expected_values) == len(predictions):

                match = np.allclose(
                    predictions,
                    expected_values,
                    rtol=0,
                    atol=1e-6
                )

                if match:
                    st.success(
                        "All predictions match Google Colab!"
                    )
                else:
                    st.warning(
                        "Predictions differ from the saved "
                        "Google Colab results."
                    )
            else:
                st.info(
                    "The expected predictions file has "
                    "a different number of rows. "
                    "Automatic comparison skipped."
                )

        # -----------------------------------
        # 6. Export predictions
        # -----------------------------------
        st.download_button(
            label="Download Predictions",
            data=results.to_csv(index=False),
            file_name="hydro_one_predictions.csv",
            mime="text/csv"
        )

else:
    st.info(
        "Upload your sample_inputs.csv file "
        "to begin testing."
    )
