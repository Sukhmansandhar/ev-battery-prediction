import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="EV Battery Health Predictor",
    page_icon="🔋",
    layout="wide"
)

# Load trained models
@st.cache_resource
def load_models():
    reg_model = joblib.load(
        "soh_regression_model_compressed.joblib"
    )
    clf_model = joblib.load(
        "degradation_classifier_compressed.joblib"
    )

    with open("model_config.json", "r") as file:
        config = json.load(file)

    return reg_model, clf_model, config


try:
    reg_model, clf_model, config = load_models()
except Exception as error:
    st.error("Unable to load the trained models.")
    st.exception(error)
    st.stop()

# Project heading
st.title("🔋 EV Battery Health Predictor")
st.subheader("Machine Learning Based Battery Analysis")

st.info(
    "Enter battery details to predict State of Health "
    "and the degradation category."
)

st.divider()

# Battery inputs
st.subheader("⚙️ Enter Battery Details")

with st.form("battery_form"):
    col1, col2 = st.columns(2)

    with col1:
        cycle_count = st.number_input(
            "Cycle Count",
            min_value=0,
            value=500
        )

        temperature = st.number_input(
            "Average Ambient Temperature (°C)",
            value=25.0
        )

    with col2:
        voltage = st.number_input(
            "Average Voltage (V)",
            min_value=0.0,
            value=3.7
        )

        battery_age = st.number_input(
            "Battery Age (Days)",
            min_value=0,
            value=365
        )

    submitted = st.form_submit_button(
        "🔍 Predict Battery Health"
    )

# Prediction results
if submitted:
    try:
        input_values = {
            "cycle_count": cycle_count,
            "avg_ambient_temp_c": temperature,
            "voltage_avg_v": voltage,
            "battery_age_days": battery_age
        }

        features = config["features"]

        input_data = pd.DataFrame(
            [[input_values[name] for name in features]],
            columns=features
        )

        health = float(reg_model.predict(input_data)[0])
        category = clf_model.predict(input_data)[0]

        st.divider()
        st.subheader("📊 Prediction Results")

        result1, result2 = st.columns(2)

        with result1:
            st.metric(
                "Predicted State of Health",
                f"{health:.2f}%"
            )

        with result2:
            st.metric(
                "Degradation Category",
                str(category)
            )

        st.progress(
            min(max(health / 100, 0.0), 1.0)
        )
       
        st.caption(
            "These are model predictions, not a physical "
            "battery test or a guarantee of performance."
        )

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)

st.divider()
st.caption("EV Battery Health Predictor | Machine Learning Project")
