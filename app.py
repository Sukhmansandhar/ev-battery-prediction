import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="EV Battery Predictor", page_icon="🔋")

st.title("🔋 EV Battery Health Predictor")
st.write("Predict battery health and degradation using Machine Learning.")

# Load the trained models
reg_model = joblib.load("soh_regression_model_compressed.joblib")
clf_model = joblib.load("degradation_classifier_compressed.joblib")

# Load model settings
with open("model_config.json", "r") as file:
    config = json.load(file)

st.subheader("Enter Battery Details")

cycle_count = st.number_input(
    "Cycle Count", min_value=0, value=500
)

temperature = st.number_input(
    "Average Ambient Temperature (°C)", value=25.0
)

voltage = st.number_input(
    "Average Voltage (V)", min_value=0.0, value=3.7
)

battery_age = st.number_input(
    "Battery Age (Days)", min_value=0, value=365
)

if st.button("Predict Battery Health"):
    input_data = pd.DataFrame(
        [[cycle_count, temperature, voltage, battery_age]],
        columns=config["features"]
    )

    health = reg_model.predict(input_data)[0]
    category = clf_model.predict(input_data)[0]

    st.subheader("Prediction Results")
    st.metric("Predicted State of Health", f"{health:.2f}%")
    st.success(f"Degradation Category: {category}")
