import streamlit as st
import requests

st.set_page_config(
    page_title="EV Battery Health Predictor",
    page_icon="🔋",
    layout="wide"
)

API_URL = "https://ev-battery-api-n313.onrender.com/predict"

st.title("🔋 EV Battery Health Predictor")
st.subheader("Machine Learning Based Battery Analysis")

st.info(
    "Enter battery details to predict State of Health "
    "and the degradation category."
)

st.divider()
st.subheader("⚙️ Enter Battery Details")

with st.form("battery_form"):
    col1, col2 = st.columns(2)

    with col1:
        cycle_count = st.number_input(
            "Cycle Count", min_value=0, value=500
        )
        temperature = st.number_input(
            "Average Ambient Temperature (°C)",
            value=25.0
        )

    with col2:
        voltage = st.number_input(
            "Average Voltage (V)",
            min_value=0.0, value=3.7
        )
        battery_age = st.number_input(
            "Battery Age (Days)", min_value=0, value=365
        )

    submitted = st.form_submit_button(
        "🔍 Predict Battery Health"
    )

if submitted:
    payload = {
        "cycle_count": int(cycle_count),
        "avg_ambient_temp_c": float(temperature),
        "voltage_avg_v": float(voltage),
        "battery_age_days": int(battery_age)
    }

    try:
        with st.spinner("Analyzing battery..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=120
            )

        if response.status_code == 200:
            result = response.json()
            health = float(result["state_of_health"])
            category = str(result["degradation_category"])

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
                    category
                )

            st.progress(min(max(health / 100, 0.0), 1.0))

            st.caption(
                "These are model predictions, not a physical "
                "battery test or a guarantee of performance."
            )

        else:
            st.error(f"API returned HTTP {response.status_code}")
            st.code(response.text)

    except requests.exceptions.RequestException as error:
        st.error("Could not connect to the Render API.")
        st.caption(str(error))

st.divider()
st.caption("EV Battery Health Predictor | Machine Learning Project")
      
