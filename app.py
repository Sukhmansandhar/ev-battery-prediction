import streamlit as st
import pandas as pd
import joblib
import json

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="EV Battery Health Predictor",
    page_icon="🔋",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #f1f5f9;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
h1, h2, h3, p, label {
    color: #f1f5f9;
}
.hero {
    background: linear-gradient(135deg, #13283c, #124e52);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #246b70;
    margin-bottom: 25px;
}
.hero p {
    color: #cbd5e1;
}
div[data-testid="stMetric"] {
    background: #142235;
    border: 1px solid #294158;
    padding: 20px;
    border-radius: 14px;
}
div.stButton > button {
    background: #14b8a6;
    color: #071521;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: bold;
    width: 100%;
}
div.stButton > button:hover {
    background: #2dd4bf;
    color: #071521;
    border: none;
}
section[data-testid="stSidebar"] {
    background-color: #101b2b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
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
except Exception as e:
    st.error("The models could not be loaded.")
    st.code(str(e))
    st.stop()

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🔋 EV Battery Health Predictor</h1>
    <p>
        Machine Learning Based Battery Health Analysis
        and Degradation Prediction
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🔋 Battery AI")
    st.caption("Machine Learning Project")
    st.divider()
    st.markdown("### About this project")
    st.write(
        "This application uses trained Machine Learning "
        "models to estimate battery State of Health and "
        "predict a degradation category."
    )
    st.divider()
    st.caption("Developed for academic demonstration")

# ---------------- INPUT SECTION ----------------
st.subheader("⚙️ Battery Information")
st.write("Enter the battery details below to generate a prediction.")

with st.form("battery_form"):
    col1, col2 = st.columns(2)

    with col1:
        cycle_count = st.number_input(
            "🔄 Cycle Count",
            min_value=0,
            value=500,
            step=10
        )

        temperature = st.number_input(
            "🌡️ Average Ambient Temperature (°C)",
            value=25.0,
            step=1.0
        )

    with col2:
        voltage = st.number_input(
            "⚡ Average Voltage (V)",
            min_value=0.0,
            value=3.7,
            step=0.1
        )

        battery_age = st.number_input(
            "📅 Battery Age (Days)",
            min_value=0,
            value=365,
            step=30
        )

    submitted = st.form_submit_button(
        "🔍 Predict Battery Health"
    )

# ---------------- PREDICTIONS ----------------
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
                "Predicted Degradation Category",
                str(category)
            )

        st.markdown("### Battery Health Indicator")
        st.progress(min(max(health / 100, 0.0), 1.0))

        if health >= 80:
            st.success(
                "The model predicts a State of Health "
                "of 80% or higher."
            )
        else:
            st.warning(
                "The model predicts a State of Health "
                "below 80%. Review the result in context."
            )

        st.caption(
            "These are model predictions, not a physical "
            "battery test or a guarantee of actual performance."
        )

    except Exception as e:
        st.error("Prediction could not be completed.")
        st.code(str(e))

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "EV Battery Health Predictor | "
    "Powered by Machine Learning"
)
