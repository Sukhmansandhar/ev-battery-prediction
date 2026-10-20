from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache
import joblib
import json
import pandas as pd

app = FastAPI(title="EV Battery Prediction API")


class BatteryInput(BaseModel):
    cycle_count: int
    avg_ambient_temp_c: float
    voltage_avg_v: float
    battery_age_days: int


@lru_cache(maxsize=1)
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


@app.get("/")
def home():
    return {"message": "EV Battery Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: BatteryInput):
    reg_model, clf_model, config = load_models()

    values = {
        "cycle_count": data.cycle_count,
        "avg_ambient_temp_c": data.avg_ambient_temp_c,
        "voltage_avg_v": data.voltage_avg_v,
        "battery_age_days": data.battery_age_days,
    }

    input_data = pd.DataFrame(
        [[values[name] for name in config["features"]]],
        columns=config["features"],
    )

    health_value = float(reg_model.predict(input_data)[0])
    category = str(clf_model.predict(input_data)[0])

    return {
        "state_of_health": round(health_value, 2),
        "degradation_category": category,
    }
