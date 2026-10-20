from pathlib import Path
from functools import lru_cache
import json
import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="EV Battery Prediction API")

BASE_DIR = Path(__file__).resolve().parent


class BatteryInput(BaseModel):
    cycle_count: int
    avg_ambient_temp_c: float
    voltage_avg_v: float
    battery_age_days: int


@lru_cache(maxsize=1)
def load_models():
    reg_model = joblib.load(
        BASE_DIR / "soh_regression_model_compressed.joblib"
    )
    clf_model = joblib.load(
        BASE_DIR / "degradation_classifier_compressed.joblib"
    )

    with open(BASE_DIR / "model_config.json", "r") as file:
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

    features = config["features"]
    input_data = np.array(
        [[values[name] for name in features]],
        dtype=float,
    )

    health_value = float(reg_model.predict(input_data)[0])
    category = str(clf_model.predict(input_data)[0])

    return {
        "state_of_health": round(health_value, 2),
        "degradation_category": category,
    }
